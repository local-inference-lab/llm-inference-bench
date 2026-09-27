"""Is this container running the code of its release, or a modified copy?

Runtime images carry ``runtime-files.json.gz``: the SHA-256 of every file of
the Python environment, torch, triton and the LIL launcher and bench, written
at the end of the image build (so the build's own patches are the
reference). The same file is published with the release, and the site
compares its hash with the one reported here, so an edited reference is
detected as well.

This check finds edits, added and removed files, overlays mounted over the
code, and interpreter hooks (``PYTHONPATH``, ``LD_PRELOAD``, sitecustomize).
It keeps honest comparisons honest; it cannot stop deliberate forgery on the
user's own machine.
"""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import time
from pathlib import Path

REFERENCE = "/opt/venv/share/lil-runtime/runtime-files.json.gz"
TEXT_SUFFIXES = {".py", ".pyi", ".sh", ".yaml", ".yml", ".json", ".txt", ".cfg", ".toml", ".ini", ".cu", ".cuh", ".h", ".cpp", ""}
MAX_TEXT_BYTES = 256 * 1024
MAX_TOTAL_TEXT = 2 * 1024 * 1024
MAX_LISTED = 500
HOOK_ENV = ("PYTHONPATH", "PYTHONSTARTUP", "PYTHONUSERBASE", "PYTHONHOME", "PYTHONEXECUTABLE", "LD_PRELOAD", "LD_AUDIT")
USER_SITE = "/root/.local/lib/python3.12/site-packages"  # imported before the image's packages


def skip(path: str) -> bool:
    """Bytecode caches are rebuilt by the interpreter and never compared."""
    return "/__pycache__/" in path or path.endswith((".pyc", ".pyo"))


def file_hash(path: str) -> str | None:
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError:
        return None
    return digest.hexdigest()


def scan(roots: list[str]) -> dict[str, str]:
    files: dict[str, str] = {}
    for root in roots:
        if os.path.isfile(root) and not os.path.islink(root):
            value = file_hash(root)
            if value:
                files[root] = value
            continue
        for directory, dirs, names in os.walk(root):
            dirs[:] = [d for d in dirs if d != "__pycache__"]
            for name in names:
                path = os.path.join(directory, name)
                if os.path.islink(path) or skip(path):
                    continue
                value = file_hash(path)
                if value:
                    files[path] = value
    return files


def load_reference(path: str = REFERENCE) -> tuple[dict | None, str | None]:
    try:
        raw = Path(path).read_bytes()
    except OSError:
        return None, None
    try:
        return json.loads(gzip.decompress(raw)), hashlib.sha256(raw).hexdigest()
    except (OSError, ValueError):
        return None, hashlib.sha256(raw).hexdigest()


def process_start(pid: int | None, proc_root: str = "/proc") -> float | None:
    """Wall-clock start time of a process (from /proc/<pid>/stat and uptime)."""
    if not pid:
        return None
    try:
        fields = Path(proc_root, str(pid), "stat").read_text().rsplit(")", 1)[1].split()
        ticks = int(fields[19])
        uptime = float(Path(proc_root, "uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        return None
    return time.time() - uptime + ticks / os.sysconf("SC_CLK_TCK")


CODE_PREFIXES = ("/opt/venv", "/opt/lil", "/usr/local/lib/python3", "/usr/local/bin", "/root/.local")


def code_mounts(roots: list[str], proc_root: str = "/proc") -> list[dict]:
    """Bind mounts that place other files over the image's code."""
    found = []
    try:
        lines = Path(proc_root, "self/mountinfo").read_text().splitlines()
    except OSError:
        return found
    prefixes = tuple(r.rstrip("/") for r in roots) + CODE_PREFIXES
    for line in lines:
        parts = line.split()
        if len(parts) < 5:
            continue
        mount_point = parts[4]
        if mount_point.startswith(prefixes):
            found.append({"mount_point": mount_point, "source": parts[3],
                          "fs": parts[parts.index("-") + 1] if "-" in parts else None})
    return found


def hooks(pid: int | None, proc_root: str = "/proc") -> dict:
    """Interpreter and loader hooks in the serving process environment."""
    found = {}
    if pid:
        try:
            raw = Path(proc_root, str(pid), "environ").read_bytes()
        except OSError:
            raw = b""
        for item in raw.split(b"\0"):
            key, sep, value = item.decode(errors="replace").partition("=")
            if sep and key in HOOK_ENV and value:
                found[key] = value
    return found


def text_of(path: str) -> str | None:
    if Path(path).suffix not in TEXT_SUFFIXES:
        return None
    try:
        if os.path.getsize(path) > MAX_TEXT_BYTES:
            return None
        data = Path(path).read_bytes()
    except OSError:
        return None
    if b"\0" in data:
        return None
    return data.decode("utf-8", errors="replace")


def packages(site_packages: str) -> dict[str, str]:
    import importlib.metadata as md
    try:
        return {d.metadata["Name"].lower(): d.version for d in md.distributions(path=[site_packages]) if d.metadata["Name"]}
    except Exception:  # noqa: BLE001
        return {}


def check(pid: int | None = None, reference_path: str = REFERENCE, proc_root: str = "/proc",
          user_site: str = USER_SITE) -> dict:
    started = time.time()
    reference, reference_sha = load_reference(reference_path)
    result: dict = {"reference": {"path": reference_path, "present": reference is not None, "sha256": reference_sha},
                    "mounts": [], "hooks": hooks(pid, proc_root), "changed": [], "added": [], "removed": []}
    if reference is None:
        result.update(status="unverified", reason="the image has no runtime file manifest (built before lil-bench integrity checks)")
        return result
    roots = reference.get("roots") or []
    expected: dict[str, str] = reference.get("files") or {}
    actual = scan(roots)
    user_site_files = scan([user_site]) if os.path.isdir(user_site) else {}
    server_start = process_start(pid, proc_root)
    text_budget = MAX_TOTAL_TEXT

    def describe(path: str, sha: str | None, expected_sha: str | None) -> dict:
        nonlocal text_budget
        item = {"path": path, "sha256": sha, "expected_sha256": expected_sha}
        try:
            stat = os.stat(path)
            item.update(size=stat.st_size, mtime=round(stat.st_mtime, 1))
            if server_start:
                item["after_server_start"] = stat.st_mtime > server_start
        except OSError:
            pass
        content = text_of(path) if len(result["changed"]) + len(result["added"]) < MAX_LISTED else None
        if content is not None and len(content) <= text_budget:
            item["content"] = content
            text_budget -= len(content)
        return item

    for path in sorted(set(expected) | set(actual)):
        want, have = expected.get(path), actual.get(path)
        if want == have:
            continue
        if want and have:
            result["changed"].append(describe(path, have, want))
        elif have:
            result["added"].append(describe(path, have, None))
        else:
            result["removed"].append({"path": path, "expected_sha256": want})
    for path, sha in sorted(user_site_files.items()):
        result["added"].append(describe(path, sha, None))
    site = next((r for r in roots if r.endswith("site-packages")), None)
    if site and reference.get("packages"):
        now = packages(site)
        result["packages"] = {name: {"expected": reference["packages"].get(name), "actual": now.get(name)}
                              for name in sorted(set(now) | set(reference["packages"]))
                              if now.get(name) != reference["packages"].get(name)}
    result["mounts"] = code_mounts(roots, proc_root)
    # The launcher itself sets hooks that point into the verified image
    # (for example LD_PRELOAD of the bundled NCCL); only foreign ones count.
    verified, foreign = {}, {}
    for key, value in result["hooks"].items():
        entries = [e for e in value.replace(" ", ":").split(":") if e]
        inside = all(e in expected and expected.get(e) == actual.get(e) or
                     (os.path.isdir(e) and any(e.rstrip("/").startswith(r.rstrip("/")) for r in roots if os.path.isdir(r)))
                     for e in entries)
        (verified if entries and inside else foreign)[key] = value
    result["hooks"], result["hooks_verified"] = foreign, verified
    modified = bool(result["changed"] or result["added"] or result["removed"] or result["mounts"]
                    or result["hooks"] or result.get("packages"))
    result.update(status="modified" if modified else "stock", files_checked=len(actual),
                  files_expected=len(expected), scan_seconds=round(time.time() - started, 1),
                  server_started=server_start, image_build=reference.get("created"))
    for key in ("changed", "added", "removed"):
        result[f"{key}_count"] = len(result[key])
        result[key] = result[key][:MAX_LISTED]
    return result


def summary_lines(result: dict, limit: int = 12) -> list[str]:
    lines = []
    for key, label in (("changed", "changed"), ("added", "added"), ("removed", "removed")):
        for item in result.get(key, [])[:limit]:
            late = " (after the server started)" if item.get("after_server_start") else ""
            lines.append(f"{label}: {item['path']}{late}")
    for mount in result.get("mounts", []):
        lines.append(f"mounted over the image: {mount['mount_point']}")
    for key, value in (result.get("hooks") or {}).items():
        lines.append(f"{key}={value}")
    for name, versions in (result.get("packages") or {}).items():
        lines.append(f"package {name}: {versions['expected'] or 'absent'} → {versions['actual'] or 'removed'}")
    return lines
