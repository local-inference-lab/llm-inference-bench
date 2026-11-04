"""Flare servers: the build, package and startup choices of a running Flare.

Flare images (``ghcr.io/local-inference-lab/flare``) contain only the
``flare`` binary, so ``lil-bench`` runs beside the container, on the host or
in another container with the Docker socket, and benchmarks the server over
its HTTP port. Everything else comes from what already exists:

* the container (Docker Engine API, or the docker CLI): the image reference,
  digest and OCI revision label (the Flare git revision), the serving
  command, its GPUs and mounts;
* the startup log (``docker logs`` since the container started, or
  ``--server-log`` for a server started outside Docker): the publisher and
  signed source of the package, the Hugging Face commit a hub package
  resolved to, the cache size and the collective profiles startup tuning
  selected;
* the package file, hashed so the site can match it to a published
  revision when nothing names one.

Startup tuning races the collective profiles a package prepared and logs one
uint32 per tuning group: the transport mode in the upper 16 bits and the
number of active CTAs in the lower 16 (FlashInfer b12x joint PCIe kernels).
All-reduce and column all-gather number their modes independently and the
log does not say which operation a value belongs to, so modes 0 and 1 keep
both readings and are shown by their raw value.
"""

from __future__ import annotations

import hashlib
import http.client
import ipaddress
import json
import os
import re
import shutil
import socket
import subprocess
import urllib.parse
from datetime import datetime
from pathlib import Path

from . import server

BINARY = "/usr/local/bin/flare"
DOCKER_SOCKET = "/var/run/docker.sock"
LOG_LIMIT = 16 * 1024 * 1024  # startup log bytes read at most (a terminal dashboard can follow it)
LOG_KEEP = 128 * 1024  # startup log bytes stored with the result
HASH_LIMIT = 2 * 1024**3  # a package file is hashed up to this size
ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]|\x1b\][^\x07\x1b]*(?:\x07|\x1b\\)")
HOOK_ENV = ("LD_PRELOAD", "LD_AUDIT", "LD_LIBRARY_PATH")
# Paths whose replacement changes the code that runs: the image keeps its binary
# in /usr/local/bin. (The NVIDIA runtime adds driver files under /usr/lib,
# /usr/bin and /etc/ld.so.conf.d to every GPU container; those are expected.)
CODE_PATHS = ("/usr/local", "/etc/ld.so.preload")

# FlashInfer b12x joint PCIe profiles: mode << 16 | active CTAs. All-reduce
# modes and column all-gather modes are numbered independently.
ALL_REDUCE_MODES = {
    0: "full", 1: "payload_rsag", 2: "bulk_push", 3: "bulk_pull", 4: "pull",
    5: "full_phased", 6: "payload_rsag_phased",
}
ALL_GATHER_MODES = {0: "barrier", 1: "word_exchange"}


# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

class DockerError(Exception):
    pass


class _UnixConnection(http.client.HTTPConnection):
    def __init__(self, path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self.unix_path = path

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        sock.connect(self.unix_path)
        self.sock = sock


class Docker:
    """Read-only access to containers: the Engine API on its Unix socket, else the docker CLI."""

    def __init__(self, socket_path: str | None = None, cli: str | None = None):
        self.socket_path = socket_path
        self.cli = cli

    @classmethod
    def open(cls) -> "Docker | None":
        host = os.environ.get("DOCKER_HOST", "")
        path = host[len("unix://"):] if host.startswith("unix://") else ("" if host else DOCKER_SOCKET)
        if path and os.access(path, os.R_OK | os.W_OK):
            return cls(socket_path=path)
        cli = shutil.which("docker")
        return cls(cli=cli) if cli else None

    def describe(self) -> str:
        return f"Docker API ({self.socket_path})" if self.socket_path else "docker CLI"

    def _get(self, path: str, timeout: float = 20.0):
        connection = _UnixConnection(self.socket_path, timeout)
        try:
            connection.request("GET", path, headers={"Host": "docker"})
            response = connection.getresponse()
        except OSError as error:
            connection.close()
            raise DockerError(f"{self.socket_path}: {error}") from error
        if response.status >= 400:
            try:
                detail = json.loads(response.read(65536) or b"{}").get("message")
            except ValueError:
                detail = None
            connection.close()
            raise DockerError(detail or f"HTTP {response.status} for {path}")
        return connection, response

    def _json(self, path: str):
        connection, response = self._get(path)
        try:
            return json.loads(response.read())
        except ValueError as error:
            raise DockerError(f"invalid JSON from {path}") from error
        except (OSError, http.client.HTTPException) as error:
            raise DockerError(f"{path}: {error}") from error
        finally:
            connection.close()

    def _cli(self, *args: str, timeout: float = 30.0) -> str:
        try:
            proc = subprocess.run([self.cli, *args], capture_output=True, text=True, timeout=timeout, check=False)
        except (OSError, subprocess.TimeoutExpired) as error:
            raise DockerError(f"docker {args[0]}: {error}") from error
        if proc.returncode != 0:
            raise DockerError((proc.stderr or proc.stdout).strip()[:500] or f"docker {args[0]} failed")
        return proc.stdout

    def inspect(self, name: str) -> dict:
        if self.socket_path:
            return self._json(f"/containers/{urllib.parse.quote(name, safe='')}/json")
        found = json.loads(self._cli("inspect", "--type", "container", name) or "[]")
        if not found:
            raise DockerError(f"no container {name}")
        return found[0]

    def image(self, reference: str) -> dict:
        """``docker image inspect``; the API also lists the platform manifests (containerd image store)."""
        if self.socket_path:
            return self._json(f"/images/{urllib.parse.quote(reference, safe='')}/json?manifests=1")
        found = json.loads(self._cli("image", "inspect", reference) or "[]")
        if not found:
            raise DockerError(f"no image {reference}")
        return found[0]

    def running(self) -> list[dict]:
        """Every running container, as ``inspect`` returns it."""
        if self.socket_path:
            ids = [item["Id"] for item in self._json("/containers/json")]
        else:
            ids = self._cli("ps", "-q", "--no-trunc").split()
        found = []
        for container_id in ids:
            try:
                found.append(self.inspect(container_id))
            except DockerError:
                continue  # stopped meanwhile
        return found

    def changes(self, container_id: str) -> list[dict] | None:
        """Files the container changed in its writable layer (``docker diff``)."""
        if self.socket_path:
            return self._json(f"/containers/{container_id}/changes") or []
        kinds = {"C": 0, "A": 1, "D": 2}
        return [{"Path": line[2:], "Kind": kinds.get(line[:1])}
                for line in self._cli("diff", container_id).splitlines() if len(line) > 2]

    def logs(self, container: dict, done=None, limit: int = LOG_LIMIT) -> str:
        """The container's output since it started; reading stops once ``done(text)`` holds."""
        since = _unix_time(((container.get("State") or {}).get("StartedAt")))
        since_arg = str(int(since) - 1) if since else "0"
        tty = bool((container.get("Config") or {}).get("Tty"))
        chunks: list[bytes] = []
        size = 0

        def finished(chunk: bytes) -> bool:
            # ``done`` sees the last two chunks: enough for a line split across them.
            nonlocal size
            chunks.append(chunk)
            size += len(chunk)
            return size >= limit or bool(done and done(b"".join(chunks[-2:]).decode(errors="replace")))

        if self.socket_path:
            connection, response = self._get(
                f"/containers/{container['Id']}/logs?stdout=1&stderr=1&since={since_arg}", timeout=60)
            try:
                if tty:
                    while chunk := response.read(65536):
                        if finished(chunk):
                            break
                else:
                    while (header := _read_exact(response, 8)) and len(header) == 8:
                        if finished(_read_exact(response, int.from_bytes(header[4:8], "big"))):
                            break
            except (OSError, http.client.HTTPException) as error:
                if not chunks:
                    raise DockerError(f"docker logs: {error}") from error
            finally:
                connection.close()
        else:
            try:
                proc = subprocess.Popen([self.cli, "logs", "--since", since_arg, container["Id"]],
                                        stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            except OSError as error:
                raise DockerError(f"docker logs: {error}") from error
            try:
                while chunk := proc.stdout.read(65536):
                    if finished(chunk):
                        break
            finally:
                proc.kill()
                proc.wait(timeout=10)
        return b"".join(chunks)[:limit].decode(errors="replace")


def _read_exact(response, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = response.read(size - len(data))
        if not chunk:
            break
        data += chunk
    return data


def _unix_time(text: str | None) -> float | None:
    """Docker's RFC 3339 times carry nanoseconds; Python parses microseconds."""
    match = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)(\.\d+)?(Z|[+-]\d\d:\d\d)?$", text or "")
    if not match or match.group(1).startswith("0001"):
        return None
    zone = match.group(3) or "Z"
    # Docker trims trailing zeros; Python 3.10 parses exactly six fraction digits.
    fraction = ((match.group(2) or ".")[1:] + "000000")[:6]
    stamp = datetime.fromisoformat(f"{match.group(1)}.{fraction}" + ("+00:00" if zone == "Z" else zone))
    return stamp.timestamp()


# ---------------------------------------------------------------------------
# Which server: container or process, by port
# ---------------------------------------------------------------------------

def run_options(argv: list[str]) -> dict:
    """Package and options of ``flare run`` (the same argument forms as ``vllm serve``)."""
    return server.parse_serve_args(list(argv), command="run")


def is_flare(container: dict) -> bool:
    """A container whose process is ``flare run``."""
    args = container.get("Args") or []
    if os.path.basename(container.get("Path") or "") == "flare":
        return args[:1] == ["run"]
    image = ((container.get("Config") or {}).get("Image") or "").split("@")[0]
    return image.rsplit(":", 1)[0].endswith("/flare") and "run" in args


def flare_port(options: dict) -> int:
    value = options.get("port")
    return int(value) if isinstance(value, str) and value.isdigit() else 8000


def container_ports(container: dict) -> list[int]:
    """Host ports on which the container's Flare HTTP server listens."""
    port = flare_port(run_options([container.get("Path") or "", *(container.get("Args") or [])])["options"])
    if (container.get("HostConfig") or {}).get("NetworkMode") == "host":
        return [port]
    bindings = ((container.get("NetworkSettings") or {}).get("Ports") or {}).get(f"{port}/tcp") or []
    return sorted({int(b["HostPort"]) for b in bindings if str(b.get("HostPort", "")).isdigit()})


def local_host(host: str | None) -> bool:
    """A URL host that names this machine, so its port identifies a local server."""
    if not host or host == "localhost":
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return host in (socket.gethostname(), socket.getfqdn())
    if address.is_loopback or address.is_unspecified:
        return True
    try:
        return host in socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        return False


def url_port(base_url: str) -> tuple[str | None, int]:
    parts = urllib.parse.urlsplit(base_url)
    return parts.hostname, parts.port or (443 if parts.scheme == "https" else 80)


def find_container(docker: Docker, base_url: str | None) -> tuple[dict | None, str | None]:
    """The running Flare container serving ``base_url`` (any when None), or why there is none."""
    try:
        candidates = [c for c in docker.running() if is_flare(c)]
    except DockerError as error:
        return None, f"cannot list containers: {error}"
    if base_url:
        host, port = url_port(base_url)
        if not local_host(host):
            return None, f"{host} is not this machine; pass --container to name the Flare container"
        candidates = [c for c in candidates if port in container_ports(c)]
    if len(candidates) == 1:
        return candidates[0], None
    names = ", ".join(c.get("Name", "").lstrip("/") for c in candidates)
    if candidates:
        return None, f"several Flare containers match ({names}); pass --container"
    return None, "no running Flare container" + (f" publishes port {url_port(base_url)[1]}" if base_url else "")


def container_url(container: dict) -> str | None:
    ports = container_ports(container)
    return f"http://127.0.0.1:{ports[0]}" if ports else None


def find_processes(base_url: str | None, proc_root: str = "/proc") -> list[dict]:
    """``flare run`` processes, on the URL's port when given, lowest PID first."""
    port = url_port(base_url)[1] if base_url else None
    found = []
    for pid in sorted((p for p in os.listdir(proc_root) if p.isdigit()), key=int):
        argv = server.read_cmdline(pid, proc_root)
        if len(argv) < 2 or os.path.basename(argv[0]) != "flare" or argv[1] != "run":
            continue
        if port is None or flare_port(run_options(argv)["options"]) == port:
            found.append({"pid": int(pid), "argv": argv})
    return found


def find_process(base_url: str | None, proc_root: str = "/proc") -> dict | None:
    """The ``flare run`` process outside Docker on the URL's port; None unless exactly one matches."""
    found = find_processes(base_url, proc_root)
    return found[0] if len(found) == 1 else None


# ---------------------------------------------------------------------------
# Startup log
# ---------------------------------------------------------------------------

STEP = re.compile(r"^\[\s*(OK|FAILED|PENDING)\s*\]\s(.+?)\.(?:\s{2}(.*?))?\s+(\d+(?:\.\d+)?)s$")
# A terminal redraws the row in place, so the line may start after a carriage return.
START_DONE = re.compile(r"(?m)(?:^|\r)\[\s*OK\s*\] Start server\.")
TUNING = re.compile(r"^Startup tuning: (\d+) graphs? completed in ([\d.]+)s$")
CHOICES = re.compile(r"^Collective choices · device (\d+)")
GRAPH = re.compile(r"^(\S+) · (\w+) b=(\d+) t=(\d+) rows=(\d+) drafts=(\d+) prefill=(\d+) · (.+)$")
GROUP = re.compile(r"^(.+): (\d+) groups?$")
SELECTOR = re.compile(r"0x[0-9a-f]{8}|raw=\[[^\]]*\]")
TRUST_KEYS = {"Publisher": "publisher", "Identity": "identity", "Signature": "signature", "Source": "source",
              "License": "license", "Stored data": "stored_data"}


def clean_lines(text: str) -> list[str]:
    """Log lines without colors; a redrawn terminal row keeps its last state."""
    lines = []
    for raw in ANSI.sub("", text.replace("\r\n", "\n")).split("\n"):
        line = raw.split("\r")[-1].rstrip()
        # The framed trust prompt of an interactive terminal: │ text │
        if line.startswith("│"):
            line = line.strip("│").strip()
        lines.append(line)
    return lines


def startup_block(lines: list[str]) -> tuple[list[str], list[str]]:
    """This start's lines (up to the server start) and the lines of earlier starts.

    Docker keeps the output of earlier starts of the same container; the
    newest start is the one after the last completed one.
    """
    ends = [i for i, line in enumerate(lines) if START_DONE.match(line)]
    end = ends[-1] + 1 if ends else len(lines)
    begin = ends[-2] + 1 if len(ends) > 1 else 0
    return lines[begin:end], lines[:begin]


def parse_trust(lines: list[str]) -> dict:
    """The publisher disclosure printed before a signing key is first accepted."""
    found: dict = {}
    for index, line in enumerate(lines):
        if line.strip() == "Signing key:" and index + 1 < len(lines) and re.fullmatch(r"[0-9a-f]{64}", lines[index + 1].strip()):
            found["fingerprint"] = lines[index + 1].strip()
            continue
        key, sep, value = line.partition(":")
        name = TRUST_KEYS.get(key.strip())
        if not sep or not name or not value.strip():
            continue
        found[name] = value.strip()
        if name == "license" and index + 1 < len(lines) and lines[index + 1].strip().startswith("https://"):
            found["license_link"] = lines[index + 1].strip()
    publisher = re.fullmatch(r"(.+) \(([\w.-]+)\)", found.get("publisher", ""))
    if publisher:
        found["name"], found["namespace"] = publisher.groups()
    if "identity" in found:
        found["verified"] = found["identity"].startswith("Verified")
        found["development"] = found["identity"].startswith("Local development")
    return found


def parse_collectives(lines: list[str]) -> dict | None:
    """Startup tuning and the selector tables of every device block."""
    tuning, devices, device = None, [], None
    for line in lines:
        text = line.strip()
        match = TUNING.match(text)
        if match:
            tuning = {"graphs": int(match.group(1)), "seconds": float(match.group(2))}
            continue
        match = CHOICES.match(text)
        if match:
            device = {"device": int(match.group(1)), "groups": [], "graphs": []}
            devices.append(device)
            continue
        if device is None:
            continue
        match = GRAPH.match(text)
        if match:
            name, kind, *numbers, selected = match.groups()
            batch, tokens, rows, drafts, prefill = (int(n) for n in numbers)
            device["graphs"].append({"name": name, "kind": kind, "batch": batch, "tokens": tokens, "rows": rows,
                                     "drafts": drafts, "prefill": prefill,
                                     "selectors": SELECTOR.findall(selected)})
            continue
        if not device["groups"] and not device["graphs"] and SELECTOR.search(text):
            for item in text.split(" · "):
                group = GROUP.match(item)
                if group:
                    device["groups"].append({"selectors": SELECTOR.findall(group.group(1)),
                                             "count": int(group.group(2))})
            continue
        device = None  # the block ends at the first other line
    if tuning is None and not devices:
        return None
    return {"tuning": tuning, "devices": devices}


def decode_selector(raw: str) -> dict:
    """One startup tuning value: transport mode, active CTAs and what the mode can mean."""
    if not re.fullmatch(r"0x[0-9a-f]{8}", raw):
        return {"raw": raw, "mode": None, "ctas": None, "name": None, "ambiguous": False, "known": False}
    value = int(raw, 16)
    mode, ctas = value >> 16, value & 0xFFFF
    reduce, gather = ALL_REDUCE_MODES.get(mode), ALL_GATHER_MODES.get(mode)
    item = {"raw": raw, "mode": mode, "ctas": ctas, "all_reduce": reduce, "all_gather": gather,
            "ambiguous": bool(reduce and gather), "known": bool(reduce or gather)}
    item["name"] = None if item["ambiguous"] else (reduce or gather)
    item["operation"] = None if item["ambiguous"] else ("all_reduce" if reduce else "all_gather" if gather else None)
    return item


def selector_label(item: dict) -> str:
    if item.get("name"):
        return f"{item['name']} ×{item['ctas']}"
    if item.get("ambiguous"):
        return f"{item['raw']} ({item['all_reduce']} or {item['all_gather']} ×{item['ctas']})"
    return item["raw"]


def decoded_profiles(collectives: dict | None) -> dict:
    values = {v for d in (collectives or {}).get("devices") or [] for g in d["groups"] + d["graphs"] for v in g["selectors"]}
    return {raw: decode_selector(raw) for raw in sorted(values)}


def step_details(lines: list[str]) -> dict:
    """Completed startup steps by name: their detail text and seconds."""
    steps = {}
    for line in lines:
        match = STEP.match(line.strip())
        if match and match.group(1) != "PENDING":
            steps[match.group(2)] = {"status": match.group(1), "detail": match.group(3) or "",
                                     "seconds": float(match.group(4))}
    return steps


def parse_log(text: str) -> dict:
    """Everything lil-bench reads from a Flare startup log (the newest start in it)."""
    lines = clean_lines(text)
    block, earlier = startup_block(lines)
    steps = step_details(block)
    result: dict = {"steps": [{"name": name, **step} for name, step in steps.items()],
                    "started": bool(steps.get("Start server", {}).get("status") == "OK"),
                    "earlier_starts": sum(1 for line in earlier if START_DONE.match(line))}
    trust = parse_trust(block)
    if not trust.get("publisher") and earlier:
        trust = parse_trust(earlier)
        if trust:
            trust["from_earlier_start"] = True
    if trust:
        result["trust"] = trust
    hub = {}
    for line in block:
        match = re.match(r"^(?:Package source: |Checking model origin: |Updating model from )([\w.-]+/[\w.-]+)@(\S+)$", line)
        if match:
            hub.setdefault("repository", match.group(1))
            hub.setdefault("revision", match.group(2))
            continue
        match = re.match(r"^Resolved package: (?:https://huggingface\.co/)?([\w.-]+/[\w.-]+)@([0-9a-f]{40}) / (\S+)$", line)
        if match:
            hub.update(repository=match.group(1), commit=match.group(2), filename=match.group(3))
            continue
        match = re.match(r"^Warning: downloaded from (\S+); the publisher signed (\S+) as its source\.$", line)
        if match:
            hub.update(downloaded_from=match.group(1), signed_source=match.group(2))
    if hub:
        result["hub"] = hub
    detail = lambda name: (steps.get(name) or {}).get("detail") or ""  # noqa: E731
    if re.match(r"^[\w.-]+/[\w.-]+$", detail("Read model configuration")):
        result["model"] = detail("Read model configuration")
    match = re.match(r"^(\d+) ranks?", detail("Prepare model"))
    if match:
        result["ranks"] = int(match.group(1))
    match = re.search(r"concurrency (\d+)", detail("Prepare execution"))
    if match:
        result["concurrency"] = int(match.group(1))
    match = re.search(r"(\d+) KV tokens", detail("Allocate cache"))
    if match:
        result["kv_tokens"] = int(match.group(1))
    match = re.search(r"(\d+) tokens/request", detail("Start server"))
    if match:
        result["tokens_per_request"] = int(match.group(1))
    gpus = re.findall(r"GPU (\d+) · (SM\d+)", detail("Initialize GPUs"))
    if gpus:
        result["gpus"] = [{"ordinal": int(ordinal), "architecture": arch} for ordinal, arch in gpus]
    collectives = parse_collectives(block)
    if collectives:
        result["collectives"] = collectives
    kept = "\n".join(startup_lines(block))
    result["text"] = kept[:LOG_KEEP]
    result["truncated"] = len(kept) > LOG_KEEP
    return result


# Lines of a start that lil-bench recognizes; others (paths, unrelated terminal output) stay local.
KNOWN_LINE = re.compile(
    r"^(?:\[\s*(?:OK|FAILED)\s*\] |Signing key acceptance$|Signing key accepted for |Signing key:$|[0-9a-f]{64}$"
    r"|(?:Publisher|Identity|Signature|Source|License|Stored data): |Startup tuning: |Collective choices · "
    r"|Package source: |Resolved package: |Checking model origin: |Updating model from |Warning: downloaded from )")


def startup_lines(block: list[str]) -> list[str]:
    """The step results, publisher disclosure, package and startup tuning lines of one start."""
    kept, previous = [], ""
    for line in block:
        text = line.strip()
        if KNOWN_LINE.match(text) or GRAPH.match(text) or (SELECTOR.match(text) and GROUP.match(text.split(" · ")[0])):
            kept.append(line)
        elif text.startswith("https://") and previous.startswith("License:"):
            kept.append(line)  # the license link under its name
        previous = text
    return kept


# ---------------------------------------------------------------------------
# GPUs, package, image and integrity
# ---------------------------------------------------------------------------

def container_devices(container: dict) -> list[str] | None:
    """GPU UUIDs or host indices exposed to the container; None for every GPU."""
    ids: list[str] = []
    for request in (container.get("HostConfig") or {}).get("DeviceRequests") or []:
        capabilities = {c for group in request.get("Capabilities") or [] for c in group}
        if "gpu" not in capabilities and request.get("Driver") not in ("nvidia", "cdi"):
            continue
        if request.get("Count") == -1 and not request.get("DeviceIDs"):
            return None
        ids += [device.rsplit("=", 1)[-1] for device in request.get("DeviceIDs") or []]  # CDI: nvidia.com/gpu=0
    if ids:
        return ids
    env = dict(item.split("=", 1) for item in (container.get("Config") or {}).get("Env") or [] if "=" in item)
    visible = env.get("NVIDIA_VISIBLE_DEVICES", "")
    return visible.split(",") if visible and visible not in ("all", "none", "void") else None


def nvml_devices() -> list[dict]:
    """Every GPU of this machine: index, UUID and the PIDs of its compute processes."""
    try:
        import pynvml
        pynvml.nvmlInit()
    except Exception:  # noqa: BLE001 - without NVML the GPUs cannot be told apart
        return []
    devices = []
    for index in range(pynvml.nvmlDeviceGetCount()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(index)
        uuid = pynvml.nvmlDeviceGetUUID(handle)
        try:
            pids = {p.pid for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)}
        except Exception:  # noqa: BLE001
            pids = set()
        devices.append({"index": index, "uuid": uuid.decode() if isinstance(uuid, bytes) else uuid, "pids": pids})
    return devices


def server_gpus(pid: int | None, container: dict | None, options: dict, ranks: int | None,
                devices: list[dict]) -> tuple[list[str] | None, str]:
    """UUIDs of the GPUs the server runs on and how they were found.

    NVML lists the server process on its GPUs when lil-bench shares the host's
    process namespace. Otherwise the container's devices, in PCI order, are
    narrowed by ``--gpus`` (or the package's rank count) the way Flare
    numbers them.
    """
    if not devices:
        return None, "NVML unavailable"
    if pid:
        used = [d["uuid"] for d in devices if pid in d["pids"]]
        if used:
            return used, "NVML processes"
    exposed = container_devices(container) if container else None
    visible = devices if exposed is None else [
        d for d in devices if d["uuid"] in exposed or str(d["index"]) in exposed]
    if container is None and pid is None:
        return None, "no server process"
    requested = [int(i) for i in str(options.get("gpus") or "").split(",") if i.strip().isdigit()]
    if requested:
        chosen = [visible[i] for i in requested if i < len(visible)]
        source = "--gpus"
    else:
        chosen = visible[:ranks] if ranks else visible
        source = "package ranks" if ranks else "container devices"
    # NVML order, as the inventory, telemetry and p2pmark number them (not the rank order of --gpus).
    return ([d["uuid"] for d in sorted(chosen, key=lambda d: d["index"])] or None), source


def container_path(path: str, pid: int | None, container: dict | None, proc_root: str = "/proc") -> Path | None:
    """A path inside the server's filesystem as this process can open it."""
    if not path.startswith("/"):
        workdir = ((container or {}).get("Config") or {}).get("WorkingDir") or "/"
        path = os.path.join(workdir, path)
    candidates = []
    if pid:
        candidates.append(Path(proc_root, str(pid), "root", path.lstrip("/")))
    for mount in sorted((container or {}).get("Mounts") or [], key=lambda m: -len(m.get("Destination") or "")):
        target, source = mount.get("Destination") or "", mount.get("Source") or ""
        if source and (path == target or path.startswith(target.rstrip("/") + "/")):
            candidates.append(Path(source + path[len(target):]))
            break
    if not container and not pid:
        candidates.append(Path(path))
    for candidate in candidates:
        try:
            if candidate.exists():
                return candidate
        except OSError:
            continue
    return None


def file_identity(path: Path | None, name: str) -> dict:
    if path is None:
        return {"name": name, "readable": False}
    try:
        size = path.stat().st_size
    except OSError:
        return {"name": name, "readable": False}
    item = {"name": name, "size": size, "readable": True}
    if size > HASH_LIMIT:
        item["sha256"] = None
        item["note"] = f"not hashed: larger than {HASH_LIMIT // 1024**3} GiB"
        return item
    digest = hashlib.sha256()
    try:
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(1 << 20), b""):
                digest.update(block)
    except OSError as error:
        item.update(readable=False, error=str(error))
        return item
    item["sha256"] = digest.hexdigest()
    return item


# Content-addressed payload files beside a package's primary file.
PAYLOAD = re.compile(r"[0-9a-f]{64}\.safetensors|.*-data-\d+-of-\d+\.safetensors")


def primary_file(directory: Path) -> Path | None:
    """The package file of a package directory: its only top-level tensor file that is not a payload."""
    try:
        found = [p for p in directory.iterdir() if p.suffix in (".safetensors", ".flaretensors")
                 and not PAYLOAD.fullmatch(p.name) and p.is_file()]
    except OSError:
        return None
    return found[0] if len(found) == 1 else None


def hf_download_commit(path: Path | None) -> str | None:
    """The commit ``hf download --local-dir`` recorded for a file (.cache/huggingface/download)."""
    if path is None:
        return None
    metadata = path.parent / ".cache" / "huggingface" / "download" / f"{path.name}.metadata"
    try:
        first = metadata.read_text(errors="replace").split("\n", 1)[0].strip()
    except OSError:
        return None
    return first if re.fullmatch(r"[0-9a-f]{40}", first) else None


def publisher_keys(path: Path | None) -> list[dict]:
    """Signing keys a server was given with --publisher-keys, fingerprinted."""
    if path is None:
        return []
    try:
        entries = json.loads(path.read_text())
    except (OSError, ValueError):
        return []
    keys = []
    for entry in entries if isinstance(entries, list) else []:
        key = entry.get("public_key") if isinstance(entry, dict) else None
        if isinstance(key, list) and len(key) == 32 and all(isinstance(b, int) and 0 <= b < 256 for b in key):
            keys.append({"namespace": entry.get("namespace"), "name": entry.get("name"),
                         "fingerprint": hashlib.sha256(bytes(key)).hexdigest(),
                         "development": bool(entry.get("development"))})
    return keys


def package_identity(serve: dict, log: dict, pid: int | None, container: dict | None,
                     proc_root: str = "/proc") -> dict:
    """Repository, revision, commit, file hash and publisher of the served package."""
    argument = serve.get("model") or ""
    hub = log.get("hub") or {}
    trust = log.get("trust") or {}
    source = trust.get("source") or hub.get("signed_source") or ""
    signed = re.fullmatch(r"https://huggingface\.co/([\w.-]+/[\w.-]+)/?", source)
    package: dict = {"argument": argument, "source": source or None,
                     "repository": signed.group(1) if signed else hub.get("repository"),
                     "revision": hub.get("revision"), "commit": hub.get("commit"),
                     "commit_source": "log" if hub.get("commit") else None,
                     "model": log.get("model")}
    local = container_path(argument, pid, container, proc_root) if argument else None
    if local is not None and local.is_dir():
        primary = primary_file(local)
        package["kind"] = "local"
        if primary is None:
            package["file"] = {"name": None, "readable": False, "note": "no single primary package file in the directory"}
        local = primary
    if local is not None and local.is_file():
        package["kind"] = "local"
        package["file"] = file_identity(local, local.name)
        commit = hf_download_commit(local)
        if commit and not package["commit"]:
            package.update(commit=commit, commit_source="hf download metadata")
    elif "kind" not in package:
        package["kind"] = "hub" if hub or re.fullmatch(r"[\w.-]+/[\w.-]+(?:@\S+)?", argument) else "unknown"
        if "@" in argument and not package["revision"]:
            package["revision"] = argument.split("@", 1)[1]
        name = hub.get("filename") or serve.get("options", {}).get("filename")
        if isinstance(name, str):
            package["file"] = {"name": name, "readable": False}
    publisher = {key: trust.get(key) for key in ("name", "namespace", "identity", "signature", "fingerprint",
                                                  "verified", "development", "from_earlier_start") if key in trust}
    if not publisher:
        keys_file = serve.get("options", {}).get("publisher-keys")
        keys = publisher_keys(container_path(keys_file, pid, container, proc_root)) if isinstance(keys_file, str) else []
        if len(keys) == 1:
            publisher = {**keys[0], "identity": "declared with --publisher-keys; the startup log does not show "
                                                "the acceptance", "verified": None}
    package["publisher"] = publisher or None
    if trust.get("license"):
        package["license"] = {"name": trust["license"], "link": trust.get("license_link")}
    return package


def parse_reference(reference: str) -> dict:
    """repository, tag and digest of an image reference such as repo:tag@sha256:…"""
    name, _, digest = reference.partition("@")
    repository, tag = name, None
    if ":" in name.rsplit("/", 1)[-1]:
        repository, tag = name.rsplit(":", 1)
    return {"repository": repository, "tag": tag, "digest": digest or None}


def image_identity(docker: Docker | None, container: dict | None) -> dict:
    """The Flare image a container runs: reference, digests and OCI labels."""
    if not container:
        return {"engine": "flare", "alias": None, "known": False}
    reference = (container.get("Config") or {}).get("Image") or ""
    item = {"engine": "flare", "known": True, "alias": reference, "reference": reference,
            **parse_reference(reference), "id": container.get("Image")}
    details: dict = {}
    if docker:
        try:
            details = docker.image(container.get("Image") or reference)
        except DockerError as error:
            item["error"] = str(error)
    labels = (details.get("Config") or {}).get("Labels") or (container.get("Config") or {}).get("Labels") or {}
    # An image pulled by a per-architecture tag is itself an index; the release
    # index lists the platform manifest inside it.
    platforms = [m["Descriptor"]["digest"] for m in details.get("Manifests") or []
                 if m.get("Kind") == "image" and (m.get("Descriptor") or {}).get("digest")]
    # The revision label names the Flare build; other labels and local tags stay on this machine.
    item.update(
        descriptor=(details.get("Descriptor") or {}).get("digest"),
        platform_digests=platforms,
        repo_digests=sorted(details.get("RepoDigests") or []),
        created=details.get("Created"),
        architecture=details.get("Architecture"),
        revision=labels.get("org.opencontainers.image.revision"),
    )
    return item


def integrity(docker: Docker | None, container: dict | None, pid: int | None, proc_root: str = "/proc") -> dict:
    """Does the container run its image's own binary, without hooks or replaced code?

    The site compares the image digest with the published Flare releases;
    this records what can change the code inside a container: another
    entrypoint, mounts or changed files where the binary lives, and loader
    hooks in the server's environment.
    """
    result: dict = {"engine": "flare", "binary": None, "entrypoint": None, "mounts": [], "changes": [], "hooks": {}}
    if container is None:
        result.update(status="unverified", reason="no container: the server's image and binary are unknown")
        return result
    path = container.get("Path") or ""
    result["entrypoint"] = path
    result["mounts"] = [{"destination": m.get("Destination"), "type": m.get("Type"), "read_only": not m.get("RW", True)}
                        for m in container.get("Mounts") or []
                        if _code_path(m.get("Destination") or "") or (m.get("Destination") or "") == "/"]
    env = dict(item.split("=", 1) for item in (container.get("Config") or {}).get("Env") or [] if "=" in item)
    if pid:
        try:
            raw = Path(proc_root, str(pid), "environ").read_bytes()
            env.update(item.decode(errors="replace").split("=", 1) for item in raw.split(b"\0") if b"=" in item)
        except OSError:
            pass
    result["hooks"] = {key: env[key] for key in HOOK_ENV if env.get(key)}
    if docker:
        try:
            result["changes"] = [c["Path"] for c in docker.changes(container["Id"]) or [] if _code_path(c.get("Path") or "")]
        except (DockerError, KeyError) as error:
            result["changes_error"] = str(error)
    if pid:
        result["binary"] = file_identity(Path(proc_root, str(pid), "exe"), os.path.basename(path) or "flare")
    problems = []
    if path != BINARY:
        problems.append(f"entrypoint {path} instead of {BINARY}")
    problems += [f"mount over {m['destination']}" for m in result["mounts"]]
    problems += [f"changed in the container: {p}" for p in result["changes"]]
    problems += [f"{k} set" for k in result["hooks"]]
    if problems:
        result.update(status="modified", reason="; ".join(problems))
    else:
        result.update(status="unmodified", reason="the image's own flare binary, no mounts or changed files over "
                                                  "it, no loader hooks")
    return result


def _code_path(path: str) -> bool:
    return any(path == p or path.startswith(p + "/") for p in CODE_PATHS)


# ---------------------------------------------------------------------------
# The whole picture
# ---------------------------------------------------------------------------

def collective_summary(collectives: dict | None) -> str:
    """One line: groups per decoded profile mode, for the terminal."""
    devices = (collectives or {}).get("devices") or []
    if not devices:
        return ""
    counts: dict[str, int] = {}
    for group in devices[0]["groups"]:
        for raw in group["selectors"]:
            item = decode_selector(raw)
            label = item["name"] or (f"{item['all_reduce']}/{item['all_gather']}" if item["ambiguous"] else raw)
            counts[label] = counts.get(label, 0) + group["count"]
    parts = [f"{label} ×{count}" for label, count in sorted(counts.items(), key=lambda kv: -kv[1])]
    total = sum(g["count"] for g in devices[0]["groups"])
    return f"device {devices[0]['device']}: {total} groups · " + " · ".join(parts)


def describe(state: dict, base_url: str, *, docker: Docker | None, container_name: str | None = None,
             log_file: str | None = None, devices: list[dict] | None = None, proc_root: str = "/proc") -> dict:
    """The container or process, build, package, startup log and GPUs of a Flare server.

    ``docker`` is None without Docker access. Returns the pieces ``lil-bench``
    stores: ``flare`` (the document's Flare block), ``server``, ``image``,
    ``integrity``, ``gpus`` (UUIDs of the server's GPUs, or None) and their
    NVML ``gpu_indices``, plus ``notes`` for the terminal.
    """
    notes: list[str] = []
    container = None
    if container_name:
        if docker is None:
            raise DockerError("no Docker access (socket or CLI) to inspect --container")
        container = docker.inspect(container_name)
    elif docker is not None:
        container, why = find_container(docker, base_url)
        if why:
            notes.append(why)
    elif not log_file:
        notes.append("no Docker access: the image, package and startup choices are unknown (use --server-log)")
    process = None
    if container:
        process = {"pid": (container.get("State") or {}).get("Pid") or None,
                   "argv": [container.get("Path") or "", *(container.get("Args") or [])]}
    else:
        process = find_process(base_url, proc_root)
    pid = (process or {}).get("pid")
    argv = (process or {}).get("argv") or []
    serve = run_options(argv) if argv else {"model": None, "options": {}}

    text, log_source = "", None
    if log_file:
        text, log_source = Path(log_file).read_text(errors="replace"), "file"
    elif container and docker:
        try:
            text = docker.logs(container, done=lambda t: bool(START_DONE.search(ANSI.sub("", t))))
            log_source = "docker logs"
        except DockerError as error:
            notes.append(f"startup log unavailable: {error}")
    log = parse_log(text) if text else {}
    if text and not log.get("started"):
        notes.append("the log has no completed server start")

    options = serve["options"]
    api = state.get("flare") or {}
    concurrency = api.get("max_concurrency") or log.get("concurrency")
    flag = options.get("max-concurrency")
    ranks = log.get("ranks")
    gpu_flag = [g for g in str(options.get("gpus") or "").split(",") if g.strip().isdigit()]
    limits = {
        "port": url_port(base_url)[1],
        "max_num_seqs": concurrency or None,
        "max_num_seqs_source": "flare_argv" if isinstance(flag, str) and flag.isdigit() else "flare_package",
        "max_model_len_arg": None,
        "tensor_parallel_size": ranks or len(gpu_flag) or None,
        "served_model_name": options.get("served-model-name") if isinstance(options.get("served-model-name"), str) else None,
    }
    devices = nvml_devices() if devices is None else devices
    gpus, gpu_source = server_gpus(pid, container, options, ranks, devices)
    if limits["tensor_parallel_size"] is None and gpus:
        limits["tensor_parallel_size"] = len(gpus)
    if container:
        environment = server.filter_environment((container.get("Config") or {}).get("Env") or [])
    else:
        environment = server.filtered_environment(pid, proc_root) if pid else {}
    image = image_identity(docker, container)
    code = integrity(docker, container, pid, proc_root)
    package = package_identity(serve, log, pid, container, proc_root)
    collectives = log.get("collectives")
    flare = {
        "source": "container" if container else ("process" if process else "api"),
        "log_source": log_source,
        "image": {k: image.get(k) for k in ("reference", "repository", "tag", "digest", "id", "descriptor",
                                             "platform_digests", "repo_digests", "created", "architecture",
                                             "revision")
                  if image.get(k) is not None},
        "revision": image.get("revision"),
        "container": {
            "network_mode": (container.get("HostConfig") or {}).get("NetworkMode"),
            "tty": bool((container.get("Config") or {}).get("Tty")),
            "started": (container.get("State") or {}).get("StartedAt"),
            "restarts": container.get("RestartCount"),
            "user": (container.get("Config") or {}).get("User") or None,
        } if container else None,
        "package": package,
        "gpus": {"source": gpu_source, "count": len(gpus) if gpus else None},
        "startup": {k: log[k] for k in ("steps", "started", "earlier_starts", "ranks", "concurrency", "kv_tokens",
                                       "tokens_per_request", "gpus", "model") if k in log},
        "collectives": {**collectives, "profiles": decoded_profiles(collectives),
                        "encoding": "b12x joint PCIe profile: transport mode << 16 | active CTAs; all-reduce and "
                                    "all-gather number their modes independently"}
        if collectives else {"status": "not reported", "reason": (
            "no startup collective tuning in the log: the package fixed its collectives at export" if text
            else "no startup log")},
        "log": {"text": log.get("text", ""), "truncated": bool(log.get("truncated"))} if text else None,
    }
    if container is not None:
        flare["container"] = {k: v for k, v in flare["container"].items() if v is not None}
    server_block = {"pid": pid, "argv": argv, "environment": environment, "model": serve.get("model"),
                    "options": options, "limits": limits, "api": state, "url": base_url}
    return {"flare": flare, "server": server_block, "image": image, "integrity": code, "gpus": gpus,
            "gpu_indices": [d["index"] for d in devices if gpus and d["uuid"] in gpus],
            "notes": notes, "container_name": (container or {}).get("Name", "").lstrip("/") or None}
