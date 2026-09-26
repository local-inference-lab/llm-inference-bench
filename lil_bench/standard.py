"""``lil-bench``: the standardized benchmark of a running Karmic Kraken container.

Run it in the serving container itself::

    docker exec -it -e LIL_BENCH_TOKEN=lilb_... <container> lil-bench

It checks the uploader identity, reads the exact serving command from
``/proc``, records hardware and PCIe topology, runs p2pmark, the standard
prefill/decode matrix (``llm_decode_bench.py``) while sampling GPU clocks and
throttle reasons, then saves and uploads one result document.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import os
import re
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from . import DEFAULT_SITE, SCHEMA, STANDARD, TOKEN_URL_PATH, VERSION
from . import inventory, server, telemetry, upload

ROOT = Path(__file__).resolve().parent.parent
BENCH = ROOT / "llm_decode_bench.py"
TOKEN_FORMAT = re.compile(r"^lilb_[A-Za-z0-9_-]{20,64}$")
RESULT_DIRS = ("/cache/lil-bench", os.path.expanduser("~/.cache/lil-bench"), "/tmp/lil-bench")
EXIT_NO_TOKEN = 2
EXIT_SERVER = 3
EXIT_BENCH = 4
EXIT_UPLOAD = 5

# The standard matrix. Changing any value is a new STANDARD version.
PROFILES = {
    "standard": {
        "prefill_contexts": [32768, 131072],
        "concurrency": [1, 8, 16],
        "contexts": [0, 65536, 131072],
        "duration": 30.0,
        "prefill_duration": 20.0,
        "max_tokens": 2048,
        "idle_check_s": 5,
    },
    # Minutes instead of half an hour; uploaded runs are flagged and kept out of statistics.
    "quick": {
        "prefill_contexts": [8192],
        "concurrency": [1, 2],
        "contexts": [0, 8192],
        "duration": 5.0,
        "prefill_duration": 4.0,
        "max_tokens": 256,
        "idle_check_s": 1,
    },
}
P2P_SIZES_MIB = (256, 128, 64, 32, 16)
P2P_OVERHEAD_MIB = 1024  # CUDA contexts plus NCCL buffers of the p2pmark process


class Output:
    """Plain, line-oriented progress that survives ``docker exec`` without a TTY."""

    def __init__(self, stream=None):
        self.stream = stream or sys.stdout
        self.color = self.stream.isatty() and os.environ.get("NO_COLOR") is None
        self.lock = threading.Lock()

    def _paint(self, text: str, code: str) -> str:
        return f"\033[{code}m{text}\033[0m" if self.color else text

    def line(self, text: str = "", code: str = "") -> None:
        with self.lock:
            print(self._paint(text, code) if code else text, file=self.stream, flush=True)

    def step(self, text: str) -> None:
        self.line(f"▶ {text}", "1;36")

    def info(self, text: str) -> None:
        self.line(f"  {text}")

    def ok(self, text: str) -> None:
        self.line(f"  ✓ {text}", "32")

    def warn(self, text: str) -> None:
        self.line(f"  ! {text}", "33")

    def error_panel(self, title: str, lines: list[str]) -> None:
        width = max(len(title), *(len(line) for line in lines)) + 4
        bar = "━" * width
        self.line(bar, "1;31")
        self.line(f"  {title}", "1;31")
        self.line(bar, "1;31")
        for line in lines:
            self.line(f"  {line}", "31" if not line.startswith("  ") else "1")
        self.line(bar, "1;31")


def fmt_ctx(tokens: int) -> str:
    return "0" if tokens == 0 else f"{tokens // 1024}k" if tokens >= 1024 else str(tokens)


def fmt_duration(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}" if seconds < 3600 else f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


# ---------------------------------------------------------------------------
# The plan
# ---------------------------------------------------------------------------

def fit_context(requested: int, output_tokens: int, max_model_len: int | None) -> tuple[int | None, str | None]:
    """The context that fits the model limit, trimmed by at most 10 %."""
    if not max_model_len:
        return requested, None
    room = max_model_len - output_tokens - 64
    if requested <= room:
        return requested, None
    if requested > 0 and room >= requested * 0.9:
        return room, f"trimmed to {room:,} tokens by max_model_len {max_model_len:,}"
    return None, f"context {fmt_ctx(requested)} + {output_tokens} output tokens exceeds max_model_len {max_model_len:,}"


def build_plan(profile: dict, limits: dict) -> list[dict]:
    """Every standard cell with its effective context or the reason it is skipped.

    ``limits``: max_model_len, max_num_seqs, kv_tokens (KV budget in tokens,
    DCP included). KV skips mirror llm_decode_bench's own capacity rule.
    """
    cells = []
    max_model_len = limits.get("max_model_len")
    for ctx in profile["prefill_contexts"]:
        effective, note = fit_context(ctx, 1, max_model_len)
        cells.append({"kind": "prefill", "requested_context": ctx, "context": effective,
                      "status": "planned" if effective else "skipped", "reason": note})
    kv = limits.get("kv_tokens") or 0
    max_num_seqs = limits.get("max_num_seqs") or 0
    for ctx in profile["contexts"]:
        effective, note = fit_context(ctx, profile["max_tokens"], max_model_len)
        for conc in profile["concurrency"]:
            cell = {"kind": "decode", "concurrency": conc, "requested_context": ctx, "context": effective,
                    "status": "planned", "reason": note}
            if effective is None:
                cell["status"] = "skipped"
            elif max_num_seqs and conc > max_num_seqs:
                cell.update(status="skipped", reason=f"concurrency {conc} is above max-num-seqs {max_num_seqs}")
            elif kv and not (effective == 0 and max_num_seqs and conc <= max_num_seqs) and \
                    conc * (effective + profile["max_tokens"]) > kv:
                need = conc * (effective + profile["max_tokens"])
                cell.update(status="skipped", reason=f"needs {need:,} KV tokens, the server has {kv:,}")
            cells.append(cell)
    return cells


def estimate_seconds(plan: list[dict], profile: dict) -> float:
    total = 90.0  # inventory, p2pmark, calibration
    decode_contexts = set()
    for cell in plan:
        if cell["status"] != "planned":
            continue
        if cell["kind"] == "prefill":
            total += profile["prefill_duration"] + 10 + cell["context"] / 10000
        else:
            decode_contexts.add(cell["context"])
            total += profile["duration"] + 12 + cell["context"] / 20000
    total += 15 * len(decode_contexts)  # one prefix-cache scout per context
    return total


def p2p_buffer_size(free_mib: list[int]) -> tuple[int | None, str | None]:
    if len(free_mib) < 2:
        return None, "single GPU: nothing to measure between GPUs"
    lowest = min(free_mib)
    for size in P2P_SIZES_MIB:
        if lowest >= P2P_OVERHEAD_MIB + 4 * size:
            return size, None
    return None, f"insufficient free GPU memory with the model loaded ({lowest} MiB free on the fullest GPU)"


# ---------------------------------------------------------------------------
# Events and phases
# ---------------------------------------------------------------------------

EVENT_PATTERNS = [
    ("prefill_start", re.compile(r"^prefill start ctx=(\S+)")),
    ("prefill_done", re.compile(r"^prefill (?:done|skipped) ctx=(\S+)")),
    ("cell_start", re.compile(r"^cell start C=(\d+) ctx=(\S+)")),
    ("cell_ready", re.compile(r"^(?:ready|measure start) C=(\d+) ctx=(\S+)")),
    ("cell_end", re.compile(r"^cell end C=(\d+) ctx=(\S+)")),
    ("cell_skipped", re.compile(r"^cell skipped C=(\d+) ctx=(\S+)")),
    ("decode_warmup", re.compile(r"^decode warmup (start|done)")),
]


def classify(event: str):
    for kind, pattern in EVENT_PATTERNS:
        match = pattern.match(event)
        if match:
            return kind, match.groups()
    return None, ()


def read_events(path: Path) -> list[dict]:
    events = []
    try:
        for line in path.read_text().splitlines():
            try:
                events.append(json.loads(line))
            except ValueError:
                continue
    except OSError:
        pass
    return events


def phases_from_events(events: list[dict]) -> list[dict]:
    """Prefill and decode phases with wall-clock windows (measured part separately)."""
    phases: list[dict] = []
    open_phase: dict | None = None
    decode_warmup = False
    for item in events:
        kind, groups = classify(item["event"])
        t = item["t"]
        if kind == "decode_warmup":
            decode_warmup = groups[0] == "start"
            continue
        if kind == "prefill_start":
            open_phase = {"kind": "prefill", "context": groups[0], "start": t, "measure_start": t}
        elif kind == "prefill_done" and open_phase and open_phase["kind"] == "prefill":
            open_phase["end"] = t
            phases.append(open_phase)
            open_phase = None
        elif kind == "cell_start" and not decode_warmup:
            open_phase = {"kind": "decode", "concurrency": int(groups[0]), "context": groups[1], "start": t}
        elif kind == "cell_ready" and open_phase and open_phase["kind"] == "decode":
            open_phase["measure_start"] = t
        elif kind == "cell_end" and open_phase and open_phase["kind"] == "decode" and not decode_warmup:
            open_phase["end"] = t
            open_phase.setdefault("measure_start", open_phase["start"])
            phases.append(open_phase)
            open_phase = None
    return phases


class EventFollower(threading.Thread):
    """Prints a progress line for every benchmark phase as it starts and ends."""

    def __init__(self, path: Path, out: Output, total_steps: int, first_step: int, deadline_estimate: float):
        super().__init__(name="lil-bench-events", daemon=True)
        self.path, self.out = path, out
        self.step = first_step
        self.total = total_steps
        self.estimate = deadline_estimate
        self.started = time.time()
        self.stop_event = threading.Event()
        self.offset = 0
        self.warmup = False

    def run(self) -> None:
        while not self.stop_event.is_set():
            self.poll()
            self.stop_event.wait(0.5)
        self.poll()

    def stop(self) -> None:
        self.stop_event.set()
        self.join(timeout=5)

    def eta(self) -> str:
        elapsed = time.time() - self.started
        left = max(0.0, self.estimate - elapsed)
        finish = datetime.now().astimezone().timestamp() + left
        return f"elapsed {fmt_duration(elapsed)}, ~{fmt_duration(left)} left (≈{datetime.fromtimestamp(finish).strftime('%H:%M')})"

    def poll(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as handle:
                handle.seek(self.offset)
                chunk = handle.read()
                self.offset = handle.tell()
        except OSError:
            return
        for line in chunk.splitlines():
            try:
                event = json.loads(line)["event"]
            except (ValueError, KeyError):
                continue
            self.show(event)

    def show(self, event: str) -> None:
        kind, groups = classify(event)
        if kind == "decode_warmup":
            self.warmup = groups[0] == "start"
            if self.warmup:
                self.out.info("decode warmup …")
            return
        if self.warmup and kind in ("cell_start", "cell_end", "cell_ready"):
            return
        if kind == "prefill_start":
            self.step += 1
            self.out.step(f"[{self.step}/{self.total}] prefill {groups[0]} — {self.eta()}")
        elif kind == "cell_start":
            self.step += 1
            self.out.step(f"[{self.step}/{self.total}] decode C{groups[0]} @ ctx {groups[1]} — {self.eta()}")
        elif kind == "cell_skipped":
            self.out.warn(f"decode C{groups[0]} @ ctx {groups[1]} skipped by the benchmark (KV capacity)")
        elif event.startswith("prefill done"):
            self.out.ok(event.replace("prefill done ", "prefill "))
        elif kind == "cell_end":
            match = re.search(r"tps=([-\d.]+)", event)
            if match and float(match.group(1)) >= 0:
                self.out.ok(f"C{groups[0]} @ {groups[1]}: {float(match.group(1)):,.1f} tok/s")
            else:
                self.out.warn(f"C{groups[0]} @ {groups[1]}: no valid measurement (see the result for the reason)")
        elif event.startswith(("calibration", "startup WARNING")):
            self.out.info(event)


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def load_bench_module():
    spec = importlib.util.spec_from_file_location("llm_decode_bench", BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def free_gpu_memory() -> list[int]:
    try:
        import pynvml
        pynvml.nvmlInit()
        free = []
        for i in range(pynvml.nvmlDeviceGetCount()):
            memory = pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i))
            free.append(memory.free // 2**20)
        return free
    except Exception:  # noqa: BLE001
        return []


def run_p2pmark(out: Output, gpu_count: int) -> dict:
    free = free_gpu_memory()
    size, reason = p2p_buffer_size(free if free else [0] * gpu_count)
    result: dict = {"free_mib": free, "with_model_loaded": True}
    try:
        bench = load_bench_module()
        result["nvidia_p2p_override"] = bench.detect_nvidia_p2p_override()
    except Exception as error:  # noqa: BLE001
        return {**result, "status": "error", "error": f"cannot load llm_decode_bench: {error}"}
    if size is None:
        out.warn(f"p2pmark skipped: {reason}")
        return {**result, "status": "skipped", "reason": reason}
    out.info(f"buffer {size} MiB per transfer ({min(free)} MiB free on the fullest GPU)")
    args = argparse.Namespace(
        p2pmark_bin="", p2pmark_mode="all", p2pmark_size_mb=size, p2pmark_iters=20, p2pmark_warmup=5,
        p2pmark_latency_iters=10000, p2pmark_allreduce_sizes_mb="full", p2pmark_max_gpus=0,
        p2pmark_timeout=300.0, p2pmark_detail=False,
    )
    from rich.console import Console
    try:
        diagnostic = bench.run_p2pmark_diagnostic(args, Console(force_terminal=out.color, width=120))
    except Exception as error:  # noqa: BLE001
        diagnostic = {"status": "error", "error": f"{type(error).__name__}: {error}"}
    if diagnostic.get("status") != "ok":
        out.warn(f"p2pmark {diagnostic.get('status')}: {(diagnostic.get('error') or diagnostic.get('stderr') or '')[:300]}")
    return {**result, "buffer_mib": size, **diagnostic}


def bench_command(base_url: str, model_id: str, profile: dict, plan: list[dict], output: Path, dcp: int | None) -> list[str]:
    prefill = [c["context"] for c in plan if c["kind"] == "prefill" and c["status"] == "planned"]
    decode = [c for c in plan if c["kind"] == "decode" and c["status"] == "planned"]
    contexts = sorted({c["context"] for c in decode})
    concurrency = sorted({c["concurrency"] for c in decode})
    cmd = [
        sys.executable, str(BENCH),
        "--host", base_url, "--model", model_id,
        "--concurrency", ",".join(map(str, concurrency or [1])),
        "--contexts", ",".join(map(str, contexts or [0])),
        "--duration", str(profile["duration"]),
        "--max-tokens", str(profile["max_tokens"]),
        "--display-mode", "plain",
        "--output", str(output),
        "--no-resume",
        "--no-calibration-cache",
        "--hw-monitor-interval", "2",
    ]
    if prefill:
        cmd += ["--standalone-prefill", "--prefill-contexts", ",".join(map(str, prefill)),
                "--prefill-duration", str(profile["prefill_duration"])]
    else:
        cmd += ["--skip-prefill"]
    if dcp and dcp > 1:
        cmd += ["--dcp-size", str(dcp)]
    return cmd


def result_dir() -> Path:
    for candidate in RESULT_DIRS:
        path = Path(candidate)
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write-test"
            probe.write_text("")
            probe.unlink()
            return path
        except OSError:
            continue
    raise OSError("no writable directory for results")


def save(document: dict, directory: Path) -> Path:
    path = directory / f"{document['run_id']}.json.gz"
    path.write_bytes(upload.compress(document))
    return path


def summarize(document: dict) -> dict:
    """Headline numbers, stored with the result and printed at the end."""
    bench = document.get("results", {}).get("bench") or {}
    prefill = {}
    for ctx, row in (bench.get("prefill") or {}).items():
        if isinstance(row, dict):
            prefill[ctx] = {k: row.get(k) for k in ("tok_per_sec", "ttft_seconds", "prompt_tokens", "samples")}
    decode = []
    for row in bench.get("results") or []:
        decode.append({
            "concurrency": row.get("concurrency"), "context": row.get("context_tokens"),
            "tok_s": row.get("aggregate_tps"), "per_user_tok_s": row.get("per_request_avg_tps"),
            "ttft_p50": row.get("ttft_p50"), "capacity_limited": row.get("capacity_limited"),
            "failure": row.get("failure_reason") or None,
            "spec_accept_length": row.get("server_spec_accept_length") or None,
        })
    verdicts = [p["analysis"]["verdict"] for p in document.get("phases", []) if "analysis" in p]
    order = ["no_data", "ok", "power_capped", "thermal", "hw_slowdown"]
    return {
        "prefill": prefill,
        "decode": decode,
        "throttle_verdict": max(verdicts, key=order.index) if verdicts else "no_data",
        "p2pmark_status": (document.get("results", {}).get("p2pmark") or {}).get("status"),
    }


def print_summary(out: Output, document: dict) -> None:
    summary = document["summary"]
    out.line()
    out.line("Results", "1")
    for ctx, row in summary["prefill"].items():
        tps = row.get("tok_per_sec")
        label = fmt_ctx(int(ctx)) if str(ctx).isdigit() else ctx
        value = f"{tps:,.0f} tok/s, TTFT {row.get('ttft_seconds')} s" if isinstance(tps, (int, float)) else "—"
        out.info(f"prefill {label:>5}: {value}")
    rows = {}
    for row in summary["decode"]:
        rows.setdefault(row["context"], {})[row["concurrency"]] = row
    concs = sorted({r["concurrency"] for r in summary["decode"]})
    if concs:
        out.info("decode tok/s  " + "".join(f"{'C' + str(c):>10}" for c in concs))
        for ctx in sorted(rows):
            cells = []
            for c in concs:
                row = rows[ctx].get(c)
                value = row and row.get("tok_s")
                cells.append(f"{value:>10,.1f}" if isinstance(value, (int, float)) and value >= 0 else f"{'—':>10}")
            out.info(f"ctx {fmt_ctx(ctx):>8}  " + "".join(cells))
    for cell in document["plan"]:
        if cell["status"] == "skipped":
            label = f"prefill {fmt_ctx(cell['requested_context'])}" if cell["kind"] == "prefill" else \
                f"decode C{cell['concurrency']} @ {fmt_ctx(cell['requested_context'])}"
            out.info(f"skipped {label}: {cell['reason']}")
    verdict = summary["throttle_verdict"]
    (out.ok if verdict in ("ok", "no_data") else out.warn)(f"GPU clocks during the run: {verdict}")
    for phase in document["phases"]:
        analysis = phase.get("analysis") or {}
        if analysis.get("verdict") not in (None, "ok", "no_data"):
            out.warn(f"  {phase['name']}: {analysis['verdict']}")
    for item in document["hardware"].get("tuning", []) + document["analysis"].get("overclock", []):
        out.warn(f"GPU {item['gpu']}: {item['detail']}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def no_token(out: Output, site: str, reason: str) -> int:
    out.error_panel("No LIL benchmark identifier", [
        reason,
        "Results are uploaded under your GitHub account; the container needs your identifier.",
        f"1. Open {site}{TOKEN_URL_PATH} and sign in with GitHub.",
        "2. Run the command shown there, for example:",
        "     docker exec -it -e LIL_BENCH_TOKEN=lilb_… <container> lil-bench",
        "To measure without uploading: lil-bench --no-upload",
    ])
    return EXIT_NO_TOKEN


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="lil-bench", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", nargs="?", default="run", choices=("run", "upload", "inventory", "whoami"))
    parser.add_argument("file", nargs="?", help="result file for 'upload'")
    parser.add_argument("--token", default=os.environ.get("LIL_BENCH_TOKEN", ""),
                        help="uploader identifier (default: $LIL_BENCH_TOKEN)")
    parser.add_argument("--site", default=os.environ.get("LIL_BENCH_SITE", DEFAULT_SITE))
    parser.add_argument("--url", default="", help="server URL (default: the port of the vLLM process in this container)")
    parser.add_argument("--profile", choices=sorted(PROFILES), default="standard",
                        help="'quick' is for testing the setup; its uploads are kept out of statistics")
    parser.add_argument("--no-upload", action="store_true", help="measure and save locally only")
    parser.add_argument("--no-p2pmark", action="store_true")
    parser.add_argument("--allow-busy", action="store_true", help="measure even if the server has other requests")
    parser.add_argument("--note", default="", help="free text stored with the result (max 500 characters)")
    return parser.parse_args(argv)


def check_identity(out: Output, args) -> tuple[int, dict | None]:
    if not args.token:
        return no_token(out, args.site, "LIL_BENCH_TOKEN is not set in this container."), None
    if not TOKEN_FORMAT.match(args.token):
        return no_token(out, args.site, "LIL_BENCH_TOKEN does not look like an identifier (lilb_…)."), None
    try:
        who = upload.whoami(args.site, args.token)
    except upload.UploadError as error:
        if error.status in (401, 403):
            return no_token(out, args.site, f"The identifier was rejected: {error}. It may have been regenerated or revoked."), None
        out.warn(f"cannot verify the identifier now ({error}); the result will be saved and can be uploaded later")
        return 0, {"login": None, "offline": True}
    out.ok(f"uploading as @{who.get('login')}")
    return 0, who


def command_upload(out: Output, args) -> int:
    if not args.file:
        out.line("usage: lil-bench upload <file.json.gz>")
        return 1
    if not args.token:
        return no_token(out, args.site, "LIL_BENCH_TOKEN is not set in this container.")
    payload = Path(args.file).read_bytes()
    if not payload.startswith(b"\x1f\x8b"):
        payload = gzip.compress(payload, compresslevel=9)
    try:
        reply = upload.upload_bytes(args.site, args.token, payload)
    except upload.UploadError as error:
        out.error_panel("Upload failed", [str(error)])
        return EXIT_UPLOAD
    out.ok(f"uploaded: {reply.get('url')}")
    return 0


def command_run(out: Output, args) -> int:
    profile = PROFILES[args.profile]
    out.line(f"lil-bench {VERSION} · {STANDARD} ({args.profile})", "1")
    identity = None
    if not args.no_upload:
        code, identity = check_identity(out, args)
        if code:
            return code

    out.step("Serving process")
    process = server.find_server()
    if not process and not args.url:
        out.error_panel("No vLLM server in this container", [
            "lil-bench runs inside the serving container: docker exec -it … <container> lil-bench",
            "Use --url http://127.0.0.1:PORT to measure another server (its command line is then unknown).",
        ])
        return EXIT_SERVER
    serve = server.parse_serve_args(process["argv"]) if process else {"model": None, "options": {}}
    limits = server.serve_limits(serve["options"])
    base_url = args.url.rstrip("/") or f"http://127.0.0.1:{limits['port']}"
    try:
        state = server.server_state(base_url)
    except Exception as error:  # noqa: BLE001
        out.error_panel("The server does not answer", [f"{base_url}/v1/models: {error}",
                                                         "Wait until the model has loaded and try again."])
        return EXIT_SERVER
    dcp = None
    for key in ("decode-context-parallel-size", "dcp"):
        value = serve["options"].get(key)
        if isinstance(value, str) and value.isdigit():
            dcp = int(value)
    kv_tokens = (state.get("kv_tokens") or 0) * (dcp or 1)
    limits.update(max_model_len=state.get("max_model_len"), kv_tokens=kv_tokens or None)
    out.ok(f"model {state.get('model_id')} · max_model_len {state.get('max_model_len')} · "
           f"max-num-seqs {limits['max_num_seqs']} ({limits['max_num_seqs_source']}) · KV {kv_tokens:,} tokens")
    image = server.image_identity()
    out.ok(f"image {image.get('alias') or 'unknown'} · assembly {(image.get('assembly_sha256') or '?')[:16]}")

    busy = 0.0
    for _ in range(profile["idle_check_s"]):
        try:
            busy = max(busy, server.busy_requests(base_url))
        except Exception:  # noqa: BLE001
            break
        time.sleep(1)
    if busy and not args.allow_busy:
        out.error_panel("The server is busy", [f"{busy:.0f} request(s) are running or waiting.",
                                                "Other traffic would distort the measurement; stop it and retry,",
                                                "or pass --allow-busy (the result is then flagged)."])
        return EXIT_SERVER

    plan = build_plan(profile, limits)
    planned = [c for c in plan if c["status"] == "planned"]
    estimate = estimate_seconds(plan, profile)
    out.step(f"Plan: {len(planned)} measurements, about {fmt_duration(estimate)} "
             f"(finishes ≈{datetime.fromtimestamp(time.time() + estimate).strftime('%H:%M')})")
    for cell in plan:
        label = (f"prefill {fmt_ctx(cell['requested_context'])}" if cell["kind"] == "prefill"
                 else f"decode C{cell['concurrency']} @ ctx {fmt_ctx(cell['requested_context'])}")
        if cell["status"] == "skipped":
            out.info(f"– {label}: skipped, {cell['reason']}")
        elif cell.get("reason"):
            out.info(f"· {label}: {cell['reason']}")

    started = datetime.now(timezone.utc)
    recorder = telemetry.Recorder(interval=0.5).start()
    workdir = result_dir()
    run_id = str(uuid.uuid4())
    phases: list[dict] = []
    status = "complete"
    bench_json: dict = {}
    p2p: dict = {"status": "not_run"}
    hardware: dict = {}
    cmd: list[str] | None = None
    bench_log_tail = ""
    total_steps = len(planned) + 2
    try:
        out.step(f"[1/{total_steps}] Hardware and PCIe topology")
        hardware = inventory.collect()
        gpus = hardware.get("gpus", [])
        recorder.link_bdfs = list(dict.fromkeys(
            hop["bdf"] for path in hardware["pcie"]["gpu_paths"].values() for hop in path["chain"]))
        out.ok(f"{len(gpus)} × {gpus[0]['name'] if gpus else '?'} · "
               f"{hardware['cpu'].get('lscpu', {}).get('Model name', '?')} · "
               f"{(hardware['memory'].get('mem_total_mib') or 0) // 1024} GiB RAM")
        for path in hardware["pcie"]["gpu_paths"].values():
            s = path["summary"]
            out.info(f"GPU {path['gpu']}: {s['hops']} PCIe hops{' behind a switch' if s['behind_switch'] else ''}, "
                     f"narrowest {s['min_current_gt_s']} GT/s x{s['min_current_width']}")
        for item in hardware.get("tuning", []):
            out.warn(f"GPU {item['gpu']}: {item['detail']}")

        out.step(f"[2/{total_steps}] p2pmark (GPU-to-GPU bandwidth and latency)")
        if args.no_p2pmark:
            p2p = {"status": "skipped", "reason": "--no-p2pmark"}
        else:
            p2p_start = time.time()
            p2p = run_p2pmark(out, len(gpus))
            phases.append({"kind": "p2pmark", "name": "p2pmark", "start": p2p_start, "measure_start": p2p_start,
                           "end": time.time()})

        events = workdir / f"{run_id}.events.jsonl"
        output = workdir / f"{run_id}.bench.json"
        log = workdir / f"{run_id}.bench.log"
        cmd = bench_command(base_url, state.get("model_id") or serve.get("model") or "", profile, plan, output, dcp)
        env = {**os.environ, "LLM_BENCH_EVENT_FILE": str(events), "LLM_BENCH_NO_UPDATE_CHECK": "1",
               "LLM_BENCH_CACHE_DIR": str(workdir / "bench-cache"), "TERM": "dumb"}
        env.pop("LIL_BENCH_TOKEN", None)
        follower = EventFollower(events, out, total_steps, 2, estimate - 90)
        follower.start()
        with open(log, "w") as log_handle:
            proc = subprocess.Popen(cmd, stdout=log_handle, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, env=env)
            try:
                returncode = proc.wait()
            except KeyboardInterrupt:
                proc.terminate()
                proc.wait(timeout=30)
                raise
            finally:
                follower.stop()
        if output.exists():
            bench_json = json.loads(output.read_text())
        if returncode != 0 or not bench_json:
            status = "failed"
            out.error_panel("The benchmark failed", [f"llm_decode_bench exited with {returncode}; last lines of {log}:"] +
                            ["  " + line for line in log.read_text(errors="replace").splitlines()[-25:]])
        for phase in phases_from_events(read_events(events)):
            phase["name"] = (f"prefill {phase['context']}" if phase["kind"] == "prefill"
                             else f"decode C{phase['concurrency']} @ {phase['context']}")
            phases.append(phase)
        bench_log_tail = log.read_text(errors="replace")[-20000:]
        for leftover in (events, output, log):
            leftover.unlink(missing_ok=True)
    except KeyboardInterrupt:
        status = "interrupted"
        out.warn("interrupted; the partial result is saved but not uploaded")
    finally:
        recorder.stop()

    series = recorder.series()
    limits_per_gpu = [g.get("power", {}) for g in (hardware.get("gpus") or [])]
    for phase in phases:
        phase["analysis"] = telemetry.analyze_phase(series, phase["measure_start"], phase["end"], limits_per_gpu)
    document = {
        "schema": SCHEMA,
        "standard": STANDARD,
        "profile": args.profile,
        "run_id": run_id,
        "status": status,
        "started": started.isoformat(),
        "finished": datetime.now(timezone.utc).isoformat(),
        "client": {"lil_bench": VERSION, "llm_decode_bench": _bench_version(), "argv": _safe_argv(),
                   "uploader": (identity or {}).get("login"), "note": args.note[:500],
                   "allow_busy": bool(busy and args.allow_busy)},
        "image": image,
        "server": {"pid": process and process["pid"], "argv": process and process["argv"],
                   "environment": server.filtered_environment(process["pid"]) if process else {},
                   "model": serve.get("model"), "options": serve["options"], "limits": limits, "api": state,
                   "url": base_url},
        "hardware": hardware,
        "plan": plan,
        "results": {"p2pmark": p2p, "bench": bench_json, "bench_command": cmd,
                    "bench_log_tail": bench_log_tail},
        "phases": phases,
        "telemetry": series,
        "analysis": {"overclock": telemetry.overclock_signals(series, hardware.get("gpus") or [])},
    }
    document["summary"] = summarize(document)
    path = save(document, workdir)
    if status == "complete":
        print_summary(out, document)
    out.line()
    out.ok(f"saved {path} ({path.stat().st_size / 1024:.0f} kB)")
    if args.no_upload or status == "interrupted":
        return 0 if status != "failed" else EXIT_BENCH
    try:
        reply = upload.upload_bytes(args.site, args.token, path.read_bytes())
    except upload.UploadError as error:
        out.error_panel("Upload failed", [str(error), "The result is saved; upload it later with:",
                                          f"  docker exec -e LIL_BENCH_TOKEN=… <container> lil-bench upload {path}"])
        return EXIT_UPLOAD
    out.ok(f"uploaded: {reply.get('url')}")
    return 0 if status == "complete" else EXIT_BENCH


def _bench_version() -> str | None:
    try:
        match = re.search(r'^VERSION = "([^"]+)"', BENCH.read_text(errors="replace")[:5000], re.M)
        return match.group(1) if match else None
    except OSError:
        return None


def _safe_argv() -> list[str]:
    argv, skip = [], False
    for arg in sys.argv[1:]:
        if skip:
            argv.append("***")
            skip = False
        elif arg == "--token":
            argv.append(arg)
            skip = True
        elif arg.startswith("--token="):
            argv.append("--token=***")
        else:
            argv.append(arg)
    return argv


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out = Output()
    if args.command == "upload":
        return command_upload(out, args)
    if args.command == "inventory":
        print(json.dumps(inventory.collect(), indent=2, default=str))
        return 0
    if args.command == "whoami":
        code, who = check_identity(out, args)
        return code
    return command_run(out, args)


if __name__ == "__main__":
    sys.exit(main())
