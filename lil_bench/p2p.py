"""p2pmark next to a loaded model: what fits, running it safely, reusing a result.

``llm_p2pmark`` is a separate process that opens a CUDA context on every GPU
and allocates copy buffers and NCCL communicators. Next to a loaded vLLM
server the GPUs are nearly full, and the server allocates more whenever it
gets work: PyTorch's caching allocator grows to its high-water mark under load
and keeps it (the first lil-bench run leaves the server ~600 MiB larger per
GPU). If p2pmark held the last free memory at such a moment, the CUDA
out-of-memory error would hit the server, not p2pmark. So:

* p2pmark is sized from its measured footprint and must leave ``RESERVE_MIB``
  free on every GPU; when only the copy and latency tests fit, the NCCL
  all-reduce comparison is left out;
* it starts only while the server has no requests, and is killed at once when
  the server gets one or a GPU falls below ``FLOOR_MIB`` free (the killed
  process returns its memory; the server is never touched);
* when it cannot run now, the p2pmark result of an earlier run on the same
  GPUs, driver, P2P and ACS settings since the last boot is reused and marked
  as such. Typically the first run after the server starts measures and later
  runs reuse it.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Peak memory of llm_p2pmark on every GPU, measured with NVML next to vLLM in
# uploaded runs (RTX PRO 6000 Blackwell, driver 610, CUDA 13): the CUDA context
# takes 555–566 MiB, the NCCL all-reduce comparison another 336–414 MiB, and the
# all-to-all copy test holds one buffer per GPU (peaks: 2 GPUs 971 MiB in the
# NCCL phase, 3 × 256 MiB 1,326, 4 × 256 MiB 1,583, 8 × 256 MiB 2,612).
CONTEXT_MIB = 600
NCCL_MIB = 480
RESERVE_MIB = 256  # kept free on every GPU for the server while p2pmark runs
FLOOR_MIB = 128  # p2pmark is killed if any GPU falls below this while it runs
SIZES_MIB = (256, 128, 64, 32, 16, 8, 4)
MAX_GPUS = 8  # llm_p2pmark measures at most eight GPUs
PARTS = ("bandwidth", "latency", "allreduce")
REUSE_SCAN_LIMIT = 50
# NCCL names the CUDA error behind a failure only with warnings on.
P2PMARK_ENV = {"NCCL_DEBUG": os.environ.get("NCCL_DEBUG") or "WARN"}
# Copied from an earlier result when it is reused.
REUSED_KEYS = ("binary", "mode", "cmd", "commands", "returncode", "elapsed_seconds", "data", "parse_error",
               "buffer_mib", "parts", "skipped_parts", "latency_error")


def footprint_mib(gpus: int, size_mib: int, allreduce: bool = True) -> int:
    """Peak GPU memory of llm_p2pmark on each GPU."""
    n = min(max(gpus, 2), MAX_GPUS)
    return CONTEXT_MIB + max(n * size_mib, NCCL_MIB if allreduce else 0)


def plan(free_mib: list[int], reserve_mib: int = RESERVE_MIB) -> dict:
    """The largest p2pmark that leaves ``reserve_mib`` free on every GPU."""
    if len(free_mib) < 2:
        return {"run": False, "why": "single_gpu", "reason": "single GPU: nothing to measure between GPUs"}
    gpus = min(len(free_mib), MAX_GPUS)
    fullest = min(range(len(free_mib)), key=lambda i: free_mib[i])
    lowest = free_mib[fullest]
    base = {"gpus": gpus, "fullest_gpu": fullest, "free_mib": lowest, "reserve_mib": reserve_mib}
    for allreduce in (True, False):
        for size in SIZES_MIB:
            need = footprint_mib(gpus, size, allreduce) + reserve_mib
            if lowest < need:
                continue
            chosen = {**base, "run": True, "size_mib": size, "need_mib": need,
                      "parts": list(PARTS if allreduce else PARTS[:2])}
            if not allreduce:
                need_nccl = footprint_mib(gpus, SIZES_MIB[-1]) + reserve_mib
                chosen["skipped_parts"] = {"allreduce": (
                    f"the NCCL all-reduce comparison needs {need_nccl:,} MiB free per GPU and GPU {fullest} "
                    f"has {lowest:,} MiB")}
            return chosen
    need = footprint_mib(gpus, SIZES_MIB[-1], allreduce=False) + reserve_mib
    reason = (f"not enough free GPU memory: GPU {fullest} has {lowest:,} MiB free, p2pmark needs {need:,} MiB "
              f"({CONTEXT_MIB} MiB CUDA context, {gpus} × {SIZES_MIB[-1]} MiB buffers, {reserve_mib} MiB kept "
              "free for the server)")
    hint = ("Run lil-bench right after the server starts, before other load: later runs reuse that measurement "
            f"until the next reboot. Or free {need - lowest:,} MiB more per GPU (a smaller KV cache or "
            "--gpu-memory-utilization).")
    return {**base, "run": False, "why": "memory", "need_mib": need, "reason": reason, "hint": hint}


# ---------------------------------------------------------------------------
# Reusing an earlier measurement
# ---------------------------------------------------------------------------

def boot_time(proc_root: str = "/proc") -> float | None:
    try:
        for line in Path(proc_root, "stat").read_text().splitlines():
            if line.startswith("btime "):
                return float(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return None


def parse_time(value) -> float | None:
    try:
        stamp = datetime.fromisoformat(str(value))
    except (TypeError, ValueError):
        return None
    return (stamp if stamp.tzinfo else stamp.replace(tzinfo=timezone.utc)).timestamp()


def _gpu_keys(hardware: dict | None) -> list[tuple]:
    return [(g.get("uuid_hash"), g.get("bdf"), g.get("name")) for g in (hardware or {}).get("gpus") or []]


def _driver(hardware: dict | None) -> str | None:
    return ((hardware or {}).get("nvml_system") or {}).get("driver_version")


def setup_difference(hardware: dict, override: dict | None, earlier: dict) -> str | None:
    """Why an earlier run's p2pmark does not describe this machine now (None: it does)."""
    before = earlier.get("hardware") or {}
    if not _gpu_keys(hardware) or _gpu_keys(hardware) != _gpu_keys(before):
        return "different GPUs"
    if _driver(hardware) != _driver(before):
        return "different driver"
    then = ((earlier.get("results") or {}).get("p2pmark") or {}).get("nvidia_p2p_override") or {}
    now = override or {}
    if bool(now.get("effective")) != bool(then.get("effective")) or (
            now.get("runtime") is not None and then.get("runtime") is not None and now["runtime"] != then["runtime"]):
        return "different NVIDIA P2P settings"
    # ACS is compared when both runs could read it (docker exec --privileged).
    acs_now = (hardware.get("pcie") or {}).get("acs") or {}
    acs_then = (before.get("pcie") or {}).get("acs") or {}
    if acs_now.get("hops_with_acs") and acs_then.get("hops_with_acs") and \
            (acs_now.get("redirect") or {}) != (acs_then.get("redirect") or {}):
        return "different PCIe ACS settings"
    return None


def find_earlier(directory: Path | None, hardware: dict, override: dict | None, proc_root: str = "/proc",
                 limit: int = REUSE_SCAN_LIMIT) -> dict | None:
    """The newest successful p2pmark measured on this machine since the last boot."""
    booted = boot_time(proc_root)
    if directory is None or booted is None:
        return None
    try:
        files = sorted(Path(directory).glob("*.json.gz"), key=lambda p: p.stat().st_mtime, reverse=True)[:limit]
    except OSError:
        return None
    for path in files:
        try:
            if path.stat().st_mtime < booted:
                break
            document = json.loads(gzip.decompress(path.read_bytes()))
            result = (document.get("results") or {}).get("p2pmark") or {}
            if result.get("status") != "ok" or not (result.get("data") or {}).get("peer_access"):
                continue
            origin = result.get("reused") or {}
            measured_at = origin.get("measured_at") or result.get("measured_at") or document.get("started")
            measured = parse_time(measured_at)
            if measured is None or measured < booted or setup_difference(hardware, override, document):
                continue
        except Exception:  # noqa: BLE001 - an unreadable or foreign file is not a result
            continue
        return {"result": result, "run_id": origin.get("run_id") or document.get("run_id"),
                "measured_at": measured_at, "measured": measured}
    return None


def reused_result(earlier: dict, why_not_now: str, now: float) -> dict:
    """The earlier measurement, marked with where it comes from and why it was not repeated."""
    source = earlier["result"]
    result = {key: source[key] for key in REUSED_KEYS if key in source}
    result.update(status="ok", measured_at=earlier["measured_at"], reused={
        "run_id": earlier["run_id"], "measured_at": earlier["measured_at"],
        "age_s": round(now - earlier["measured"]), "reason": why_not_now,
        "free_mib": (source.get("reused") or {}).get("free_mib") or source.get("free_mib")})
    return result


# ---------------------------------------------------------------------------
# Running it
# ---------------------------------------------------------------------------

def server_busy(busy_fn) -> str | None:
    """Why the server does not count as idle: it has requests, or they cannot be read."""
    if busy_fn is None:
        return None
    try:
        busy = busy_fn()
    except Exception as error:  # noqa: BLE001 - a server too slow to answer is not idle
        return f"the server's request counts cannot be read ({type(error).__name__})"
    if busy is None:
        return "the server does not report its running and waiting requests"
    return f"the server has {busy:.0f} running or waiting request(s)" if busy else None


def memory_short(free_fn, need_mib: int) -> str | None:
    free = free_fn()
    if not free:
        return "free GPU memory is unknown (NVML unavailable)"
    gpu = min(range(len(free)), key=lambda i: free[i])
    return f"GPU {gpu} has {free[gpu]:,} MiB free, p2pmark needs {need_mib:,} MiB" if free[gpu] < need_mib else None


def guard(free_fn, busy_fn=None, floor_mib: int = FLOOR_MIB):
    """Polled while p2pmark runs: why it must stop now (the server is not idle, a GPU is nearly full)."""
    def check() -> str | None:
        reason = server_busy(busy_fn)
        if reason:
            return f"{reason} while p2pmark ran"
        free = free_fn()
        if free and min(free) < floor_mib:
            gpu = min(range(len(free)), key=lambda i: free[i])
            return f"GPU {gpu} fell to {free[gpu]} MiB free while p2pmark ran"
        return None
    return check


def launched(result: dict) -> bool:
    """llm_p2pmark was started (not refused before launch, missing, or unable to start)."""
    return "returncode" in result or "stdout" in result


def _diagnostic(bench, console, mode: str, size: int, abort, before) -> dict:
    """One llm_p2pmark run; ``before`` is checked right before the launch."""
    reason = before()
    if reason:
        return {"status": "aborted", "mode": mode, "reason": f"not started: {reason}"}
    args = argparse.Namespace(
        p2pmark_bin="", p2pmark_mode=mode, p2pmark_size_mb=size, p2pmark_iters=20, p2pmark_warmup=5,
        p2pmark_latency_iters=10000, p2pmark_allreduce_sizes_mb="full", p2pmark_max_gpus=0,
        p2pmark_timeout=300.0, p2pmark_detail=False,
    )
    try:
        return bench.run_p2pmark_diagnostic(args, console, abort=abort, summary=False, extra_env=P2PMARK_ENV)
    except Exception as error:  # noqa: BLE001
        return {"status": "error", "error": f"{type(error).__name__}: {error}"}


def execute(bench, console, chosen: dict, abort, before) -> dict:
    """Run the planned parts: one ``all`` run, or ``bandwidth`` then ``latency``."""
    size = chosen["size_mib"]
    if "allreduce" in chosen["parts"]:
        return _diagnostic(bench, console, "all", size, abort, before)
    copies = _diagnostic(bench, console, "bandwidth", size, abort, before)
    if copies.get("status") != "ok":
        return copies
    latency = _diagnostic(bench, console, "latency", size, abort, before)
    data = {**(copies.get("data") or {}), "mode": "bandwidth+latency"}
    merged = {**copies, "mode": "bandwidth+latency", "data": data,
              "commands": [copies.get("cmd"), latency.get("cmd")],
              "elapsed_seconds": round((copies.get("elapsed_seconds") or 0) + (latency.get("elapsed_seconds") or 0), 3)}
    if latency.get("status") == "ok":
        data["latency"] = (latency.get("data") or {}).get("latency") or {}
    else:
        merged["latency_error"] = {k: latency[k] for k in ("status", "reason", "error", "stderr") if latency.get(k)}
    return merged


def out_of_memory(result: dict) -> bool:
    text = " ".join(str(result.get(k) or "") for k in ("stderr", "error", "stdout")).lower()
    return "out of memory" in text or "cudaerrormemoryallocation" in text


def summary_line(result: dict) -> str:
    data = result.get("data") or {}
    bandwidth = (data.get("bandwidth_summary") or {}).get("avg_offdiag_gbps")
    latency = (data.get("latency") or {}).get("avg_sequential_us")
    parts = [f"{bandwidth:.1f} GB/s GPU↔GPU" if isinstance(bandwidth, (int, float)) and bandwidth else None,
             f"{latency:.2f} µs latency" if isinstance(latency, (int, float)) and latency else None]
    return ", ".join(p for p in parts if p) or "no bandwidth data"


def _age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 3600:
        return f"{seconds // 60} min"
    if seconds < 86400:
        return f"{seconds // 3600} h {seconds % 3600 // 60:02d} min"
    return f"{seconds // 86400} d {seconds % 86400 // 3600} h"


def _brief(attempt: dict) -> dict:
    brief = {k: attempt[k] for k in ("status", "reason", "returncode", "error", "elapsed_seconds") if attempt.get(k)}
    if attempt.get("stderr"):
        brief["stderr"] = attempt["stderr"][-2000:]
    return brief


def measure(out, hardware: dict, bench, *, free_fn, busy_fn=None, directory: Path | None = None,
            console=None, proc_root: str = "/proc", now=time.time) -> dict:
    """p2pmark for this run: measured now, reused from an earlier run, or skipped with the reason.

    ``ran`` in the returned dict is True when llm_p2pmark was started (the
    caller records a telemetry phase for it and removes the key).
    """
    free = free_fn()
    chosen = plan(free) if free else {"run": False, "why": "unknown",
                                       "reason": "free GPU memory is unknown (NVML unavailable)"}
    if chosen.get("why") == "single_gpu":
        out.warn(f"p2pmark skipped: {chosen['reason']}")
        return {"status": "skipped", "reason": chosen["reason"], "free_mib": free, "ran": False}
    result: dict = {"free_mib": free, "with_model_loaded": True, "ran": False, "memory": {
        "free_mib": free, "need_mib": chosen.get("need_mib"), "reserve_mib": RESERVE_MIB,
        "context_mib": CONTEXT_MIB, "nccl_mib": NCCL_MIB}}
    try:
        result["nvidia_p2p_override"] = bench.detect_nvidia_p2p_override()
    except Exception as error:  # noqa: BLE001
        return {**result, "status": "error", "error": f"cannot read the NVIDIA P2P settings: {error}"}

    attempt = None
    why_not = chosen.get("reason")
    if chosen["run"]:
        busy = server_busy(busy_fn)
        if busy:
            why_not = f"{busy}; p2pmark only runs next to an idle server"
        else:
            out.info(f"{' + '.join(chosen['parts'])} with {chosen['size_mib']} MiB buffers: {chosen['free_mib']:,} MiB "
                     f"free on the fullest GPU, {chosen['reserve_mib']} MiB stay free for the server")
            for part, reason in (chosen.get("skipped_parts") or {}).items():
                out.info(f"no {part}: {reason}")
            started = now()
            before = lambda: server_busy(busy_fn) or memory_short(free_fn, chosen["need_mib"])  # noqa: E731
            attempt = execute(bench, console, chosen, guard(free_fn, busy_fn), before)
            attempt.update(buffer_mib=chosen["size_mib"], parts=chosen["parts"],
                           measured_at=datetime.fromtimestamp(started, timezone.utc).isoformat())
            if chosen.get("skipped_parts"):
                attempt["skipped_parts"] = chosen["skipped_parts"]
            result["ran"] = launched(attempt)
            status = attempt.get("status")
            if status == "ok":
                if console is not None:
                    try:
                        bench.print_p2pmark_summary(console, attempt)
                    except Exception:  # noqa: BLE001 - the tables are cosmetic
                        pass
                out.ok(f"p2pmark: {summary_line(attempt)}")
                if attempt.get("latency_error"):
                    error = attempt["latency_error"]
                    out.warn(f"p2pmark latency test {error.get('status')}: "
                             f"{(error.get('reason') or error.get('error') or error.get('stderr') or '')[:300]}")
                return {**result, **attempt}
            if status == "aborted":
                why_not = f"stopped: {attempt.get('reason')}"
            elif out_of_memory(attempt):
                why_not = "p2pmark ran out of GPU memory"
            else:  # a failure unrelated to memory is a finding; never hide it behind an old result
                out.warn(f"p2pmark {status}: {(attempt.get('error') or attempt.get('stderr') or '')[:300]}")
                return {**result, **attempt}

    try:
        earlier = find_earlier(directory, hardware, result.get("nvidia_p2p_override"), proc_root)
    except Exception:  # noqa: BLE001 - never fail the run over an old result
        earlier = None
    if earlier:
        reused = reused_result(earlier, why_not, now())
        when = datetime.fromtimestamp(earlier["measured"]).strftime("%H:%M")
        out.ok(f"p2pmark reused from {when} ({_age(now() - earlier['measured'])} ago; same GPUs, driver and P2P "
               f"settings since boot): {summary_line(reused)}")
        out.info(f"not measured now: {why_not}")
        return {**result, **reused, **({"attempt": _brief(attempt)} if attempt else {})}
    reason = why_not + (f". {chosen['hint']}" if chosen.get("hint") else "")
    status = attempt.get("status") if attempt else "skipped"
    out.warn(f"p2pmark {status}: {reason}")
    return {**result, **(attempt or {}), "status": status, "reason": reason}
