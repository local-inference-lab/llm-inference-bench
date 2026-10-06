"""lil-bench: plan, discovery, inventory, telemetry, upload and the run itself."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lil_bench import inventory, p2p, server, standard, telemetry, upload  # noqa: E402

STANDARD = standard.PROFILES["standard"]


# ---------------------------------------------------------------------------
# Plan
# ---------------------------------------------------------------------------

def by_key(plan):
    return {(c["kind"], c.get("concurrency"), c["requested_context"]): c for c in plan}


def test_plan_full_server_runs_everything():
    plan = standard.build_plan(STANDARD, {"max_model_len": 262144, "max_num_seqs": 32, "kv_tokens": 10_000_000})
    assert [c["status"] for c in plan] == ["planned"] * 11
    assert all(c["context"] == c["requested_context"] for c in plan)


def test_plan_trims_128k_to_model_limit_and_skips_by_limits():
    plan = by_key(standard.build_plan(STANDARD, {"max_model_len": 131072, "max_num_seqs": 8, "kv_tokens": 500_000}))
    cell = plan[("decode", 1, 131072)]
    assert cell["status"] == "planned" and cell["context"] == 131072 - 2048 - 64
    assert "trimmed" in cell["reason"]
    assert plan[("prefill", None, 131072)]["context"] == 131072 - 1 - 64
    assert plan[("decode", 16, 0)]["status"] == "skipped"
    assert "max-num-seqs 8" in plan[("decode", 16, 0)]["reason"]
    # 8 × (64k + 2048) = 540,672 > 500,000
    assert plan[("decode", 8, 65536)]["status"] == "skipped"
    assert "KV tokens" in plan[("decode", 8, 65536)]["reason"]
    # Context 0 within max-num-seqs is never a KV skip (same rule as the bench).
    assert plan[("decode", 8, 0)]["status"] == "planned"


def test_plan_skips_context_far_above_model_limit():
    plan = by_key(standard.build_plan(STANDARD, {"max_model_len": 65536, "max_num_seqs": 64}))
    assert plan[("decode", 1, 131072)]["status"] == "skipped"
    assert "exceeds max_model_len" in plan[("decode", 1, 131072)]["reason"]
    assert plan[("prefill", None, 131072)]["status"] == "skipped"
    assert plan[("prefill", None, 32768)]["status"] == "planned"
    # 64k + 2048 output does not fit 65536 but is within 10 %: trimmed.
    assert plan[("decode", 1, 65536)]["context"] == 65536 - 2048 - 64


def test_estimate_grows_with_context():
    small = standard.build_plan(standard.PROFILES["quick"], {"max_model_len": 32768})
    big = standard.build_plan(STANDARD, {"max_model_len": 262144, "max_num_seqs": 64})
    assert standard.estimate_seconds(small, standard.PROFILES["quick"]) < 300
    assert 500 < standard.estimate_seconds(big, STANDARD) < 3600


def test_concurrency_above_max_num_seqs_is_explained_once():
    plan = standard.build_plan(STANDARD, {"max_model_len": 1_000_000, "max_num_seqs": 8, "kv_tokens": 10_000_000,
                                          "max_num_seqs_source": "argv"})
    reason = by_key(plan)[("decode", 16, 131072)]["reason"]
    assert reason == ("the server runs at most 8 requests at once (--max-num-seqs 8), so C16 would measure a queue, "
                      "not 16 concurrent users; expected with this configuration")
    assert standard.skipped_lines(plan) == [f"decode C16 @ ctx 0, 64k, 128k: {reason}"]
    default = standard.concurrency_skip(16, 8, "vllm_default")
    assert "vLLM's default --max-num-seqs 8" in default


# ---------------------------------------------------------------------------
# p2pmark next to a loaded model
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("free,parts,size", [
    ([997, 1037], ["bandwidth", "latency"], 64),  # GLM Spark TP2, first run after start
    ([1062, 1062], ["bandwidth", "latency"], 64),
    ([1485, 1627], ["bandwidth", "latency", "allreduce"], 256),
    ([2826] * 4, ["bandwidth", "latency", "allreduce"], 256),
    ([4178] * 8, ["bandwidth", "latency", "allreduce"], 256),
    ([1200] * 8, ["bandwidth", "latency"], 32),
])
def test_p2p_plan_leaves_the_reserve_free(free, parts, size):
    chosen = p2p.plan(free)
    assert chosen["run"] and chosen["parts"] == parts and chosen["size_mib"] == size
    assert min(free) - p2p.footprint_mib(len(free), size, "allreduce" in parts) >= p2p.RESERVE_MIB
    assert ("allreduce" in chosen.get("skipped_parts", {})) == ("allreduce" not in parts)


def test_p2p_plan_without_room_says_what_it_needs():
    chosen = p2p.plan([397, 437])  # the same server after its first benchmark
    assert not chosen["run"] and chosen["need_mib"] == 600 + 2 * 4 + 256
    assert chosen["reason"].startswith("not enough free GPU memory: GPU 0 has 397 MiB free, p2pmark needs 864 MiB")
    assert "right after the server starts" in chosen["hint"] and "467 MiB more per GPU" in chosen["hint"]
    assert p2p.plan([733, 33, 13, 33])["fullest_gpu"] == 2
    assert p2p.plan([90000])["why"] == "single_gpu"


@pytest.mark.parametrize("gpus,size,allreduce,measured_peak_mib", [
    (2, 32, True, 971), (2, 128, True, 896), (3, 256, True, 1326), (4, 256, True, 1583), (8, 256, True, 2612)])
def test_p2p_footprint_covers_measured_peaks(gpus, size, allreduce, measured_peak_mib):
    # Peaks of llm_p2pmark next to vLLM in uploaded runs (NVML memory used minus the server's).
    assert measured_peak_mib <= p2p.footprint_mib(gpus, size, allreduce) <= measured_peak_mib * 1.25


BOOT = 1_790_000_000
OVERRIDE = {"effective": False, "runtime": {"ForceP2P": "", "EnableResizableBar": "0"}}
P2P_DATA = {"tool": "llm_p2pmark", "peer_access": [[1, 1], [1, 1]], "bandwidth_gbps": [[0, 53.8], [53.9, 0]],
            "bandwidth_summary": {"avg_offdiag_gbps": 53.8}, "latency": {"avg_sequential_us": 0.91}, "allreduce": []}


def p2p_hardware(uuid="u0"):
    return {"gpus": [{"index": i, "name": "RTX PRO 6000", "uuid_hash": f"{uuid}{i}", "bdf": f"0000:0{i}:00.0"}
                     for i in range(2)],
            "nvml_system": {"driver_version": "610.57.04"},
            "pcie": {"acs": {"readable": True, "hops_with_acs": ["0000:00:01.1"], "redirect": {}}}}


def write_run(directory, name, mtime, p2pmark, hardware=None, started=None):
    document = {"run_id": name, "started": datetime_iso(started or mtime), "hardware": hardware or p2p_hardware(),
                "results": {"p2pmark": p2pmark}}
    path = directory / f"{name}.json.gz"
    path.write_bytes(gzip.compress(json.dumps(document).encode()))
    os.utime(path, (mtime, mtime))
    return path


def datetime_iso(stamp):
    from datetime import datetime, timezone
    return datetime.fromtimestamp(stamp, timezone.utc).isoformat()


def ok_p2pmark(**extra):
    return {"status": "ok", "data": P2P_DATA, "buffer_mib": 64, "parts": ["bandwidth", "latency"],
            "elapsed_seconds": 2.1, "free_mib": [997, 1037], "nvidia_p2p_override": OVERRIDE, **extra}


@pytest.fixture
def boot_proc(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "stat").write_text(f"cpu  1 2 3 4\nbtime {BOOT}\nprocesses 5\n")
    return str(proc)


def test_find_earlier_needs_same_boot_gpus_and_settings(tmp_path, boot_proc):
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, "first", BOOT + 100, ok_p2pmark())
    write_run(runs, "skipped", BOOT + 500, {"status": "skipped", "reason": "no room"})
    write_run(runs, "other-gpus", BOOT + 400, ok_p2pmark(), hardware=p2p_hardware("x"))
    write_run(runs, "last-boot", BOOT - 50, ok_p2pmark())
    found = p2p.find_earlier(runs, p2p_hardware(), OVERRIDE, boot_proc)
    assert found["run_id"] == "first" and found["measured"] == BOOT + 100
    # A reused result points at the original measurement, never at the run that copied it.
    write_run(runs, "second", BOOT + 900, {**ok_p2pmark(free_mib=[397, 437]), "reused": {
        "run_id": "first", "measured_at": datetime_iso(BOOT + 100), "free_mib": [997, 1037]}})
    found = p2p.find_earlier(runs, p2p_hardware(), OVERRIDE, boot_proc)
    assert found["run_id"] == "first" and found["measured"] == BOOT + 100
    assert p2p.reused_result(found, "no room", BOOT + 1000)["reused"]["free_mib"] == [997, 1037]
    assert p2p.find_earlier(runs, p2p_hardware(), {"effective": True, "runtime": {}}, boot_proc) is None
    (runs / "broken.json.gz").write_bytes(b"not gzip")
    (runs / "truncated.json.gz").write_bytes(gzip.compress(b'{"results": {}}')[:12] + b"garbage" * 10)
    (runs / "list.json.gz").write_bytes(gzip.compress(b"[1, 2]"))
    write_run(runs, "foreign", BOOT + 950, ok_p2pmark(), hardware={"gpus": ["not a dict"]})
    write_run(runs, "odd", BOOT + 960, {**ok_p2pmark(), "reused": True})
    for name in ("broken", "truncated", "list"):
        os.utime(runs / f"{name}.json.gz", (BOOT + 970, BOOT + 970))
    assert p2p.find_earlier(runs, p2p_hardware(), OVERRIDE, boot_proc)["run_id"] == "first"
    only_old = tmp_path / "old"
    only_old.mkdir()
    write_run(only_old, "last-boot", BOOT - 50, ok_p2pmark())
    assert p2p.find_earlier(only_old, p2p_hardware(), OVERRIDE, boot_proc) is None


def test_setup_difference():
    earlier = {"hardware": p2p_hardware(), "results": {"p2pmark": {"nvidia_p2p_override": OVERRIDE}}}
    assert p2p.setup_difference(p2p_hardware(), OVERRIDE, earlier) is None
    assert p2p.setup_difference(p2p_hardware("y"), OVERRIDE, earlier) == "different GPUs"
    newer_driver = {**p2p_hardware(), "nvml_system": {"driver_version": "610.60"}}
    assert p2p.setup_difference(newer_driver, OVERRIDE, earlier) == "different driver"
    changed = {"effective": False, "runtime": {"ForceP2P": "0x11", "EnableResizableBar": "0"}}
    assert p2p.setup_difference(p2p_hardware(), changed, earlier) == "different NVIDIA P2P settings"
    redirect = p2p_hardware()
    redirect["pcie"]["acs"]["redirect"] = {"0000:00:01.1": ["ReqRedir"]}
    assert p2p.setup_difference(redirect, OVERRIDE, earlier) == "different PCIe ACS settings"
    unreadable = p2p_hardware()
    unreadable["pcie"]["acs"] = {"readable": False, "unreadable": ["0000:00:01.1"]}
    assert p2p.setup_difference(unreadable, OVERRIDE, earlier) is None


class FakeBench:
    """llm_decode_bench's p2pmark entry points; records every run."""

    def __init__(self, replies=None, after_call=None):
        self.calls = []
        self.replies = replies or {}
        self.after_call = after_call

    def detect_nvidia_p2p_override(self):
        return OVERRIDE

    def run_p2pmark_diagnostic(self, args, console, abort=None, summary=True, extra_env=None):
        assert callable(abort) and summary is False and extra_env["NCCL_DEBUG"]
        self.calls.append((args.p2pmark_mode, args.p2pmark_size_mb))
        if self.after_call:
            self.after_call()
        reply = self.replies.get(args.p2pmark_mode)
        if reply is not None:
            return dict(reply)
        data = {**P2P_DATA, "mode": args.p2pmark_mode}
        if args.p2pmark_mode == "bandwidth":
            data["latency"] = {}
        if args.p2pmark_mode == "latency":
            data.update(bandwidth_gbps=[], bandwidth_summary={}, latency={"avg_sequential_us": 0.93})
        return {"status": "ok", "mode": args.p2pmark_mode, "cmd": ["llm_p2pmark", "--mode", args.p2pmark_mode],
                "returncode": 0, "stdout": json.dumps(data), "elapsed_seconds": 1.0, "data": data}

    def print_p2pmark_summary(self, console, result):
        pass


class Lines(standard.Output):
    def __init__(self):
        super().__init__(stream=open(os.devnull, "w"))
        self.lines = []

    def line(self, text="", code=""):
        self.lines.append(text)


def measure(free, bench, directory=None, busy=None, proc=None, now=BOOT + 3600):
    out = Lines()
    result = p2p.measure(out, p2p_hardware(), bench, free_fn=lambda: list(free), busy_fn=busy, directory=directory,
                         proc_root=proc or "/nonexistent", now=lambda: now)
    return result, "\n".join(out.lines)


def test_p2pmark_runs_copy_and_latency_tests_when_nccl_does_not_fit():
    bench = FakeBench()
    result, text = measure([997, 1037], bench, busy=lambda: 0.0)
    assert bench.calls == [("bandwidth", 64), ("latency", 64)]
    assert result["status"] == "ok" and result["ran"] and result["parts"] == ["bandwidth", "latency"]
    assert result["data"]["bandwidth_gbps"] == P2P_DATA["bandwidth_gbps"]
    assert result["data"]["latency"] == {"avg_sequential_us": 0.93} and result["mode"] == "bandwidth+latency"
    assert "allreduce" in result["skipped_parts"] and result["buffer_mib"] == 64 and result["measured_at"]
    assert result["memory"]["reserve_mib"] == p2p.RESERVE_MIB
    assert "bandwidth + latency with 64 MiB buffers: 997 MiB free on the fullest GPU, 256 MiB stay free" in text
    assert "p2pmark: 53.8 GB/s GPU↔GPU, 0.93 µs latency" in text


def test_p2pmark_full_run_when_everything_fits():
    bench = FakeBench()
    result, _ = measure([4178] * 8, bench)
    assert bench.calls == [("all", 256)] and result["status"] == "ok" and "skipped_parts" not in result


def test_p2pmark_reuses_the_first_run_when_the_server_has_grown(tmp_path, boot_proc):
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, "first", BOOT + 100, ok_p2pmark())
    bench = FakeBench()
    result, text = measure([397, 437], bench, directory=runs, proc=boot_proc)
    assert bench.calls == [] and result["ran"] is False
    assert result["status"] == "ok" and result["data"] == P2P_DATA and result["buffer_mib"] == 64
    assert result["reused"]["run_id"] == "first" and result["reused"]["age_s"] == 3500
    assert result["reused"]["reason"].startswith("not enough free GPU memory: GPU 0 has 397 MiB free")
    assert result["free_mib"] == [397, 437] and result["reused"]["free_mib"] == [997, 1037]
    assert "p2pmark reused from" in text and "58 min ago" in text


def test_p2pmark_skip_without_an_earlier_result_is_actionable(tmp_path, boot_proc):
    result, text = measure([397, 437], FakeBench(), directory=tmp_path, proc=boot_proc)
    assert result["status"] == "skipped"
    assert "GPU 0 has 397 MiB free, p2pmark needs 864 MiB" in result["reason"]
    assert "Run lil-bench right after the server starts" in result["reason"] and "p2pmark skipped" in text


def unreadable_metrics():
    raise TimeoutError("timed out")


@pytest.mark.parametrize("busy,why", [
    (lambda: 2.0, "the server has 2 running or waiting request(s)"),
    (lambda: None, "the server does not report its running and waiting requests"),
    (unreadable_metrics, "the server's request counts cannot be read (TimeoutError)"),
])
def test_p2pmark_never_starts_unless_the_server_is_known_to_be_idle(tmp_path, boot_proc, busy, why):
    bench = FakeBench()
    result, _ = measure([4000, 4000], bench, directory=tmp_path, busy=busy, proc=boot_proc)
    assert bench.calls == [] and result["status"] == "skipped" and result["ran"] is False
    assert result["reason"] == f"{why}; p2pmark only runs next to an idle server"


def test_p2pmark_latency_run_is_not_started_after_the_server_got_work():
    state = {"busy": 0.0}
    bench = FakeBench(after_call=lambda: state.update(busy=1.0))
    result, text = measure([997, 1037], bench, busy=lambda: state["busy"])
    assert bench.calls == [("bandwidth", 64)] and result["status"] == "ok" and result["ran"]
    assert result["latency_error"] == {"status": "aborted",
                                       "reason": "not started: the server has 1 running or waiting request(s)"}
    assert "p2pmark latency test aborted: not started" in text


def test_p2pmark_that_never_started_records_no_phase(tmp_path, boot_proc):
    bench = FakeBench({"all": {"status": "missing_binary", "error": "p2pmark binary not found"}})
    result, _ = measure([4000, 4000], bench, directory=tmp_path, proc=boot_proc)
    assert result["status"] == "missing_binary" and result["ran"] is False


def test_p2pmark_failure_is_reported_not_replaced(tmp_path, boot_proc):
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, "first", BOOT + 100, ok_p2pmark())
    bench = FakeBench({"all": {"status": "failed", "returncode": 1, "stderr": "NCCL error: unhandled system error"}})
    result, _ = measure([4000, 4000], bench, directory=runs, proc=boot_proc)
    assert result["status"] == "failed" and "reused" not in result and result["ran"]


@pytest.mark.parametrize("attempt,why", [
    ({"status": "aborted", "reason": "the server got 1 request(s) while p2pmark ran", "stdout": ""},
     "stopped: the server got 1 request(s) while p2pmark ran"),
    ({"status": "failed", "returncode": 1, "stderr": "CUDA error: out of memory"}, "p2pmark ran out of GPU memory"),
])
def test_p2pmark_stopped_for_the_server_falls_back_to_the_earlier_result(tmp_path, boot_proc, attempt, why):
    runs = tmp_path / "runs"
    runs.mkdir()
    write_run(runs, "first", BOOT + 100, ok_p2pmark())
    result, _ = measure([4000, 4000], FakeBench({"all": attempt}), directory=runs, proc=boot_proc)
    assert result["status"] == "ok" and result["reused"]["reason"] == why and result["ran"]
    assert result["attempt"]["status"] == attempt["status"]
    alone, _ = measure([4000, 4000], FakeBench({"all": attempt}), directory=tmp_path / "none", proc=boot_proc)
    assert alone["status"] == attempt["status"] and alone["reason"] == why


def test_p2pmark_guard_stops_for_requests_unknown_state_and_full_gpus():
    frees = iter([[900, 900], [900, 100]])
    check = p2p.guard(lambda: next(frees), lambda: 0.0)
    assert check() is None
    assert check() == "GPU 1 fell to 100 MiB free while p2pmark ran"
    assert p2p.guard(lambda: [900], lambda: 1.0)() == \
        "the server has 1 running or waiting request(s) while p2pmark ran"
    assert p2p.guard(lambda: [900, 900], unreadable_metrics)() == \
        "the server's request counts cannot be read (TimeoutError) while p2pmark ran"


def load_decode_bench():
    import importlib.util
    spec = importlib.util.spec_from_file_location("llm_decode_bench_p2p", standard.BENCH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fake_binary(tmp_path, body):
    path = tmp_path / "llm_p2pmark"
    path.write_text("#!/bin/sh\n" + body + "\n")
    path.chmod(0o755)
    return str(path)


def p2pmark_args(binary, timeout=30.0):
    import argparse
    return argparse.Namespace(p2pmark_bin=binary, p2pmark_mode="all", p2pmark_size_mb=4, p2pmark_iters=1,
                              p2pmark_warmup=0, p2pmark_latency_iters=1, p2pmark_allreduce_sizes_mb="full",
                              p2pmark_max_gpus=0, p2pmark_timeout=timeout, p2pmark_detail=False)


def test_p2pmark_diagnostic_is_killed_when_aborted(tmp_path):
    import io
    from rich.console import Console
    bench = load_decode_bench()
    console = Console(file=io.StringIO(), width=120)
    ok = bench.run_p2pmark_diagnostic(p2pmark_args(fake_binary(tmp_path, "echo '{\"peer_access\": [[1]]}'")), console)
    assert ok["status"] == "ok" and ok["data"] == {"peer_access": [[1]]}
    calls = []

    def abort():
        calls.append(1)
        return "the server got 1 request(s)" if len(calls) >= 2 else None
    started = time.monotonic()
    stopped = bench.run_p2pmark_diagnostic(p2pmark_args(fake_binary(tmp_path, "exec sleep 30")), console, abort=abort)
    assert stopped["status"] == "aborted" and stopped["reason"] == "the server got 1 request(s)"
    assert time.monotonic() - started < 5
    late = bench.run_p2pmark_diagnostic(p2pmark_args(fake_binary(tmp_path, "exec sleep 30"), timeout=0.3), console)
    assert late["status"] == "timeout" and late["timeout_seconds"] == 0.3


# ---------------------------------------------------------------------------
# Serving process
# ---------------------------------------------------------------------------

def fake_proc(tmp_path: Path, processes: dict) -> Path:
    root = tmp_path / "proc"
    for pid, (argv, env) in processes.items():
        d = root / str(pid)
        d.mkdir(parents=True)
        (d / "cmdline").write_bytes(b"\0".join(a.encode() for a in argv) + b"\0")
        (d / "environ").write_bytes(b"\0".join(f"{k}={v}".encode() for k, v in env.items()) + b"\0")
    return root


def test_find_server_reads_argv_and_filters_environment(tmp_path):
    argv = ["/opt/venv/bin/python", "-m", "vllm.entrypoints.cli.main", "serve", "org/Model", "--port", "5091",
            "--max-num-seqs=32", "--enable-prefix-caching", "--tensor-parallel-size", "2"]
    env = {"VLLM_USE_B12X": "1", "HF_TOKEN": "hf_secret", "LIL_BENCH_TOKEN": "lilb_x", "NCCL_P2P_LEVEL": "SYS",
           "PATH": "/usr/bin", "MODEL": "org/Model", "B12X_API_KEY": "k"}
    proc = fake_proc(tmp_path, {1: (["/sbin/docker-init", "--", "/usr/local/bin/lil-entrypoint"], {}),
                                7: (argv, env), 843: (["VLLM::EngineCore"], {})})
    found = server.find_server(str(proc))
    assert found == {"pid": 7, "argv": argv}
    assert server.filtered_environment(7, str(proc)) == {"MODEL": "org/Model", "NCCL_P2P_LEVEL": "SYS", "VLLM_USE_B12X": "1"}
    serve = server.parse_serve_args(argv)
    assert serve["model"] == "org/Model"
    assert serve["options"]["enable-prefix-caching"] is True
    limits = server.serve_limits(serve["options"])
    assert limits["port"] == 5091 and limits["max_num_seqs"] == 32 and limits["max_num_seqs_source"] == "argv"
    assert limits["tensor_parallel_size"] == 2


def test_serve_limits_default_max_num_seqs():
    limits = server.serve_limits(server.parse_serve_args(["vllm", "serve", "m"])["options"])
    assert limits == {**limits, "port": 8000, "max_num_seqs": server.VLLM_DEFAULT_MAX_NUM_SEQS,
                      "max_num_seqs_source": "vllm_default"}


def test_image_identity(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"assembly": {"alias": "ghcr.io/x:karmic-kraken-beta", "assembly_sha256": "ab" * 32,
                                                 "channel": "karmic-kraken"}}))
    identity = server.image_identity(str(manifest), str(tmp_path / "missing.json"))
    assert identity["assembly_sha256"] == "ab" * 32 and identity["contract"] is None


def test_metrics_parsing():
    text = """# HELP x
vllm:num_requests_running{engine="0",model_name="m"} 2.0
vllm:num_requests_waiting{engine="0",model_name="m"} 1.0
vllm:cache_config_info{block_size="64",num_gpu_blocks="1000",enable_prefix_caching="True"} 1.0
"""
    samples = server.parse_metrics(text)
    assert server.metric_sum(samples, "vllm:num_requests_running") == 2.0
    assert server.kv_capacity(samples)["kv_tokens"] == 64000
    assert server.kv_capacity(samples)["kv_tokens_source"] == "blocks"


def test_kv_capacity_is_vllms_own_for_a_hybrid_model():
    """Qwen3.8-Flash-Next TP4: its layer groups share the block pool, so blocks
    × block size (100M) is 15× what the server holds (vLLM logs 6,471,460)."""
    text = ('vllm:cache_config_info{_block_size_resolved="True",block_size="1472",'
            'kv_cache_size_tokens="6471460",mamba_block_size="64",num_gpu_blocks="68308",'
            'num_gpu_blocks_override="None"} 1.0\n')
    capacity = server.kv_capacity(server.parse_metrics(text))
    assert capacity["kv_tokens"] == 6_471_460 and capacity["kv_tokens_source"] == "vllm"
    assert capacity["num_gpu_blocks"] * capacity["block_size"] == 100_549_376


@pytest.mark.parametrize("source,expected", [("vllm", 1_000_000), ("blocks", 4_000_000)])
def test_server_kv_tokens_counts_dcp_once(source, expected):
    state = {"kv_tokens": 1_000_000, "kv_tokens_source": source}
    assert standard.server_kv_tokens(state, 4) == expected


# ---------------------------------------------------------------------------
# Inventory
# ---------------------------------------------------------------------------

def fake_sysfs(tmp_path: Path) -> Path:
    """Root port → Broadcom switch (upstream, downstream) → GPU."""
    sys_root = tmp_path / "sys"
    path = sys_root / "devices" / "pci0000:00"
    hops = [("0000:00:01.1", "0x1022", "0x0604"), ("0000:01:00.0", "0x1000", "0x0604"),
            ("0000:02:10.0", "0x1000", "0x0604"), ("0000:03:00.0", "0x10de", "0x030200")]
    (sys_root / "bus/pci/devices").mkdir(parents=True)
    for bdf, vendor, klass in hops:
        path = path / bdf
        path.mkdir(parents=True)
        values = {"vendor": vendor, "device": "0xc030", "class": klass, "current_link_speed": "32.0 GT/s PCIe",
                  "max_link_speed": "32.0 GT/s PCIe", "current_link_width": "16", "max_link_width": "16",
                  "numa_node": "0"}
        if bdf == "0000:02:10.0":
            values["current_link_speed"] = "16.0 GT/s PCIe"
        for name, value in values.items():
            (path / name).write_text(value + "\n")
        os.symlink(path, sys_root / "bus/pci/devices" / bdf)
    (sys_root / "class/dmi/id").mkdir(parents=True)
    (sys_root / "class/dmi/id/board_name").write_text("GENOA2D24G\n")
    (sys_root / "class/dmi/id/board_serial").write_text("SECRET123\n")
    (sys_root / "class/dmi/id/product_uuid").write_text("uuid\n")
    dimm = sys_root / "devices/system/edac/mc/mc0/dimm0"
    dimm.mkdir(parents=True)
    (dimm / "dimm_mem_type").write_text("Registered-DDR5\n")
    (dimm / "size").write_text("65536\n")
    return sys_root


def test_pci_chain_walks_switch(tmp_path):
    sys_root = fake_sysfs(tmp_path)
    chain = inventory.pci_chain("0000:03:00.0", str(sys_root))
    assert [h["bdf"] for h in chain] == ["0000:00:01.1", "0000:01:00.0", "0000:02:10.0", "0000:03:00.0"]
    assert chain[0]["host_bridge"] == "pci0000:00"
    assert chain[1]["switch_vendor"] == "Broadcom" and chain[-1]["role"] == "gpu"
    summary = inventory.link_summary(chain)
    assert summary == {"hops": 4, "bridges": 3, "behind_switch": True, "switches": "Broadcom switch",
                       "min_current_gt_s": 16.0, "min_current_width": 16}
    links = telemetry.read_links([h["bdf"] for h in chain], str(sys_root))
    assert links["0000:02:10.0"] == "16.0x16"


def test_dmi_and_memory_never_read_serials(tmp_path):
    sys_root = fake_sysfs(tmp_path)
    board = inventory.dmi_inventory(str(sys_root))
    assert board == {"board_name": "GENOA2D24G"}
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text("MemTotal:       792092672 kB\nHugePages_Total:       0\n")
    memory = inventory.memory_inventory(str(sys_root), str(proc))
    assert memory["memory_types"] == ["Registered-DDR5"] and memory["dimm_count"] == 1
    assert memory["mem_total_mib"] == 792092672 // 1024


def test_tuning_indicators():
    gpus = [{"index": 0, "clocks": {"gpc_vf_offset_mhz": 150, "mem_vf_offset_mhz": 0, "app_sm_mhz": 2280,
                                    "default_app_sm_mhz": 2280},
             "power": {"enforced_limit_w": 325.0, "default_limit_w": 300.0}},
            {"index": 1, "clocks": {}, "power": {"enforced_limit_w": 600.0, "default_limit_w": 600.0}}]
    kinds = [(f["gpu"], f["kind"]) for f in inventory.tuning_indicators(gpus)]
    assert kinds == [(0, "gpc_clock_offset"), (0, "power_limit")]


def test_identity_hash_is_stable_and_not_the_uuid():
    value = inventory.identity_hash("GPU-d8438b2d-f000-a617-5dcc-0197ce0365a3")
    assert value == inventory.identity_hash("GPU-d8438b2d-f000-a617-5dcc-0197ce0365a3")
    assert "d8438b2d" not in value and len(value) == 24


# ---------------------------------------------------------------------------
# Telemetry
# ---------------------------------------------------------------------------

class FakeSource:
    name = "fake"

    def __init__(self, rows_per_call):
        self.calls = iter(rows_per_call)

    def sample(self):
        return next(self.calls)


def row(sm, reasons=0, power=3000, temp=60):
    return {"sm_mhz": sm, "mem_mhz": 14001, "gr_mhz": sm, "power_dw": power, "temp_c": temp, "util_pct": 99,
            "mem_util_pct": 50, "reasons": reasons, "pcie_gen": 5, "pcie_width": 16, "pstate": 0,
            "energy_j": 100, "mem_used_mib": 90000, "fan_pct": None}


def with_errors(rows):
    for i, r in enumerate(rows):
        r.update(pcie_replay=0 if i < 5 else 3, pcie_corr_err=0, pcie_recovery=0)
    return rows


def series_of(rows, interval=0.5):
    recorder = telemetry.Recorder(interval=interval, source=FakeSource([[r] for r in rows]))
    recorder.t0 = 1000.0
    for index in range(len(rows)):
        recorder.record_once()
        recorder.times[-1] = int(index * interval * 1000)
    return recorder.series()


def test_analyze_power_capped_phase():
    cap = telemetry.REASONS["sw_power_cap"]
    series = series_of([row(2800)] * 4 + [row(2400, cap, power=3250)] * 16)
    result = telemetry.analyze_phase(series, 1000.0, 1010.0, [{"enforced_limit_w": 325.0}])
    gpu = result["gpus"][0]
    assert result["verdict"] == "power_capped" and gpu["verdict"] == "power_capped"
    assert gpu["sm_mhz"]["median"] == 2400 and gpu["sm_clock_drop_pct"] > 10
    assert gpu["power_to_limit_max"] == 1.0
    assert gpu["reason_share"]["sw_power_cap"] == 0.8


def test_unknown_reason_bits_are_kept():
    series = series_of([row(2800, 0x400)] * 5 + [row(2800)] * 5)
    gpu = telemetry.analyze_phase(series, 1000.0, 1010.0, [{}])["gpus"][0]
    assert gpu["reason_share"] == {"bit_0x400": 0.5} and gpu["verdict"] == "ok"


def test_analyze_hw_slowdown_wins_and_idle_is_ignored():
    series = series_of([row(2800, telemetry.REASONS["gpu_idle"] | telemetry.REASONS["hw_slowdown"])] * 10 +
                       [row(1200, telemetry.REASONS["hw_slowdown"])] * 2 + [row(2800)] * 8)
    assert telemetry.analyze_phase(series, 1000.0, 1010.0, [{}])["verdict"] == "hw_slowdown"
    calm = series_of([row(2800, telemetry.REASONS["gpu_idle"])] * 10)
    assert telemetry.analyze_phase(calm, 1000.0, 1010.0, [{}])["verdict"] == "ok"


def test_analyze_window_without_samples():
    series = series_of([row(2800)] * 4)
    assert telemetry.analyze_phase(series, 2000.0, 2010.0, [{}])["verdict"] == "no_data"


def test_overclock_signals():
    series = series_of([row(3200)] * 3)
    found = telemetry.overclock_signals(series, [{"clocks": {"max_sm_mhz": 3090, "max_mem_mhz": 14001}}])
    assert [f["kind"] for f in found] == ["sm_clock_above_rated"]


def test_recorder_thread_samples_links(tmp_path):
    sys_root = fake_sysfs(tmp_path)
    rows = [[row(2800)] for _ in range(1000)]
    recorder = telemetry.Recorder(interval=0.01, source=FakeSource(rows), link_bdfs=["0000:02:10.0"],
                                  link_every=5, sys_root=str(sys_root)).start()
    import time
    time.sleep(0.2)
    recorder.stop()
    series = recorder.series()
    assert len(series["t_ms"]) >= 5 and len(series["gpus"][0]["sm_mhz"]) == len(series["t_ms"])
    assert series["pcie_links"][0]["links"] == {"0000:02:10.0": "16.0x16"}


# ---------------------------------------------------------------------------
# Phases from benchmark events
# ---------------------------------------------------------------------------

EVENTS = [
    (1.0, "prefill warmup start"), (2.0, "prefill start ctx=32k"), (22.0, "prefill done ctx=32k 9,000 tok/s"),
    (23.0, "prefill start ctx=128k"), (50.0, "prefill skipped ctx=128k"),
    (51.0, "decode warmup start"), (52.0, "cell start C=1 ctx=64k"), (55.0, "cell end C=1 ctx=64k tps=1.0"),
    (56.0, "decode warmup done C=1 ctx=64k"),
    (60.0, "cell start C=1 ctx=0"), (63.0, "ready C=1 ctx=0 stable"), (93.0, "cell end C=1 ctx=0 tps=150.2"),
    (94.0, "cell skipped C=16 ctx=128k"),
    (95.0, "cell start C=8 ctx=64k"), (140.0, "cell end C=8 ctx=64k tps=-1.0"),
]


def test_phases_from_events_skip_warmup():
    phases = standard.phases_from_events([{"t": t, "event": e} for t, e in EVENTS])
    assert [(p["kind"], p.get("concurrency"), p["context"]) for p in phases] == [
        ("prefill", None, "32k"), ("prefill", None, "128k"), ("decode", 1, "0"), ("decode", 8, "64k")]
    assert phases[2]["measure_start"] == 63.0 and phases[2]["end"] == 93.0
    assert phases[3]["measure_start"] == 95.0  # no ready event: the whole cell


def test_event_follower_prints_progress(tmp_path):
    path = tmp_path / "events.jsonl"
    path.write_text("".join(json.dumps({"t": t, "event": e}) + "\n" for t, e in EVENTS))
    lines = []

    class Capture(standard.Output):
        def line(self, text="", code=""):
            lines.append(text)

    follower = standard.EventFollower(path, Capture(stream=open(os.devnull, "w")), 6, 2, 600)
    follower.poll()
    text = "\n".join(lines)
    assert "[3/6] prefill 32k" in text and "[5/6] decode C1 @ ctx 0" in text and "[6/6] decode C8 @ ctx 64k" in text
    assert "C1 @ 0: 150.2 tok/s" in text and "skipped by the benchmark" in text and "decode warmup" in text
    assert "C8 @ 64k: no valid measurement" in text and "tps=1.0" not in text


# ---------------------------------------------------------------------------
# Site API client (fake site)
# ---------------------------------------------------------------------------

class FakeSite:
    def __init__(self):
        self.uploads: list[dict] = []
        self.fail_next = 0
        site = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, body):
                data = json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def authorized(self):
                return self.headers.get("Authorization") == "Bearer " + GOOD_TOKEN

            def do_GET(self):
                if self.path == "/api/bench/whoami" and self.authorized():
                    return self.reply(200, {"login": "tester"})
                return self.reply(401, {"error": "unknown identifier"})

            def do_POST(self):
                body = self.rfile.read(int(self.headers["Content-Length"]))
                if not self.authorized():
                    return self.reply(401, {"error": "unknown identifier"})
                if site.fail_next:
                    site.fail_next -= 1
                    return self.reply(503, {"error": "busy"})
                document = json.loads(gzip.decompress(body))
                site.uploads.append(document)
                return self.reply(201, {"id": "r1", "url": "https://site/bench/runs/r1"})

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()


GOOD_TOKEN = "lilb_" + "a" * 32


@pytest.fixture
def site():
    fake = FakeSite()
    yield fake
    fake.close()


def test_whoami_and_upload_retry(site):
    assert upload.whoami(site.url, GOOD_TOKEN) == {"login": "tester"}
    with pytest.raises(upload.UploadError) as error:
        upload.whoami(site.url, "lilb_" + "b" * 32)
    assert error.value.status == 401 and "unknown identifier" in str(error.value)
    site.fail_next = 1
    sleeps = []
    reply = upload.upload_bytes(site.url, GOOD_TOKEN, upload.compress({"x": 1}), sleep=sleeps.append)
    assert reply["id"] == "r1" and site.uploads == [{"x": 1}] and sleeps == [3.0]


def test_upload_does_not_retry_client_errors(site):
    sleeps = []
    with pytest.raises(upload.UploadError) as error:
        upload.upload_bytes(site.url, "lilb_" + "c" * 32, b"x", sleep=sleeps.append)
    assert error.value.status == 401 and sleeps == []


def test_unreachable_site_is_retryable():
    with pytest.raises(upload.UploadError) as error:
        upload.upload_bytes("http://127.0.0.1:9", GOOD_TOKEN, b"x", attempts=1)
    assert error.value.retry and "cannot reach" in str(error.value)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def run_main(argv, monkeypatch, capsys, env=None):
    for key in ("LIL_BENCH_TOKEN", "LIL_BENCH_SITE"):
        monkeypatch.delenv(key, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    code = standard.main(argv)
    return code, capsys.readouterr().out


def test_missing_token_is_a_visible_error(monkeypatch, capsys):
    code, out = run_main(["run"], monkeypatch, capsys)
    assert code == standard.EXIT_NO_TOKEN
    assert "No LIL benchmark identifier" in out
    assert "https://docker.local-inference-lab.ai/bench/token" in out
    assert "docker exec --privileged -it -e LIL_BENCH_TOKEN=" in out


def test_malformed_and_rejected_tokens(monkeypatch, capsys, site):
    code, out = run_main(["run"], monkeypatch, capsys, {"LIL_BENCH_TOKEN": "hello"})
    assert code == standard.EXIT_NO_TOKEN and "does not look like" in out
    code, out = run_main(["run", "--site", site.url], monkeypatch, capsys, {"LIL_BENCH_TOKEN": "lilb_" + "z" * 32})
    assert code == standard.EXIT_NO_TOKEN and "rejected" in out and f"{site.url}/bench/token" in out


def test_upload_command(monkeypatch, capsys, site, tmp_path):
    path = tmp_path / "run.json.gz"
    path.write_bytes(upload.compress({"run_id": "x"}))
    code, out = run_main(["upload", str(path), "--site", site.url], monkeypatch, capsys, {"LIL_BENCH_TOKEN": GOOD_TOKEN})
    assert code == 0 and "https://site/bench/runs/r1" in out and site.uploads == [{"run_id": "x"}]


STUB_BENCH = r'''
import json, os, sys, time
args = sys.argv[1:]
out = args[args.index("--output") + 1]
events = os.environ["LLM_BENCH_EVENT_FILE"]
assert os.environ["LLM_BENCH_NO_UPDATE_CHECK"] == "1" and "LIL_BENCH_TOKEN" not in os.environ
def emit(e):
    with open(events, "a") as f:
        f.write(json.dumps({"t": time.time(), "event": e}) + "\n")
    time.sleep(0.05)
emit("prefill start ctx=8k"); emit("prefill done ctx=8k 5,000 tok/s")
emit("cell start C=1 ctx=0"); emit("ready C=1 ctx=0 x"); emit("cell end C=1 ctx=0 tps=100.0")
json.dump({"metadata": {"version": "0.7.0"}, "prefill": {"8192": {"tok_per_sec": 5000.0, "ttft_seconds": 1.6}},
           "results": [{"concurrency": 1, "context_tokens": 0, "aggregate_tps": 100.0, "per_request_avg_tps": 100.0}],
           "argv": args}, open(out, "w"))
print("stub bench done")
'''


@pytest.mark.parametrize("p2pmark", ["disabled", "measured", "reused"])
def test_quick_run_end_to_end_with_stub_bench(monkeypatch, capsys, site, tmp_path, p2pmark):
    stub = tmp_path / "llm_decode_bench.py"
    stub.write_text(STUB_BENCH)
    monkeypatch.setattr(standard, "BENCH", stub)
    monkeypatch.setattr(standard, "RESULT_DIRS", (str(tmp_path / "results"),))
    # The reused case also stands for rank 0 of a server that spans two machines.
    nodes = ["--nnodes", "2", "--master-addr", "10.200.0.11"] if p2pmark == "reused" else []
    monkeypatch.setattr(server, "find_server", lambda: {"pid": os.getpid(), "argv": ["vllm", "serve", "org/M", "--port", "5091", "--max-num-seqs", "4", *nodes]})
    monkeypatch.setattr(server, "server_state", lambda url: {"model_id": "org/M", "max_model_len": 32768, "kv_tokens": 100000})
    monkeypatch.setattr(server, "busy_requests", lambda url: 0.0)
    monkeypatch.setattr(server, "image_identity", lambda: {"alias": "ghcr.io/x:kk-beta", "assembly_sha256": "ab" * 32})
    monkeypatch.setattr(standard.time, "sleep", lambda s: None)
    counters = {"pcie_replay": 0, "pcie_corr_err": 0, "pcie_recovery": 7}
    monkeypatch.setattr(telemetry, "open_source", lambda: FakeSource([[{**row(2800), **counters}] for _ in range(100000)]))
    monkeypatch.setattr(inventory, "collect", lambda: {
        "gpus": [{"index": 0, "name": "GPU", "power": {"enforced_limit_w": 300.0}, "clocks": {}}],
        "tuning": [], "pcie": {"gpu_paths": {}}, "cpu": {"lscpu": {"Model name": "CPU"}}, "memory": {"mem_total_mib": 1024}})
    p2p_calls = []

    def fake_p2pmark(out, hardware, base_url, directory):
        p2p_calls.append((base_url, directory))
        if p2pmark == "measured":
            return {"status": "ok", "ran": True, "data": P2P_DATA}
        return {"status": "ok", "ran": False, "data": P2P_DATA, "reused": {"run_id": "earlier"}}
    monkeypatch.setattr(standard, "run_p2pmark", fake_p2pmark)
    argv = ["run", "--profile", "quick", "--site", site.url] + (["--no-p2pmark"] if p2pmark == "disabled" else [])
    code, out = run_main(argv, monkeypatch, capsys, {"LIL_BENCH_TOKEN": GOOD_TOKEN})
    assert code == 0, out
    assert "uploading as @tester" in out and "Plan: 5 measurements" in out
    assert "PCIe links: no replays, errors or retraining under load" in out
    assert "prefill 8k" in out and "C1 @ 0: 100.0 tok/s" in out and "https://site/bench/runs/r1" in out
    document = site.uploads[0]
    assert document["schema"] == "lil-bench-result/1" and document["profile"] == "quick"
    assert document["status"] == "complete" and document["client"]["uploader"] == "tester"
    assert document["server"]["limits"]["max_num_seqs"] == 4
    if nodes:
        assert document["server"]["nodes"] == {"nnodes": 2, "node_rank": 0, "headless": False}
        assert "one tensor-parallel group across 2 machines; this is node rank 0" in out
    else:
        assert "nodes" not in document["server"]
    assert document["summary"]["decode"][0]["tok_s"] == 100.0
    names = [p["name"] for p in document["phases"]]
    assert names == (["p2pmark"] if p2pmark == "measured" else []) + ["prefill 8k", "decode C1 @ 0"]
    assert all("analysis" in p for p in document["phases"])
    assert document["analysis"]["pcie"][0]["issues"] == []
    measured = document["results"]["p2pmark"]
    assert "ran" not in measured and p2p_calls == ([] if p2pmark == "disabled" else
                                                   [("http://127.0.0.1:5091", tmp_path / "results")])
    assert measured["status"] == ("skipped" if p2pmark == "disabled" else "ok")
    command = document["results"]["bench_command"]
    assert command[command.index("--concurrency") + 1] == "1,2"
    assert command[command.index("--prefill-contexts") + 1] == "8192"
    assert GOOD_TOKEN not in json.dumps(document)
    saved = list((tmp_path / "results").glob("*.json.gz"))
    assert len(saved) == 1 and json.loads(gzip.decompress(saved[0].read_bytes()))["run_id"] == document["run_id"]
    assert not list((tmp_path / "results").glob("*.events.jsonl"))


def test_busy_server_is_refused(monkeypatch, capsys, tmp_path):
    monkeypatch.setattr(server, "find_server", lambda: {"pid": 1, "argv": ["vllm", "serve", "m"]})
    monkeypatch.setattr(server, "server_state", lambda url: {"model_id": "m", "max_model_len": 4096})
    monkeypatch.setattr(server, "busy_requests", lambda url: 3.0)
    monkeypatch.setattr(server, "image_identity", lambda: {})
    monkeypatch.setattr(standard.time, "sleep", lambda s: None)
    code, out = run_main(["run", "--no-upload"], monkeypatch, capsys)
    assert code == standard.EXIT_SERVER and "The server is busy" in out


def test_phase_pcie_errors():
    series = series_of(with_errors([row(2800) for _ in range(20)]))
    gpu = telemetry.analyze_phase(series, 1000.0, 1010.0, [{}])["gpus"][0]
    assert gpu["pcie_errors"] == {"pcie_replay": 3}


# PCIe link states (gen, width, P-state): an idle link sits at Gen1 in P8.
IDLE, LOAD, GEN4 = (1, 16, 8), (5, 16, 1), (4, 16, 1)
GEN5_X16 = [{"max_gen": 5, "max_width": 16}]


def link_series(states, recovery, replay=None):
    """Synthetic telemetry: one GPU, 0.5 s samples, cumulative NVML counters."""
    rows = []
    for index, ((gen, width, pstate), recoveries) in enumerate(zip(states, recovery)):
        sample = row(2800)
        sample.update(pcie_gen=gen, pcie_width=width, pstate=pstate, util_pct=0 if pstate == 8 else 90,
                      pcie_recovery=recoveries, pcie_replay=replay[index] if replay else 0, pcie_corr_err=0)
        rows.append(sample)
    return series_of(rows)


def link_phase(series, links=GEN5_X16, start=1000.0, end=1010.0, errors_from=None):
    return telemetry.analyze_phase(series, start, end, [{}], links=links, errors_from=errors_from)["gpus"][0]


@pytest.mark.parametrize("ramp_at", [3, 4])
def test_idle_link_ramping_up_under_load_is_not_a_recovery_error(ramp_at):
    # The counter moves once when the idle Gen1 link trains up to Gen5 as load starts. NVML may report the
    # increment one sample before the new speed (the counter is read after the link state).
    series = link_series([IDLE] * 4 + [LOAD] * 16, [1253] * ramp_at + [1254] * (20 - ramp_at))
    gpu = link_phase(series)
    assert gpu["pcie_errors"] == {} and gpu["pcie_link_speed_changes"] == 1
    health = telemetry.link_health(telemetry.link_events(series, GEN5_X16))
    assert health[0]["issues"] == [] and health[0]["under_load"] == "Gen5 x16" and health[0]["link_speed_changes"] == 1


def test_link_dropping_back_to_idle_is_not_an_error():
    series = link_series([LOAD] * 10 + [IDLE] * 10, [7] * 10 + [8] * 10)
    assert link_phase(series)["pcie_errors"] == {}


def test_links_without_nvml_counters_are_not_called_healthy():
    health = telemetry.link_health(telemetry.link_events(series_of([row(2800)] * 3)))
    assert health[0]["error_counters"] is False and health[0]["issues"] == []


def test_recoveries_on_a_steady_link_are_errors():
    series = link_series([LOAD] * 20, [100] * 10 + [103] * 10, replay=[0] * 15 + [2] * 5)
    assert link_phase(series)["pcie_errors"] == {"pcie_replay": 2, "pcie_recovery": 3}
    issues = telemetry.link_health(telemetry.link_events(series, GEN5_X16))[0]["issues"]
    assert issues == ["3 link recoveries (retraining) outside speed changes", "2 replayed PCIe packets"]


def test_a_speed_change_explains_only_its_own_recovery():
    # A marginal link retrains many times while it ramps up; only the change itself is power management.
    series = link_series([IDLE] * 4 + [LOAD] * 16, [0] * 4 + [29] * 16)
    gpu = link_phase(series)
    assert gpu["pcie_errors"] == {"pcie_recovery": 27} and gpu["pcie_link_speed_changes"] == 2


def test_recovery_counter_wraps_at_16_bits():
    series = link_series([LOAD] * 6, [65381, 65440, 65513, 45, 123, 198])
    assert link_phase(series)["pcie_errors"] == {"pcie_recovery": 353}
    assert telemetry.increments([65530, 4], 16) == [0, 10]
    assert telemetry.increments([None, 5, None, 7, 2**20, 3]) == [0, 0, 0, 2, 2**20 - 7, 3]  # a reset


def test_link_downgrade_under_load_is_reported():
    # Gen5 under load, then 3 s at Gen4 while still busy (P1): retraining at a steady P-state is not power
    # management, so both retrains and the downgrade are findings.
    series = link_series([LOAD] * 8 + [GEN4] * 6 + [LOAD] * 6, [0] * 8 + [1] * 6 + [2] * 6)
    gpu = link_phase(series)
    assert gpu["pcie_errors"] == {"pcie_recovery": 2, "pcie_downgrade": 1} and "pcie_link_speed_changes" not in gpu
    health = telemetry.link_health(telemetry.link_events(series, GEN5_X16))[0]
    assert health["downgrades"] == 1 and health["issues"] == [
        "2 link recoveries (retraining) outside speed changes", "link dropped to Gen4 x16 under load"]


def test_retraining_at_a_steady_pstate_is_not_excused():
    # A one-sample dip to Gen4 under steady P1, and P0/P2 flapping without any link change.
    dip = link_series([LOAD] * 8 + [GEN4] + [LOAD] * 11, [0] * 8 + [2] * 2 + [4] * 10)
    assert link_phase(dip)["pcie_errors"] == {"pcie_recovery": 4}
    flapping = [(5, 16, 0), (5, 16, 2)] * 10
    series = link_series(flapping, [i // 2 for i in range(20)])
    assert link_phase(series)["pcie_errors"] == {"pcie_recovery": 9}


def test_error_counter_reset_is_not_a_wrap():
    series = link_series([LOAD] * 6, [0] * 6, replay=[276, 276, 0, 0, 1, 1])
    assert link_phase(series)["pcie_errors"] == {"pcie_replay": 1}


def test_idle_gpu_in_a_performance_state_is_not_downgraded():
    # P1 without load while the link sits at Gen1 (no utilization): nothing is running over it.
    idle_p1 = link_series([LOAD] * 4 + [(1, 16, 1)] * 6 + [LOAD] * 4, [0] * 4 + [1] * 6 + [2] * 4)
    for index in range(4, 10):
        idle_p1["gpus"][0]["util_pct"][index] = 0
    assert "pcie_downgrade" not in link_phase(idle_p1)["pcie_errors"]


def test_link_limits_use_the_upstream_port():
    # NVML reports x16 for a Gen4 x16 card in a slot wired x8.
    hardware = {"gpus": [{"bdf": "0000:09:00.0", "pcie": {"max_gen": 4, "max_width": 16}}],
                "pcie": {"gpu_paths": {"0000:09:00.0": {"chain": [
                    {"bdf": "0000:00:03.1", "max_link_speed": "16.0 GT/s PCIe", "max_link_width": "8"},
                    {"bdf": "0000:09:00.0", "max_link_speed": "32.0 GT/s PCIe", "max_link_width": "16"}]}}}}
    assert telemetry.link_limits(hardware) == [{"max_gen": 4, "max_width": 8}]
    assert telemetry.link_limits({"gpus": [{"pcie": {"max_gen": 5, "max_width": 16}}]}) == [
        {"max_gen": 5, "max_width": 16}]


def test_one_sample_below_full_speed_while_ramping_is_not_a_downgrade():
    # P-state already P1 while the link still reads Gen1: values read on both sides of the ramp.
    series = link_series([IDLE] * 3 + [(1, 16, 1)] + [LOAD] * 16, [0] * 3 + [1] * 17)
    assert link_phase(series)["pcie_errors"] == {}


def test_link_below_its_maximum_for_the_whole_run_is_reported_once():
    # A Gen4 x16 link that trains to x8 under load: not a per-phase event, one finding for the run.
    x8 = (4, 8, 1)
    series = link_series([(1, 8, 8)] * 2 + [x8] * 18, [5] * 2 + [6] * 18)
    links = [{"max_gen": 4, "max_width": 16}]  # e.g. a riser that trains only 8 of 16 lanes
    assert link_phase(series, links)["pcie_errors"] == {}
    assert telemetry.link_health(telemetry.link_events(series, links))[0]["issues"] == [
        "Gen4 x8 under load; the link maximum is Gen4 x16"]


def test_errors_count_from_the_cell_start_and_at_window_edges():
    # A genuine retrain seen by the first sample of the window (increment between the samples at
    # 2.0 s and 2.5 s) belongs to the phase; errors_from extends the window to the cell's warmup.
    series = link_series([LOAD] * 20, [0] * 5 + [1] * 15)
    assert link_phase(series, start=1002.2)["pcie_errors"] == {"pcie_recovery": 1}
    assert link_phase(series, start=1004.0)["pcie_errors"] == {}
    assert link_phase(series, start=1004.0, errors_from=1002.0)["pcie_errors"] == {"pcie_recovery": 1}


def test_pcie_recorder_parses_dmon_rounds():
    recorder = telemetry.PcieRecorder(t0=100.0)
    lines = ["# gpu  rxpci  txpci", "# Idx   MB/s   MB/s", "    0   1200   1300", "    1   1100   1000",
             "    0   9000   9100", "    1   8000   8100", "    0      -      -", "    1      5      6"]
    for k, line in enumerate(lines):
        recorder.feed(line, 100.0 + k // 2)
    series = recorder.series()
    assert series["t_ms"] == [1000, 2000, 3000]
    assert series["gpus"][0] == {"rx_mbs": [1200.0, 9000.0, None], "tx_mbs": [1300.0, 9100.0, None]}
    assert series["gpus"][1]["rx_mbs"] == [1100.0, 8000.0, 5.0]
    rates = telemetry.pcie_rates(series, 100.5, 102.5)
    assert rates[0] == {"rx": 5.1, "rx_peak": 9.0, "tx": 5.2, "tx_peak": 9.1}


def test_pcie_recorder_without_nvidia_smi():
    recorder = telemetry.PcieRecorder(0.0, command=["definitely-not-installed"]).start()
    recorder.stop()
    assert recorder.series()["error"] == "definitely-not-installed not found"


def test_server_recorder_samples_metrics_and_cpu(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    stats = iter(["cpu 100 0 100 800 0 0 0 0 0 0\n", "cpu 200 0 200 1000 0 0 0 0 0 0\n"])
    (proc / "stat").write_text(next(stats))
    hwmon = tmp_path / "sys/class/hwmon/hwmon0"
    hwmon.mkdir(parents=True)
    (hwmon / "name").write_text("k10temp\n")
    (hwmon / "temp1_input").write_text("71500\n")
    (hwmon / "temp3_input").write_text("64000\n")
    pages = iter(["vllm:generation_tokens_total{engine=\"0\"} 100\nvllm:generation_tokens_total{engine=\"1\"} 50\n"
                  "vllm:num_requests_running{engine=\"0\"} 8\n", RuntimeError("down")])

    def fetch():
        page = next(pages)
        if isinstance(page, Exception):
            raise page
        return page
    recorder = telemetry.ServerRecorder("http://x", 0.0, fetch=fetch, proc_root=str(proc), sys_root=str(tmp_path / "sys"))
    (proc / "stat").write_text(next(stats))
    recorder.record_once()
    recorder.record_once()
    series = recorder.series()
    assert series["metrics"]["vllm:generation_tokens_total"] == [150.0, None]
    assert series["metrics"]["vllm:num_requests_running"] == [8.0, None]
    assert series["host"]["cpu_pct"][0] == 50.0  # 200 busy of 400 jiffies
    assert series["host"]["cpu_temp_c"] == [71.5, 71.5] and series["errors"] == 1


# ---------------------------------------------------------------------------
# Image integrity
# ---------------------------------------------------------------------------

from lil_bench import integrity  # noqa: E402


def make_reference(tmp_path):
    site = tmp_path / "venv/lib/python3.12/site-packages"
    (site / "vllm").mkdir(parents=True)
    (site / "b12x").mkdir()
    (site / "vllm/model.py").write_text("def forward():\n    return 1\n")
    (site / "vllm/_C.so").write_bytes(b"\0\1\2binary")
    (site / "b12x/kernel.py").write_text("SPEED = 1\n")
    roots = [str(site)]
    files = integrity.scan(roots)
    ref = tmp_path / "runtime-files.json.gz"
    ref.write_bytes(gzip.compress(json.dumps({"schema": "lil-runtime-files/1", "roots": roots, "files": files,
                                              "packages": {}}).encode()))
    return site, ref


def proc_without_mounts(tmp_path, extra=""):
    proc = tmp_path / "proc"
    (proc / "self").mkdir(parents=True)
    (proc / "self/mountinfo").write_text("22 1 0:21 / / rw - overlay overlay rw\n" + extra)
    return proc


def test_integrity_stock_image(tmp_path):
    site, ref = make_reference(tmp_path)
    (site / "vllm/__pycache__").mkdir()
    (site / "vllm/__pycache__/model.cpython-312.pyc").write_bytes(b"cache")
    result = integrity.check(None, str(ref), str(proc_without_mounts(tmp_path)), user_site=str(tmp_path / "none"))
    assert result["status"] == "stock" and result["files_checked"] == 3
    assert result["reference"]["sha256"] == hashlib.sha256(ref.read_bytes()).hexdigest()


def test_integrity_reports_changes_with_content(tmp_path):
    site, ref = make_reference(tmp_path)
    (site / "vllm/model.py").write_text("def forward():\n    return 2  # faster!\n")
    (site / "b12x/kernel.py").unlink()
    (site / "vllm/patch.py").write_text("import vllm\n")
    (site / "vllm/_C.so").write_bytes(b"other binary")
    proc = proc_without_mounts(tmp_path, "90 22 0:50 /patched /opt/venv/lib/python3.12/site-packages/vllm/model.py ro - ext4 /dev/md1 rw\n")
    result = integrity.check(None, str(ref), str(proc), user_site=str(tmp_path / "none"))
    assert result["status"] == "modified"
    changed = {Path(c["path"]).name: c for c in result["changed"]}
    assert changed["model.py"]["content"].endswith("return 2  # faster!\n") and "content" not in changed["_C.so"]
    assert changed["model.py"]["expected_sha256"] != changed["model.py"]["sha256"]
    assert [Path(a["path"]).name for a in result["added"]] == ["patch.py"]
    assert [Path(r["path"]).name for r in result["removed"]] == ["kernel.py"]
    assert result["mounts"][0]["mount_point"].endswith("vllm/model.py")
    lines = integrity.summary_lines(result)
    assert any(line.startswith("changed:") for line in lines) and any("mounted over" in line for line in lines)


def test_integrity_hooks_and_missing_reference(tmp_path):
    proc = proc_without_mounts(tmp_path)
    (proc / "7").mkdir()
    (proc / "7/environ").write_bytes(b"PATH=/usr/bin\0PYTHONPATH=/work/overlay\0LD_PRELOAD=/x.so\0")
    assert integrity.hooks(7, str(proc)) == {"PYTHONPATH": "/work/overlay", "LD_PRELOAD": "/x.so"}
    result = integrity.check(7, str(tmp_path / "missing.json.gz"), str(proc))
    assert result["status"] == "unverified" and result["hooks"]["LD_PRELOAD"] == "/x.so"


def test_integrity_marks_edits_after_server_start(tmp_path, monkeypatch):
    site, ref = make_reference(tmp_path)
    monkeypatch.setattr(integrity, "process_start", lambda pid, proc_root="/proc": time.time() - 60)
    (site / "vllm/model.py").write_text("changed later\n")
    result = integrity.check(1, str(ref), str(proc_without_mounts(tmp_path)), user_site=str(tmp_path / "none"))
    assert result["changed"][0]["after_server_start"] is True


def test_hooks_into_the_verified_image_are_allowed(tmp_path):
    site, ref = make_reference(tmp_path)
    proc = proc_without_mounts(tmp_path)
    (proc / "9").mkdir()
    (proc / "9/environ").write_bytes(f"LD_PRELOAD={site / 'vllm/_C.so'}\0PYTHONPATH={site}:/work/overlay\0".encode())
    result = integrity.check(9, str(ref), str(proc), user_site=str(tmp_path / "none"))
    assert result["hooks_verified"] == {"LD_PRELOAD": str(site / "vllm/_C.so")}
    assert result["hooks"] == {"PYTHONPATH": f"{site}:/work/overlay"} and result["status"] == "modified"


def test_acs_parsing_and_summary(monkeypatch):
    outputs = {
        "0000:ce:01.1": "ce:01.1 PCI bridge: AMD\n\tCapabilities: [2a0] Access Control Services\n"
                        "\t\tACSCap:\tSrcValid+ TransBlk+ ReqRedir+ CmpltRedir+ UpstreamFwd+ EgressCtrl- DirectTrans+\n"
                        "\t\tACSCtl:\tSrcValid- TransBlk- ReqRedir- CmpltRedir- UpstreamFwd- EgressCtrl- DirectTrans-\n",
        "0000:d5:00.0": "d5:00.0 PCI bridge: Broadcom\n\t\tACSCap:\tSrcValid+ ReqRedir+ CmpltRedir+\n"
                        "\t\tACSCtl:\tSrcValid+ TransBlk- ReqRedir+ CmpltRedir+ UpstreamFwd+ EgressCtrl- DirectTrans-\n",
        "0000:d6:00.0": "d6:00.0 VGA compatible controller: NVIDIA\n\tCapabilities: <access denied>\n",
    }
    monkeypatch.setattr(inventory.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(inventory, "run", lambda cmd, timeout=15.0, limit=200_000: {"stdout": outputs[cmd[-1]]})
    detail = inventory.lspci_detail(list(outputs))
    assert detail["0000:ce:01.1"]["acs_ctl"]["ReqRedir"] is False and detail["0000:ce:01.1"]["acs_redirect"] == []
    assert detail["0000:d5:00.0"]["acs_redirect"] == ["ReqRedir", "CmpltRedir"]
    summary = inventory.acs_summary(detail)
    assert summary == {"readable": False, "unreadable": ["0000:d6:00.0"],
                       "hops_with_acs": ["0000:ce:01.1", "0000:d5:00.0"],
                       "redirect": {"0000:d5:00.0": ["ReqRedir", "CmpltRedir"]}}


# ---------------------------------------------------------------------------
# GB10 (DGX Spark): integrated GPU, unified memory, several machines
# ---------------------------------------------------------------------------

class NVMLError(Exception):
    pass


class FakeNVML:
    """pynvml with a GB10 (no memory info) and an RTX PRO 6000; unknown calls are not supported."""

    NVML_CLOCK_SM, NVML_CLOCK_MEM, NVML_CLOCK_GRAPHICS = 1, 2, 0

    def __init__(self, gb10_memory=None):
        self.gb10_memory = gb10_memory
        self.devices = [{"name": "NVIDIA GB10", "bus": "0000000F:01:00.0", "memory": gb10_memory},
                        {"name": "NVIDIA RTX PRO 6000 Blackwell Workstation Edition", "bus": "00000000:41:00.0",
                         "memory": type("Memory", (), {"total": 97887 * 2**20, "free": 1000 * 2**20})()}]

    def __getattr__(self, name):
        def unsupported(*args):
            raise NVMLError("Not Supported")
        return unsupported

    def nvmlInit(self):
        pass

    def nvmlDeviceGetCount(self):
        return len(self.devices)

    def nvmlDeviceGetHandleByIndex(self, index):
        return self.devices[index]

    def nvmlDeviceGetName(self, handle):
        return handle["name"]

    def nvmlDeviceGetPciInfo_v3(self, handle):
        return type("Pci", (), {"busId": handle["bus"].encode(), "pciDeviceId": 0x2B8510DE, "pciSubSystemId": 0})()

    def nvmlDeviceGetMemoryInfo(self, handle):
        if handle["memory"] is None:
            raise NVMLError("Not Supported")
        return handle["memory"]

    def nvmlDeviceGetMaxPcieLinkGeneration(self, handle):
        return 5

    def nvmlDeviceGetMaxPcieLinkWidth(self, handle):
        return 16

    def nvmlDeviceGetCurrPcieLinkGeneration(self, handle):
        return 1

    def nvmlDeviceGetCurrPcieLinkWidth(self, handle):
        return 1


def meminfo_proc(tmp_path):
    proc = tmp_path / "proc"
    proc.mkdir()
    (proc / "meminfo").write_text("MemTotal:       125751720 kB\nMemAvailable:   98000000 kB\n")
    return str(proc)


@pytest.mark.parametrize("gb10_memory", [None, type("Memory", (), {"total": 0, "free": 0})()])
def test_gb10_reports_unified_memory_and_no_pcie_link(monkeypatch, tmp_path, gb10_memory):
    monkeypatch.setitem(sys.modules, "pynvml", FakeNVML(gb10_memory))
    gb10, rtx = inventory.nvml_gpus(meminfo_proc(tmp_path))["gpus"]
    assert gb10["integrated"] is True and gb10["unified_memory"] is True
    assert gb10["memory_total_mib"] == 125751720 // 1024 and gb10["bdf"] == "000f:01:00.0"
    assert gb10["pcie"] == dict.fromkeys(inventory.PCIE_KEYS)
    # Discrete GPUs keep exactly what they reported before.
    assert "integrated" not in rtx and "unified_memory" not in rtx
    assert rtx["memory_total_mib"] == 97887
    assert rtx["pcie"] == {"max_gen": 5, "gpu_max_gen": None, "max_width": 16, "current_gen": 1,
                           "current_width": 1, "replay_counter": None}


def test_gb10_has_no_pcie_path(monkeypatch, tmp_path):
    gpus = [{"index": 0, "name": "NVIDIA GB10", "bdf": "000f:01:00.0", "integrated": True}]
    monkeypatch.setattr(inventory, "nvml_gpus", lambda proc_root: {"gpus": gpus})
    monkeypatch.setattr(inventory, "run", lambda cmd, **kwargs: {"cmd": cmd, "error": "not installed"})
    hardware = inventory.collect(str(fake_sysfs(tmp_path)), meminfo_proc(tmp_path))
    assert hardware["pcie"]["gpu_paths"] == {} and hardware["pcie"]["lspci_detail"] == {}


def test_gb10_link_is_never_a_downgrade():
    hardware = {"gpus": [{"bdf": "000f:01:00.0", "integrated": True, "pcie": dict.fromkeys(inventory.PCIE_KEYS)}]}
    links = telemetry.link_limits(hardware)
    assert links == [{"max_gen": None, "max_width": None, "integrated": True}]
    # Whatever NVML samples for the integrated GPU's link, falling from it is not reported.
    series = link_series([LOAD] * 8 + [GEN4] * 12, [0] * 20)
    health = telemetry.link_health(telemetry.link_events(series, links))[0]
    assert health["downgrades"] == 0 and health["issues"] == [] and health["link_max"] is None


def test_p2pmark_on_one_gpu_without_memory_info_is_the_single_gpu_skip():
    out = Lines()
    hardware = {"gpus": [{"index": 0, "name": "NVIDIA GB10", "integrated": True}]}
    result = p2p.measure(out, hardware, FakeBench(), free_fn=lambda: [])
    assert result == {"status": "skipped", "reason": "single GPU: nothing to measure between GPUs",
                      "free_mib": [], "ran": False}
    # Without NVML the number of GPUs is unknown, and so is the reason.
    unknown = p2p.measure(Lines(), {"gpus": []}, FakeBench(), free_fn=lambda: [], proc_root="/nonexistent")
    assert unknown["status"] == "skipped" and unknown["reason"] == "free GPU memory is unknown (NVML unavailable)"


@pytest.mark.parametrize("args, nodes", [
    (["--tensor-parallel-size", "2"], None),
    (["--nnodes", "1"], None),
    (["--nnodes", "2", "--master-addr", "10.200.0.11"], {"nnodes": 2, "node_rank": 0, "headless": False}),
    (["--nnodes=2", "--node-rank=1", "--headless"], {"nnodes": 2, "node_rank": 1, "headless": True}),
    (["--nnodes", "x"], None),
])
def test_serve_nodes_from_argv(args, nodes):
    options = server.parse_serve_args(["vllm", "serve", "org/M", *args])["options"]
    assert server.serve_nodes(options) == nodes


def test_a_worker_rank_is_refused_with_directions(monkeypatch, capsys):
    monkeypatch.setattr(server, "find_server", lambda: {"pid": 1, "argv": [
        "vllm", "serve", "m", "--nnodes", "2", "--node-rank", "1", "--headless"]})
    code, out = run_main(["run", "--no-upload"], monkeypatch, capsys)
    assert code == standard.EXIT_SERVER and "node rank 1" in out and "node rank 0, which serves the API" in out
