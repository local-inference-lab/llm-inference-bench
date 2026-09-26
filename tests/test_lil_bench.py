"""lil-bench: plan, discovery, inventory, telemetry, upload and the run itself."""

from __future__ import annotations

import gzip
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lil_bench import inventory, server, standard, telemetry, upload  # noqa: E402

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


@pytest.mark.parametrize("free,size", [([1500, 1600], 64), ([4000, 4000], 256), ([1100, 9000], 16), ([1000, 1000], None), ([90000], None)])
def test_p2p_buffer_size(free, size):
    assert standard.p2p_buffer_size(free)[0] == size


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
    assert summary == {"hops": 4, "bridges": 3, "behind_switch": True, "min_current_gt_s": 16.0, "min_current_width": 16}
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
    assert "docker exec -it -e LIL_BENCH_TOKEN=" in out


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


def test_quick_run_end_to_end_with_stub_bench(monkeypatch, capsys, site, tmp_path):
    stub = tmp_path / "llm_decode_bench.py"
    stub.write_text(STUB_BENCH)
    monkeypatch.setattr(standard, "BENCH", stub)
    monkeypatch.setattr(standard, "RESULT_DIRS", (str(tmp_path / "results"),))
    monkeypatch.setattr(server, "find_server", lambda: {"pid": os.getpid(), "argv": ["vllm", "serve", "org/M", "--port", "5091", "--max-num-seqs", "4"]})
    monkeypatch.setattr(server, "server_state", lambda url: {"model_id": "org/M", "max_model_len": 32768, "kv_tokens": 100000})
    monkeypatch.setattr(server, "busy_requests", lambda url: 0.0)
    monkeypatch.setattr(server, "image_identity", lambda: {"alias": "ghcr.io/x:kk-beta", "assembly_sha256": "ab" * 32})
    monkeypatch.setattr(standard.time, "sleep", lambda s: None)
    monkeypatch.setattr(telemetry, "open_source", lambda: FakeSource([[row(2800)] for _ in range(100000)]))
    monkeypatch.setattr(inventory, "collect", lambda: {
        "gpus": [{"index": 0, "name": "GPU", "power": {"enforced_limit_w": 300.0}, "clocks": {}}],
        "tuning": [], "pcie": {"gpu_paths": {}}, "cpu": {"lscpu": {"Model name": "CPU"}}, "memory": {"mem_total_mib": 1024}})
    code, out = run_main(["run", "--profile", "quick", "--site", site.url, "--no-p2pmark"], monkeypatch, capsys,
                         {"LIL_BENCH_TOKEN": GOOD_TOKEN})
    assert code == 0, out
    assert "uploading as @tester" in out and "Plan: 5 measurements" in out
    assert "prefill 8k" in out and "C1 @ 0: 100.0 tok/s" in out and "https://site/bench/runs/r1" in out
    document = site.uploads[0]
    assert document["schema"] == "lil-bench-result/1" and document["profile"] == "quick"
    assert document["status"] == "complete" and document["client"]["uploader"] == "tester"
    assert document["server"]["limits"]["max_num_seqs"] == 4
    assert document["summary"]["decode"][0]["tok_s"] == 100.0
    assert [p["name"] for p in document["phases"]] == ["prefill 8k", "decode C1 @ 0"]
    assert all("analysis" in p for p in document["phases"])
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
