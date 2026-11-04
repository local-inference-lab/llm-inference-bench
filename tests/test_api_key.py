"""API keys: lil-bench measures a server started with one and never records or uploads the key."""

from __future__ import annotations

import json
import os
import sys
import threading
import types
import urllib.error
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from lil_bench import flare, inventory, server, standard, telemetry  # noqa: E402
from test_flare import DEVICES, FakeDocker, container_doc, package_dir  # noqa: E402
from test_lil_bench import GOOD_TOKEN, FakeSource, row, run_main, site  # noqa: E402,F401

KEY = "sk-lil-" + "k3y" * 8


class KeyServer:
    """An OpenAI-style API that answers only with ``Authorization: Bearer KEY`` (HTTP 401 otherwise).

    ``open_metrics``: /metrics and /version answer without the key, as on vLLM.
    ``redirect``: every request is redirected to that server instead.
    Every request's path and Authorization header are kept in ``requests``.
    """

    def __init__(self, open_metrics=True, flare=False, redirect=None):
        self.requests: list[tuple[str, str | None]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, body, kind="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", kind)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                auth = self.headers.get("Authorization")
                fake.requests.append((self.path, auth))
                if redirect:
                    self.send_response(302)
                    self.send_header("Location", redirect + self.path)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                if auth != f"Bearer {KEY}" and not (open_metrics and self.path in ("/metrics", "/version")):
                    return self.reply(401, {"error": {"message": "missing or invalid API key",
                                                      "type": "authentication_error"}})
                if self.path == "/v1/models":
                    entry = {"id": "org/M", "max_model_len": 32768, **({"flare": {"max_concurrency": 4}} if flare else {})}
                    return self.reply(200, {"object": "list", "data": [entry]})
                if self.path == "/version":
                    return self.reply(200, {"version": "0.0.0-key"})
                if self.path == "/metrics":
                    text = ("flare_active_requests 0\nflare_pending_requests 0\nflare_total_kv_tokens 100000\n" if flare
                            else 'vllm:num_requests_running{engine="0"} 0\nvllm:num_requests_waiting{engine="0"} 0\n'
                                 'vllm:cache_config_info{block_size="16",kv_cache_size_tokens="100000",'
                                 'num_gpu_blocks="6250"} 1\n')
                    return self.reply(200, text.encode(), "text/plain")
                return self.reply(404, {"error": "not found"})

        self.httpd = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self.httpd.server_address[1]}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


@pytest.fixture
def api():
    fake = KeyServer()
    yield fake
    fake.close()


# ---------------------------------------------------------------------------
# lil-bench's own requests
# ---------------------------------------------------------------------------

def test_probes_send_the_key(api):
    with pytest.raises(urllib.error.HTTPError) as error:
        server.server_state(api.url)
    assert error.value.code == 401
    state = server.server_state(api.url, api_key=KEY)
    assert state["model_id"] == "org/M" and state["version"] == "0.0.0-key" and state["kv_tokens"] == 100000
    assert server.busy_requests(api.url, api_key=KEY) == 0.0
    assert ("/v1/models", f"Bearer {KEY}") in api.requests and ("/metrics", f"Bearer {KEY}") in api.requests


def test_a_flare_server_whose_metrics_need_the_key():
    fake = KeyServer(open_metrics=False, flare=True)
    try:
        assert server.is_flare(fake.url) is False  # neither /v1/models nor /metrics answer
        assert server.is_flare(fake.url, api_key=KEY) is True
        recorder = telemetry.ServerRecorder(fake.url, 0.0, names=telemetry.FLARE_SERVER_METRICS, api_key=KEY)
        recorder.record_once()
        assert recorder.errors == 0 and recorder.series()["metrics"]["flare_active_requests"] == [0.0]
        without = telemetry.ServerRecorder(fake.url, 0.0, names=telemetry.FLARE_SERVER_METRICS)
        without.record_once()
        assert without.errors == 1
    finally:
        fake.close()


def test_the_key_does_not_follow_a_redirect(api):
    redirecting = KeyServer(redirect=api.url)
    try:
        with pytest.raises(urllib.error.HTTPError) as error:
            server.server_state(redirecting.url, api_key=KEY)
        assert error.value.code == 401
        assert redirecting.requests == [("/v1/models", f"Bearer {KEY}")] and api.requests == [("/v1/models", None)]
    finally:
        redirecting.close()


# ---------------------------------------------------------------------------
# Where the key comes from
# ---------------------------------------------------------------------------

def args_of(*argv):
    return standard.parse_args(["run", *argv])


def test_the_key_comes_from_a_file_or_the_environment(monkeypatch, tmp_path):
    monkeypatch.delenv("LIL_BENCH_API_KEY", raising=False)
    assert standard.read_api_key(args_of()) == ("", None)
    monkeypatch.setenv("LIL_BENCH_API_KEY", f" {KEY}\n")
    assert standard.read_api_key(args_of()) == (KEY, "LIL_BENCH_API_KEY")
    path = tmp_path / "flare-api-key"
    path.write_text("key-from-the-file-0123\n")  # Flare trims its --api-key-file the same way
    assert standard.read_api_key(args_of("--api-key-file", str(path))) == (
        "key-from-the-file-0123", f"--api-key-file {path}")


@pytest.mark.parametrize("content, message", [
    (b"", "is empty"), (b" \n", "is empty"), (b"first-line\nsecond-line\n", "several lines"),
    (b"\xffk3y", "not UTF-8"),
])
def test_unusable_key_files_are_refused_without_quoting_them(tmp_path, content, message):
    path = tmp_path / "key"
    path.write_bytes(content)
    with pytest.raises(ValueError, match=message) as error:
        standard.read_api_key(args_of("--api-key-file", str(path)))
    assert not any(text in str(error.value) for text in ("first-line", "second-line", "k3y"))


def test_a_missing_key_file_is_refused(tmp_path):
    with pytest.raises(ValueError, match="cannot read .*missing: No such file"):
        standard.read_api_key(args_of("--api-key-file", str(tmp_path / "missing")))


@pytest.mark.parametrize("argv", [["--api-key", KEY], [f"--api-key={KEY}"]])
def test_a_key_on_the_command_line_is_refused(capsys, argv):
    with pytest.raises(SystemExit) as exit_:
        standard.parse_args(["run", *argv])
    error = capsys.readouterr().err
    assert exit_.value.code == 2 and "use --api-key-file PATH or set LIL_BENCH_API_KEY" in error and KEY not in error


# ---------------------------------------------------------------------------
# What is recorded
# ---------------------------------------------------------------------------

def test_credentials_in_serving_commands_are_masked():
    argv = ["vllm", "serve", "org/M", "--api-key", "k1", "k2", "--port", "8000", "--hf-token=hf_x",
            "--ssl-keyfile", "/k.pem", "--max-num-batched-tokens", "8192", "--api_key=k3"]
    assert server.redact_argv(argv) == ["vllm", "serve", "org/M", "--api-key", "***", "***", "--port", "8000",
                                        "--hf-token=***", "--ssl-keyfile", "/k.pem", "--max-num-batched-tokens",
                                        "8192", "--api_key=***"]
    options = server.parse_serve_args(argv)["options"]
    assert server.redact_options(options) == {**options, "api-key": "***", "hf-token": "***"}
    # A key file path and the publisher's key fingerprint are not secrets.
    flare_argv = ["/usr/local/bin/flare", "run", "/models/model.safetensors", "--api-key-file",
                  "/run/secrets/flare_api_key", "--dangerously-approve-key", "9" * 64, "--port", "8000"]
    assert server.redact_argv(flare_argv) == flare_argv


def test_a_flare_containers_key_file_stays_and_key_values_go(tmp_path):
    package_dir(tmp_path)
    doc = container_doc(tmp_path)
    doc["Args"] += ["--api-key-file", "/run/secrets/flare_api_key", "--api-key", "abc123"]
    doc["Config"]["Env"] += ["FLARE_API_KEY=abc123", "VLLM_API_KEY=abc123"]
    described = flare.describe({"flare": {"max_concurrency": 8}}, "http://127.0.0.1:5092",
                               docker=FakeDocker([doc]), devices=DEVICES)
    recorded = described["server"]
    assert recorded["argv"][-4:] == ["--api-key-file", "/run/secrets/flare_api_key", "--api-key", "***"]
    assert recorded["options"]["api-key-file"] == "/run/secrets/flare_api_key" and recorded["options"]["api-key"] == "***"
    assert "abc123" not in json.dumps(described, default=str)


def test_key_variables_are_never_recorded():
    items = ["VLLM_API_KEY=k", "OPENAI_API_KEY=k", "LIL_BENCH_API_KEY=k", "VLLM_USE_B12X=1"]
    assert server.filter_environment(items) == {"VLLM_USE_B12X": "1"}


def test_lil_bench_argv_masks_its_token_and_a_key(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["lil-bench", "run", "--token", "lilb_x", f"--api-key={KEY}", "--tok", "lilb_y",
                                      "--api-key", KEY, "--api-key-file", "/run/secrets/k", "--note", "n"])
    assert standard._safe_argv() == ["run", "--token", "***", "--api-key=***", "--tok", "***", "--api-key", "***",
                                     "--api-key-file", "/run/secrets/k", "--note", "n"]


def test_scrub_masks_every_copy():
    document = {"a": [f"Bearer {KEY}", 1, None, {KEY: (KEY,)}], "b": 2.5}
    assert standard.scrub(document, [KEY]) == {"a": ["Bearer ***", 1, None, {"***": ["***"]}], "b": 2.5}
    assert standard.scrub(document, []) is document


def test_the_benchmark_takes_its_key_from_the_environment(monkeypatch):
    bench = standard.load_bench_module()
    monkeypatch.setenv("LLM_BENCH_API_KEY", KEY)
    monkeypatch.setattr(sys, "argv", ["llm_decode_bench.py"])
    assert bench.parse_args().api_key == KEY
    monkeypatch.setattr(sys, "argv", ["llm_decode_bench.py", "--api-key", "other"])
    assert bench.parse_args().api_key == "other"


def test_benchmark_diagnostics_leave_out_key_variables(monkeypatch):
    bench = standard.load_bench_module()
    monkeypatch.setattr(bench.shutil, "which", lambda name: None)  # no nvidia-smi queries
    for name, value in (("VLLM_API_KEY", KEY), ("NCCL_SOCKET_TOKEN", KEY), ("VLLM_USE_B12X", "1")):
        monkeypatch.setenv(name, value)
    args = types.SimpleNamespace(concurrency="1", contexts="0", max_tokens=16, duration=1.0)
    env = bench.collect_startup_diagnostics(args, "http://x")["env"]
    assert env["VLLM_USE_B12X"] == "1" and "VLLM_API_KEY" not in env and "NCCL_SOCKET_TOKEN" not in env


# ---------------------------------------------------------------------------
# The whole run
# ---------------------------------------------------------------------------

STUB_BENCH = r'''
import json, os, sys, time
args = sys.argv[1:]
key = os.environ["LLM_BENCH_API_KEY"]
assert not any(key in arg for arg in args), args
assert "LIL_BENCH_API_KEY" not in os.environ and "LIL_BENCH_TOKEN" not in os.environ
out, events = args[args.index("--output") + 1], os.environ["LLM_BENCH_EVENT_FILE"]
for event in ("cell start C=1 ctx=0", "ready C=1 ctx=0 x", "cell end C=1 ctx=0 tps=100.0"):
    with open(events, "a") as f:
        f.write(json.dumps({"t": time.time(), "event": event}) + "\n")
# A benchmark that echoes the key in its log and its result must not leak it.
print("request headers: Authorization: Bearer " + key)
json.dump({"metadata": {"version": "0.8.0"}, "startup_diagnostics": {"env": {"VLLM_API_KEY": key}},
           "results": [{"concurrency": 1, "context_tokens": 0, "aggregate_tps": 100.0}]}, open(out, "w"))
sys.exit(int(os.environ.get("STUB_EXIT", "0")))
'''


@pytest.mark.parametrize("fails", [False, True])
def test_a_run_against_a_server_with_a_key(monkeypatch, capsys, site, tmp_path, fails):  # noqa: F811
    api = KeyServer(open_metrics=False)
    stub = tmp_path / "llm_decode_bench.py"
    stub.write_text(STUB_BENCH)
    monkeypatch.setattr(standard, "BENCH", stub)
    monkeypatch.setattr(standard, "RESULT_DIRS", (str(tmp_path / "results"),))
    # The recorded command is masked by itself, also where a key is too short to be masked everywhere.
    argv = ["vllm", "serve", "org/M", "--api-key", "abc123", "--max-num-seqs", "4", "--hf-token=hf_x"]
    monkeypatch.setattr(server, "find_server", lambda: {"pid": os.getpid(), "argv": argv})
    monkeypatch.setattr(server, "image_identity", lambda: {"alias": "ghcr.io/x:kk", "assembly_sha256": "ab" * 32})
    monkeypatch.setattr(standard.time, "sleep", lambda s: None)
    monkeypatch.setattr(telemetry, "open_source", lambda: FakeSource([[row(2800)] for _ in range(100000)]))
    monkeypatch.setattr(inventory, "collect", lambda: {
        "gpus": [{"index": 0, "name": "GPU", "power": {"enforced_limit_w": 300.0}, "clocks": {}}],
        "tuning": [], "pcie": {"gpu_paths": {}}, "cpu": {"lscpu": {"Model name": "CPU"}}, "memory": {"mem_total_mib": 1024}})
    if fails:
        monkeypatch.setenv("STUB_EXIT", "1")
    try:
        code, out = run_main(["run", "--profile", "quick", "--site", site.url, "--url", api.url, "--no-p2pmark"],
                             monkeypatch, capsys, {"LIL_BENCH_TOKEN": GOOD_TOKEN, "LIL_BENCH_API_KEY": KEY})
    finally:
        api.close()
    assert code == (standard.EXIT_BENCH if fails else 0), out
    assert "API key from LIL_BENCH_API_KEY" in out and KEY not in out
    if fails:
        assert "The benchmark failed" in out and "Authorization: Bearer ***" in out
    # Every request lil-bench made carried the key (this server answers nothing without it).
    assert api.requests and all(auth == f"Bearer {KEY}" for _, auth in api.requests), api.requests
    document = site.uploads[0]
    assert not any(secret in json.dumps(document) for secret in (KEY, "abc123", "hf_x"))
    assert document["server"]["argv"] == ["vllm", "serve", "org/M", "--api-key", "***", "--max-num-seqs", "4",
                                          "--hf-token=***"]
    assert document["server"]["options"]["api-key"] == "***" and document["server"]["limits"]["max_num_seqs"] == 4
    assert document["server"]["api"]["model_id"] == "org/M" and document["server_telemetry"]["errors"] == 0
    assert document["results"]["bench"]["startup_diagnostics"]["env"] == {"VLLM_API_KEY": "***"}
    assert "Authorization: Bearer ***" in document["results"]["bench_log_tail"]
    assert not any(KEY in arg for arg in document["results"]["bench_command"])


def test_a_missing_or_wrong_key_is_named(monkeypatch, capsys, api):
    monkeypatch.setattr(server, "find_server", lambda: None)
    code, out = run_main(["run", "--no-upload", "--url", api.url], monkeypatch, capsys)
    assert code == standard.EXIT_SERVER and "The server needs an API key" in out and "HTTP 401" in out
    assert "lil-bench --api-key-file PATH, or LIL_BENCH_API_KEY" in out
    wrong = "wrong-key-0123456789"
    code, out = run_main(["run", "--no-upload", "--url", api.url], monkeypatch, capsys, {"LIL_BENCH_API_KEY": wrong})
    assert code == standard.EXIT_SERVER and "The server rejected the API key" in out and wrong not in out
    assert "The key from LIL_BENCH_API_KEY is not the one the server was started with." in out


def test_an_unreadable_key_file_stops_the_run(monkeypatch, capsys, tmp_path):
    code, out = run_main(["run", "--no-upload", "--api-key-file", str(tmp_path / "missing")], monkeypatch, capsys)
    assert code == standard.EXIT_SERVER and "Cannot read the API key" in out and "No such file" in out
