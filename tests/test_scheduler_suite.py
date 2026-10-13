"""Tests for the scheduler tests (--scheduler).

FakeSchedulerServer is an OpenAI-compatible SSE stub with a stepped engine: at
most max_num_seqs requests run, the rest wait FCFS; every step prefills up to
prefill_chunk prompt tokens (FCFS over the running requests) and decodes one
token for every running request past its prefill. A step that prefills takes
longer, so decode slows down while a long prompt is in prefill. /metrics
exports the vLLM gauges, counters and a TTFT histogram the bench samples;
/tokenize and the dev endpoint /prefill_fairness are optional. With
wedge_after=N the engine stops stepping after admitting N requests, like the
lane deadlock of vLLM #959: requests stay running and nothing moves.
No GPU or model is needed.
"""
import importlib.util
import io
import json
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import queue
import sys
import threading
import time

import pytest
from rich.console import Console


MODULE_PATH = Path(__file__).resolve().parents[1] / "llm_decode_bench.py"
SPEC = importlib.util.spec_from_file_location("llm_decode_bench_scheduler", MODULE_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCH)

TTFT_BUCKETS = (0.05, 0.1, 0.5, 1.0, 5.0)


class FakeRequest:
    def __init__(self, prompt_tokens, max_tokens, token_ids, chat, images):
        self.prompt_tokens = prompt_tokens
        self.max_tokens = max_tokens
        self.token_ids = token_ids
        self.chat = chat
        self.images = images
        self.remaining_prefill = prompt_tokens
        self.generated = 0
        self.events = queue.Queue()
        self.aborted = False
        self.t_arrival = time.monotonic()


class FakeSchedulerServer:
    def __init__(self, *, max_num_seqs=4, prefill_chunk=8192, prefill_tok_s=2_000_000.0, step_s=0.003,
                 max_model_len=300_000, kv_tokens=2_000_000, wedge_after=None, tokenize=True,
                 fairness=True, metrics=True, fairness_share="auto"):
        self.max_num_seqs = max_num_seqs
        self.prefill_chunk = prefill_chunk
        self.prefill_tok_s = prefill_tok_s
        self.step_s = step_s
        self.max_model_len = max_model_len
        self.kv_tokens = kv_tokens
        self.wedge_after = wedge_after
        self.tokenize = tokenize
        self.fairness = fairness
        self.metrics = metrics
        self.fairness_state = {"prefill_compute_share": fairness_share, "prefill_compute_half_life": "smooth",
                               "max_parallel_prefills": 2, "effective_max_parallel_prefills": 2,
                               "prefill_policy": "fcfs"}
        self.fairness_posts = []
        self.lock = threading.Lock()
        self.waiting = deque()
        self.running = []
        self.admitted = 0
        self.wedged = False
        self.prompt_tokens_total = 0
        self.generation_tokens_total = 0
        self.success_total = 0
        self.aborted_total = 0
        self.ttfts = []
        self.prompts = []
        self.chat_requests = []
        self.closed = False
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, code, body, content_type="application/json"):
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(code)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def do_GET(self):
                if self.path == "/v1/models":
                    return self.reply(200, {"data": [{"id": "fake-model", "max_model_len": server.max_model_len}]})
                if self.path == "/version":
                    return self.reply(200, {"version": "0.0.0-fake"})
                if self.path == "/metrics" and server.metrics:
                    return self.reply(200, server.metrics_text().encode(), "text/plain")
                if self.path == "/prefill_fairness" and server.fairness:
                    with server.lock:
                        return self.reply(200, dict(server.fairness_state))
                return self.reply(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/tokenize" and server.tokenize:
                    return self.reply(200, {"tokens": list(range(100, 400)), "count": 300})
                if self.path == "/prefill_fairness" and server.fairness:
                    with server.lock:
                        server.fairness_posts.append(payload)
                        server.fairness_state["prefill_compute_share"] = payload.get("prefill_compute_share")
                        if "prefill_compute_half_life" in payload:
                            server.fairness_state["prefill_compute_half_life"] = payload["prefill_compute_half_life"]
                        return self.reply(200, {**server.fairness_state, "applied": True})
                if self.path in ("/v1/completions", "/v1/chat/completions"):
                    return server.stream(self, payload, chat=self.path.endswith("chat/completions"))
                return self.reply(404, {"error": "not found"})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()
        self.engine = threading.Thread(target=self.engine_loop, daemon=True)
        self.engine.start()

    def close(self):
        self.closed = True
        self.httpd.shutdown()
        self.httpd.server_close()

    # --- engine -------------------------------------------------------------

    def engine_loop(self):
        while not self.closed:
            prefilled = 0
            with self.lock:
                if self.wedged:
                    pass
                else:
                    for r in [r for r in self.running if r.aborted]:
                        self.running.remove(r)
                    self.waiting = deque(r for r in self.waiting if not r.aborted)
                    while self.waiting and len(self.running) < self.max_num_seqs:
                        self.running.append(self.waiting.popleft())
                        self.admitted += 1
                        if self.wedge_after and self.admitted >= self.wedge_after:
                            self.wedged = True
                    if not self.wedged:
                        budget = self.prefill_chunk
                        finished = []
                        for r in self.running:
                            if r.remaining_prefill > 0:
                                take = min(budget, r.remaining_prefill)
                                r.remaining_prefill -= take
                                budget -= take
                                prefilled += take
                                if r.remaining_prefill == 0:
                                    self.prompt_tokens_total += r.prompt_tokens
                                    self.ttfts.append(time.monotonic() - r.t_arrival)
                                    self.emit(r, finished)
                            else:
                                self.emit(r, finished)
                        for r in finished:
                            self.running.remove(r)
                            self.success_total += 1
            time.sleep(self.step_s + prefilled / self.prefill_tok_s)

    def emit(self, r, finished):
        r.generated += 1
        self.generation_tokens_total += 1
        done = r.generated >= r.max_tokens
        r.events.put((r.generated, done))
        if done:
            finished.append(r)

    def metrics_text(self):
        with self.lock:
            running, waiting = len(self.running), len(self.waiting)
            used = sum(r.prompt_tokens - r.remaining_prefill + r.generated for r in self.running)
            lines = [
                f'vllm:num_requests_running{{engine="0",model_name="fake-model"}} {running}',
                f'vllm:num_requests_waiting{{engine="0",model_name="fake-model"}} {waiting}',
                f'vllm:num_requests_waiting_by_reason{{engine="0",model_name="fake-model",reason="capacity"}} {waiting}',
                'vllm:num_requests_waiting_by_reason{engine="0",model_name="fake-model",reason="deferred"} 0',
                f'vllm:kv_cache_usage_perc{{engine="0",model_name="fake-model"}} {used / self.kv_tokens}',
                'vllm:num_preemptions_total{engine="0",model_name="fake-model"} 0',
                f'vllm:prompt_tokens_total{{engine="0",model_name="fake-model"}} {self.prompt_tokens_total}',
                f'vllm:generation_tokens_total{{engine="0",model_name="fake-model"}} {self.generation_tokens_total}',
                f'vllm:prefix_cache_queries_total{{engine="0",model_name="fake-model"}} {self.prompt_tokens_total}',
                'vllm:prefix_cache_hits_total{engine="0",model_name="fake-model"} 0',
                f'vllm:request_success_total{{engine="0",finished_reason="length",model_name="fake-model"}} {self.success_total}',
                f'vllm:cache_config_info{{block_size="16",engine="0",kv_cache_size_tokens="{self.kv_tokens}",'
                f'num_gpu_blocks="{self.kv_tokens // 16}"}} 1.0',
            ]
            for bound in TTFT_BUCKETS:
                count = sum(1 for t in self.ttfts if t <= bound)
                lines.append(f'vllm:time_to_first_token_seconds_bucket{{engine="0",le="{bound}"}} {count}')
            lines.append(f'vllm:time_to_first_token_seconds_bucket{{engine="0",le="+Inf"}} {len(self.ttfts)}')
            lines.append(f'vllm:time_to_first_token_seconds_count{{engine="0"}} {len(self.ttfts)}')
            lines.append(f'vllm:time_to_first_token_seconds_sum{{engine="0"}} {sum(self.ttfts)}')
        return "\n".join(lines) + "\n"

    def busy(self):
        with self.lock:
            return len(self.running) + len(self.waiting)

    # --- HTTP streaming -------------------------------------------------------

    def stream(self, handler, payload, *, chat):
        images = 0
        if chat:
            tokens = 0
            for message in payload.get("messages") or []:
                content = message.get("content")
                parts = content if isinstance(content, list) else [{"type": "text", "text": str(content or "")}]
                for part in parts:
                    if part.get("type") == "image_url":
                        images += 1
                        tokens += 256
                    else:
                        tokens += len(part.get("text") or "") // 4
            prompt_tokens = max(1, tokens)
        else:
            prompt = payload.get("prompt")
            prompt_tokens = len(prompt) if isinstance(prompt, list) else max(1, len(str(prompt)) // 4)
            with self.lock:
                self.prompts.append(prompt if isinstance(prompt, list) else None)
        if prompt_tokens + int(payload.get("max_tokens") or 16) > self.max_model_len:
            return handler.reply(400, {"error": "prompt too long"})
        r = FakeRequest(prompt_tokens, int(payload.get("max_tokens") or 16),
                        bool(payload.get("return_tokens_as_token_ids")), chat, images)
        with self.lock:
            if chat:
                self.chat_requests.append({"images": images, "prompt_tokens": prompt_tokens})
            self.waiting.append(r)
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        rid = f"cmpl-{id(r)}"
        sent = 0
        try:
            while True:
                try:
                    generated, done = r.events.get(timeout=0.05)
                except queue.Empty:
                    handler.wfile.write(b": keepalive\n\n")
                    handler.wfile.flush()
                    continue
                n = generated - sent
                sent = generated
                choice = {"index": 0, "finish_reason": "length" if done else None}
                if chat:
                    choice["delta"] = {"content": "x" * n}
                else:
                    choice["text"] = "x" * n
                    if r.token_ids:
                        choice["logprobs"] = {"tokens": [f"token_id:{1000 + generated - n + k}" for k in range(n)]}
                chunk = {"id": rid, "choices": [choice],
                         "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": generated,
                                   "total_tokens": prompt_tokens + generated}}
                handler.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                handler.wfile.flush()
                if done:
                    handler.wfile.write(b"data: [DONE]\n\n")
                    handler.wfile.flush()
                    return
        except (BrokenPipeError, ConnectionResetError, OSError):
            with self.lock:
                r.aborted = True
                self.aborted_total += 1


@pytest.fixture
def make_server():
    servers = []

    def factory(**kwargs):
        server = FakeSchedulerServer(**kwargs)
        servers.append(server)
        return server

    yield factory
    for server in servers:
        server.close()


FAST = [
    "--sched-metrics-interval", "0.2", "--sched-idle-timeout", "10", "--sched-request-timeout", "30",
    "--sched-stall-seconds", "1", "--sched-stop-after-stall", "2",
]


def parse(monkeypatch, server, *extra):
    argv = ["llm_decode_bench.py", "--host", "127.0.0.1", "--port", str(server.port), *FAST, *extra]
    monkeypatch.setattr(sys, "argv", argv)
    return BENCH.parse_args()


def run_mode(monkeypatch, server, *extra):
    args = parse(monkeypatch, server, *extra)
    out = io.StringIO()
    report, code = BENCH.run_scheduler_mode(args, Console(file=out, width=200))
    return report, code, out.getvalue()


def only(report, name):
    results = [r for r in report["scheduler"]["scenarios"] if r["scenario"] == name]
    assert len(results) == 1, results
    return results[0]


# --- pure helpers -----------------------------------------------------------------

def test_parse_metrics_sums_engines_and_reads_labels_and_histograms():
    text = "\n".join([
        "# HELP vllm:num_requests_running x",
        'vllm:num_requests_running{engine="0"} 3',
        'vllm:num_requests_running{engine="1"} 2',
        'vllm:kv_cache_usage_perc{engine="0"} 0.5',
        'vllm:kv_cache_usage_perc{engine="1"} 0.7',
        'vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 4',
        'vllm:num_requests_waiting_by_reason{engine="0",reason="deferred"} 1',
        'vllm:scheduler_compute_seconds_total{engine="0",class="prefill"} 12.5',
        'vllm:time_to_first_token_seconds_bucket{le="0.1"} 5',
        'vllm:time_to_first_token_seconds_bucket{le="1.0"} 9',
        'vllm:time_to_first_token_seconds_bucket{le="+Inf"} 10',
        "vllm:time_to_first_token_seconds_count 10",
        "vllm:time_to_first_token_seconds_sum 4.0",
        "vllm:gpu_cache_usage_perc 0.99",
    ])
    snap = BENCH.sched_parse_metrics(text, BENCH.ENGINE_VLLM)
    assert snap["running"] == 5
    assert snap["kv_usage"] == pytest.approx(0.6)  # averaged, the legacy name is ignored
    assert snap["waiting_capacity"] == 4 and snap["waiting_deferred"] == 1
    assert snap["compute_seconds_prefill"] == 12.5
    hist = BENCH.sched_histogram_delta(None, snap["histograms"]["ttft"])
    assert hist["count"] == 10 and hist["mean_s"] == pytest.approx(0.4)
    assert hist["p50_s"] == pytest.approx(0.1)
    assert hist["p90_s"] == pytest.approx(1.0)
    assert hist["p99_s"] == pytest.approx(1.0)  # +Inf bucket reports its lower bound
    start = {"sum": 1.0, "count": 5, "buckets": {"0.1": 5, "1.0": 5, "+Inf": 5}}
    delta = BENCH.sched_histogram_delta(start, snap["histograms"]["ttft"])
    assert delta["count"] == 5 and delta["p50_s"] == pytest.approx(0.1 + 0.9 * 2.5 / 4)


def test_sglang_metric_names():
    text = "sglang:num_running_reqs 7\nsglang:num_queue_reqs 2\nsglang:token_usage 0.25\n"
    snap = BENCH.sched_parse_metrics(text, BENCH.ENGINE_SGLANG)
    assert (snap["running"], snap["waiting"], snap["kv_usage"]) == (7, 2, 0.25)


def test_watchdog_stall_ends_and_wedge_stops():
    tracker = BENCH.SchedulerClientTracker()
    dog = BENCH.SchedulerStallWatchdog(stall_seconds=10, stop_after_seconds=30)
    busy = {"running": 2, "waiting": 0, "prompt_tokens": 100, "generation_tokens": 50}
    assert not dog.observe(0, busy, tracker)
    assert not dog.observe(9, busy, tracker) and not dog.stalls
    assert not dog.observe(12, busy, tracker)
    assert len(dog.stalls) == 1 and dog.stalls[0]["seconds"] == 12
    moved = dict(busy, generation_tokens=60)
    assert not dog.observe(14, moved, tracker)
    assert dog.stalls[0]["end_s"] == 14 and dog.current is None
    idle = dict(moved, running=0)
    assert not dog.observe(100, idle, tracker)  # an idle server is never stalled
    tracker.in_flight = 1  # no metrics: client-side progress only
    assert not dog.observe(101, None, tracker)
    assert not dog.observe(120, None, tracker)
    assert dog.observe(131.5, None, tracker) and dog.wedged
    assert dog.stalls[-1]["stopped_run"]


def test_ranges_arms_and_ab_schedule():
    assert BENCH.parse_token_range("40k-100k") == (40960, 102400)
    assert BENCH.parse_token_range("0-3", floats=True) == (0.0, 3.0)
    assert BENCH.parse_token_range("5k") == (5120, 5120)
    with pytest.raises(ValueError):
        BENCH.parse_token_range("9-3")
    assert BENCH.parse_fairness_arm("off") == ("off", None)
    assert BENCH.parse_fairness_arm("AUTO") == ("auto", "auto")
    assert BENCH.parse_fairness_arm("0.40") == ("0.4", 0.4)
    with pytest.raises(ValueError):
        BENCH.parse_fairness_arm("1.5")
    settings = BENCH.SchedulerSettings(ab_arms=[("off", None), ("0.4", 0.4)], ab_rounds=2)
    assert [(r, label) for r, label, _ in BENCH.sched_arm_schedule(settings)] == [
        (0, "off"), (0, "0.4"), (1, "0.4"), (1, "off")]
    assert BENCH.sched_order_inversions([1.0, 2.0, 3.0]) == 0.0
    assert BENCH.sched_order_inversions([3.0, 2.0, 1.0]) == 1.0


def test_result_values_and_lines_do_not_wrap(monkeypatch, make_server):
    assert BENCH.sched_format_value("max_waiting", 294.0) == "294"
    assert BENCH.sched_format_value("decode_tok_s_during_prefill", 1396.7) == "1,397"
    assert BENCH.sched_format_value("ttft_p50_s", 5.538) == "5.54"
    server = make_server(max_num_seqs=2)
    args = parse(monkeypatch, server, "--scheduler", "burst", "--sched-burst-requests", "2",
                 "--sched-burst-context", "1k", "--sched-burst-max-tokens", "2")
    out = io.StringIO()
    BENCH.run_scheduler_mode(args, Console(file=out, width=80))
    result_lines = [line for line in out.getvalue().splitlines() if line.startswith("  burst: PASS")]
    assert len(result_lines) == 1 and "order_inversions" in result_lines[0]


def test_cli_validation(monkeypatch):
    def error_of(*argv):
        monkeypatch.setattr(sys, "argv", ["llm_decode_bench.py", *argv])
        with pytest.raises(SystemExit):
            BENCH.parse_args()

    error_of("--scheduler", "nope")
    error_of("--sched-duration", "5")
    error_of("--scheduler", "burst", "--tool-eval")
    error_of("--scheduler", "burst", "--mixed-prefill-contexts", "32k")
    error_of("--scheduler", "burst", "--sched-ab-fairness", "0.4")
    error_of("--scheduler", "burst", "--sched-ab-fairness", "0.4,0.40")
    error_of("--scheduler", "burst", "--sched-stall-seconds", "60", "--sched-stop-after-stall", "30")
    error_of("--scheduler", "agents", "--sched-agent-tool", "6k-1k")
    monkeypatch.setattr(sys, "argv", ["llm_decode_bench.py", "--scheduler", "all,burst"])
    args = BENCH.parse_args()
    assert args.scheduler_list == list(BENCH.SCHEDULER_SCENARIOS)
    monkeypatch.setattr(sys, "argv", ["llm_decode_bench.py", "--scheduler", "burst",
                                      "--sched-ab-fairness", "off,auto,0.4"])
    assert [label for label, _ in BENCH.parse_args().sched_ab_arms] == ["off", "auto", "0.4"]


# --- scenarios against the fake server ------------------------------------------------

def test_burst_passes_and_reads_server_limits(monkeypatch, make_server):
    server = make_server(max_num_seqs=2, prefill_tok_s=100_000.0)
    report, code, out = run_mode(monkeypatch, server, "--scheduler", "burst", "--sched-burst-requests", "6",
                                 "--sched-burst-context", "16k", "--sched-burst-max-tokens", "8",
                                 "--sched-max-running", "2")
    result = only(report, "burst")
    assert code == 0, out
    assert result["status"] == "pass", result
    assert result["metrics"]["completed"] == 6
    assert result["metrics"]["order_inversions"] is not None
    assert result["server"]["max_running"] == 2 and result["server"]["max_waiting"] >= 1
    assert result["server"]["prompt_tokens"] == 6 * 16384
    assert result["server"]["histograms"]["ttft"]["count"] == 6
    assert report["server"]["kv_tokens"] == server.kv_tokens
    assert report["server"]["prompt_mode"] == "token_ids"
    assert report["metadata"]["mode"] == "scheduler" and report["scheduler"]["verdict"] == "pass"
    assert all(len(p) == 16384 and set(p) <= set(range(100, 400)) for p in server.prompts)
    assert len({tuple(p) for p in server.prompts}) == 6  # unique prompts


def test_burst_wedge_is_detected_and_later_scenarios_skipped(monkeypatch, make_server):
    server = make_server(max_num_seqs=4, wedge_after=2)
    report, code, out = run_mode(monkeypatch, server, "--scheduler", "burst,abort-storm",
                                 "--sched-burst-requests", "6", "--sched-burst-context", "2k",
                                 "--sched-burst-max-tokens", "4")
    burst = only(report, "burst")
    assert code == 2
    assert burst["status"] == "wedged", burst
    assert burst["stalls"] and burst["stalls"][0]["stopped_run"]
    stalled = burst["stalls"][0]["server"]
    assert stalled["running"] >= 1 and stalled["waiting"] >= 3
    assert burst["elapsed_s"] < 10
    abort = only(report, "abort-storm")
    assert abort["status"] == "skipped" and "wedged in burst" in abort["reason"]
    assert report["scheduler"]["verdict"] == "fail"
    assert "WEDGED" in out


def test_arrivals_measures_decode_during_prefill(monkeypatch, make_server):
    server = make_server(max_num_seqs=8, prefill_chunk=4096, prefill_tok_s=200_000.0)
    report, code, out = run_mode(monkeypatch, server, "--scheduler", "arrivals", "--sched-streams", "3",
                                 "--sched-warmup", "0.5", "--sched-duration", "3", "--sched-arrival-context", "60k",
                                 "--sched-arrival-period", "1", "--sched-stream-prompt-tokens", "64",
                                 "--sched-stream-max-tokens", "100000")
    result = only(report, "arrivals")
    assert code == 0, out
    m = result["metrics"]
    assert result["status"] == "pass", result
    assert m["arrivals"] >= 2 and m["arrival_ttft_s"]["p50"] > 0
    assert m["decode_tok_s"] > 0 and m["decode_tok_s_during_prefill"] is not None
    assert m["decode_kept"] < 0.9  # prefill steps are slower in the fake engine
    assert m["gap_ms"]["count"] > 10 and m["gap_ms"]["max"] >= m["gap_ms"]["p50"]
    assert m["arrival_prompt_tokens"]["p50"] == 61440


def test_abort_storm_returns_to_idle(monkeypatch, make_server):
    server = make_server(max_num_seqs=4, prefill_tok_s=400_000.0)
    report, code, out = run_mode(monkeypatch, server, "--scheduler", "abort-storm", "--sched-abort-waves", "2",
                                 "--sched-abort-requests", "6", "--sched-abort-context", "8k")
    result = only(report, "abort-storm")
    assert code == 0, out
    assert result["status"] == "pass", result
    m = result["metrics"]
    assert len(m["waves"]) == 2 and all(w["idle"] for w in m["waves"])
    assert m["by_mode"]["complete"]["completed"] == 4
    assert m["by_mode"]["decode"]["aborted_after_first_token"] + m["by_mode"]["prefill"]["sent"] > 0
    assert m["probe_after"]["ok"]
    assert server.busy() == 0


def test_overload_drains_and_recovers(monkeypatch, make_server):
    server = make_server(max_num_seqs=4)
    report, code, out = run_mode(monkeypatch, server, "--scheduler", "overload", "--sched-duration", "2",
                                 "--sched-overload-probe-seconds", "1.5", "--sched-overload-prompt-tokens", "512",
                                 "--sched-overload-max-tokens", "20", "--sched-overload-factor", "3",
                                 "--sched-max-running", "4")
    result = only(report, "overload")
    assert code == 0, out
    m = result["metrics"]
    assert result["status"] == "pass", result
    assert m["capacity_rps"] > 0 and m["offered_rps"] == pytest.approx(3 * m["capacity_rps"], rel=1e-3)
    assert m["sent"] > 0 and m["failed"] == 0
    assert m["max_waiting"] >= 1
    assert m["idle_after_drain"]["idle"] and m["probe_after"]["ok"]
    assert len(m["ttft_s_by_third"]) == 3


def test_agents_extend_their_exact_conversation(monkeypatch, make_server):
    server = make_server(max_num_seqs=4)
    report, code, out = run_mode(monkeypatch, server, "--scheduler", "agents", "--sched-agents", "2",
                                 "--sched-agents-duration", "3", "--sched-agent-start", "2k-3k",
                                 "--sched-agent-tool", "100-200", "--sched-agent-response", "20-40",
                                 "--sched-agent-think", "0-0.1")
    result = only(report, "agents")
    assert code == 0, out
    m = result["metrics"]
    assert result["status"] == "pass", result
    assert m["exact_history"] and m["turns_ok"] >= 4 and m["windows"]
    # every later prompt of an agent starts with its previous prompt plus the exact response IDs
    prompts = [p for p in server.prompts if p]
    extended = [(a, b) for a in prompts for b in prompts if len(b) > len(a) and b[:len(a)] == a]
    assert extended
    a, b = extended[0]
    response = b[len(a):]
    assert response[0] == 1000  # the fake server's first output token ID


def test_short_under_long_reports_both_phases(monkeypatch, make_server):
    server = make_server(max_num_seqs=8, prefill_chunk=4096, prefill_tok_s=150_000.0)
    report, code, out = run_mode(monkeypatch, server, "--scheduler", "short-under-long",
                                 "--sched-short-rate", "15", "--sched-short-baseline-seconds", "1",
                                 "--sched-duration", "2.5", "--sched-arrival-context", "60k",
                                 "--sched-arrival-period", "1", "--sched-short-prompt-tokens", "64",
                                 "--sched-short-max-tokens", "4")
    result = only(report, "short-under-long")
    assert code == 0, out
    m = result["metrics"]
    assert result["status"] == "pass", result
    assert m["short_ttft_baseline_s"]["count"] > 3 and m["short_ttft_under_load_s"]["count"] > 3
    assert m["longs"] >= 1 and m["short_ttft_during_long_prefill_s"]["count"] >= 1
    assert m["short_ttft_during_long_prefill_s"]["p50"] > m["short_ttft_baseline_s"]["p50"]


def test_soak_mix_text_and_images(monkeypatch, make_server):
    pytest.importorskip("PIL")
    server = make_server(max_num_seqs=4, max_model_len=40_000)
    report, code, out = run_mode(monkeypatch, server, "--scheduler", "soak-mix", "--sched-soak-duration", "2.5",
                                 "--sched-soak-concurrency", "3", "--sched-soak-images")
    result = only(report, "soak-mix")
    assert code == 0, out
    m = result["metrics"]
    assert result["status"] == "pass", result
    assert m["prompt_lengths"] == [1024, 8192, 30720]  # 92k / 160k exceed the model length
    assert m["requests"] > 0 and m["errors"] == 0
    assert m["images"]


def test_ab_fairness_switches_interleaves_and_restores(monkeypatch, make_server):
    server = make_server(max_num_seqs=2)
    report, code, out = run_mode(monkeypatch, server, "--scheduler", "burst", "--sched-burst-requests", "3",
                                 "--sched-burst-context", "1k", "--sched-burst-max-tokens", "4",
                                 "--sched-ab-fairness", "off,0.4", "--sched-ab-rounds", "2")
    assert code == 0, out
    shares = [post.get("prefill_compute_share") for post in server.fairness_posts]
    assert shares == [None, 0.4, 0.4, None, "auto"]
    assert server.fairness_posts[-1]["prefill_compute_half_life"] == "smooth"
    runs = [(r["arm"], r["round"]) for r in report["scheduler"]["scenarios"]]
    assert runs == [("off", 1), ("0.4", 1), ("0.4", 2), ("off", 2)]
    assert report["scheduler"]["scenarios"][1]["fairness"]["prefill_compute_share"] == 0.4
    ab = report["scheduler"]["ab"]["burst"]
    assert set(ab) == {"off", "0.4"} and len(ab["off"]["wall_s"]["values"]) == 2
    assert "A/B burst" in out


def test_ab_needs_the_dev_endpoint(monkeypatch, make_server):
    server = make_server(fairness=False)
    report, code, out = run_mode(monkeypatch, server, "--scheduler", "burst", "--sched-ab-fairness", "off,0.4")
    assert code == 1
    assert "VLLM_SERVER_DEV_MODE=1" in report["scheduler"]["error"]
    assert not server.prompts


def test_without_metrics_or_tokenize_still_runs(monkeypatch, make_server):
    server = make_server(max_num_seqs=2, metrics=False, tokenize=False, fairness=False)
    report, code, out = run_mode(monkeypatch, server, "--scheduler", "burst,abort-storm",
                                 "--sched-burst-requests", "3", "--sched-burst-context", "1k",
                                 "--sched-burst-max-tokens", "4", "--sched-abort-waves", "1",
                                 "--sched-abort-requests", "3", "--sched-abort-context", "1k")
    assert code == 0, out
    assert report["server"]["prompt_mode"] == "text" and not report["server"]["metrics"]
    abort = only(report, "abort-storm")
    assert abort["status"] == "pass" and abort["metrics"]["waves"][0] == {"wave": 0, "checked": False}


def test_main_writes_the_report(monkeypatch, make_server, tmp_path):
    server = make_server(max_num_seqs=2)
    output = tmp_path / "sched.json"
    monkeypatch.setenv("LLM_BENCH_NO_UPDATE_CHECK", "1")
    monkeypatch.setattr(sys, "argv", ["llm_decode_bench.py", "--host", "127.0.0.1", "--port", str(server.port),
                                      "--api-key", "secret-key", *FAST, "--scheduler", "burst",
                                      "--sched-burst-requests", "2", "--sched-burst-context", "1k",
                                      "--sched-burst-max-tokens", "2", "--output", str(output)])
    BENCH.main()
    report = json.loads(output.read_text())
    assert report["metadata"]["version"] == BENCH.VERSION
    assert "secret-key" not in output.read_text()
    assert report["scheduler"]["verdict"] == "pass"
    assert set(report["methodology"]) == {"common", "burst"}
