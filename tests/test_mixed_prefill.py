"""Tests for the mixed prefill + decode phase (--mixed-prefill-contexts).

The phase sends one long prompt alone, then the same length again while N background
streams decode, and reports both TTFTs and the background decode tok/s before, during
and after the prompt's prefill. Nothing here needs a GPU or a model: FakeStreamingServer
is an OpenAI-compatible SSE stub whose TTFT grows with the number of decoding requests,
whose decode steps slow down while a long prompt is in prefill, and whose chunks carry
several tokens each (like MTP / speculative decoding).
"""
import asyncio
import importlib.util
import io
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import random
import re
import sys
import threading
import time
import uuid

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "llm_decode_bench.py"
SPEC = importlib.util.spec_from_file_location("llm_decode_bench_mixed_prefill", MODULE_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCH)

WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliett kilo lima mike november "
    "oscar papa quebec romeo sierra tango uniform victor whiskey xray yankee zulu amber basalt "
    "cobalt dune ember fjord granite harbor iris jade kelp lagoon meadow nectar onyx prairie"
).split()


class FakeStreamingServer:
    """OpenAI-compatible streaming stub with prefill/decode interference.

    * prompt tokens = characters of all message contents // 4 + 16 (a fake tokenizer)
    * TTFT = prompt_tokens / prefill_tok_s * (1 + load_ttft_factor * requests decoding)
    * decode: one chunk every step_s with tokens_per_chunk tokens; while a prompt of at
      least long_prompt_tokens is in prefill, a step takes prefill_slowdown times longer
    * usage on every chunk when stream_options.continuous_usage_stats is set (and
      continuous_usage is True), in a final usage chunk when include_usage is set
    """

    def __init__(self, *, prefill_tok_s=100_000.0, load_ttft_factor=0.5, step_s=0.01,
                 tokens_per_chunk=3, long_prompt_tokens=4096, prefill_slowdown=4.0,
                 continuous_usage=True, metrics=False, max_model_len=1_000_000,
                 fail_background=False):
        self.prefill_tok_s = prefill_tok_s
        self.load_ttft_factor = load_ttft_factor
        self.step_s = step_s
        self.tokens_per_chunk = tokens_per_chunk
        self.long_prompt_tokens = long_prompt_tokens
        self.prefill_slowdown = prefill_slowdown
        self.continuous_usage = continuous_usage
        self.metrics = metrics
        self.max_model_len = max_model_len
        self.fail_background = fail_background
        self.lock = threading.Lock()
        self.decoding = 0
        self.in_flight = 0
        self.long_prefills = 0
        self.requests = []
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
                    with server.lock:
                        running = server.in_flight
                    text = (
                        f'vllm:num_requests_running{{model_name="fake-model"}} {running}\n'
                        'vllm:num_requests_waiting{model_name="fake-model"} 0\n'
                    )
                    return self.reply(200, text.encode(), "text/plain")
                return self.reply(404, {"error": "not found"})

            def do_POST(self):
                length = int(self.headers.get("Content-Length") or 0)
                payload = json.loads(self.rfile.read(length) or b"{}")
                if self.path == "/v1/chat/completions":
                    return server.chat(self, payload)
                return self.reply(404, {"error": "not found"})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        self.port = self.httpd.server_address[1]
        self.base_url = f"http://127.0.0.1:{self.port}"
        threading.Thread(target=self.httpd.serve_forever, daemon=True).start()

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()

    def tagged(self, prefix: str) -> list:
        with self.lock:
            return [r for r in self.requests if r["tag"].startswith(prefix)]

    def chat(self, handler, payload: dict) -> None:
        text = "".join(str(m.get("content") or "") for m in payload.get("messages") or [])
        prompt_tokens = len(text) // 4 + 16
        tag = re.search(r"\[((?:MIXED|BENCH|WARMUP)_[^\]]*)\]", text)
        options = payload.get("stream_options") or {}
        record = {
            "tag": tag.group(1) if tag else "",
            "prompt_tokens": prompt_tokens,
            "max_tokens": payload.get("max_tokens"),
            "stream": bool(payload.get("stream")),
            "stream_options": options,
            "ignore_eos": payload.get("ignore_eos"),
            "authorization": handler.headers.get("Authorization"),
        }
        with self.lock:
            record["decoding_at_start"] = self.decoding
            self.requests.append(record)
        if not payload.get("stream"):
            return handler.reply(200, {
                "id": "cmpl-calibration",
                "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}, "finish_reason": "length"}],
                "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": 1, "total_tokens": prompt_tokens + 1},
            })
        if self.fail_background and "_BG" in record["tag"]:
            return handler.reply(500, {"error": "background requests are refused"})
        continuous = self.continuous_usage and bool(options.get("continuous_usage_stats"))
        include_usage = bool(options.get("include_usage"))
        max_tokens = int(payload.get("max_tokens") or 16)
        long_prompt = prompt_tokens >= self.long_prompt_tokens
        rid = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        rng = random.Random(rid)
        decoding = False
        handler.send_response(200)
        handler.send_header("Content-Type", "text/event-stream")
        handler.end_headers()
        with self.lock:
            self.in_flight += 1
            if long_prompt:
                self.long_prefills += 1
            load = self.decoding

        def usage(n: int) -> dict:
            return {"prompt_tokens": prompt_tokens, "completion_tokens": n, "total_tokens": prompt_tokens + n}

        def send(obj: dict) -> None:
            handler.wfile.write(f"data: {json.dumps(obj)}\n\n".encode())
            handler.wfile.flush()

        try:
            time.sleep(prompt_tokens / self.prefill_tok_s * (1.0 + self.load_ttft_factor * load))
            with self.lock:
                if long_prompt:
                    self.long_prefills -= 1
                    long_prompt = False
                self.decoding += 1
                decoding = True
            first = {"id": rid, "choices": [{"index": 0, "delta": {"role": "assistant", "content": ""},
                                             "finish_reason": None}]}
            if continuous:
                first["usage"] = usage(0)
            send(first)
            generated = 0
            while generated < max_tokens:
                n = min(self.tokens_per_chunk, max_tokens - generated)
                generated += n
                chunk = {"id": rid, "choices": [{
                    "index": 0,
                    "delta": {"content": "".join(" " + rng.choice(WORDS) for _ in range(n))},
                    "finish_reason": "length" if generated >= max_tokens else None,
                }]}
                if continuous:
                    chunk["usage"] = usage(generated)
                send(chunk)
                if generated < max_tokens:
                    with self.lock:
                        slow = self.long_prefills > 0
                    time.sleep(self.step_s * (self.prefill_slowdown if slow else 1.0))
            if include_usage:
                send({"id": rid, "choices": [], "usage": usage(generated)})
            handler.wfile.write(b"data: [DONE]\n\n")
            handler.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            with self.lock:
                self.in_flight -= 1
                if decoding:
                    self.decoding -= 1
                if long_prompt:
                    self.long_prefills -= 1


@pytest.fixture
def fake_server():
    servers = []

    def start(**kwargs):
        server = FakeStreamingServer(**kwargs)
        servers.append(server)
        return server

    yield start
    for server in servers:
        server.close()


def parse(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["llm_decode_bench.py", *argv])
    return BENCH.parse_args()


def parse_error(monkeypatch, capsys, *argv) -> str:
    with pytest.raises(SystemExit) as exc:
        parse(monkeypatch, *argv)
    assert exc.value.code == 2
    return capsys.readouterr().err


# ---------------------------------------------------------------------------
# Window rates and token counting
# ---------------------------------------------------------------------------

def test_window_counts_samples_in_half_open_interval():
    samples = [(0.5, 3), (1.0, 3), (1.5, 2), (2.0, 3), (2.5, 4)]
    window = BENCH.mixed_window(samples, 1.0, 2.0)
    assert window == {"tokens": 5, "seconds": 1.0, "tok_s": 5.0}
    # Start is inclusive, end exclusive: the 2.0 sample belongs to the next window.
    assert BENCH.mixed_window(samples, 2.0, 3.0)["tokens"] == 7
    assert BENCH.mixed_window(samples, 0.0, 10.0) == {"tokens": 15, "seconds": 10.0, "tok_s": 1.5}
    assert BENCH.mixed_window(samples, 3.0, 4.0)["tok_s"] == 0.0
    assert BENCH.mixed_window([], 0.0, 1.0)["tok_s"] == 0.0


def test_empty_or_inverted_window_has_no_rate():
    samples = [(1.0, 3)]
    assert BENCH.mixed_window(samples, 2.0, 2.0) == {"tokens": 0, "seconds": 0.0, "tok_s": None}
    assert BENCH.mixed_window(samples, 2.0, 1.0)["tok_s"] is None


def test_timeline_bins_align_to_the_arrival():
    samples = [(9.2, 4), (9.8, 2), (10.1, 3), (10.6, 3), (11.5, 6)]
    timeline = BENCH.mixed_timeline(samples, origin=10.0, start=8.7, end=12.4)
    assert timeline["first_bin_start_s"] == -1.0
    assert timeline["tok_s"] == [6.0, 6.0, 6.0]
    assert BENCH.mixed_timeline(samples, origin=10.0, start=10.0, end=10.5)["tok_s"] == []


def chunk(content=None, completion_tokens=None, finish=None, reasoning=None):
    delta = {}
    if content is not None:
        delta["content"] = content
    if reasoning is not None:
        delta["reasoning_content"] = reasoning
    data = {"choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
    if completion_tokens is not None:
        data["usage"] = {"prompt_tokens": 100, "completion_tokens": completion_tokens}
    return data


def test_usage_deltas_count_every_token_of_multi_token_chunks():
    counter = BENCH.UsageTokenCounter()
    stream = [
        chunk(content="", completion_tokens=0),          # role chunk
        chunk(content=" a b c", completion_tokens=3),    # MTP: 3 tokens in one chunk
        chunk(reasoning=" d e", completion_tokens=5),
        chunk(content=" f", completion_tokens=6),
        chunk(content=" g h i j", completion_tokens=10, finish="length"),
        {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 10}},  # final usage chunk
    ]
    deltas = [counter.observe(data) for data in stream]
    assert deltas == [0, 3, 2, 1, 4, 0]
    assert counter.tokens == 10
    assert counter.usage_chunks == 6 and counter.fallback_chunks == 0
    # Four chunks carried text, but ten tokens were generated.
    assert sum(1 for d in deltas if d) == 4


def test_chunks_without_usage_count_one_token_until_usage_corrects_the_total():
    counter = BENCH.UsageTokenCounter()
    stream = [
        {"choices": [{"index": 0, "delta": {"role": "assistant", "content": ""}}], "usage": None},
        chunk(content=" a b"),
        chunk(content=" c"),
        chunk(content=" d e f"),
        {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 9}},
    ]
    deltas = [counter.observe(data) for data in stream]
    assert deltas == [0, 1, 1, 1, 6]
    assert counter.tokens == 9
    assert counter.fallback_chunks == 3 and counter.usage_chunks == 1


def test_usage_never_counts_backwards():
    counter = BENCH.UsageTokenCounter()
    assert counter.observe(chunk(content="x", completion_tokens=4)) == 4
    assert counter.observe(chunk(content="y", completion_tokens=3)) == 0
    assert counter.observe(chunk(content="z", completion_tokens=6)) == 2
    assert counter.tokens == 6


def test_first_output_detection_ignores_the_role_chunk():
    assert not BENCH.stream_chunk_has_output(chunk(content="", completion_tokens=0))
    assert BENCH.stream_chunk_has_output(chunk(content="hi"))
    assert BENCH.stream_chunk_has_output(chunk(reasoning="think"))
    assert BENCH.stream_chunk_has_output(chunk(content="", completion_tokens=1))
    assert BENCH.stream_chunk_has_output(chunk(finish="length"))
    assert not BENCH.stream_chunk_has_output({"choices": [], "usage": None})


# ---------------------------------------------------------------------------
# Summary of one context from raw timings
# ---------------------------------------------------------------------------

def test_summary_windows_slowdown_and_timeline():
    settings = BENCH.MixedPrefillSettings(contexts=[32768], decode_streams=2, window_seconds=10.0)
    stats = BENCH.MixedBackgroundStats()
    t0 = 1000.0
    # Two streams decode from t0+1; 100 tok/s each before, 25 tok/s during the prefill, 90 after.
    for stream in range(2):
        stats.first_token_at[stream] = t0 + 1.0 + stream * 0.1
    t = t0 + 1.0
    while t < t0 + 50.0:
        rate = 25 if t0 + 30.0 <= t < t0 + 34.0 else (90 if t >= t0 + 34.0 else 100)
        stats.samples.append((t, rate * 2 // 10))   # both streams, one sample every 0.1 s
        t = round(t + 0.1, 6)
    stats.prompt_tokens = [2100, 2100, 2101]
    stats.requests_started, stats.requests_completed, stats.usage_chunks = 6, 4, 900
    alone = {"ttft_s": 2.0, "prompt_tokens": 32800, "cached_tokens": 0, "t_send": t0 - 5, "t_first": t0 - 3}
    arrival = {"ttft_s": 4.0, "prompt_tokens": 32800, "cached_tokens": None,
               "t_send": t0 + 30.0, "t_first": t0 + 34.0}
    result = BENCH.summarize_mixed_prefill_context(
        32768, settings, alone=alone, arrival=arrival, stats=stats,
        t_background_start=t0, t_stop=t0 + 44.2,
    )
    assert result["status"] == "ok"
    assert result["ttft_alone_s"] == 2.0 and result["ttft_under_load_s"] == 4.0
    assert result["slowdown"] == 2.0
    assert result["prefill_tok_s_alone"] == 16400.0 and result["prefill_tok_s_under_load"] == 8200.0
    assert result["bg_tok_s_before"] == pytest.approx(200.0, rel=0.02)
    assert result["bg_tok_s_during"] == pytest.approx(50.0, rel=0.02)
    assert result["bg_tok_s_after"] == pytest.approx(180.0, rel=0.02)
    assert result["bg_during_vs_before"] == pytest.approx(0.25, rel=0.03)
    windows = result["windows"]
    assert windows["before"]["start_s"] == -10.0 and windows["before"]["end_s"] == 0.0
    assert windows["during"]["start_s"] == 0.0 and windows["during"]["end_s"] == 4.0
    assert windows["after"]["start_s"] == 4.0 and windows["after"]["end_s"] == 14.0
    background = result["background"]
    assert background["streams_decoding_at_arrival"] == 2
    assert background["prompt_tokens"] == 2100 and background["token_source"] == "usage"
    assert background["warmup_s"] == 30.0
    timeline = result["timeline"]
    assert timeline["first_bin_start_s"] == -30.0 and len(timeline["tok_s"]) == 44
    assert timeline["tok_s"][31] == pytest.approx(50.0, rel=0.05)    # bin [+1, +2): inside the prefill
    json.dumps(result)


def test_summary_clips_the_before_window_to_the_steady_state():
    settings = BENCH.MixedPrefillSettings(contexts=[8192], decode_streams=2, window_seconds=10.0)
    stats = BENCH.MixedBackgroundStats(samples=[(103.0 + i * 0.5, 5) for i in range(20)])
    stats.first_token_at = {0: 101.0, 1: 103.0}
    result = BENCH.summarize_mixed_prefill_context(
        8192, settings,
        alone={"ttft_s": 1.0}, arrival={"ttft_s": 1.0, "t_send": 106.0, "t_first": 107.0},
        stats=stats, t_background_start=100.0, t_stop=118.0,
    )
    # Warmup (6 s) is shorter than the window: before starts when the last stream decoded.
    assert result["windows"]["before"]["start_s"] == -3.0
    assert result["windows"]["before"]["seconds"] == 3.0
    assert result["bg_tok_s_before"] == pytest.approx(10.0)


def test_summary_of_a_failed_arrival_is_an_error_without_rates():
    settings = BENCH.MixedPrefillSettings(contexts=[8192], decode_streams=2)
    result = BENCH.summarize_mixed_prefill_context(
        8192, settings,
        alone={"ttft_s": 1.0, "prompt_tokens": 8200},
        arrival={"ttft_s": None, "t_send": 50.0, "t_first": None, "error": "HTTP 500: boom"},
        stats=BENCH.MixedBackgroundStats(), t_background_start=30.0, t_stop=51.0,
        errors=["prompt under load: HTTP 500: boom"],
    )
    assert result["status"] == "error"
    assert result["slowdown"] is None and result["bg_tok_s_during"] is None
    assert result["errors"] == ["prompt under load: HTTP 500: boom"]
    assert "ERROR prompt under load" in BENCH.format_mixed_prefill_line(result)


# ---------------------------------------------------------------------------
# Command line
# ---------------------------------------------------------------------------

def test_phase_is_off_by_default(monkeypatch):
    args = parse(monkeypatch)
    assert args.mixed_prefill_context_list == []
    assert not args.mixed_only
    assert not BENCH.MixedPrefillSettings.from_args(args).enabled


def test_defaults_and_context_parsing(monkeypatch):
    args = parse(monkeypatch, "--mixed-prefill-contexts", "32k, 128k,32768")
    assert args.mixed_prefill_context_list == [32768, 131072]
    settings = BENCH.MixedPrefillSettings.from_args(args)
    assert (settings.decode_streams, settings.decode_prompt_tokens, settings.decode_max_tokens) == (8, 2048, 1024)
    assert (settings.warmup_seconds, settings.window_seconds, settings.arrival_max_tokens) == (20.0, 10.0, 16)
    assert settings.ignore_eos and settings.loop_detection
    assert settings.prompt_sizes() == [2048, 8192, 32768, 131072]
    assert settings.kv_tokens_needed(32768) == 8 * (2048 + 1024) + 32768 + 16
    assert not args.skip_prefill


def test_every_option_is_applied(monkeypatch):
    args = parse(
        monkeypatch, "--mixed-prefill-contexts", "64k", "--mixed-decode-streams", "16",
        "--mixed-decode-prompt-tokens", "4096", "--mixed-decode-max-tokens", "512",
        "--mixed-warmup-seconds", "5", "--mixed-window-seconds", "3.5", "--mixed-arrival-max-tokens", "4",
        "--mixed-only", "--respect-eos", "--temperature", "0", "--no-loop-detection",
    )
    settings = BENCH.MixedPrefillSettings.from_args(args)
    assert settings.metadata() == {
        "contexts": [65536], "decode_streams": 16, "decode_prompt_tokens": 4096, "decode_max_tokens": 512,
        "warmup_seconds": 5.0, "window_seconds": 3.5, "arrival_max_tokens": 4, "ignore_eos": False,
        "temperature": 0.0, "forced_token_id": None, "loop_detection": False,
        "stream_options": {"include_usage": True, "continuous_usage_stats": True},
    }
    assert args.mixed_only and args.skip_prefill
    payload = BENCH.mixed_request_payload("m", 4096, "text", 512, settings)
    assert payload["stream"] and payload["max_tokens"] == 512 and payload["temperature"] == 0.0
    assert payload["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
    assert "ignore_eos" not in payload


def test_warmup_abbreviation_is_accepted(monkeypatch):
    args = parse(monkeypatch, "--mixed-prefill-contexts", "8k", "--mixed-warmup", "7")
    assert args.mixed_warmup_seconds == 7.0


@pytest.mark.parametrize("argv, message", [
    (["--mixed-only"], "--mixed-only needs --mixed-prefill-contexts"),
    (["--mixed-decode-streams", "4", "--mixed-only"], "--mixed-decode-streams, --mixed-only need --mixed-prefill-contexts"),
    (["--mixed-prefill-contexts", ","], "needs at least one prompt length"),
    (["--mixed-prefill-contexts", "abc"], "cannot read 'abc'"),
    (["--mixed-prefill-contexts", "0"], "values must be > 0"),
    (["--mixed-prefill-contexts", "8k", "--mixed-decode-streams", "0"], "--mixed-decode-streams must be >= 1"),
    (["--mixed-prefill-contexts", "8k", "--mixed-decode-prompt-tokens", "0"], "must be >= 1"),
    (["--mixed-prefill-contexts", "8k", "--mixed-decode-max-tokens", "0"], "must be >= 1"),
    (["--mixed-prefill-contexts", "8k", "--mixed-warmup-seconds", "-1"], "must be >= 0"),
    (["--mixed-prefill-contexts", "8k", "--mixed-window-seconds", "0"], "must be > 0"),
    (["--mixed-prefill-contexts", "8k", "--mixed-arrival-max-tokens", "0"], "must be >= 1"),
    (["--mixed-prefill-contexts", "8k", "--test-profile", "gsm8k"], "cannot be combined with --completion-stats"),
    (["--mixed-prefill-contexts", "8k", "--p2pmark-only"], "cannot be combined with --p2pmark-only"),
    (["--mixed-prefill-contexts", "8k", "--mixed-only", "--prefill-only"], "cannot be combined with --prefill-only"),
    (["--mixed-prefill-contexts", "8k", "--mixed-only", "--run-burst"], "cannot be combined with --run-burst"),
    (["--mixed-prefill-contexts", "8k", "--mixed-only", "--request-count", "4"], "--request-count"),
])
def test_invalid_combinations_are_refused(monkeypatch, capsys, argv, message):
    assert message in parse_error(monkeypatch, capsys, *argv)


def test_mixed_phase_combines_with_prefill_only(monkeypatch):
    args = parse(monkeypatch, "--prefill-only", "--mixed-prefill-contexts", "8k")
    assert args.standalone_prefill and args.mixed_prefill_context_list == [8192]


def test_skip_reasons():
    settings = BENCH.MixedPrefillSettings(contexts=[8192], decode_streams=4, decode_prompt_tokens=2048,
                                          decode_max_tokens=1024, arrival_max_tokens=16)
    assert BENCH.mixed_prefill_skip_reason(8192, settings) == ""
    assert "model context length" in BENCH.mixed_prefill_skip_reason(8192, settings, server_context_length=8000)
    assert "background requests" in BENCH.mixed_prefill_skip_reason(8192, settings, server_context_length=3000)
    reason = BENCH.mixed_prefill_skip_reason(8192, settings, kv_budget=20000)
    assert "needs 20,496 tokens, budget 20,000" in reason
    assert BENCH.mixed_prefill_skip_reason(8192, settings, kv_budget=20496) == ""


def test_arrival_must_fit_the_running_request_limit():
    # Four streams plus the arrival need five slots; with four the arrival
    # would queue and its TTFT would measure scheduler admission.
    settings = BENCH.MixedPrefillSettings(contexts=[8192], decode_streams=4, decode_prompt_tokens=2048,
                                          decode_max_tokens=1024, arrival_max_tokens=16)
    reason = BENCH.mixed_prefill_skip_reason(8192, settings, max_running_requests=4)
    assert "exceed the server's 4 running requests" in reason
    assert "--mixed-decode-streams 3" in reason
    assert BENCH.mixed_prefill_skip_reason(8192, settings, max_running_requests=5) == ""


# ---------------------------------------------------------------------------
# Streams against the fake server
# ---------------------------------------------------------------------------

def test_background_stream_counts_usage_tokens_and_restarts_with_fresh_prompts(fake_server):
    server = fake_server(tokens_per_chunk=3, step_s=0.005)
    settings = BENCH.MixedPrefillSettings(contexts=[8192], decode_max_tokens=20)
    prompts = []

    def make_payload(index, ordinal):
        prompts.append((index, ordinal))
        return BENCH.mixed_request_payload("fake-model", 64, f"[MIXED_t_BG{index}_R{ordinal}] " + "x" * 256, 20, settings)

    async def run():
        stats = BENCH.MixedBackgroundStats()
        stop = asyncio.Event()
        async with BENCH.httpx.AsyncClient() as client:
            task = asyncio.create_task(BENCH.mixed_background_stream(
                client, f"{server.base_url}/v1/chat/completions", 0, make_payload, stop, stats,
                loop_check=BENCH.LoopCheckState(),
            ))
            while stats.requests_completed < 3:
                await asyncio.sleep(0.01)
            stop.set()
            await asyncio.wait_for(task, timeout=5)
        return stats

    stats = asyncio.run(run())
    assert stats.errors == [] and stats.alive == 0
    assert stats.requests_completed >= 3 and stats.fallback_chunks == 0
    # 20 tokens per request in chunks of 3,3,3,3,3,3,2: tokens come from usage, not chunk counts.
    completed_tokens = 20 * stats.requests_completed
    counted = sum(delta for _, delta in stats.samples)
    assert counted >= completed_tokens
    assert counted > 2.5 * len(stats.samples)
    assert [t for t, _ in stats.samples] == sorted(t for t, _ in stats.samples)
    tags = [r["tag"] for r in server.tagged("MIXED_t_BG")]
    assert len(tags) == len(set(tags)) == stats.requests_started
    assert all(r["stream_options"]["continuous_usage_stats"] for r in server.tagged("MIXED_t_BG"))


def test_background_stream_without_continuous_usage_falls_back_to_chunks(fake_server):
    server = fake_server(tokens_per_chunk=3, continuous_usage=False, step_s=0.005)
    settings = BENCH.MixedPrefillSettings(contexts=[8192])

    async def run():
        stats = BENCH.MixedBackgroundStats()
        stop = asyncio.Event()
        async with BENCH.httpx.AsyncClient() as client:
            task = asyncio.create_task(BENCH.mixed_background_stream(
                client, f"{server.base_url}/v1/chat/completions", 0,
                lambda i, n: BENCH.mixed_request_payload("fake-model", 64, f"[MIXED_u_{n}] x", 12, settings),
                stop, stats,
            ))
            while stats.requests_completed < 2:
                await asyncio.sleep(0.01)
            stop.set()
            await asyncio.wait_for(task, timeout=5)
        return stats

    stats = asyncio.run(run())
    assert stats.fallback_chunks > 0 and stats.usage_chunks >= 2
    # Chunks count one token each and the final usage chunk adds the rest: totals stay exact.
    assert sum(delta for _, delta in stats.samples) >= 12 * stats.requests_completed


def test_ttft_request_records_first_token_and_usage(fake_server):
    server = fake_server(prefill_tok_s=50_000.0)
    settings = BENCH.MixedPrefillSettings(contexts=[8192])
    payload = BENCH.mixed_request_payload("fake-model", 8192, "y" * 32768, 16, settings)

    async def run():
        first = asyncio.Event()
        async with BENCH.httpx.AsyncClient() as client:
            return await BENCH.mixed_ttft_request(client, f"{server.base_url}/v1/chat/completions", payload,
                                                  first_token=first), first.is_set()

    result, signalled = asyncio.run(run())
    assert signalled and result["error"] == ""
    assert result["prompt_tokens"] == server.requests[0]["prompt_tokens"]
    assert result["completion_tokens"] == 16
    assert result["ttft_s"] >= result["prompt_tokens"] / 50_000.0
    assert result["t_send"] < result["t_first"] <= result["t_end"]


def test_ttft_request_reports_http_errors(fake_server):
    server = fake_server()

    async def run():
        async with BENCH.httpx.AsyncClient() as client:
            return await BENCH.mixed_ttft_request(client, f"{server.base_url}/v1/nope", {"stream": True})

    result = asyncio.run(run())
    assert result["error"].startswith("HTTP 404") and result["ttft_s"] is None


# ---------------------------------------------------------------------------
# End to end
# ---------------------------------------------------------------------------

MIXED_ARGV = [
    "--mixed-prefill-contexts", "8k,16k", "--mixed-decode-streams", "4",
    "--mixed-decode-prompt-tokens", "256", "--mixed-decode-max-tokens", "48",
    "--mixed-warmup-seconds", "0.6", "--mixed-window-seconds", "0.6",
    "--display-mode", "plain", "--no-hw-monitor", "--no-calibration-cache", "--no-resume",
]


def run_main(monkeypatch, server, tmp_path, *extra):
    output = tmp_path / "report.json"
    monkeypatch.setenv("LLM_BENCH_NO_UPDATE_CHECK", "1")
    monkeypatch.setenv("COLUMNS", "200")
    monkeypatch.delenv("LLM_BENCH_EVENT_FILE", raising=False)
    # No nvidia-smi query from the startup diagnostics on a test host.
    monkeypatch.setattr(BENCH, "collect_startup_diagnostics", lambda args, base_url: {"stub": True})
    monkeypatch.setattr(sys, "argv", [
        "llm_decode_bench.py", "--host", "127.0.0.1", "--port", str(server.port),
        *MIXED_ARGV, *extra, "--output", str(output),
    ])
    BENCH.main()
    return json.loads(output.read_text())


CONTEXT_KEYS = {
    "context", "status", "ttft_alone_s", "ttft_under_load_s", "slowdown",
    "prompt_tokens_alone", "prompt_tokens_under_load", "cached_tokens_alone", "cached_tokens_under_load",
    "prefill_tok_s_alone", "prefill_tok_s_under_load",
    "bg_tok_s_before", "bg_tok_s_during", "bg_tok_s_after", "bg_during_vs_before",
    "windows", "background", "timeline", "errors", "warnings", "loop_detected", "loop_diagnostics",
    "hardware_summary",
}
BACKGROUND_KEYS = {
    "streams", "streams_decoding_at_arrival", "prompt_tokens", "max_tokens", "warmup_s",
    "requests_started", "requests_completed", "errors", "token_source",
}
WINDOW_KEYS = {"start_s", "end_s", "seconds", "tokens", "tok_s"}


def check_context_result(row: dict, context: int, streams: int = 4):
    assert set(row) == CONTEXT_KEYS and set(row["background"]) == BACKGROUND_KEYS
    assert all(set(window) == WINDOW_KEYS for window in row["windows"].values())
    assert set(row["timeline"]) == {"bin_seconds", "first_bin_start_s", "tok_s"}
    assert row["context"] == context and row["status"] == "ok", row
    assert row["errors"] == [] and row["warnings"] == [] and not row["loop_detected"]
    assert row["ttft_under_load_s"] > row["ttft_alone_s"] > 0
    assert row["slowdown"] > 1.5
    assert abs(row["prompt_tokens_alone"] - context) < 0.1 * context
    assert abs(row["prompt_tokens_under_load"] - row["prompt_tokens_alone"]) <= 4
    assert row["bg_tok_s_before"] > 0 and row["bg_tok_s_after"] > 0
    assert row["bg_tok_s_during"] < 0.7 * row["bg_tok_s_before"]
    assert row["bg_tok_s_after"] > row["bg_tok_s_during"]
    assert row["bg_during_vs_before"] == pytest.approx(row["bg_tok_s_during"] / row["bg_tok_s_before"], abs=0.01)
    assert set(row["windows"]) == {"before", "during", "after"}
    assert row["windows"]["during"]["seconds"] == pytest.approx(row["ttft_under_load_s"], abs=0.002)
    assert row["windows"]["after"]["seconds"] == pytest.approx(0.6, abs=0.01)
    background = row["background"]
    assert background["streams"] == streams and background["streams_decoding_at_arrival"] == streams
    assert background["token_source"] == "usage" and background["max_tokens"] == 48
    # Estimate targeting: short prompts come out longer by the fixed template/question overhead.
    assert 256 <= background["prompt_tokens"] < 256 + 200
    assert background["requests_completed"] >= streams and background["errors"] == 0
    assert background["warmup_s"] >= 0.6
    assert row["timeline"]["bin_seconds"] == 1.0


def test_mixed_only_end_to_end_through_main(monkeypatch, capsys, fake_server, tmp_path):
    server = fake_server()
    report = run_main(monkeypatch, server, tmp_path, "--mixed-only", "--api-key", "sk-test")
    out = capsys.readouterr().out

    # JSON: one entry per context, config in metadata, methodology entry, no decode matrix.
    assert list(report["mixed_prefill"]) == ["8192", "16384"]
    check_context_result(report["mixed_prefill"]["8192"], 8192)
    check_context_result(report["mixed_prefill"]["16384"], 16384)
    assert report["results"] == [] and report["prefill"] == {}
    meta = report["metadata"]
    assert meta["mixed_only"] is True and meta["prefill_mode"] == "skipped"
    assert meta["mixed_prefill"]["contexts"] == [8192, 16384]
    assert meta["mixed_prefill"]["decode_streams"] == 4
    assert report["methodology"]["mixed_prefill"]["present"] is True

    # Requests: unique uncached prompts, the settings of the methodology.
    refs = server.tagged("MIXED_")
    tags = [r["tag"] for r in refs]
    assert len(tags) == len(set(tags))
    for context in (8192, 16384):
        ref = [r for r in refs if r["tag"].endswith(f"_C{context}_REF")]
        arrivals = [r for r in refs if r["tag"].endswith(f"_C{context}_ARR")]
        background = [r for r in refs if f"_C{context}_BG" in r["tag"]]
        assert len(ref) == 1 and len(arrivals) == 1
        assert ref[0]["decoding_at_start"] == 0          # alone: nothing else decodes
        assert arrivals[0]["decoding_at_start"] >= 2     # under load
        assert ref[0]["max_tokens"] == arrivals[0]["max_tokens"] == 16
        assert len(background) > 4                       # streams restarted with new prompts
        assert all(r["max_tokens"] == 48 and r["ignore_eos"] is True for r in background)
        assert all(r["stream_options"] == {"include_usage": True, "continuous_usage_stats": True}
                   for r in background)
    assert any(r["tag"].endswith("_WARMUP") for r in refs)
    with server.lock:
        assert {r["authorization"] for r in server.requests} == {"Bearer sk-test"}

    # Plain output: per-context lines and the final table.
    flat = " ".join(out.split())
    assert "Mixed prefill + decode 8k: TTFT" in flat and "Mixed prefill + decode 16k: TTFT" in flat
    assert "Phase 4" in flat and "TTFT alone s" in flat and "slowdown" in flat
    assert "Decode: skipped (--mixed-only)" in flat
    assert "Results saved to" in flat


def test_mixed_phase_runs_after_the_decode_matrix(monkeypatch, capsys, fake_server, tmp_path):
    server = fake_server(metrics=True)
    report = run_main(
        monkeypatch, server, tmp_path,
        "--concurrency", "2", "--contexts", "0", "--duration", "0.5", "--max-tokens", "64",
        "--skip-prefill", "--decode-warmup-seconds", "0", "--mixed-prefill-contexts", "8k",
    )
    out = capsys.readouterr().out
    assert len(report["results"]) == 1 and report["results"][0]["aggregate_tps"] > 0
    assert list(report["mixed_prefill"]) == ["8192"]
    check_context_result(report["mixed_prefill"]["8192"], 8192)
    assert report["metadata"]["mixed_only"] is False
    # Detailed table after Phase 3, compact one again at the end of the primary summary.
    assert out.count("Mixed prefill + decode") >= 2
    assert out.index("Phase 4") > out.index("Phase 3")
    # The decode matrix ran before the mixed phase.
    decode_requests = [i for i, r in enumerate(server.requests) if r["tag"] == "" and r["max_tokens"] == 64]
    first_mixed = next(i for i, r in enumerate(server.requests) if r["tag"].startswith("MIXED_"))
    assert decode_requests and max(decode_requests) < first_mixed


def test_failed_background_streams_are_reported(monkeypatch, capsys, fake_server, tmp_path):
    server = fake_server(fail_background=True)
    report = run_main(monkeypatch, server, tmp_path, "--mixed-only")
    out = capsys.readouterr().out
    row = report["mixed_prefill"]["8192"]
    assert row["status"] == "error"
    assert row["ttft_alone_s"] > 0 and row["ttft_under_load_s"] is None
    assert any("no background stream was running" in e for e in row["errors"])
    assert any("HTTP 500" in e for e in row["errors"])
    assert row["background"]["errors"] == 4
    assert "ERROR" in out


def test_contexts_beyond_the_model_length_are_skipped(monkeypatch, capsys, fake_server, tmp_path):
    server = fake_server(max_model_len=12000)
    report = run_main(monkeypatch, server, tmp_path, "--mixed-only")
    capsys.readouterr()
    check_context_result(report["mixed_prefill"]["8192"], 8192)
    skipped = report["mixed_prefill"]["16384"]
    assert skipped["status"] == "skipped" and "model context length 12,000" in skipped["reason"]
    # Same keys as a measured context, without values.
    assert set(skipped) == set(report["mixed_prefill"]["8192"]) | {"reason"}
    assert skipped["ttft_alone_s"] is None and skipped["bg_tok_s_before"] is None
    assert not [r for r in server.tagged("MIXED_") if "_C16384_" in r["tag"]]


def test_report_tables_render_without_a_decode_matrix():
    row = {
        "context": 32768, "status": "ok", "ttft_alone_s": 3.21, "ttft_under_load_s": 4.87, "slowdown": 1.517,
        "prompt_tokens_alone": 32790, "prompt_tokens_under_load": 32790,
        "bg_tok_s_before": 512.3, "bg_tok_s_during": 201.7, "bg_tok_s_after": 498.1, "bg_during_vs_before": 0.394,
        "windows": {"before": {"seconds": 10.0}, "during": {"seconds": 4.87}, "after": {"seconds": 10.0}},
        "background": {"streams": 8, "prompt_tokens": 2061, "max_tokens": 1024},
        "errors": [], "warnings": [], "loop_detected": False,
    }
    skipped = {"context": 131072, "status": "skipped", "reason": "KV cache: needs 1 tokens"}
    console = BENCH.Console(file=io.StringIO(), width=160, color_system=None)
    BENCH.print_final_results([], [1], [0], console, {}, mixed_prefill_results={32768: row, 131072: skipped})
    text = " ".join(console.file.getvalue().split())
    assert "Mixed prefill + decode" in text and "8 background streams" in text
    assert "32,790" in text and "3.21" in text and "4.87" in text and "1.52x" in text
    assert "512" in text and "202" in text and "498" in text and "39%" in text
    assert "128k" in text and "skip" in text and "KV cache: needs 1 tokens" in text
    assert "bg kept = prefill / before" in text


def test_live_dashboard_shows_the_mixed_phase():
    state = BENCH.TUIState(
        engine=BENCH.ENGINE_VLLM, model_name="m", server_url="x:1", total_tests=2,
        overall_start=time.monotonic(), hw_monitor_enabled=False, metrics_available=False,
    )
    state.mixed_phase = state.cell_running = True
    state.cell_start = time.monotonic()
    state.current_context, state.current_concurrency = 131072, 8
    state.mixed_step = "prefill under load 3.2s [with brackets]"
    state.mixed_bg_tps, state.mixed_bg_decoding = 512.0, 8
    state.mixed_last = "TTFT alone 14.20s (131,050 tokens)"
    console = BENCH.Console(file=io.StringIO(), width=160, height=40, color_system=None)
    console.print(BENCH.build_display(state))
    text = console.file.getvalue()
    assert "MIXED PREFILL+DECODE" in text and "ctx=128k" in text and "background C=8" in text
    assert "prefill under load 3.2s [with brackets]" in text
    assert "512 tok/s" in text and "decoding 8/8" in text and "TTFT alone 14.20s" in text


def test_skip_key_ends_the_context_during_the_warmup(fake_server):
    server = fake_server()
    settings = BENCH.MixedPrefillSettings(
        contexts=[8192], decode_streams=2, decode_prompt_tokens=64, decode_max_tokens=32,
        warmup_seconds=30.0, window_seconds=1.0,
    )
    state = BENCH.TUIState(metrics_available=False, hw_monitor_enabled=False)

    async def run():
        async def press_skip():
            while len(server.tagged("MIXED_t_C8192_BG")) < 2:
                await asyncio.sleep(0.02)
            BENCH._skip_event.set()

        async with BENCH.httpx.AsyncClient() as client:
            presser = asyncio.create_task(press_skip())
            result = await BENCH.run_mixed_prefill_context(
                client, 8192, settings, model="fake-model", base_url=server.base_url,
                engine=BENCH.ENGINE_VLLM, prompt_text=lambda tokens, tag: f"[MIXED_t_{tag}] " + "z" * (tokens * 4),
                state=state, live=BENCH.NullLive(),
            )
            await presser
        return result

    BENCH._skip_event.clear()
    started = time.monotonic()
    result = asyncio.run(run())
    assert time.monotonic() - started < 10
    assert result["status"] == "skipped" and result["reason"] == "skipped by user"
    assert result["ttft_alone_s"] > 0 and result["ttft_under_load_s"] is None
    assert not server.tagged("MIXED_t_C8192_ARR")
    assert not state.mixed_phase and not state.cell_running
    assert not BENCH._skip_event.is_set()
