"""Decode results must belong to the benchmark's streams, not other server work."""

import asyncio
import importlib.util
import inspect
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest


@pytest.fixture
def bench(monkeypatch):
    source = Path(os.environ.get("BENCH_SOURCE", Path(__file__).resolve().parents[1] / "llm_decode_bench.py"))
    spec = importlib.util.spec_from_file_location("decode_ownership_test_module", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "build_display", lambda state: None)

    async def idle(*args, **kwargs):
        return None

    monkeypatch.setattr(module, "require_decode_server_idle", idle, raising=False)
    return module


def run_cell(bench, monkeypatch, mode):
    scrapes = []
    state = bench.TUIState(engine=bench.ENGINE_VLLM, metrics_available=True)
    live_values = []

    async def metrics(*args, **kwargs):
        scrapes.append(len(scrapes))
        return {
            "vllm:num_requests_running": 1,
            "vllm:num_requests_waiting": int(mode == "queued"),
            "vllm:avg_generation_throughput_toks_per_s": 500,
            "vllm:generation_tokens_total": 10000 * len(scrapes),
        }

    async def stream(client, url, payload, index, cancel_event, chunks,
                     active, samples, **kwargs):
        count = 0
        if mode == "queued":
            await cancel_event.wait()
        else:
            active[0] = 1
            while not cancel_event.is_set():
                chunks[0] += 1
                count += 1
                kwargs["shared_token_last_time"][0] = time.monotonic()
                if mode == "usage":
                    kwargs["shared_usage_token_count"][0] += 2
                    kwargs["shared_usage_last_time"][0] = time.monotonic()
                await asyncio.sleep(0.01)
        return bench.StreamResult(total_tokens=count)

    monkeypatch.setattr(bench, "scrape_metrics", metrics)
    monkeypatch.setattr(bench, "stream_one_request", stream)
    live = SimpleNamespace(update=lambda value: live_values.append(state.cell_live_tps))

    async def execute():
        async with httpx.AsyncClient() as client:
            kwargs = {
                "client": client, "base_url": "http://unused.invalid", "concurrency": 1,
                "context_tokens": 0, "context_text": "", "duration": 0.05, "max_tokens": 1024,
                "model": "test", "state": state, "live": live, "engine": bench.ENGINE_VLLM,
                "cell_warmup_timeout_seconds": 0.01,
            }
            if "loop_detection" in inspect.signature(bench.run_one_cell).parameters:
                kwargs["loop_detection"] = False
            return await bench.run_one_cell(**kwargs)

    return execute, state, live_values, scrapes


def test_queued_request_cannot_borrow_another_requests_throughput(bench, monkeypatch):
    execute, state, live, scrapes = run_cell(bench, monkeypatch, "queued")
    with pytest.raises(RuntimeError, match="No output tokens received"):
        asyncio.run(execute())
    assert len(scrapes) >= 2
    assert state.srv_gen_throughput == 500
    assert live and max(live) == 0
    assert not state.cell_running


@pytest.mark.parametrize("mode,source", [
    ("usage", "openai_continuous_usage"),
    ("chunks", "openai_stream_chunks_fallback"),
])
def test_received_output_owns_rate_and_token_total(bench, monkeypatch, mode, source):
    execute, _, _, _ = run_cell(bench, monkeypatch, mode)
    cell = asyncio.run(execute())
    assert cell.aggregate_source == source
    assert cell.aggregate_tps > 0
    assert cell.total_tokens == cell.client_output_tokens
    assert cell.server_output_tokens > cell.total_tokens


def test_resume_discards_unattributed_server_results(bench, tmp_path):
    if not hasattr(bench, "load_resume_checkpoint"):
        pytest.skip("This source revision does not implement resume checkpoints")
    args = SimpleNamespace(output=str(tmp_path / "results.json"), resume_signature={"case": "ownership"})
    valid = {"aggregate_source": "openai_continuous_usage", "client_output_tokens": 10}
    invalid = {"aggregate_source": "prometheus_fallback", "client_output_tokens": 0}
    path = Path(bench.resume_checkpoint_path(args))
    path.write_text(json.dumps({"signature": args.resume_signature,
                                "results": [valid, invalid], "prefill_results": {}}))
    assert bench.load_resume_checkpoint(args)["results"] == [valid]
