"""Exercise readiness and timeout precedence through the sustained cell runner."""
import asyncio
from dataclasses import asdict
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import time
from unittest.mock import AsyncMock

import pytest


SPEC = importlib.util.spec_from_file_location(
    "llm_decode_bench_warmup", Path(__file__).resolve().parents[1] / "llm_decode_bench.py"
)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


async def run_cell(monkeypatch, *, engine, metrics_available=True,
                   polls=(12.0, 15.0), ready=True, queue=0):
    clock = SimpleNamespace(now=100.0)
    ticks = iter((*polls, polls[-1] + 2.0))

    async def poll_sleep(_):
        # Only the benchmark's polling clock advances; asyncio keeps its real clock.
        clock.now = 100.0 + next(ticks)
        await asyncio.sleep(0)

    async def stream(client, url, payload, index, cancel, tokens, active, *args, **kwargs):
        tokens[0] += 1
        if ready:
            active[0] += 1
        await cancel.wait()
        return BENCH.StreamResult(total_tokens=1)

    metrics = {
        BENCH.metric_name(engine, "running_reqs"): 2,
        BENCH.metric_name(engine, "queue_reqs"): queue,
        BENCH.metric_name(engine, "gen_throughput"): 100,
    }
    client = AsyncMock()
    monkeypatch.setattr(BENCH, "time", SimpleNamespace(
        monotonic=lambda: clock.now, strftime=time.strftime, time=time.time,
    ))
    monkeypatch.setattr(BENCH, "asyncio", SimpleNamespace(**(vars(asyncio) | {"sleep": poll_sleep})))
    monkeypatch.setattr(BENCH.httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(BENCH, "stream_one_request", stream)
    monkeypatch.setattr(BENCH, "scrape_metrics", AsyncMock(return_value=metrics))
    monkeypatch.setattr(BENCH, "require_decode_server_idle", AsyncMock())
    monkeypatch.setattr(BENCH, "build_display", lambda state: None)
    state = BENCH.TUIState(metrics_available=metrics_available, hw_monitor_enabled=False)
    cell = await BENCH.run_one_cell(
        client, "http://fixture", 2, 0, "", 1.0, 10000, "model",
        state, BENCH.NullLive(), engine=engine, cell_warmup_timeout_seconds=15.0,
    )
    client.aclose.assert_awaited_once()
    return cell, state


@pytest.mark.parametrize("engine", [BENCH.ENGINE_SGLANG, BENCH.ENGINE_VLLM, BENCH.ENGINE_FLARE])
@pytest.mark.parametrize("polls", [(11.0, 14.0), (12.0, 15.0), (13.0, 16.0)])
def test_readiness_wins_before_at_and_after_timeout(monkeypatch, engine, polls):
    cell, state = asyncio.run(run_cell(monkeypatch, engine=engine, polls=polls))
    assert not cell.warmup_timed_out
    assert not cell.capacity_limited
    assert cell.timeout_reason == ""
    assert "stable=3.0s" in cell.ready_reason
    assert cell.warmup_duration == polls[-1]
    assert cell.measurement_wall_seconds == 2.0
    assert sum("ready C=2" in event for event in state.events) == 1
    assert not any("warmup timeout" in event for event in state.events)
    report = json.loads(json.dumps(asdict(cell)))
    assert not report["warmup_timed_out"] and not report["capacity_limited"]
    assert state.queue_info[(0, 2)][2] is False


def test_client_only_readiness_wins_at_timeout(monkeypatch):
    cell, state = asyncio.run(run_cell(
        monkeypatch, engine=BENCH.ENGINE_SGLANG, metrics_available=False,
    ))
    assert not cell.warmup_timed_out
    assert not cell.capacity_limited
    assert "metrics_unavailable, active_streams=2/2" in cell.ready_reason
    assert not any("warmup timeout" in event for event in state.events)


@pytest.mark.parametrize("kwargs,reason", [
    ({"ready": False}, "active_streams=0/2"),
    ({"queue": 1}, "queue_reqs=1"),
    ({"polls": (13.0, 15.0)}, "not_stable"),
])
def test_unready_cells_still_time_out(monkeypatch, kwargs, reason):
    cell, state = asyncio.run(run_cell(monkeypatch, engine=BENCH.ENGINE_SGLANG, **kwargs))
    assert cell.warmup_timed_out
    assert cell.capacity_limited
    assert cell.ready_reason == "warmup_timeout"
    assert cell.timeout_reason == reason
    assert sum("warmup timeout" in event for event in state.events) == 1
    assert not any("ready C=2" in event for event in state.events)
