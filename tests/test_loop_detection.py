"""Exact-repetition guard, cancellation, and invalid-result contracts."""

import asyncio
from contextlib import asynccontextmanager
import importlib.util
import io
import json
from pathlib import Path
import random
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, patch


MODULE_PATH = Path(__file__).resolve().parents[1] / "llm_decode_bench.py"
SPEC = importlib.util.spec_from_file_location("bench_loop_tests", MODULE_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


def detect(text, fragment_size=31):
    detector = BENCH.StreamingLoopDetector()
    findings = []
    for offset in range(0, len(text), fragment_size):
        finding = detector.feed(text[offset:offset + fragment_size])
        if finding:
            findings.append(finding)
            if finding["status"] == "confirmed":
                break
    finding = detector.finish()
    if finding:
        findings.append(finding)
    return detector, findings


class ExactRepetitionTests(unittest.TestCase):
    def test_fragmentation_does_not_change_confirmation_or_offsets(self):
        text = "Introduction.\n" + "duct" * 1200
        evidence = []
        for size in (1, 7, 256, 901, len(text)):
            _, findings = detect(text, size)
            self.assertEqual(findings[-1]["status"], "confirmed")
            self.assertEqual(findings[-1]["period_chars"], 4)
            evidence.append(findings[-1])
        self.assertTrue(all(f == evidence[0] for f in evidence))
        self.assertEqual(evidence[0]["observed_start_char"], len("Introduction.\n"))

    def test_four_cycles_and_4096_characters_are_both_required(self):
        rng = random.Random(491)
        cycle = "".join(rng.choices("abcdefghijklmnopqrstuvwxyz", k=1500))
        _, findings = detect(cycle * 3)
        self.assertEqual(findings[-1]["status"], "suspected")
        _, findings = detect(cycle * 4)
        self.assertEqual(findings[-1]["status"], "confirmed")
        _, findings = detect("duct" * 1023)
        self.assertEqual(findings[-1]["status"], "suspected")
        _, findings = detect("duct" * 1024)
        self.assertEqual(findings[-1]["status"], "confirmed")

    def test_long_cycles_partial_tail_and_unicode(self):
        rng = random.Random(752)
        for length in (222, 384, 9560, BENCH.LOOP_MAX_PERIOD_CHARS):
            cycle = "".join(rng.choices("abcde č漢🙂\n", k=length))
            text = "Unique prefix!" + cycle * max(5, 5000 // length + 1) + cycle[:117]
            _, findings = detect(text, 41)
            self.assertEqual(findings[-1]["status"], "confirmed", length)
            self.assertEqual(findings[-1]["period_chars"], length)
            self.assertEqual(findings[-1]["coordinate_unit"], "unicode_code_points_in_channel")

    def test_numbered_code_whitespace_and_nonidentical_sections(self):
        for text in (
            " \n\t" * 10000,
            "\n".join(f"assert compute({i}) == {i * i}  # case {i}" for i in range(6000)),
            "\n".join(f"Section {i}: " + "describes a distinct result. " * 10 for i in range(100)),
        ):
            detector, findings = detect(text)
            self.assertNotIn("confirmed", [f["status"] for f in findings])
            self.assertLessEqual(len(detector.text), BENCH.LOOP_WINDOW_CHARS)
            self.assertLess(detector.pending_chars, BENCH.LOOP_CHECK_CHARS)

    def test_evidence_is_bounded_and_keeps_confirmed_findings(self):
        state = BENCH.LoopCheckState()
        for i in range(100):
            state.record({"status": "suspected"}, stream_index=i, request_ordinal=0, channel="content")
        state.record({"status": "confirmed"}, stream_index=200, request_ordinal=0, channel="content")
        self.assertTrue(state.confirmed)
        self.assertEqual(len(state.findings), 64)
        self.assertEqual(state.findings[-1]["status"], "confirmed")
        self.assertEqual(state.omitted, 37)

    def test_default_enabled_and_explicit_opt_out(self):
        with patch("sys.argv", [str(MODULE_PATH)]):
            self.assertTrue(BENCH.parse_args().loop_detection)
        with patch("sys.argv", [str(MODULE_PATH), "--no-loop-detection"]):
            self.assertFalse(BENCH.parse_args().loop_detection)


class FakeStreamingClient:
    """OpenAI SSE fixture with observable request ownership at abort time."""

    def __init__(self, deltas, *, done=True, before_delta=None):
        self.deltas = deltas
        self.done = done
        self.requests = []
        self.open_requests = set()
        self.aborts = []
        self.closed = False
        self.before_delta = before_delta

    @asynccontextmanager
    async def stream(self, method, url, *, json, timeout):
        rid = f"fixture-{len(self.requests)}"
        self.requests.append(json)
        self.open_requests.add(rid)

        async def lines():
            for index, delta in enumerate(self.deltas):
                if self.before_delta:
                    await self.before_delta(index)
                await asyncio.sleep(0)
                yield "data: " + BENCH.json.dumps({
                    "id": rid, "choices": [{"delta": delta}],
                    "usage": {"completion_tokens": index + 1, "prompt_tokens": 21},
                })
            if self.done:
                yield "data: [DONE]"

        try:
            yield SimpleNamespace(status_code=200, aiter_lines=lines)
        finally:
            self.open_requests.remove(rid)

    async def post(self, url, **kwargs):
        rid = kwargs["json"]["rid"]
        self.aborts.append((rid, rid in self.open_requests))
        return SimpleNamespace(status_code=200)

    async def aclose(self):
        self.closed = True


class StreamingGuardTests(unittest.IsolatedAsyncioTestCase):
    async def stream(self, fake, *, count=1, check=None, viewer=None):
        check = check if check is not None else BENCH.LoopCheckState()
        cancel = asyncio.Event()
        result = await BENCH.stream_one_request(
            fake, "http://fixture/v1/chat/completions", {"stream": True}, 0,
            cancel, [0], shared_started_count=[0], target_request_count=count,
            abort_url="http://fixture/abort_request", loop_check=check,
            output_viewer=viewer,
        )
        return result, check, cancel

    async def test_loop_cancels_while_request_owned_and_does_not_restart(self):
        fake = FakeStreamingClient([{"content": "duct" * 128}] * 20)
        result, check, cancel = await self.stream(fake, count=3)
        self.assertEqual(result.error, "LoopDetected")
        self.assertTrue(cancel.is_set())
        self.assertTrue(check.confirmed)
        self.assertEqual(len(fake.requests), 1)
        self.assertEqual(fake.aborts, [("fixture-0", True)])
        self.assertEqual(check.findings[-1]["request_id"], "fixture-0")
        self.assertFalse(fake.open_requests)

    async def test_content_and_reasoning_are_separate(self):
        fake = FakeStreamingClient([{"reasoning_content": "duct" * 600, "content": "duct" * 600}])
        result, check, cancel = await self.stream(fake)
        self.assertIsNone(result.error)
        self.assertFalse(cancel.is_set())
        self.assertFalse(check.confirmed)
        self.assertEqual({f["channel"] for f in check.findings}, {"reasoning", "content"})

    async def test_history_resets_between_requests(self):
        fake = FakeStreamingClient([{"content": "duct" * 600}])
        result, check, cancel = await self.stream(fake, count=3)
        self.assertIsNone(result.error)
        self.assertFalse(check.confirmed)
        self.assertEqual(len(fake.requests), 3)
        self.assertEqual({f["request_ordinal"] for f in check.findings}, {0, 1, 2})

    async def test_disabled_guard_does_not_disable_preview_or_change_payload(self):
        fake = FakeStreamingClient([{"content": "duct" * 2000}])
        viewer = BENCH.OutputViewer()
        result, check, cancel = await self.stream(fake, check=BENCH.LoopCheckState(enabled=False), viewer=viewer)
        self.assertIsNone(result.error)
        self.assertFalse(check.findings)
        self.assertFalse(cancel.is_set())
        self.assertIn("duct" * 2000, "".join(viewer.parts))
        self.assertEqual(fake.requests[0], {"stream": True})

    async def test_final_partial_detector_block_is_checked_at_eof(self):
        cycle = "".join(random.Random(751).choices("abcdefghijklm", k=1100))
        fake = FakeStreamingClient([{"content": cycle * 4}], done=False)
        result, check, cancel = await self.stream(fake)
        self.assertTrue(check.confirmed)
        self.assertEqual(result.error, "LoopDetected")
        self.assertEqual(fake.aborts, [("fixture-0", True)])

    async def test_shared_guard_stops_concurrent_streams(self):
        fake = FakeStreamingClient([{"content": "duct" * 128}] * 20)
        check, cancel = BENCH.LoopCheckState(), asyncio.Event()
        results = await asyncio.gather(*[
            BENCH.stream_one_request(
                fake, "http://fixture/v1/chat/completions", {}, index, cancel, [0],
                loop_check=check, abort_url="http://fixture/abort_request",
            ) for index in range(8)
        ])
        self.assertTrue(check.confirmed)
        self.assertEqual(len(fake.requests), 8)
        self.assertFalse(fake.open_requests)
        self.assertTrue(all(owned for _, owned in fake.aborts))
        self.assertTrue(any(r.error == "LoopDetected" for r in results))


class CellGuardTests(unittest.IsolatedAsyncioTestCase):
    async def test_sustained_duration_stop_preserves_valid_speed(self):
        state = BENCH.TUIState(metrics_available=False, hw_monitor_enabled=False)
        fake = FakeStreamingClient([{"content": "An ordinary response."}])
        with patch.object(BENCH.httpx, "AsyncClient", return_value=fake), patch.object(BENCH, "require_decode_server_idle", AsyncMock()):
            cell = await BENCH.run_one_cell(
                fake, "http://fixture", 1, 0, "", 0.1, 10000, "model",
                state, BENCH.NullLive(), temperature=1.0,
                cell_warmup_timeout_seconds=0.01,
            )
        self.assertGreater(cell.aggregate_tps, 0)
        self.assertFalse(cell.failure_reason)
        self.assertFalse(cell.loop_detected)
        self.assertTrue(fake.closed)

    async def test_sustained_stream_error_after_warmup_invalidates_speed(self):
        state = BENCH.TUIState(metrics_available=False, hw_monitor_enabled=False)

        async def fail_during_measurement(index):
            if index == 1:
                async def ready():
                    while state.cell_warmup:
                        await asyncio.sleep(0.01)
                await asyncio.wait_for(ready(), timeout=5)
                raise BENCH.httpx.RemoteProtocolError("engine disconnected")

        fake = FakeStreamingClient(
            [{"content": "Beginning."}, {"content": "Unreachable."}],
            before_delta=fail_during_measurement,
        )
        with patch.object(BENCH.httpx, "AsyncClient", return_value=fake), patch.object(BENCH, "require_decode_server_idle", AsyncMock()):
            cell = await BENCH.run_one_cell(
                fake, "http://fixture", 1, 0, "", 30, 10000, "model",
                state, BENCH.NullLive(), temperature=1.0,
                cell_warmup_timeout_seconds=0.01,
            )
        self.assertEqual(cell.aggregate_tps, BENCH.ERROR_CELL_TPS)
        self.assertEqual(cell.ready_reason, "stream_failure_during_measurement")
        self.assertIn("engine disconnected", cell.failure_reason)

    async def test_stream_failure_before_measurement_is_error_not_speed(self):
        await self.assert_stream_failure_cell()

    async def test_burst_stream_failures_are_errors_during_warmup_and_measurement(self):
        for warmup in (0, 2):
            with self.subTest(warmup=warmup):
                await self.assert_stream_failure_cell(request_count=2, warmup=warmup)

    async def assert_stream_failure_cell(self, *, request_count=0, warmup=0):
        async def fail_after_one_token(index):
            if index == 1:
                raise BENCH.httpx.RemoteProtocolError("engine disconnected")

        fake = FakeStreamingClient(
            [{"content": "Beginning."}, {"content": "Unreachable."}],
            before_delta=fail_after_one_token,
        )
        state = BENCH.TUIState(metrics_available=False, hw_monitor_enabled=False)
        drain = AsyncMock()
        with patch.object(BENCH.httpx, "AsyncClient", return_value=fake), patch.object(BENCH, "require_decode_server_idle", drain):
            cell = await BENCH.run_one_cell(
                fake, "http://fixture", 2, 0, "", 30, 10000, "model",
                state, BENCH.NullLive(), temperature=1.0,
                request_count=request_count, warmup_request_count=warmup,
            )
        self.assertLess(cell.aggregate_tps, 0)
        self.assertEqual(cell.num_errors, 2)
        self.assertIn("engine disconnected", cell.failure_reason)
        self.assertEqual(BENCH.invalid_decode_label(cell), "ERROR")
        self.assertFalse(cell.loop_detected)
        self.assertFalse(state.cell_running)
        self.assertEqual(state.cell_live_tps, 0)
        self.assertEqual(drain.await_count, 2)
        self.assertTrue(fake.closed)
        console = BENCH.Console(file=io.StringIO(), width=140, color_system=None)
        BENCH.print_final_results([cell], [2], [0], console)
        self.assertIn("ERROR", console.file.getvalue())
        with patch("sys.argv", [str(MODULE_PATH), "--concurrency", "2", "--contexts", "0"]):
            args = BENCH.parse_args()
        with tempfile.TemporaryDirectory() as directory:
            args.output = str(Path(directory) / "report.json")
            BENCH.save_results([cell], args, args.output)
            report = json.loads(Path(args.output).read_text())
        self.assertIsNone(report["summary_table"]["0"]["2"])
        self.assertIn("engine disconnected", report["results"][0]["failure_reason"])

    async def run_cell(self, *, request_count=0, warmup=0, loop_detection=True):
        fake = FakeStreamingClient([{"content": "duct" * 1500}])
        state = BENCH.TUIState(metrics_available=False, hw_monitor_enabled=False)
        BENCH._skip_event.clear()
        drain = AsyncMock()
        with patch.object(BENCH.httpx, "AsyncClient", return_value=fake), patch.object(BENCH, "require_decode_server_idle", drain):
            cell = await BENCH.run_one_cell(
                fake, "http://fixture", 2, 0, "", 30, 10000, "model",
                state, BENCH.NullLive(), request_count=request_count,
                warmup_request_count=warmup, loop_detection=loop_detection,
            )
        return cell, state, fake, drain

    async def test_loop_during_sustained_warmup_is_error_not_throughput(self):
        cell, state, fake, drain = await self.run_cell()
        self.assertTrue(cell.loop_detected)
        self.assertTrue(cell.loop_detection_enabled)
        self.assertEqual(cell.aggregate_tps, BENCH.LOOP_CELL_TPS)
        self.assertEqual(state.results[(0, 2)], BENCH.LOOP_CELL_TPS)
        self.assertEqual(state.cell_live_tps, 0)
        self.assertFalse(state.cell_running)
        self.assertEqual(cell.ready_reason, "loop_detected_during_warmup")
        self.assertEqual(drain.await_count, 2)
        self.assertTrue(fake.closed)
        self.assertFalse(fake.open_requests)

    async def test_loop_during_burst_warmup_prevents_measured_requests(self):
        cell, state, fake, drain = await self.run_cell(request_count=5, warmup=2)
        self.assertTrue(cell.loop_detected)
        self.assertEqual(cell.ready_reason, "loop_detected_during_warmup")
        self.assertEqual(len(fake.requests), 2)
        self.assertEqual(drain.await_count, 2)

    async def test_loop_during_burst_measurement_is_error(self):
        cell, state, fake, drain = await self.run_cell(request_count=5)
        self.assertTrue(cell.loop_detected)
        self.assertEqual(cell.ready_reason, "loop_detected_during_measurement")
        self.assertEqual(len(fake.requests), 2)

    async def test_loop_after_sustained_measurement_begins_invalidates_rate(self):
        state = BENCH.TUIState(metrics_available=False, hw_monitor_enabled=False)

        async def await_measurement(index):
            if index == 1:
                async def ready():
                    while state.cell_warmup:
                        await asyncio.sleep(0.01)
                await asyncio.wait_for(ready(), timeout=5)

        fake = FakeStreamingClient(
            [{"content": "Starting answer."}, {"content": "duct" * 1500}],
            before_delta=await_measurement,
        )
        with patch.object(BENCH.httpx, "AsyncClient", return_value=fake), patch.object(BENCH, "require_decode_server_idle", AsyncMock()):
            cell = await BENCH.run_one_cell(
                fake, "http://fixture", 1, 0, "", 30, 10000, "model",
                state, BENCH.NullLive(), cell_warmup_timeout_seconds=0.01,
            )
        self.assertTrue(cell.loop_detected)
        self.assertEqual(cell.aggregate_tps, BENCH.LOOP_CELL_TPS)
        self.assertEqual(cell.ready_reason, "loop_detected_during_measurement")
        self.assertEqual(state.cell_live_tps, 0)
        self.assertFalse(fake.open_requests)

    async def test_explicit_opt_out_preserves_completed_measurement(self):
        cell, state, fake, drain = await self.run_cell(request_count=2, loop_detection=False)
        self.assertFalse(cell.loop_detection_enabled)
        self.assertFalse(cell.loop_detected)
        self.assertGreater(cell.aggregate_tps, 0)
        self.assertEqual(cell.completed_request_count, 2)

    async def test_drain_failure_does_not_overwrite_loop_with_numeric_rate(self):
        state = BENCH.TUIState(metrics_available=True, hw_monitor_enabled=False)
        fake = FakeStreamingClient([{"content": "duct" * 1500}])
        drain = AsyncMock(side_effect=[None, RuntimeError("scheduler did not drain")])
        with patch.object(BENCH.httpx, "AsyncClient", return_value=fake), patch.object(BENCH, "require_decode_server_idle", drain):
            cell = await BENCH.run_one_cell(
                fake, "http://fixture", 1, 0, "", 30, 10000, "model",
                state, BENCH.NullLive(),
            )
        self.assertTrue(cell.loop_detected)
        self.assertEqual(cell.aggregate_tps, BENCH.LOOP_CELL_TPS)
        self.assertIn("scheduler did not drain", cell.timeout_reason)
        self.assertEqual(len(fake.requests), 1)

    async def test_error_survives_report_json_and_resume(self):
        cell, state, fake, drain = await self.run_cell(request_count=2)
        console = BENCH.Console(file=io.StringIO(), width=140, color_system=None)
        BENCH.print_final_results([cell], [2], [0], console)
        self.assertIn("ERROR: loop", console.file.getvalue())
        with patch("sys.argv", [str(MODULE_PATH), "--concurrency", "2", "--contexts", "0"]):
            args = BENCH.parse_args()
        with tempfile.TemporaryDirectory() as directory:
            args.output = str(Path(directory) / "report.json")
            BENCH.save_results([cell], args, args.output, burst_results=[cell])
            report = json.loads(Path(args.output).read_text())
            self.assertIsNone(report["summary_table"]["0"]["2"])
            self.assertIsNone(report["burst_summary_table"]["0"]["2"])
            self.assertTrue(report["results"][0]["loop_detected"])
            self.assertTrue(report["results"][0]["loop_diagnostics"])
            self.assertTrue(report["metadata"]["loop_detection"]["enabled"])
            BENCH.write_resume_checkpoint(args, [cell], [], {})
            checkpoint = BENCH.load_resume_checkpoint(args)
            self.assertTrue(checkpoint["results"][0]["loop_detected"])
            args.loop_detection = False
            self.assertIsNone(BENCH.load_resume_checkpoint(args))


if __name__ == "__main__":
    unittest.main()
