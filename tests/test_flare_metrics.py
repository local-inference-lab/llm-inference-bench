import asyncio
import importlib.util
from pathlib import Path
import unittest
from unittest.mock import patch


SPEC = importlib.util.spec_from_file_location(
    "flare_metrics_bench", Path(__file__).resolve().parents[1] / "llm_decode_bench.py"
)
BENCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BENCH)


class Client:
    def __init__(self, text):
        self.text = text

    async def get(self, *args, **kwargs):
        return type("Response", (), {"text": self.text})()


class FlareMetricsTests(unittest.IsolatedAsyncioTestCase):
    async def test_scrape_and_exact_metric_names(self):
        metrics = await BENCH.scrape_metrics(Client(
            'flare_generated_tokens_total 100\n'
            'flare_generated_tokens_total_extra 999\n'
            'flare_total_kv_tokens 18175232\n'
            'flare_speculative_rounds_total{batch="1",width="3"} 2\n'
            'flare_speculative_rounds_total{batch="4",width="6"} 3\n'
            'vllm:generation_tokens_total{tp_rank="1"} 500\n'
            'vllm:generation_tokens_total{tp_rank="0"} 250\n'
        ), "http://example")
        self.assertEqual(BENCH.extract_metric(metrics, "flare_generated_tokens_total"), 100)
        self.assertEqual(BENCH.sum_metric(metrics, "flare_generated_tokens_total"), 100)
        self.assertFalse(BENCH.has_metric(metrics, "flare_generated_tokens"))
        self.assertEqual(BENCH.sum_metric(metrics, "flare_speculative_rounds_total"), 5)
        self.assertEqual(BENCH.extract_metric(metrics, "vllm:generation_tokens_total"), 250)
        self.assertEqual(BENCH.extract_metric(metrics, "flare_total_kv_tokens"), 18175232)

    def test_label_extraction_does_not_match_label_suffixes(self):
        metrics = {'vllm:cache_config_info|physical_block_size="512",block_size="16"': 1}
        self.assertEqual(BENCH.extract_label(metrics, "vllm:cache_config_info", "block_size"), "16")

    def test_nonpositive_measurement_interval_is_invalid(self):
        before, after = self.snapshots()
        after["sample_time"] = before["sample_time"]
        self.assertFalse(BENCH.flare_metric_delta(before, after)["valid"])

    def snapshots(self):
        # Two batched graph executions can contain more request rounds, plus
        # padded/cancelled lanes whose raw records are not committed output.
        counters = dict(generated_tokens=15, prefill_tokens=0, draft_request_rounds=4,
                        accepted_draft_tokens=7, verification_graphs=2,
                        raw_emitted_tokens=19, raw_offered_drafts=20, raw_accepted_drafts=9)
        before = {"engine": BENCH.ENGINE_FLARE, "sample_time": 10,
                  "counters": {key: 100 for key in counters}}
        after = {"engine": BENCH.ENGINE_FLARE, "sample_time": 12,
                 "counters": {key: 100 + value for key, value in counters.items()}}
        return before, after

    def test_request_acceptance_does_not_use_padded_graph_denominator(self):
        before, after = self.snapshots()
        norm = BENCH.compute_spec_normalization(before, after, 15, 7.5)
        cell = BENCH.CellResult(concurrency=4, context_tokens=8192)
        BENCH.apply_spec_normalization(cell, norm, BENCH.ENGINE_FLARE, 0)
        self.assertEqual(cell.server_spec_drafts, 4)
        self.assertEqual(cell.server_spec_accepted_tokens, 7)
        self.assertIsNone(cell.server_spec_draft_tokens)
        self.assertEqual(cell.server_spec_accept_length, 2.75)
        self.assertEqual(cell.server_spec_accept_rate, 0.45)
        self.assertEqual(cell.server_flare_metrics["verification_graphs_per_second"], 1)
        self.assertEqual(cell.server_flare_metrics["counter_delta"]["generated_tokens"], 15)
        self.assertEqual(cell.server_flare_metrics["counter_delta"]["raw_emitted_tokens"], 19)
        self.assertEqual(cell.server_engine_steps, 0)
        self.assertEqual(cell.accept_norm_tps, 0)

    def test_counter_reset_and_disappearing_counter_are_invalid(self):
        before, after = self.snapshots()
        after["counters"]["generated_tokens"] = 1
        self.assertFalse(BENCH.flare_metric_delta(before, after)["valid"])
        before, after = self.snapshots()
        del after["counters"]["raw_offered_drafts"]
        self.assertFalse(BENCH.flare_metric_delta(before, after)["valid"])

    def test_missing_or_optional_counters_do_not_invent_acceptance(self):
        self.assertEqual(BENCH.snapshot_spec_counters({}, BENCH.ENGINE_FLARE), {})
        before = {"sample_time": 1, "counters": {"generated_tokens": 10}}
        after = {"sample_time": 2, "counters": {"generated_tokens": 20}}
        delta = BENCH.flare_metric_delta(before, after)
        self.assertTrue(delta["valid"])
        self.assertNotIn("request_acceptance_length", delta)
        self.assertNotIn("raw_draft_acceptance", delta)

    def test_vllm_normalization_is_preserved(self):
        before = dict(drafts=10, draft_tokens=50, accepted_tokens=20, pos=[])
        after = dict(drafts=14, draft_tokens=70, accepted_tokens=28, pos=[])
        norm = BENCH.compute_spec_normalization(before, after, 15, 7.5)
        self.assertEqual(norm["engine_steps"], 7)
        self.assertEqual(norm["accept_rate"], 0.4)
        self.assertEqual(norm["accept_len_spec"], 3)

    def test_failed_window_does_not_reuse_last_live_acceptance_sample(self):
        for norm in ({}, {"flare": {"valid": False, "reason": "counter reset"}}):
            cell = BENCH.CellResult(concurrency=1, context_tokens=8192,
                                    server_spec_accept_rate=0.7, server_spec_accept_length=4.5)
            BENCH.apply_spec_normalization(cell, norm, BENCH.ENGINE_FLARE, 4.5)
            self.assertFalse(cell.server_flare_metrics["valid"])
            self.assertEqual(cell.server_spec_accept_rate, 0)
            self.assertEqual(cell.server_spec_accept_length, 0)
            self.assertIsNone(cell.server_spec_draft_tokens)

    async def test_idle_barrier_rejects_retiring_work_and_missing_metrics(self):
        names = ("flare_active_requests", "flare_pending_requests", "flare_retiring_request_slots",
                 "flare_http_inflight_requests", "flare_frontend_active_workers")
        idle = {name: 0 for name in names}
        state = BENCH.TUIState()
        with patch.object(BENCH, "scrape_metrics", return_value={**idle, names[2]: 1}):
            with self.assertRaisesRegex(RuntimeError, "did not drain"):
                await BENCH.require_decode_server_idle(None, "", BENCH.ENGINE_FLARE, state, None, "test", 0)
        with patch.object(BENCH, "scrape_metrics", return_value={}):
            with self.assertRaisesRegex(RuntimeError, "unavailable"):
                await BENCH.require_decode_server_idle(None, "", BENCH.ENGINE_FLARE, state, None, "test", 0)
        with patch.object(BENCH, "scrape_metrics", return_value=idle):
            await asyncio.wait_for(
                BENCH.require_decode_server_idle(None, "", BENCH.ENGINE_FLARE, state, None, "test", 2), 3
            )


if __name__ == "__main__":
    unittest.main()
