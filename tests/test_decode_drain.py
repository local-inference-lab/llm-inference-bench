import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "llm_decode_bench.py"
SPEC = importlib.util.spec_from_file_location("llm_decode_bench", MODULE_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCH)


class _Response:
    def __init__(self, status_code: int):
        self.status_code = status_code


class _Client:
    def __init__(self, status_code: int = 200):
        self.status_code = status_code
        self.calls = []

    async def post(self, url, **kwargs):
        self.calls.append((url, kwargs))
        return _Response(self.status_code)


class SGLangAbortTests(unittest.IsolatedAsyncioTestCase):
    async def test_targeted_abort_uses_stream_request_id(self):
        client = _Client()

        ok = await BENCH.abort_sglang_request(
            client,
            "http://127.0.0.1:30000/abort_request",
            "request-123",
        )

        self.assertTrue(ok)
        self.assertEqual(len(client.calls), 1)
        url, kwargs = client.calls[0]
        self.assertEqual(url, "http://127.0.0.1:30000/abort_request")
        self.assertEqual(kwargs["json"], {"rid": "request-123"})

    async def test_empty_request_id_never_aborts_other_requests(self):
        client = _Client()

        ok = await BENCH.abort_sglang_request(
            client,
            "http://127.0.0.1:30000/abort_request",
            "",
        )

        self.assertFalse(ok)
        self.assertEqual(client.calls, [])

    async def test_non_success_response_returns_false(self):
        client = _Client(status_code=404)

        ok = await BENCH.abort_sglang_request(
            client,
            "http://127.0.0.1:30000/abort_request",
            "request-123",
        )

        self.assertFalse(ok)


class SGLangSpecNormalizationTests(unittest.TestCase):
    def test_matched_gauges_produce_verifier_rate(self):
        self.assertAlmostEqual(
            BENCH.sglang_step_rate_sample(286.0, 4.0),
            71.5,
        )

    def test_invalid_or_non_speculative_gauges_are_ignored(self):
        self.assertEqual(BENCH.sglang_step_rate_sample(0.0, 3.0), 0.0)
        self.assertEqual(BENCH.sglang_step_rate_sample(200.0, 1.0), 0.0)

    def test_sampled_rate_overrides_unmatched_final_gauge(self):
        cell = BENCH.CellResult(
            concurrency=1,
            context_tokens=0,
            aggregate_tps=270.0,
            measurement_seconds=30.0,
        )

        BENCH.apply_spec_normalization(
            cell,
            {},
            BENCH.ENGINE_SGLANG,
            gauge_accept_len=2.0,
            sampled_steps_per_s=72.0,
        )

        self.assertEqual(cell.server_steps_per_s, 72.0)
        self.assertEqual(cell.server_accept_len_effective, 3.75)
        self.assertEqual(cell.server_engine_steps, 2160.0)


if __name__ == "__main__":
    unittest.main()
