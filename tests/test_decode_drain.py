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


if __name__ == "__main__":
    unittest.main()
