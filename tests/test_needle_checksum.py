"""Tests for the needle-checksum profile: its pinned prompt, request settings, scorer and
streamed tool-call capture.

The profile rebuilds logprobz's GLM-5.3-Flash context-check probe exactly (291 REC
lines, three CANONICAL FACT needles, one submit_context_check tool call).
"""
import asyncio
from contextlib import asynccontextmanager
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "llm_decode_bench.py"
SPEC = importlib.util.spec_from_file_location("llm_decode_bench", MODULE_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCH)

PROFILE = BENCH.BUILTIN_TEST_PROFILES["needle-checksum"]
TOOL = BENCH.NEEDLE_CHECKSUM_TOOL


def call(alpha=478, beta=788, gamma=426, checksum=3332, name=TOOL) -> dict:
    return {
        "name": name,
        "arguments": json.dumps({"alpha": alpha, "beta": beta, "gamma": gamma, "checksum": checksum}),
    }


def score(tool_calls, visible: str = "", finish_reason: str = "tool_calls") -> dict:
    return BENCH.score_completion_profile(
        profile=PROFILE,
        final_answer=BENCH.extract_answer_line(visible),
        content_text=visible,
        output_text=visible,
        regex="",
        source=PROFILE["score_source"],
        finish_reason=finish_reason,
        tool_calls=tool_calls,
    )


class NeedleChecksumPromptTests(unittest.TestCase):
    def test_prompt_matches_the_pinned_sha256(self):
        prompt, source, profile = BENCH.decode_builtin_test_profile_prompt("needle-checksum")
        self.assertEqual(source, "profile:needle-checksum")
        self.assertEqual(hashlib.sha256(prompt.encode("utf-8")).hexdigest(), profile["prompt_sha256"])

    def test_messages_and_tool_reproduce_the_original_probe(self):
        messages = [
            {"role": "system", "content": PROFILE["system_prompt"]},
            {"role": "user", "content": BENCH.NEEDLE_CHECKSUM_PROMPT},
        ]
        canonical = json.dumps(
            {"messages": messages, "tools": PROFILE["request_overrides"]["tools"]},
            sort_keys=True, separators=(",", ":"),
        )
        self.assertEqual(
            hashlib.sha256(canonical.encode()).hexdigest(), BENCH.NEEDLE_CHECKSUM_PROBE_SHA256
        )

    def test_dataset_hides_three_facts_among_291_records(self):
        lines = BENCH.NEEDLE_CHECKSUM_PROMPT.splitlines()
        records = [line for line in lines if line.startswith("REC ")]
        facts = [line for line in lines if line.startswith("CANONICAL FACT ")]
        self.assertEqual(len(records), 291)
        self.assertEqual(facts, [
            "CANONICAL FACT ALPHA=478",
            "CANONICAL FACT BETA=788",
            "CANONICAL FACT GAMMA=426",
        ])
        self.assertTrue(lines[lines.index(facts[0]) - 1].startswith("REC 000014 "))
        self.assertTrue(lines[lines.index(facts[1]) - 1].startswith("REC 000145 "))
        self.assertTrue(lines[lines.index(facts[2]) - 1].startswith("REC 000276 "))
        self.assertEqual(BENCH.NEEDLE_CHECKSUM_EXPECTED, 3332)
        self.assertEqual(lines[-2:], [
            "Call submit_context_check exactly once with the three values and CHECKSUM.",
            "Do not answer in prose.",
        ])

    def test_aliases_resolve(self):
        for alias in ("needle", "checksum", "checksum-3332"):
            with self.subTest(alias=alias):
                self.assertEqual(BENCH.normalize_builtin_test_profile_name(alias), "needle-checksum")

    def test_requests_are_greedy_parallel_tool_calls(self):
        self.assertEqual(PROFILE["default_temperature"], 0.0)
        self.assertEqual(PROFILE["default_top_p"], 0.95)
        self.assertEqual(PROFILE["default_max_tokens"], 4096)
        self.assertEqual(PROFILE["default_concurrency"], 8)
        self.assertEqual(PROFILE["default_runs"], 500)
        overrides = PROFILE["request_overrides"]
        self.assertEqual(overrides["tool_choice"], "auto")
        self.assertEqual(overrides["tools"][0]["function"]["name"], TOOL)
        self.assertEqual(overrides["chat_template_kwargs"], {"thinking": True, "reasoning_effort": "max"})


class NeedleChecksumScorerTests(unittest.TestCase):
    def test_exact_tool_call(self):
        result = score([call()])
        self.assertTrue(result["correct"])
        self.assertEqual(result["score_label"], "exact")
        self.assertEqual(result["parsed_answer"], "alpha=478 beta=788 gamma=426 checksum=3332")

    def test_known_last_digit_slip_is_a_near_miss(self):
        result = score([call(checksum=3330)])
        self.assertFalse(result["correct"])
        self.assertEqual(result["score_label"], "near_miss")
        self.assertEqual(result["score_detail"], "checksum 3330, expected 3332")

    def test_other_checksum_is_wrong_sum(self):
        self.assertEqual(score([call(checksum=3342)])["score_label"], "wrong_checksum")

    def test_wrong_retrieved_value_is_wrong_facts(self):
        result = score([call(beta=787, checksum=3330)])
        self.assertEqual(result["score_label"], "wrong_facts")
        self.assertEqual(result["score_detail"], "BETA 787 (expected 788)")

    def test_prose_answer_without_a_call_fails(self):
        result = score([], visible="CHECKSUM = 3332", finish_reason="stop")
        self.assertEqual(result["score_label"], "fail")
        self.assertEqual(result["score_detail"], f"unparseable: no {TOOL} call")

    def test_malformed_calls_fail(self):
        cases = {
            "two calls": [call(), call()],
            "other tool": [call(name="other_tool")],
            "not json": [{"name": TOOL, "arguments": "{alpha: 478"}],
            "string value": [{"name": TOOL, "arguments": json.dumps(
                {"alpha": 478, "beta": 788, "gamma": 426, "checksum": "3332"})}],
            "missing key": [{"name": TOOL, "arguments": json.dumps({"alpha": 478, "beta": 788, "gamma": 426})}],
        }
        for name, calls in cases.items():
            with self.subTest(case=name):
                result = score(calls)
                self.assertFalse(result["correct"])
                self.assertEqual(result["score_label"], "fail")
                self.assertTrue(result["score_detail"].startswith("unparseable"))

    def test_incomplete_stream_is_truncated(self):
        self.assertEqual(score([], finish_reason="length")["score_label"], "truncated")
        partial = [{"name": TOOL, "arguments": '{"alpha": 478, "be'}]
        self.assertEqual(score(partial, finish_reason="length")["score_label"], "truncated")


class FakeStreamingClient:
    """OpenAI SSE fixture: one request streaming the given deltas."""

    def __init__(self, deltas, finish_reason="tool_calls"):
        self.deltas = deltas
        self.finish_reason = finish_reason
        self.requests = []

    @asynccontextmanager
    async def stream(self, method, url, *, json, timeout):
        self.requests.append(json)

        async def lines():
            for index, delta in enumerate(self.deltas):
                last = index == len(self.deltas) - 1
                yield "data: " + BENCH.json.dumps({
                    "choices": [{"delta": delta, "finish_reason": self.finish_reason if last else None}],
                    "usage": {"completion_tokens": index + 1, "prompt_tokens": 8192},
                })
            yield "data: [DONE]"

        yield SimpleNamespace(status_code=200, aiter_lines=lines)


class NeedleChecksumStreamTests(unittest.TestCase):
    def run_stream(self, deltas, finish_reason="tool_calls"):
        fake = FakeStreamingClient(deltas, finish_reason)
        return asyncio.run(BENCH.stream_completion_stats_request(
            fake, "http://fixture/v1/chat/completions", {"stream": True}, 1, "profile", 8,
            "", PROFILE["score_source"], False, profile_config=PROFILE,
        ))

    def test_streamed_tool_call_arguments_are_joined_and_scored(self):
        arguments = json.dumps({"alpha": 478, "beta": 788, "gamma": 426, "checksum": 3330})
        deltas = [
            {"reasoning_content": "478 + 1576 + 1278 = 3330"},
            {"tool_calls": [{"index": 0, "id": "call-1", "type": "function",
                             "function": {"name": TOOL, "arguments": ""}}]},
        ] + [
            {"tool_calls": [{"index": 0, "function": {"arguments": arguments[i:i + 7]}}]}
            for i in range(0, len(arguments), 7)
        ]
        run = self.run_stream(deltas)
        self.assertTrue(run.ok)
        self.assertEqual(run.score_label, "near_miss")
        self.assertEqual(run.parsed_answer, "alpha=478 beta=788 gamma=426 checksum=3330")
        self.assertIn(TOOL, run.final_answer)

    def test_reasoning_only_stream_has_no_answer(self):
        run = self.run_stream([{"reasoning_content": "thinking"}], finish_reason="length")
        self.assertEqual(run.score_label, "truncated")


class NeedleChecksumSummaryTests(unittest.TestCase):
    def _run(self, label, correct):
        return BENCH.CompletionStatsRun(
            run_index=1, phase="profile", concurrency=8, ok=True, correct=correct,
            completion_tokens=100, score_label=label,
        )

    def test_summary_counts_near_misses_as_wrong(self):
        runs = (
            [self._run("exact", True)] * 467
            + [self._run("near_miss", False)] * 31
            + [self._run("wrong_checksum", False)]
            + [self._run("wrong_facts", False)]
        )
        summary = BENCH.summarize_completion_stats_runs(runs)
        self.assertEqual(summary["correct"], 467)
        self.assertEqual(summary["near_miss"], 31)
        self.assertEqual(summary["wrong_checksum"], 1)
        self.assertEqual(summary["wrong_facts"], 1)
        self.assertAlmostEqual(summary["correct_rate"], 467 / 500)
        line = BENCH.format_completion_score_summary(summary)
        self.assertIn("EXACT 467", line)
        self.assertIn("NEAR_MISS 31", line)
        self.assertIn("WRONG_SUM 1", line)
        self.assertIn("WRONG_FACTS 1", line)
        bar = BENCH.completion_star_bar(summary)
        self.assertEqual(len(bar), 10)
        self.assertIn("✕", bar)


if __name__ == "__main__":
    unittest.main()
