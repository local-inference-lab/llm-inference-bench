"""Regression tests for the estonia (country) and hotel-lights (numeric) scorers.

The fixtures are real final answers taken from local benchmark logs, plus the
false-positive shapes reported against the pre-0.4.31 scorers:

* the estonia regex passed any last line containing "Estonia", including
  "concluded Latvia (… Mirel Instrument is in Estonia)" and a truncated
  reasoning trace that merely mentioned the word;
* the hotel-lights numeric scorer took the last number anywhere, so "Not 48"
  passed and "48 rooms out of 100" failed, and private reasoning was scored
  when the visible content was empty.
"""
import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "llm_decode_bench.py"
SPEC = importlib.util.spec_from_file_location("llm_decode_bench", MODULE_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCH)

ESTONIA = BENCH.BUILTIN_TEST_PROFILES["estonia"]
HOTEL = BENCH.BUILTIN_TEST_PROFILES["hotel-lights"]


def score_country(visible: str, finish_reason: str = "stop", reasoning: str = "") -> dict:
    """Score the way the stream reader does: visible content only."""
    return BENCH.score_completion_profile(
        profile=ESTONIA,
        final_answer=BENCH.extract_answer_line(visible),
        content_text=visible,
        output_text=reasoning + visible,
        regex="",
        source="final_answer",
        finish_reason=finish_reason,
    )


def score_number(visible: str, finish_reason: str = "stop", reasoning: str = "") -> dict:
    return BENCH.score_completion_profile(
        profile=HOTEL,
        final_answer=BENCH.extract_answer_line(visible),
        content_text=visible,
        output_text=reasoning + visible,
        regex="",
        source="final_answer",
        finish_reason=finish_reason,
    )


class AnswerLineExtractionTests(unittest.TestCase):
    def test_prefers_last_final_answer_line(self):
        text = "Working notes.\nFinal answer: Estonia\n\nThe chain runs MX-88 -> N-4 -> AR-12."
        self.assertEqual(BENCH.extract_answer_line(text), "Estonia")

    def test_strips_markdown_and_trailing_punctuation(self):
        self.assertEqual(BENCH.extract_answer_line("**Final answer:** *Estonia*."), "Estonia")
        self.assertEqual(BENCH.extract_answer_line("- **48**"), "48")
        self.assertEqual(BENCH.extract_answer_line("### Estonia."), "Estonia")

    def test_falls_back_to_last_non_empty_line(self):
        self.assertEqual(BENCH.extract_answer_line("first\n\nsecond\n\n"), "second")

    def test_negative_number_keeps_its_sign(self):
        self.assertEqual(BENCH.extract_answer_line("-5"), "-5")


class ThinkBlockSplittingTests(unittest.TestCase):
    def test_closed_think_block_is_removed(self):
        visible, unclosed = BENCH.split_visible_answer_text("<think>maybe 47, maybe 48</think>\n48")
        self.assertEqual(visible.strip(), "48")
        self.assertFalse(unclosed)

    def test_unclosed_think_block_leaves_nothing_visible(self):
        visible, unclosed = BENCH.split_visible_answer_text("<think>still reasoning about 48")
        self.assertEqual(visible, "")
        self.assertTrue(unclosed)

    def test_stray_close_tag_keeps_only_the_answer(self):
        visible, _ = BENCH.split_visible_answer_text("reasoning mentions Latvia</think>\nEstonia")
        self.assertEqual(visible.strip(), "Estonia")


class CountryScorerTests(unittest.TestCase):
    def test_bare_country_variants_pass(self):
        for text in ("Estonia", "Estonia.", "**Estonia**", "Final answer: Estonia", "Final answer: **Estonia**."):
            with self.subTest(text=text):
                result = score_country(text)
                self.assertTrue(result["correct"], text)
                self.assertEqual(result["score_label"], "pass")
                self.assertEqual(result["parsed_answer"], "Estonia")

    def test_contrast_mention_of_decoy_does_not_fail_a_correct_answer(self):
        text = (
            "The vendor registration correction note states that vendor account V-441 "
            "corresponds to Mirel Instrument, headquartered in **Estonia** (not Latvia, as an "
            "older sheet said)."
        )
        result = score_country(text)
        self.assertEqual(result["score_label"], "pass")
        self.assertEqual(result["parsed_answer"], "Estonia")

    def test_superseded_country_in_from_to_phrase_is_not_asserted(self):
        text = (
            "The manufacturer is headquartered in Estonia. This is based on the vendor "
            "registration correction that updates the country for Mirel Instrument from "
            "Latvia to Estonia."
        )
        self.assertEqual(score_country(text)["score_label"], "pass")
        self.assertEqual(score_country("Latvia was corrected to Estonia.")["parsed_answer"], "Estonia")
        self.assertEqual(score_country("Latvia -> Estonia")["parsed_answer"], "Estonia")

    def test_latvia_conclusion_with_estonia_aside_is_a_decoy_not_a_pass(self):
        # Pre-0.4.31 regex false positive: last line contains "Estonia".
        text = (
            "Mirel Industrial is headquartered in Latvia (the separate Mirel Instrument "
            "company is in Estonia)."
        )
        result = score_country(text)
        self.assertFalse(result["correct"])
        self.assertEqual(result["score_label"], "decoy")
        self.assertEqual(result["parsed_answer"], "Latvia")

    def test_plain_decoy_answers(self):
        for text in ("Latvia", "Latvia.", "Final answer: **Latvia**"):
            with self.subTest(text=text):
                result = score_country(text)
                self.assertEqual(result["score_label"], "decoy")
                self.assertFalse(result["correct"])

    def test_other_country_is_a_plain_fail_with_parsed_value(self):
        result = score_country("The answer is Finland.")
        self.assertEqual(result["score_label"], "fail")
        self.assertEqual(result["parsed_answer"], "Finland")

    def test_not_stated_answers_get_their_own_label(self):
        samples = (
            "**Not stated in the packet.** The document references the bench but does not "
            "identify a manufacturer.",
            "The information is missing or unknown.",
            "I don't have the document or report you're referring to. Could you share it?",
            "Final answer: The packet does not specify a headquarters country for Polt Metric.",
            "The text does not provide information about the manufacturer or country of "
            "headquarters for the material used by the Glass Current salinity bench.",
        )
        for text in samples:
            with self.subTest(text=text[:50]):
                result = score_country(text)
                self.assertFalse(result["correct"])
                self.assertEqual(result["score_label"], "not_stated")

    def test_two_asserted_countries_are_ambiguous_and_wrong(self):
        result = score_country("It is either Latvia or Estonia depending on which Mirel entity is meant.")
        self.assertEqual(result["score_label"], "ambiguous")
        self.assertFalse(result["correct"])

    def test_answer_followed_by_justification_without_countries(self):
        text = "Estonia\n\nThe chain is MX-88 -> N-4 -> AR-12 -> K-27 -> V-441 -> Mirel Instrument."
        self.assertEqual(score_country(text)["score_label"], "pass")

    def test_garbage_output_is_unparseable(self):
        result = score_country("</div>")
        self.assertEqual(result["score_label"], "fail")
        self.assertTrue(result["score_detail"].startswith("unparseable"))

    def test_truncated_reasoning_mentioning_estonia_is_truncated_not_pass(self):
        # Local log: mimo baseline run 2 hit max_completion_tokens inside reasoning;
        # the old scorer read the reasoning's last line and passed it.
        reasoning = (
            "The text doesn't actually identify the manufacturer of the calibration cassette "
            "or the bench itself. I'm not finding Estonia mentioned as the answer yet"
        )
        result = score_country("", finish_reason="length", reasoning=reasoning)
        self.assertFalse(result["correct"])
        self.assertEqual(result["score_label"], "truncated")

    def test_explicit_final_answer_survives_a_length_stop(self):
        result = score_country("Final answer: Estonia", finish_reason="length")
        self.assertEqual(result["score_label"], "pass")

    def test_unanchored_partial_line_on_length_stop_is_truncated(self):
        result = score_country("So far the evidence points to Estonia because", finish_reason="length")
        self.assertEqual(result["score_label"], "truncated")

    def test_watchdog_finish_reasons(self):
        self.assertEqual(score_country("", finish_reason="stalled")["score_label"], "stalled")
        self.assertEqual(score_country("", finish_reason="timeout")["score_label"], "timeout")
        cancelled = score_country("", finish_reason="client_cancelled")
        self.assertEqual(cancelled["score_label"], "cancelled")
        self.assertIsNone(cancelled["correct"])

    def test_empty_visible_answer_after_stop_is_a_fail(self):
        result = score_country("", finish_reason="stop")
        self.assertEqual(result["score_label"], "fail")
        self.assertFalse(result["correct"])

    def test_reasoning_is_never_scored_even_when_content_is_empty(self):
        result = score_country("", finish_reason="stop", reasoning="Final answer: Estonia")
        self.assertFalse(result["correct"])


class NumericScorerTests(unittest.TestCase):
    def test_real_hotel_final_answers_from_local_log(self):
        for text in ("48", "**48**", "**Blue lights: 49 − 1 = 48**", "48.", "The answer is 48."):
            with self.subTest(text=text):
                result = score_number(text)
                self.assertTrue(result["correct"], text)
                self.assertEqual(result["parsed_answer"], "48")

    def test_negated_number_does_not_pass(self):
        result = score_number("Not 48")
        self.assertFalse(result["correct"])
        self.assertTrue(result["score_detail"].startswith("unparseable"))
        self.assertFalse(score_number("47, not 48")["correct"])

    def test_hedged_line_is_not_an_answer(self):
        result = score_number("I considered 48 but cannot answer")
        self.assertFalse(result["correct"])
        self.assertTrue(result["score_detail"].startswith("unparseable"))

    def test_multiple_numbers_without_anchor_are_unparseable_not_wrong_number(self):
        # Old scorer parsed the trailing 100 and reported "expected 48, got 100".
        result = score_number("48 rooms out of 100")
        self.assertFalse(result["correct"])
        self.assertEqual(result["parsed_answer"], "")
        self.assertIn("multiple numbers", result["score_detail"])

    def test_anchored_number_wins_over_other_numbers_on_the_line(self):
        self.assertEqual(score_number("Rooms 1..100 checked, total blue = 48")["parsed_answer"], "48")
        self.assertTrue(score_number("Final answer: 48\n\n(52 of the 100 lights stay red.)")["correct"])

    def test_wrong_number_is_reported_as_wrong(self):
        result = score_number("52")
        self.assertFalse(result["correct"])
        self.assertEqual(result["score_detail"], "expected 48, got 52")

    def test_reasoning_only_output_is_not_scored(self):
        result = score_number("", finish_reason="stop", reasoning="so the count is 48")
        self.assertFalse(result["correct"])
        self.assertTrue(result["score_detail"].startswith("unparseable"))

    def test_truncated_reasoning_with_candidate_number_is_truncated(self):
        result = score_number("", finish_reason="length", reasoning="candidate 48, verifying room 96")
        self.assertEqual(result["score_label"], "truncated")
        self.assertFalse(result["correct"])

    def test_bare_number_line_survives_a_length_stop(self):
        self.assertTrue(score_number("48", finish_reason="length")["correct"])

    def test_unanchored_number_in_cut_off_prose_is_truncated(self):
        result = score_number("so there are 48 blue lights and the remaining", finish_reason="length")
        self.assertEqual(result["score_label"], "truncated")

    def test_inline_think_block_in_content_is_ignored(self):
        visible, _ = BENCH.split_visible_answer_text("<think>could be 47 or 52</think>\n48")
        self.assertTrue(score_number(visible)["correct"])


class ProfilePromptTests(unittest.TestCase):
    def test_estonia_v2_prompt_asks_for_a_final_answer_line(self):
        prompt, _source, profile = BENCH.decode_builtin_test_profile_prompt("estonia")
        self.assertTrue(prompt.endswith(BENCH.ESTONIA_V2_QUESTION_TAIL))
        self.assertIn("Final answer: <country>", prompt)
        self.assertEqual(profile["scorer"], "country_exact")
        self.assertEqual(profile["profile_version"], 2)

    def test_estonia_v1_prompt_is_the_legacy_tail(self):
        prompt, _source, profile = BENCH.decode_builtin_test_profile_prompt("estonia-v1")
        self.assertTrue(prompt.endswith(BENCH.ESTONIA_V1_QUESTION_TAIL))
        self.assertEqual(profile["profile_version"], 1)

    def test_v1_and_v2_share_the_packet(self):
        v1, _, _ = BENCH.decode_builtin_test_profile_prompt("estonia-v1")
        v2, _, _ = BENCH.decode_builtin_test_profile_prompt("estonia")
        self.assertEqual(
            v1[: -len(BENCH.ESTONIA_V1_QUESTION_TAIL)],
            v2[: -len(BENCH.ESTONIA_V2_QUESTION_TAIL)],
        )

    def test_estonia_long_wraps_the_v2_prompt(self):
        prompt, _source, _profile = BENCH.decode_builtin_test_profile_prompt("estonia-long")
        self.assertTrue(prompt.startswith(BENCH.ESTONIA_LONG_PROMPT_PREFIX.strip()))
        self.assertTrue(prompt.endswith(BENCH.ESTONIA_V2_QUESTION_TAIL))

    def test_consistency_profiles_leave_sampling_to_the_server(self):
        for name in ("estonia", "estonia-v1", "estonia-long", "hotel-lights"):
            with self.subTest(profile=name):
                profile = BENCH.BUILTIN_TEST_PROFILES[name]
                self.assertIsNone(profile.get("default_temperature"))
                self.assertIsNone(profile.get("default_top_p"))


class SummaryFormattingTests(unittest.TestCase):
    def _run(self, label, correct):
        return BENCH.CompletionStatsRun(
            run_index=1, phase="profile", concurrency=1, ok=True, correct=correct,
            completion_tokens=10, score_label=label,
            score_detail="unparseable: no number" if label == "fail" else "",
        )

    def test_summary_separates_incomplete_from_wrong(self):
        runs = (
            [self._run("pass", True)] * 19
            + [self._run("decoy", False)] * 4
            + [self._run("not_stated", False)] * 5
            + [self._run("truncated", False)] * 2
        )
        summary = BENCH.summarize_completion_stats_runs(runs)
        self.assertEqual(summary["correct"], 19)
        self.assertEqual(summary["decoy"], 4)
        self.assertEqual(summary["not_stated"], 5)
        self.assertEqual(summary["incomplete"], 2)
        self.assertAlmostEqual(summary["correct_rate"], 19 / 30)
        self.assertAlmostEqual(summary["correct_rate_completed"], 19 / 28)
        line = BENCH.format_completion_score_summary(summary)
        self.assertIn("PASS 19", line)
        self.assertIn("DECOY 4", line)
        self.assertIn("NOT_STATED 5", line)
        self.assertIn("TRUNC 2", line)
        bar = BENCH.completion_star_bar(summary)
        self.assertEqual(len(bar), 10)
        self.assertIn("⊘", bar)

    def test_cancelled_runs_do_not_count_as_wrong(self):
        runs = [self._run("pass", True), self._run("cancelled", None)]
        summary = BENCH.summarize_completion_stats_runs(runs)
        self.assertEqual(summary["wrong"], 0)
        self.assertAlmostEqual(summary["correct_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
