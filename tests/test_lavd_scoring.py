import importlib.util
from pathlib import Path
import unittest


MODULE_PATH = Path(__file__).resolve().parents[1] / "llm_decode_bench.py"
SPEC = importlib.util.spec_from_file_location("llm_decode_bench", MODULE_PATH)
BENCH = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BENCH)


class LavdAnswerParsingTests(unittest.TestCase):
    def test_prefers_labelled_result_over_minute_conversion(self):
        self.assertEqual(
            BENCH.parse_lavd_numeric_pair(
                "Result: 72 tickets, 46 hours (2,760 minutes)."
            ),
            (72, 46.0),
        )

    def test_prefers_labelled_result_over_trailing_ticket_ids(self):
        self.assertEqual(
            BENCH.parse_lavd_numeric_pair(
                "Result: 72 tickets, 46 hours.\n"
                "Zero-duration entries: 9134465, 9134421."
            ),
            (72, 46.0),
        )

    def test_accepts_two_line_summary(self):
        self.assertEqual(
            BENCH.parse_lavd_numeric_pair(
                "Matching tickets (Effective Close Date 10/1-10/15): **72**\n"
                "Total time: 2,760 minutes = **46 hours**"
            ),
            (72, 46.0),
        )

    def test_accepts_label_first_summary_before_trailing_dates(self):
        self.assertEqual(
            BENCH.parse_lavd_numeric_pair(
                "Tickets: 72\nHours: 46\nChecked 10/1-10/15"
            ),
            (72, 46.0),
        )

    def test_keeps_terse_numeric_pair_compatibility(self):
        self.assertEqual(BENCH.parse_lavd_numeric_pair("72, 46"), (72, 46.0))

    def test_preserves_genuinely_wrong_answer(self):
        self.assertEqual(
            BENCH.parse_lavd_numeric_pair("Answer: 73 tickets, 50 hours."),
            (73, 50.0),
        )

    def test_unparseable_response_remains_unparseable(self):
        self.assertEqual(BENCH.parse_lavd_numeric_pair("I could not determine it."), (None, None))


if __name__ == "__main__":
    unittest.main()
