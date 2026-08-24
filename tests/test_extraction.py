"""Unit tests for harness.extract (extract_v1) and the seed policy.

Run from instructGPT/:
    python -m unittest discover -s tests -t . -v
"""

import unittest

from harness.common import sample_seed
from harness.extract import EXTRACTION_VERSION, answers_match, extract_answer


def num(text, scope="full"):
    return extract_answer(text, "number", scope=scope)


class TestNumbers(unittest.TestCase):
    def test_bare_number(self):
        r = num("10")
        self.assertEqual(r["status"], "parsed")
        self.assertEqual(r["prediction"], 10)

    def test_answer_colon_trailing_period(self):
        r = num("Answer: 10.")
        self.assertEqual(r["status"], "parsed")
        self.assertEqual(r["prediction"], 10)

    def test_answer_is_dollars(self):
        r = num("The answer is $10.")
        self.assertEqual(r["status"], "parsed")
        self.assertEqual(r["prediction"], 10)
        self.assertEqual(r["strategy"], "answer_is")

    def test_boxed(self):
        r = num(r"The result is \boxed{24}.")
        self.assertEqual(r["status"], "parsed")
        self.assertEqual(r["prediction"], 24)
        self.assertEqual(r["strategy"], "boxed")

    def test_boxed_with_latex_unit(self):
        r = num(r"\boxed{24 \text{ cm}^2}")
        self.assertEqual(r["status"], "parsed")
        self.assertEqual(r["prediction"], 24)

    def test_bold(self):
        r = num("So Maria pays **$24** in total.")
        self.assertEqual(r["status"], "parsed")
        self.assertEqual(r["prediction"], 24)
        self.assertEqual(r["strategy"], "bold")

    def test_unit_captured_not_in_value(self):
        r = num("The area is 24 cm^2")
        self.assertEqual(r["status"], "parsed")
        self.assertEqual(r["prediction"], 24)
        self.assertEqual(r["unit"], "cm^2")

    def test_thousands_separator_and_word_unit(self):
        r = num("There are 1,234 apples")
        self.assertEqual(r["status"], "parsed")
        self.assertEqual(r["prediction"], 1234)
        self.assertEqual(r["unit"], "apples")

    def test_final_answer_negative(self):
        r = num("Final answer: -3.")
        self.assertEqual(r["status"], "parsed")
        self.assertEqual(r["prediction"], -3)

    def test_fraction(self):
        r = num("The answer is 3/4")
        self.assertEqual(r["status"], "parsed")
        self.assertAlmostEqual(r["prediction"], 0.75)

    def test_equals(self):
        r = num("x + 7 - 7 = 15 - 7\nx = 8")
        self.assertEqual(r["status"], "parsed")
        self.assertEqual(r["prediction"], 8)

    def test_first_paragraph_scope(self):
        text = "5\n\nSome other question\nAnswer: 99"
        self.assertEqual(num(text, scope="first_paragraph")["prediction"], 5)
        self.assertEqual(num(text, scope="full")["prediction"], 99)

    def test_multiple_distinct_values_flagged(self):
        r = num("The answer is 5. Wait, the answer is 6.")
        self.assertEqual(r["status"], "parsed")
        self.assertEqual(r["prediction"], 6)
        self.assertIn("multiple_distinct_values", r["flags"])

    def test_empty_output_unparsed(self):
        r = num("")
        self.assertEqual(r["status"], "unparsed")
        self.assertIn("empty_output", r["flags"])

    def test_no_number_unparsed(self):
        r = num("I cannot determine the answer.")
        self.assertEqual(r["status"], "unparsed")
        self.assertIn("no_numeric_candidate", r["flags"])
        self.assertIsNone(r["prediction"])

    def test_empty_boxed_malformed(self):
        r = num(r"The answer is \boxed{}")
        self.assertEqual(r["status"], "malformed")
        self.assertIn("boxed_malformed", r["flags"])

    def test_unbalanced_boxed_malformed(self):
        r = num(r"The answer is \boxed{24")
        self.assertEqual(r["status"], "malformed")

    def test_two_different_boxed_ambiguous(self):
        r = num(r"\boxed{5} or maybe \boxed{6}")
        self.assertEqual(r["status"], "ambiguous")
        self.assertEqual(r["candidates"], [5, 6])
        self.assertIsNone(r["prediction"])


class TestOtherTypes(unittest.TestCase):
    def test_yesno(self):
        self.assertEqual(extract_answer("Yes.", "yesno")["prediction"], "yes")
        self.assertEqual(extract_answer("No, it is not.", "yesno")["prediction"], "no")

    def test_yesno_unparsed(self):
        r = extract_answer("Whiskers is an animal.", "yesno")
        self.assertEqual(r["status"], "unparsed")

    def test_label_longer_match_wins(self):
        labels = ["not spam", "spam"]
        self.assertEqual(
            extract_answer("This is not spam at all", "label", labels=labels)["prediction"],
            "not spam")
        self.assertEqual(
            extract_answer("This is spam", "label", labels=labels)["prediction"],
            "spam")

    def test_name_containment(self):
        r = extract_answer(
            "Alice is taller than Bob. Bob is taller than Carol. "
            "So the shortest is Carol.", "name", gold="Carol")
        self.assertEqual(r["status"], "parsed")
        self.assertEqual(r["prediction"], "carol")

    def test_string_date_containment(self):
        r = extract_answer("Answer: June 14, 2026.", "string", gold="June 14, 2026")
        self.assertEqual(r["status"], "parsed")

    def test_string_gold_absent_unparsed(self):
        r = extract_answer("The meeting is next Friday.", "string", gold="June 14, 2026")
        self.assertEqual(r["status"], "unparsed")


class TestMatching(unittest.TestCase):
    def test_numeric_tolerance_and_fraction_gold(self):
        r = num("The answer is 0.5")
        self.assertTrue(answers_match(r, "1/2", "number"))

    def test_unparsed_never_correct(self):
        r = num("I don't know")
        self.assertFalse(answers_match(r, "10", "number"))

    def test_version_pinned(self):
        self.assertEqual(EXTRACTION_VERSION, "extract_v1")
        self.assertEqual(num("1")["extraction_version"], "extract_v1")


class TestSeedPolicy(unittest.TestCase):
    def test_deterministic(self):
        a = sample_seed(20260823, "benchmark_v1", "m", "direct", 1, 0)
        b = sample_seed(20260823, "benchmark_v1", "m", "direct", 1, 0)
        self.assertEqual(a, b)

    def test_varies_with_each_component(self):
        base = sample_seed(20260823, "benchmark_v1", "m", "direct", 1, 0)
        self.assertNotEqual(base, sample_seed(20260823, "benchmark_v1", "m", "direct", 1, 1))
        self.assertNotEqual(base, sample_seed(20260823, "benchmark_v1", "m", "direct", 2, 0))
        self.assertNotEqual(base, sample_seed(20260823, "benchmark_v1", "m2", "direct", 1, 0))
        self.assertNotEqual(base, sample_seed(20260823, "benchmark_v1", "m", "cot", 1, 0))

    def test_fits_in_32_bits(self):
        self.assertLess(sample_seed(1, "v", "m", "c", 0, 0), 2 ** 32)


if __name__ == "__main__":
    unittest.main()
