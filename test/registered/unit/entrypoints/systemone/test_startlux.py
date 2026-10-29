"""Unit tests for the StartLux-Decision prompt format and read-out math.

The prompt literals are the trained format of startlux-models/StartLux-Decision
checkpoints, so they are pinned here: changing one changes every answer.
"""

import unittest

from sglang.srt.entrypoints.systemone import startlux
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestStartLuxPrompt(CustomTestCase):
    def test_the_prompt_is_the_trained_format(self):
        row = startlux.from_systemone(
            "The integration keeps failing.",
            {"type": "noul", "instructions": "Urgent?"},
        )
        self.assertEqual(
            startlux.messages(row),
            [
                {
                    "role": "system",
                    "content": "Apply the criterion to the evidence. Choose exactly one "
                    "listed option. Answer with its letter only.",
                },
                {
                    "role": "user",
                    "content": "Evidence:\nThe integration keeps failing.\n\n"
                    "Question: Urgent?\nOptions:\nA) yes\nB) no",
                },
            ],
        )

    def test_choice_options_show_their_descriptions(self):
        row = startlux.from_systemone(
            "s",
            {
                "type": "choice",
                "instructions": "Which team?",
                "criteria": {"billing": "Payments", "technical": None},
            },
        )
        self.assertEqual(
            startlux.option_lines(row), ["A) billing: Payments", "B) technical"]
        )

    def test_bare_option_ids_are_hidden_when_every_option_is_described(self):
        """A line such as "A) C: Paris" would make the answer letter ambiguous."""
        row = startlux.from_systemone(
            "s",
            {
                "type": "choice",
                "instructions": "Capital?",
                "criteria": {"C": "Paris", "D": "Rome"},
            },
        )
        self.assertEqual(startlux.option_lines(row), ["A) Paris", "B) Rome"])

    def test_score_levels_start_at_the_lowest(self):
        row = startlux.from_systemone(
            "s",
            {
                "type": "score",
                "instructions": "How severe?",
                "criteria": {"10": "worst", "2": "mild", "5": "mid"},
            },
        )
        self.assertEqual(
            startlux.option_lines(row),
            ["A) 0: mild", "B) 1: mid", "C) 2: worst"],
        )
        self.assertEqual([o["id"] for o in row["options"]], ["0", "1", "2"])

    def test_the_none_candidate_is_the_last_option(self):
        row = startlux.from_systemone(
            "s",
            {"type": "choice", "instructions": "q", "criteria": {"a": None, "b": None}},
            allow_none=True,
        )
        self.assertIs(row["none_option"], True)
        self.assertEqual(
            startlux.option_lines(row),
            ["A) a", "B) b", "C) None of the above"],
        )
        without = startlux.from_systemone(
            "s",
            {"type": "choice", "instructions": "q", "criteria": {"a": None, "b": None}},
        )
        self.assertEqual(startlux.option_lines(without), ["A) a", "B) b"])

    def test_a_none_option_of_the_same_name_is_refused(self):
        with self.assertRaisesRegex(ValueError, "None of the above"):
            startlux.from_systemone(
                "s",
                {
                    "type": "choice",
                    "instructions": "q",
                    "criteria": {"None of the above": "x", "b": None},
                },
                allow_none=True,
            )

    def test_long_arrays_carry_their_index(self):
        """The model refers to items of a long JSON state by position."""
        state = {"items": [{"v": i} for i in range(8)]}
        text = startlux.state_text(state)
        self.assertIn('"_index": 7', text)
        self.assertIn('"value"', startlux.state_text({"items": list(range(8))}))
        self.assertNotIn("_index", startlux.state_text({"items": [1, 2]}))
        # An empty state renders as the literal the format expects.
        self.assertEqual(startlux.state_text(None), "")
        row = startlux.from_systemone("", {"type": "noul", "instructions": "q"})
        self.assertIn("Evidence:\n(none)", startlux.messages(row)[1]["content"])

    def test_a_blank_noul_instruction_gets_the_default_question(self):
        row = startlux.from_systemone("s", {"type": "noul", "instructions": ""})
        self.assertEqual(row["instructions"], startlux.NOUL_INSTRUCTIONS)

    def test_one_letter_per_option_and_at_most_26(self):
        for count in (1, 27):
            with self.subTest(count=count):
                row = {
                    "type": "choice",
                    "state": "s",
                    "instructions": "q",
                    "options": [
                        {"id": f"o{i}", "criterion": None} for i in range(count)
                    ],
                }
                with self.assertRaisesRegex(ValueError, "2 to 26 options"):
                    startlux.validate_row(row)


class TestStartLuxReadOut(CustomTestCase):
    def test_temperature_divides_the_logits_before_the_softmax(self):
        logprobs = [0.0, -1.0, -2.0]
        cold = startlux.probabilities(logprobs, 0.5)
        warm = startlux.probabilities(logprobs, 2.0)
        self.assertGreater(cold[0], warm[0])
        self.assertAlmostEqual(sum(cold), 1.0)
        self.assertEqual(warm, startlux.probabilities(logprobs, 2.0))

    def test_the_none_candidate_is_split_out_of_the_listed_options(self):
        listed, none, abstained = startlux.split_none([0.1, 0.3, 0.6])
        self.assertAlmostEqual(none, 0.6)
        self.assertAlmostEqual(sum(listed), 1.0)
        self.assertAlmostEqual(listed[0], 0.25)
        self.assertAlmostEqual(listed[1], 0.75)
        self.assertTrue(abstained)
        # A tie goes to the listed option.
        _, _, tie = startlux.split_none([0.5, 0.5])
        self.assertFalse(tie)
        _, _, wins = startlux.split_none([0.4, 0.6])
        self.assertTrue(wins)

    def test_choice_confidence_is_the_margin_above_an_even_split(self):
        self.assertAlmostEqual(startlux.choice_confidence([0.25] * 4), 0.0)
        self.assertAlmostEqual(startlux.choice_confidence([1.0]), 1.0)
        self.assertAlmostEqual(startlux.choice_confidence([0.5, 0.5]), 0.0)
        self.assertAlmostEqual(startlux.choice_confidence([1.0, 0.0]), 1.0)

    def test_score_confidence_and_expected_level(self):
        self.assertAlmostEqual(startlux.score_confidence([1.0, 0.0, 0.0]), 1.0)
        self.assertAlmostEqual(sum(i * p for i, p in enumerate([0.0, 1.0, 0.0])), 1.0)

    def test_every_question_kind_has_a_temperature_key(self):
        """The checkpoint's temperature_by_type is keyed by its own type names."""
        for kind in ("choice", "score", "yes_no"):
            self.assertIn(startlux.TEMPERATURE_KEYS[kind], ("choice", "score", "noul"))


if __name__ == "__main__":
    unittest.main()
