import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.simple_eval_mixed_prefix_gsm8k import (
    INVALID,
    GSM8KEval,
    get_answer_value,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=6, suite="stage-a-test-cpu-intel")


class TestGSM8KAnswerExtraction(CustomTestCase):
    def test_explicit_answers_and_legacy_default(self):
        examples = [
            ("Distance: #### 45\nLet me check: 0.5 * 30 = 1", 45, 1),
            ("#### -1,234.50\nCheck item 9", -1234.5, 9),
            ("#### 12\nCorrection: #### 13", 13, 13),
            # The fixed rule must not choose whichever answer matches the label.
            ("#### 12\nCorrection: the answer is 13", 12, 13),
            ("The answer is 42", 42, 42),
            ("#### 1/2\nThe answer is 3", 3, 3),
            ("No numeric answer", INVALID, INVALID),
        ]
        for response, explicit, legacy in examples:
            with self.subTest(response=response):
                self.assertEqual(get_answer_value(response), legacy)
                self.assertEqual(
                    get_answer_value(response, prefer_explicit=True), explicit
                )

    def test_evaluator_uses_selected_answer_mode_and_preserves_response(self):
        responses = ["#### 45\nCheck: 1", "#### 12\nCorrection: 13"]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "examples.jsonl"
            path.write_text(
                "\n".join(
                    json.dumps({"question": f"Question {i}", "answer": f"#### {gold}"})
                    for i, gold in enumerate((45, 13))
                )
            )
            for mode in ("last_number", "last_explicit"):
                with self.subTest(mode=mode):
                    sampler = Mock(side_effect=responses)
                    sampler._pack_message.side_effect = lambda **kwargs: kwargs
                    result = GSM8KEval(
                        num_examples=2,
                        num_threads=1,
                        num_shots=0,
                        data_path=str(path),
                        answer_mode=mode,
                    )(sampler)
                    self.assertEqual(result.score, 0.5)
                    self.assertEqual(
                        [convo[-1]["content"] for convo in result.convos], responses
                    )
                    if mode == "last_explicit":
                        self.assertIn("Extracted Answer: 45", result.htmls[0])
                        self.assertIn("Extracted Answer: 12", result.htmls[1])
                    else:
                        self.assertIn("Extracted Answer: 1", result.htmls[0])

    def test_invalid_mode_fails_before_loading_data(self):
        with self.assertRaisesRegex(ValueError, "Unsupported GSM8K answer mode"):
            GSM8KEval(answer_mode="pick_correct_answer")


if __name__ == "__main__":
    unittest.main()
