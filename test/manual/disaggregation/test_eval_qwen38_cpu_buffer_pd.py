"""Offline acceptance checks for the external-endpoint evaluator."""

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from eval_qwen38_cpu_buffer_pd import evaluate_gsm8k


class TestGSM8KAcceptance(unittest.TestCase):
    def run_result(self, result, passes):
        evaluator = Mock(return_value=result)
        with tempfile.TemporaryDirectory() as directory:
            args = SimpleNamespace(
                url="http://localhost:30000", model="model", output_dir=Path(directory)
            )
            with patch.dict(
                "sys.modules",
                {"sglang.test.sgl_eval_utils": SimpleNamespace(run_sgl_eval=evaluator)},
            ):
                if passes:
                    self.assertEqual(evaluate_gsm8k(args), result)
                else:
                    with self.assertRaises(AssertionError):
                        evaluate_gsm8k(args)
            # Preserve failed reports too, rather than dropping their evidence.
            self.assertEqual(
                json.loads((args.output_dir / "gsm8k.json").read_text()), result
            )
            recipe = evaluator.call_args.args[0]
            self.assertEqual(recipe.num_examples, 200)
            self.assertEqual(recipe.max_tokens, 16384)
            self.assertEqual(recipe.chat_template_kwargs, {"thinking": True})

    def test_threshold_and_truncated_responses(self):
        self.run_result({"score": 0.94, "error_rate": 0, "truncated_rate": 0.005}, True)

    def test_low_accuracy_fails(self):
        self.run_result({"score": 0.935, "error_rate": 0}, False)

    def test_request_errors_fail_even_with_high_accuracy(self):
        self.run_result({"score": 0.965, "error_rate": 0.005}, False)


if __name__ == "__main__":
    unittest.main()
