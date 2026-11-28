import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.simple_eval_mgsm import parse_answer, score_mgsm
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMGSMAnswerSign(CustomTestCase):
    def test_opposite_sign_is_not_scored_as_correct(self):
        for prefix in ("Answer", "Respuesta", "答案"):
            with self.subTest(prefix=prefix):
                prediction = parse_answer(f"{prefix}: -42", prefix)
                self.assertEqual(prediction, "-42")
                self.assertFalse(score_mgsm("42", prediction))
                self.assertTrue(score_mgsm("-42", prediction))

    def test_signed_decimal_and_comma_formatting(self):
        self.assertEqual(parse_answer("Answer: -1,234.00", "Answer"), "-1234.00")
        self.assertTrue(
            score_mgsm("-1234", parse_answer("Answer: -1,234.00", "Answer"))
        )
        self.assertTrue(score_mgsm("42", parse_answer("Answer: 42", "Answer")))
        self.assertEqual(parse_answer("No answer given", "Answer"), "")


if __name__ == "__main__":
    unittest.main()
