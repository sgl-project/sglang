import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kits.cache_hit_kit import (
    calculate_tpot,
    calculate_tpot_statistics,
    get_openai_chat_output_delta,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestCacheHitKitMetrics(CustomTestCase):
    def test_openai_chat_output_delta(self):
        self.assertEqual(get_openai_chat_output_delta({"content": "answer"}), "answer")
        self.assertEqual(
            get_openai_chat_output_delta({"reasoning_content": "think"}), "think"
        )
        self.assertEqual(get_openai_chat_output_delta({"reasoning": "think"}), "think")
        self.assertEqual(
            get_openai_chat_output_delta(
                {"reasoning_content": "think", "content": "answer"}
            ),
            "thinkanswer",
        )
        self.assertEqual(get_openai_chat_output_delta({"role": "assistant"}), "")
        self.assertEqual(get_openai_chat_output_delta(None), "")

    def test_calculate_tpot(self):
        self.assertAlmostEqual(calculate_tpot(1.1, 0.1, 101), 0.01)
        self.assertIsNone(calculate_tpot(1.1, 0.0, 101))
        self.assertIsNone(calculate_tpot(1.1, 0.1, 0))
        self.assertIsNone(calculate_tpot(1.1, 0.1, 1))
        self.assertIsNone(calculate_tpot(0.05, 0.1, 101))

    def test_calculate_tpot_statistics(self):
        stats = calculate_tpot_statistics([0.0024, 0.0025, 0.0026, 0.0035])

        self.assertAlmostEqual(stats["average_tpot"], 0.00275)
        self.assertAlmostEqual(stats["p90_tpot"], 0.00323)
        self.assertAlmostEqual(stats["p99_tpot"], 0.003473)
        self.assertAlmostEqual(stats["median_tpot"], 0.00255)
        self.assertAlmostEqual(stats["max_tpot"], 0.0035)

    def test_calculate_tpot_statistics_empty(self):
        self.assertEqual(
            calculate_tpot_statistics([]),
            {
                "average_tpot": 0.0,
                "p90_tpot": 0.0,
                "p99_tpot": 0.0,
                "median_tpot": 0.0,
                "max_tpot": 0.0,
            },
        )


if __name__ == "__main__":
    unittest.main()
