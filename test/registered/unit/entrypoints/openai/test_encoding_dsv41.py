"""Unit tests for the DeepSeek-V4.1 prompt encoder."""

import unittest

from sglang.srt.entrypoints.openai import encoding_dsv41
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestDeepSeekV41Encoding(CustomTestCase):
    def test_release_reasoning_effort_budgets(self):
        expected_budgets = {
            "low": 50,
            "high": 75,
            "xhigh": 75,
            "max": 100,
        }

        for effort, budget in expected_budgets.items():
            with self.subTest(effort=effort):
                self.assertEqual(
                    encoding_dsv41.render_reasoning_effort(
                        index=0,
                        thinking_mode="thinking",
                        effort=effort,
                    ),
                    encoding_dsv41.REASONING_EFFORT_TEMPLATE.format(budget=budget),
                )

        self.assertEqual(
            encoding_dsv41.render_reasoning_effort(
                index=0,
                thinking_mode="thinking",
                effort=None,
            ),
            encoding_dsv41.REASONING_EFFORT_TEMPLATE.format(budget=75),
        )


if __name__ == "__main__":
    unittest.main()
