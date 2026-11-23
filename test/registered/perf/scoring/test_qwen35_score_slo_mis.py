"""Bounded H100 pointwise and setwise scoring checks with MIS enabled."""

import unittest
from pathlib import Path

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.qwen35_score_utils import check_qwen35_score_perf
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=360, stage="extra-a", runner_config="1-gpu-large")


class TestQwen35ScoreSLOMIS(CustomTestCase):
    def test_pointwise_and_setwise_mis(self):
        check_qwen35_score_perf(
            self,
            enable_mis=True,
            output_dir=Path(__file__).resolve().parents[3] / "score_slo_artifacts",
        )


if __name__ == "__main__":
    unittest.main()
