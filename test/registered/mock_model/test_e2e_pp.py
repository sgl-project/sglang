from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.mock_model.utils import run_mock_model_bench_serving
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, stage="extra-a", runner_config="2-gpu-large")
register_amd_ci(est_time=67, stage="extra-a", runner_config="2-gpu-large-amd")


class TestE2EPipelineParallel(CustomTestCase):
    def test_pp_no_canary_violation(self) -> None:
        run_mock_model_bench_serving(
            extra_server_args=["--pp-size", "2"],
        )


if __name__ == "__main__":
    unittest.main()
