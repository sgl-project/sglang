from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.mock_model_utils import run_mock_model_bench_serving
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, suite="extra-a-test-1-gpu-large")


class TestE2ETensorParallel(CustomTestCase):
    def test_tp_no_canary_violation(self) -> None:
        run_mock_model_bench_serving(
            extra_server_args=["--tp", "2"],
        )


if __name__ == "__main__":
    unittest.main()
