from __future__ import annotations

import unittest

from utils import run_mock_model_bench_serving

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=600, suite="extra-a-test-1-gpu-large")


class TestE2ETensorParallel(CustomTestCase):
    def test_tp_no_canary_violation(self) -> None:
        bench_result = run_mock_model_bench_serving(
            extra_server_args=["--tp", "2"],
        )

        self.assertEqual(bench_result.result["completed"], 8)
        self.assertIsNone(bench_result.server_return_code, bench_result.log_tail())
        self.assertNotIn("kv_canary violation:", bench_result.log_text)


if __name__ == "__main__":
    unittest.main()
