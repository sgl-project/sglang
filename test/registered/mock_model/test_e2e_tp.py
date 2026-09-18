from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.mock_model.utils import run_mock_model_bench_serving
from sglang.test.test_utils import CustomTestCase, is_in_amd_ci

register_cuda_ci(est_time=111, stage="extra-a", runner_config="2-gpu-large")
register_amd_ci(est_time=167, stage="extra-a", runner_config="2-gpu-large-amd")


class TestE2ETensorParallel(CustomTestCase):
    def test_tp_no_canary_violation(self) -> None:
        run_mock_model_bench_serving(
            extra_server_args=["--tp", "2", "--mem-fraction-static", "0.88"],
        )

    @unittest.skipIf(is_in_amd_ci(), "PyNccl CUDA graph smoke test requires CUDA.")
    def test_tp_lm_head_all_to_all_cuda_graph(self) -> None:
        """Smoke-test the PyNccl all-to-all path used during graph capture."""
        run_mock_model_bench_serving(
            extra_server_args=[
                "--tp",
                "2",
                "--dp",
                "2",
                "--enable-dp-attention",
                "--enable-tp-lm-head-all-to-all",
                "--cuda-graph-max-bs-decode",
                "4",
                "--mem-fraction-static",
                "0.88",
                "--attention-backend",
                "triton",
            ],
            num_prompts=2,
            random_input_len=32,
            random_output_len=2,
        )


if __name__ == "__main__":
    unittest.main()
