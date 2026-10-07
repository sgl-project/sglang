from __future__ import annotations

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.mock_model.utils import run_mock_model_bench_serving
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=50, stage="extra-a", runner_config="2-gpu-large")


class TestE2EContextParallel(CustomTestCase):
    def test_cp_prefill_then_decode_no_canary_violation(self) -> None:
        # CP prefill enters the transformer body directly, while decode calls
        # the outer model.forward. Both need exactly one canary bracket.
        run_mock_model_bench_serving(
            extra_server_args=[
                "--tp",
                "2",
                "--attn-cp-size",
                "2",
                "--enable-prefill-cp",
                "--cp-strategy",
                "zigzag",
                "--attention-backend",
                "fa3",
                "--kv-canary-real-data",
                "all",
                "--mem-fraction-static",
                "0.2",
                "--max-total-tokens",
                "4096",
                "--max-running-requests",
                "8",
                "--context-length",
                "256",
                "--cuda-graph-max-bs-decode",
                "4",
            ],
            num_prompts=4,
            random_input_len=32,
            random_output_len=8,
        )


if __name__ == "__main__":
    unittest.main()
