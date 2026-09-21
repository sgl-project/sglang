"""Latency and accept length of EAGLE speculative decoding.

Registered for CUDA only: the one test here is skipped on ROCm.
"""

import unittest

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.perf_bench_kit import at_least, at_most, check_perf
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_TARGET_MODEL_EAGLE,
    CustomTestCase,
    run_bench_serving,
)

register_cuda_ci(est_time=145, stage="extra-a", runner_config="1-gpu-large")


class TestEagleLatency(CustomTestCase):
    @unittest.skipIf(is_hip(), "Skip Eagle test for ROCm")
    def test_online_latency_eagle(self):
        res = run_bench_serving(
            model=DEFAULT_TARGET_MODEL_EAGLE,
            num_prompts=300,
            request_rate=8,
            sharegpt_context_len=3072,
            disable_ignore_eos=True,
            dataset_name="sharegpt",
            other_server_args=[
                "--speculative-algorithm",
                "EAGLE",
                "--speculative-draft-model-path",
                DEFAULT_DRAFT_MODEL_EAGLE,
                "--speculative-num-steps",
                "5",
                "--speculative-eagle-topk",
                "4",
                "--speculative-num-draft-tokens",
                "16",
                "--mem-fraction-static",
                "0.7",
            ],
            need_warmup=True,
            seed=42,
        )

        check_perf(
            self,
            "test_online_latency_eagle",
            # No AMD bound: `skipIf(is_hip())` means this never runs on ROCm.
            at_most(
                "median_e2e_latency_ms", res["median_e2e_latency_ms"], 900, unit="ms"
            ),
            at_least("accept_length", res["accept_length"], 3.0),
        )


if __name__ == "__main__":
    unittest.main()
