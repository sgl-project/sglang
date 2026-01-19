"""
Performance tests for single GPU that need H200 (80GB) - FP8 and EAGLE tests.
"""

import unittest

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE,
    DEFAULT_MODEL_NAME_FOR_TEST_FP8,
    DEFAULT_TARGET_MODEL_EAGLE,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    run_bench_serving,
    write_github_step_summary,
)

register_cuda_ci(est_time=300, suite="stage-b-test-large-1-gpu-performance")


class TestBenchServing1GPULarge(CustomTestCase):
    def test_offline_throughput_default_fp8(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST_FP8,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[],
        )

        if is_in_ci():
            write_github_step_summary(
                f"### test_offline_throughput_default_fp8\n"
                f"Output throughput: {res['output_throughput']:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(res["output_throughput"], 3500)
            else:
                self.assertGreater(res["output_throughput"], 4300)

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

        if is_in_ci():
            write_github_step_summary(
                f"### test_online_latency_eagle\n"
                f"median_e2e_latency_ms: {res['median_e2e_latency_ms']:.2f} ms\n"
                f"accept_length: {res['accept_length']:.2f} \n"
            )
            if is_in_amd_ci():
                self.assertLess(res["median_e2e_latency_ms"], 1800)
            else:
                self.assertLess(res["median_e2e_latency_ms"], 900)
            self.assertGreater(res["accept_length"], 3.0)


if __name__ == "__main__":
    unittest.main()
