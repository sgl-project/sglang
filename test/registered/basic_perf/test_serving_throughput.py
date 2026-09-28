"""Offline throughput of the default serving path on one large GPU."""

import unittest

from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.perf_bench_kit import at_least, check_perf
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST,
    DEFAULT_MODEL_NAME_FOR_TEST_FP8,
    CustomTestCase,
    run_bench_serving,
)

register_cuda_ci(est_time=710, stage="extra-a", runner_config="1-gpu-large")
register_amd_ci(est_time=810, suite="stage-b-test-1-gpu-large-amd")


class TestServingThroughput(CustomTestCase):
    def test_offline_throughput_default(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[],
        )

        check_perf(
            self,
            at_least(
                "output_throughput",
                res["output_throughput"],
                4000,
                amd=3050,
                unit="token/s",
            ),
        )

    def test_offline_throughput_non_stream_small_batch_size(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=200,
            request_rate=float("inf"),
            other_server_args=["--max-running-requests", "10"],
            dataset_name="sharegpt",
            random_input_len=None,
            random_output_len=None,
            disable_stream=True,
            need_warmup=True,
        )

        check_perf(
            self,
            at_least(
                "output_throughput",
                res["output_throughput"],
                1100,
                amd=1000,
                unit="token/s",
            ),
        )

    def test_offline_throughput_with_triton_attention_backend(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[
                "--attention-backend",
                "triton",
                "--context-length",
                "8192",
            ],
        )

        check_perf(
            self,
            at_least(
                "output_throughput",
                res["output_throughput"],
                3730,
                amd=2700,
                unit="token/s",
            ),
        )

    def test_offline_throughput_default_fp8(self):
        res = run_bench_serving(
            model=DEFAULT_MODEL_NAME_FOR_TEST_FP8,
            num_prompts=500,
            request_rate=float("inf"),
            other_server_args=[],
        )

        check_perf(
            self,
            at_least(
                "output_throughput",
                res["output_throughput"],
                4860,
                amd=3500,
                unit="token/s",
            ),
        )


if __name__ == "__main__":
    unittest.main()
