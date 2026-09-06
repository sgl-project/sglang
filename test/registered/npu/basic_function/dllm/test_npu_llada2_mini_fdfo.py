import unittest

from sglang.test.ascend.test_ascend_utils import (
    LLaDA2_0_MINI_WEIGHTS_PATH,
    run_bench_serving,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="nightly-2-npu-a3", nightly=True)


class TestLLaDA2MiniFDFO(CustomTestCase):
    """The test used the LLaDA2.0-mini model, with --dllm-fdfo, and throughputs improved.

    [Test Category] Diffusion LLM
    [Test Target] --dllm-fdfo
    """

    def test_dlla_fdfo_vs_no_fdfo_performance(self):
        TTFTS = []
        throughputs = []
        p99_ttfts = []
        model = LLaDA2_0_MINI_WEIGHTS_PATH
        common_args = [
            [
                "--trust-remote-code",
                "--tp-size",
                2,
                "--mem-fraction-static",
                0.9,
                "--disable-radix-cache",
                "--max-running-requests",
                16,
                "--dllm-algorithm",
                "LowConfidence",
            ],
            [
                "--trust-remote-code",
                "--tp-size",
                2,
                "--disable-radix-cache",
                "--mem-fraction-static",
                0.9,
                "--max-running-requests",
                16,
                "--dllm-algorithm",
                "LowConfidence",
                "--no-dllm-fdfo",
            ],
        ]
        for common_arg in common_args:
            other_args = common_arg + (
                ["--attention-backend", "ascend", "--disable-cuda-graph"]
            )
            res = run_bench_serving(
                model=model,
                num_prompts=128,
                random_input_len=3584,
                random_output_len=1024,
                request_rate=float("inf"),
                max_concurrency=16,
                other_server_args=other_args,
            )
            TTFT = res["mean_ttft_ms"]
            TTFTS.append(TTFT)
            throughput = res["total_throughput"]
            throughputs.append(throughput)
            p99_ttft = res["p99_ttft_ms"]
            p99_ttfts.append(p99_ttft)

        # FDFO (First-Done-First-Out) scheduling improves performance by prioritizing
        # requests that complete their diffusion steps earlier, reducing waiting time.
        # With FDFO enabled:
        # - TTFT is lower because earlier-completing requests get priority and start generating tokens sooner
        # - Throughput is higher due to better hardware utilization from reduced blocking
        # - P99 TTFT is lower because FDFO prevents a single slow request from blocking others
        assert float(TTFTS[0]) < float(TTFTS[1])
        assert float(throughputs[0]) > float(throughputs[1])
        assert float(p99_ttfts[0]) < float(p99_ttfts[1])


if __name__ == "__main__":
    unittest.main()
