import unittest

from sglang.test.ascend.test_ascend_utils import (
    QWEN3_32B_WEIGHTS_PATH,
    run_bench_serving,
)
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(
    est_time=400,
    suite="nightly-2-npu-a3",
    nightly=True,
    disabled="run failed",
)


class TestAscendL2CacheTTFT(CustomTestCase):
    """The test used the Qwen3-32B model, with L2 cache enabled, and TTFT improved by 40%.

    [Test Category] Parameter
    [Test Target] --enable-hierarchical-cache
    """

    def test_no_chunked_prefill_without_radix_cache(self):
        TTFTS = []
        model = QWEN3_32B_WEIGHTS_PATH
        common_args = [
            [
                "--trust-remote-code",
                "--tp-size",
                2,
                "--mem-fraction-static",
                0.8,
                "--max-running-requests",
                16,
                "--disable-radix-cache",
                "--chunked-prefill-size",
                "-1",
                "--disable-cuda-graph",
            ],
            [
                "--trust-remote-code",
                "--tp-size",
                2,
                "--mem-fraction-static",
                0.8,
                "--max-running-requests",
                16,
                "--chunked-prefill-size",
                "-1",
                "--disable-cuda-graph",
                "--enable-hierarchical-cache",
                "--hicache-ratio",
                5,
                "--hicache-write-policy",
                "write_back",
            ],
        ]
        for common_arg in common_args:
            other_args = common_arg + (
                [
                    "--attention-backend",
                    "ascend",
                ]
            )
            res = run_bench_serving(
                model=model,
                dataset_name="generated-shared-prefix",
                num_prompts=128,
                random_input_len=3584,
                random_output_len=1,
                request_rate=float("inf"),
                max_concurrency=16,
                gsp_num_groups=1,
                gsp_prompts_per_group=128,
                gsp_system_prompt_len=1792,
                gsp_question_len=1792,
                gsp_output_len=1,
                other_server_args=other_args,
            )
            TTFT = res["mean_ttft_ms"]
            TTFTS.append(TTFT)

        assert float(TTFTS[1]) <= 0.6 * float(TTFTS[0])


if __name__ == "__main__":
    unittest.main()
