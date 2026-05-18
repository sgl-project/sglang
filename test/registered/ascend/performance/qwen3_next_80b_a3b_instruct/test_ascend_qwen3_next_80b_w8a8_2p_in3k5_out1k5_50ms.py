import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-4-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

QWEN3_NEXT_80B_A3B_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_ALGO": "level0:NA;level1:ring",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "20",
    "HCCL_BUFFSIZE": "2000",
}

QWEN3_NEXT_80B_A3B_OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    4,
    "--mem-fraction-static",
    0.685,
    "--max-running-requests",
    80,
    "--watchdog-timeout",
    9000,
    "--disable-radix-cache",
    "--cuda-graph-bs",
    80,
    "--max-prefill-tokens",
    28672,
    "--max-total-tokens",
    450560,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--chunked-prefill-size",
    -1,
]


class TestQwen3Next80BA3B(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_NEXT_80B_A3B_W8A8_MODEL_PATH
    other_args = QWEN3_NEXT_80B_A3B_OTHER_ARGS
    envs = QWEN3_NEXT_80B_A3B_ENVS
    dataset_name = "random"
    max_concurrency = 80
    num_prompts = 320
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 50
    output_token_throughput = 1410

    def test_qwen3_next_80b_a3b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
