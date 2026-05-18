import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_480B_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-16-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

QWEN3_480B_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "1600",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_NPU_PROFILING": "1",
    "SGLANG_NPU_PROFILING_BS": "1",
}

QWEN3_480B_OTHER_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--skip-server-warmup",
    "--max-running-requests",
    "1",
    "--dtype",
    "bfloat16",
    "--chunked-prefill-size",
    "-1",
    "--max-prefill-tokens",
    "16384",
    "--disable-radix-cache",
    "--enable-dp-lm-head",
    "--quantization",
    "modelslim",
    "--tp",
    "16",
    "--mem-fraction-static",
    "0.78",
    "--cuda-graph-bs",
    "1",
]


class TestQwen480B(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_480B_W8A8_MODEL_PATH
    other_args = QWEN3_480B_OTHER_ARGS
    envs = QWEN3_480B_ENVS
    dataset_name = "random"
    max_concurrency = 1
    num_prompts = 1
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 20.81
    # T: 143@50ms.   800I: 1.1*T
    output_token_throughput = 47.54

    def test_qwen3_480b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
