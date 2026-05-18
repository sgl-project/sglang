import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_8B_EAGLE_MODEL_PATH,
    QWEN3_8B_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-2-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

QWEN3_8B_ENVS = {
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "50",
}

QWEN3_8B_OTHER_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--max-running-requests",
    70,
    "--max-prefill-tokens",
    16384,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    16384,
    "--tp-size",
    1,
    "--mem-fraction-static",
    0.9,
    "--cuda-graph-bs",
    8,
    12,
    24,
    36,
    48,
    51,
    55,
    60,
    63,
    64,
    66,
    68,
    70,
    "--dtype",
    "bfloat16",
    "--speculative-draft-model-quantization",
    "unquant",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_8B_EAGLE_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
]


class TestQwen8B(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_8B_W8A8_MODEL_PATH
    other_args = QWEN3_8B_OTHER_ARGS
    envs = QWEN3_8B_ENVS
    dataset_name = "random"
    max_concurrency = 64
    num_prompts = 256
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 37
    output_token_throughput = 1586

    def test_qwen3_8b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
