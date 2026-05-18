import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_30B_A3B_MODEL_PATH,
    QWEN3_A3B_EAGLE_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-2-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

ENVS = {
    "ASCEND_LAUNCH_BLOCKING": "0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "HCCL_ALGO": "level0:NA;level1:ring",
    "DP_ROUND_ROBIN": "1",
    "SGLANG_USE_MAX_DP_ATT": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ALLOW_OVERWRITE_LONGER_CONTEXT_LEN": "1",
}

OTHER_ARGS = [
    "--trust-remote-code",
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--max-running-requests",
    168,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    8300,
    "--speculative-draft-model-quantization",
    "unquant",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_A3B_EAGLE_MODEL_PATH,
    "--speculative-num-steps",
    7,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    8,
    "--tp-size",
    2,
    "--enable-dp-attention",
    "--dp-size",
    2,
    "--mem-fraction-static",
    0.85,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    16,
    20,
    24,
    28,
    32,
    36,
    40,
    44,
    48,
    52,
    56,
    60,
    64,
    68,
    72,
    76,
    80,
    84,
    "--dtype",
    "bfloat16",
]


class TestQwen32B(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_30B_A3B_MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    dataset_name = "random"
    max_concurrency = 162
    num_prompts = 624
    input_len = 1000
    output_len = 100
    random_range_ratio = 1
    mean_e2e_latency = 10000
    output_token_throughput = 2000
    max_attempts = 4

    def test_qwen3_32b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
