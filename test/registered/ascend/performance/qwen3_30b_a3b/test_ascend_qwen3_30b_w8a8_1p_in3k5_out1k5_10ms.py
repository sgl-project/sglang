import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_30B_A3B_W8A8_VLLM_MODEL_PATH,
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
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "200",
    "HCCL_BUFFSIZE": "400",
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
    "--quantization",
    "modelslim",
    "--max-running-requests",
    162,
    "--disable-radix-cache",
    "--speculative-draft-model-quantization",
    "unquant",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    35000,
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_A3B_EAGLE_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--tp-size",
    2,
    "--mem-fraction-static",
    0.87,
    "--cuda-graph-bs",
    1,
    5,
    15,
    40,
    70,
    100,
    120,
    130,
    140,
    146,
    150,
    154,
    156,
    158,
    160,
    162,
    "--dtype",
    "bfloat16",
]


class TestQwen30B(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_30B_A3B_W8A8_VLLM_MODEL_PATH
    other_args = OTHER_ARGS
    envs = ENVS
    dataset_name = "random"
    # dataset_path = "/data/l30081563/GSM8K-in3500-bs3000_qwen3-30b.jsonl"
    max_concurrency = 1
    num_prompts = 1
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 10

    def test_qwen3_30b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
