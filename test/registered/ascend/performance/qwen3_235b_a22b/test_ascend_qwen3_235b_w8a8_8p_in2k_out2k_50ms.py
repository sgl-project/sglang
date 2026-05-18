import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    QWEN3_235B_A22B_EAGLE_MODEL_PATH,
    QWEN3_235B_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=1800,
    suite="nightly-16-npu-a3",
    nightly=True,
    disabled="Currently it is executed by the npu performance workflow.",
)

QWEN3_235B_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "HCCL_BUFFSIZE": "450",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "100",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "147456",
    "SGLANG_NPU_FUSED_MOE_MODE": "2",
    "SGLANG_NPU_PROFILING": "0",
    "SGLANG_NPU_PROFILING_BS": "39",
}

QWEN3_235B_OTHER_ARGS = [
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
    624,
    "--context-length",
    8192,
    "--dtype",
    "bfloat16",
    "--chunked-prefill-size",
    73728,
    "--max-prefill-tokens",
    458880,
    "--ep-dispatch-algorithm",
    "static",
    "--disable-radix-cache",
    "--moe-a2a-backend",
    "ascend_fuseep",
    "--ep-dispatch-algorithm",
    "static",
    "--init-expert-location",
    "/root/.cache/modelscope/hub/models/hot_map/235B_2k_decode.pt",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_235B_A22B_EAGLE_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
    "--tp",
    16,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--mem-fraction-static",
    0.83,
    "--cuda-graph-bs",
    4,
    8,
    16,
    24,
    28,
    29,
    30,
    32,
    34,
    36,
    37,
    38,
    39,
]


class TestQwen235B(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_235B_W8A8_MODEL_PATH
    other_args = QWEN3_235B_OTHER_ARGS
    envs = QWEN3_235B_ENVS
    dataset_name = "random"
    max_concurrency = 624
    num_prompts = 2496
    input_len = 2048
    output_len = 2048
    random_range_ratio = 1
    tpot = 47.56
    output_token_throughput = 9522

    def test_qwen3_235b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
