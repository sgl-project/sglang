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
    "HCCL_BUFFSIZE": "570",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "100",
    "SGLANG_NPU_PROFILING": "0",
    "SGLANG_NPU_PROFILING_BS": "27",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "188416",
    "SGLANG_NPU_FUSED_MOE_MODE": "2",
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
    432,
    "--context-length",
    8192,
    "--dtype",
    "bfloat16",
    "--chunked-prefill-size",
    94208,
    "--max-prefill-tokens",
    458880,
    "--sampling-backend",
    "ascend",
    "--ep-dispatch-algorithm",
    "static",
    "--init-expert-location",
    "/root/.cache/modelscope/hub/models/hot_map/235B_3_5k_decode.pt",
    "--disable-radix-cache",
    "--moe-a2a-backend",
    "ascend_fuseep",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    QWEN3_235B_A22B_EAGLE_MODEL_PATH,
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
    "--speculative-draft-model-quantization",
    "unquant",
    "--tp",
    "16",
    "--dp-size",
    "16",
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--mem-fraction-static",
    "0.8",
    "--cuda-graph-bs",
    "1",
    "2",
    "4",
    "8",
    "16",
    "20",
    "24",
    "26",
    "27",
]


class TestQwen235B(TestAscendPerformanceTestCaseBase):
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = QWEN3_235B_W8A8_MODEL_PATH
    other_args = QWEN3_235B_OTHER_ARGS
    envs = QWEN3_235B_ENVS
    dataset_name = "random"
    max_concurrency = 432
    num_prompts = int(max_concurrency) * 4
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 50.1
    output_token_throughput = 6189

    def test_qwen3_235b(self):
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
