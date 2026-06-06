import os
import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    MINIMAX_M2_5_EAGLE3_MODEL_PATH,
    MINIMAX_M2_5_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

MINIMAX_M2_5_LOW_LATENCY_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "TASK_QUEUE_ENABLE": "1",
    "HCCL_BUFFSIZE": "1500",
    "ASCEND_USE_FIA": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "SGLANG_NPU_FUSED_MOE_MODE": "2",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "224000",
    "PYTHONPATH": f"{MINIMAX_M2_5_EAGLE3_MODEL_PATH}:{os.environ.get('PYTHONPATH', '')}",
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "custom_eagle3",
}

MINIMAX_M2_5_LOW_LATENCY_OTHER_ARGS = [
    "--tp-size",
    16,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--mem-fraction-static",
    0.75,
    "--max-running-requests",
    256,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    8192,
    "--cuda-graph-bs",
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    "--moe-a2a-backend",
    "ascend_fuseep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--speculative-algorithm",
    "EAGLE3",
    "--speculative-draft-model-path",
    MINIMAX_M2_5_EAGLE3_MODEL_PATH,
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--speculative-draft-model-quantization",
    "unquant",
    "--dtype",
    "bfloat16",
    "--tokenizer-worker-num",
    2,
    "--prefill-delayer-max-delay-passes",
    10,
    "--enable-prefill-delayer",
]


class TestNPUMiniMaxM2_5_W8A8_8P_In3k5_Out1k5_LowLatency(
    TestAscendPerformanceTestCaseBase
):
    """Test NPU performance for MiniMax-M2.5-w8a8 8p single node low latency in3k5 out1k5"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = MINIMAX_M2_5_W8A8_MODEL_PATH
    other_args = MINIMAX_M2_5_LOW_LATENCY_OTHER_ARGS
    envs = MINIMAX_M2_5_LOW_LATENCY_ENVS
    dataset_name = "random"
    max_concurrency = 64
    num_prompts = 256
    input_len = 3500
    output_len = 1500
    random_range_ratio = 1
    tpot = 20
    output_token_throughput = 3369.59

    def test_npu_minimax_m2_5_w8a8_8p_in3k5_out1k5_low_latency(self):
        """Run NPU performance test for MiniMax-M2.5-w8a8 low latency"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
