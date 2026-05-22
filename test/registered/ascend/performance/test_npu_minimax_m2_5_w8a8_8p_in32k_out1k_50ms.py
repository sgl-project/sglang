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

MINIMAX_M2_5_32K_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "TASK_QUEUE_ENABLE": "1",
    "HCCL_BUFFSIZE": "1500",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "8",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "4096",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "384",
    "PYTHONPATH": f"{MINIMAX_M2_5_EAGLE3_MODEL_PATH}:{os.environ.get('PYTHONPATH', '')}",
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "custom_eagle3",
}

MINIMAX_M2_5_32K_OTHER_ARGS = [
    "--tp-size",
    16,
    "--enable-dp-attention",
    "--dp-size",
    8,
    "--ep-size",
    16,
    "--mem-fraction-static",
    0.6,
    "--prefill-delayer-max-delay-passes",
    200,
    "--enable-prefill-delayer",
    "--max-running-requests",
    48,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    32768,
    "--cuda-graph-max-bs",
    12,
    "--moe-a2a-backend",
    "deepep",
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
]


class TestNPUMiniMaxM2_5_W8A8_8P_In32k_Out1k_HighThroughput(
    TestAscendPerformanceTestCaseBase
):
    """Test NPU performance for MiniMax-M2.5-w8a8 8p single node high throughput in32k out1k"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = MINIMAX_M2_5_W8A8_MODEL_PATH
    other_args = MINIMAX_M2_5_32K_OTHER_ARGS
    envs = MINIMAX_M2_5_32K_ENVS
    dataset_name = "random"
    max_concurrency = 24
    num_prompts = 96
    input_len = 32768
    output_len = 1024
    random_range_ratio = 1
    tpot = 50
    output_token_throughput = 200

    def test_npu_minimax_m2_5_w8a8_8p_in32k_out1k_high_throughput(self):
        """Run NPU performance test for MiniMax-M2.5-w8a8 in32k out1k"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
