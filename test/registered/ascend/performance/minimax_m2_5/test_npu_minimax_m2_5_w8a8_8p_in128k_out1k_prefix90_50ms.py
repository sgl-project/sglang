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

MINIMAX_M2_5_128K_PREFIX_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "TASK_QUEUE_ENABLE": "1",
    "ASCEND_USE_FIA": "1",
    "HCCL_BUFFSIZE": "1600",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "640",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "64",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "SGLANG_NPU_FUSED_MOE_MODE": "2",
    "SGLANG_NPU_DEEPEP_USE_FUSED_MOE_DECODE": "1",
    "SGLANG_NPU_FUSEEP_DECODE_ONLY": "1",
    "ENABLE_PROFILING": "0",
    "PROFILING_BS": "28",
    "PROFILING_STAGE": "decode",
    "PROFILING_step": "10",
    "PYTHONPATH": f"{MINIMAX_M2_5_EAGLE3_MODEL_PATH}:{os.environ.get('PYTHONPATH', '')}",
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "custom_eagle3",
}

MINIMAX_M2_5_128K_PREFIX_OTHER_ARGS = [
    "--tp-size",
    16,
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--prefill-delayer-max-delay-passes",
    100,
    "--enable-prefill-delayer",
    "--mem-fraction-static",
    0.65,
    "--max-running-requests",
    36,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    130000,
    "--cuda-graph-bs",
    8,
    16,
    24,
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
    "--trust-remote-code",
    "--tokenizer-worker-num",
    8,
]


class TestNPUMiniMaxM2_5_W8A8_8P_In128k_Out1k_Prefix90(
    TestAscendPerformanceTestCaseBase
):
    """Test NPU performance for MiniMax-M2.5-w8a8 8p single node prefix cache in128k out1k"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = MINIMAX_M2_5_W8A8_MODEL_PATH
    other_args = MINIMAX_M2_5_128K_PREFIX_OTHER_ARGS
    envs = MINIMAX_M2_5_128K_PREFIX_ENVS
    dataset_name = "random"
    max_concurrency = 8
    num_prompts = 32
    input_len = 131072
    output_len = 1024
    random_range_ratio = 1
    aisbench_repeat_rate = 0.9
    tpot = 50
    output_token_throughput = 366.35

    def test_npu_minimax_m2_5_w8a8_8p_in128k_out1k_prefix(self):
        """Run NPU performance test for MiniMax-M2.5-w8a8 in128k out1k prefix"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
