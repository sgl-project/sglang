import os
import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    BENCHMARK_TOOL_DEFAULT,
    MINIMAX_M2_5_EAGLE3_MODEL_PATH,
    MINIMAX_M2_5_W8A8_MODEL_PATH,
    TestAscendPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="npu-performance",
    nightly=True,
)

MINIMAX_M2_5_W8A8_8P_IN128K_OUT1K_PREFIX90_20MS_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "TASK_QUEUE_ENABLE": "1",
    "ASCEND_USE_FIA": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_FUSED_MOE_MODE": "2",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "160000",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "DEEPEP_HCCL_BUFFSIZE": "1024",
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "custom_eagle3",
    "PYTHONPATH": f"{MINIMAX_M2_5_EAGLE3_MODEL_PATH}:{os.environ.get('PYTHONPATH', '')}",
}

MINIMAX_M2_5_W8A8_8P_IN128K_OUT1K_PREFIX90_20MS_OTHER_ARGS = [
    "--tp-size",
    16,
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--mem-fraction-static",
    0.65,
    "--max-running-requests",
    4,
    "--reasoning-parser",
    "minimax-append-think",
    "--tool-call-parser",
    "minimax-m2",
    "--enable-prefill-delayer",
    "--prefill-max-requests",
    4,
    "--chunked-prefill-size",
    160000,
    "--max-prefill-tokens",
    80000,
    "--cuda-graph-bs",
    2,
    4,
    6,
    8,
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
    "--tokenizer-worker-num",
    4,
    "--dtype",
    "bfloat16",
    "--reasoning-parser",
    "minimax-append-think",
    "--tool-call-parser",
    "minimax-m2",
]


class TestNPUMiniMaxM2_5W8A8_8P_In128k_Out1k_Prefix90_20ms(
    TestAscendPerformanceTestCaseBase
):
    """MiniMax-M2.5-w8a8 8p (16 die) 128k input 1k output with 90% prefix cache low latency 20ms performance test"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = MINIMAX_M2_5_W8A8_MODEL_PATH
    other_args = MINIMAX_M2_5_W8A8_8P_IN128K_OUT1K_PREFIX90_20MS_OTHER_ARGS
    envs = MINIMAX_M2_5_W8A8_8P_IN128K_OUT1K_PREFIX90_20MS_ENVS
    dataset_name = "generated-shared-prefix"
    max_concurrency = 4
    num_prompts = 16
    input_len = 131072
    output_len = 1024
    random_range_ratio = 1
    repeat_rate = 0.9
    seed = 1
    warmup_requests = 2
    tpot = 24.44
    output_token_throughput = 66.29
    request_rate = float("inf")

    def test_npu_minimax_m2_5_w8a8_8p_in128k_out1k_prefix90_20ms(self):
        """Run MiniMax-M2.5-w8a8 8p 128k/1k prefix90 low latency 20ms performance test"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
