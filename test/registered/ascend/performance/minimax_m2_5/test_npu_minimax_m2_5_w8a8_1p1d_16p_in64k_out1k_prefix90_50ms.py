import os
import unittest

from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    MINIMAX_M2_5_EAGLE3_MODEL_PATH,
    MINIMAX_M2_5_W8A8_MODEL_PATH,
    TestAscendPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="npu-performance",
    nightly=True,
)

PREFILL_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "ASCEND_USE_FIA": "1",
    "HCCL_BUFFSIZE": "2500",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "TASK_QUEUE_ENABLE": "2",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "64",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "custom_eagle3",
    "PYTHONPATH": f"{MINIMAX_M2_5_EAGLE3_MODEL_PATH}:{os.environ.get('PYTHONPATH', '')}",
    "ENABLE_PROFILING": "0",
    "PROFILING_BS": "8",
    "PROFILING_STAGE": "prefill",
    "PROFILING_step": "30",
}

DECODE_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "HCCL_BUFFSIZE": "1600",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "640",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_NPU_FUSED_MOE_MODE": "2",
    "SGLANG_DISAGGREGATION_NUM_PRE_ALLOCATE_REQS": "96",
    "SGLANG_EXTERNAL_MODEL_PACKAGE": "custom_eagle3",
    "PYTHONPATH": f"{MINIMAX_M2_5_EAGLE3_MODEL_PATH}:{os.environ.get('PYTHONPATH', '')}",
}

PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--trust-remote-code",
    "--tp-size",
    16,
    "--mem-fraction-static",
    0.43,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--disaggregation-transfer-backend",
    "ascend",
    "--max-running-requests",
    128,
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    58000,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "normal",
    "--tokenizer-worker-num",
    16,
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--dtype",
    "bfloat16",
    "--load-balance-method",
    "round_robin",
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
    "--skip-server-warmup",
]

DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--trust-remote-code",
    "--tp-size",
    16,
    "--mem-fraction-static",
    0.76,
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--quantization",
    "modelslim",
    "--disaggregation-transfer-backend",
    "ascend",
    "--max-running-requests",
    80,
    "--chunked-prefill-size",
    -1,
    "--moe-a2a-backend",
    "ascend_fuseep",
    "--deepep-mode",
    "low_latency",
    "--tokenizer-worker-num",
    16,
    "--dp-size",
    2,
    "--enable-dp-attention",
    "--dtype",
    "bfloat16",
    "--load-balance-method",
    "round_robin",
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
    "--skip-server-warmup",
    "--cuda-graph-bs",
    2,
    4,
    8,
    16,
    24,
    32,
    40,
]

ROUTER_ARGS = [
    "--policy",
    "round_robin",
    "--mini-lb",
]

ROUTER_ENVS = {}

MODEL_CONFIG = {
    "model_path": MINIMAX_M2_5_W8A8_MODEL_PATH,
    "prefill_args": PREFILL_ARGS,
    "decode_args": DECODE_ARGS,
    "prefill_envs": PREFILL_ENVS,
    "decode_envs": DECODE_ENVS,
    "router_args": ROUTER_ARGS,
    "router_envs": ROUTER_ENVS,
}


class TestNPUMiniMaxM2_5W8A8_1P1D_16P_In64k_Out1k_Prefix90_50ms(
    TestAscendPerfMultiNodePdSepTestCaseBase
):
    """MiniMax-M2.5-w8a8 PD Sep 1p1d 16p 64k input 1k output with 90% prefix cache performance test"""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    aisbench_dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model_config = MODEL_CONFIG
    dataset_name = "random"
    max_concurrency = 160
    num_prompts = 640
    input_len = 65536
    output_len = 1024
    random_range_ratio = 1
    aisbench_repeat_rate = 0.9
    tpot = 50
    output_token_throughput = 1529.48

    def test_npu_minimax_m2_5_w8a8_1p1d_16p_in64k_out1k_prefix90_50ms(self):
        """Run MiniMax-M2.5-w8a8 PD Sep 1p1d 16p 64k/1k prefix90 performance test"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
