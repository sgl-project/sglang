import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    MIMO_V2_FLASH_MODEL_PATH,
    TestNpuPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

PREFILL_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "32",
    "HCCL_BUFFSIZE": "1024",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_NPU_PROFILING": "0",
    "SGLANG_NPU_PROFILING_STAGE": "prefill",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "3600",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_BF16_DISPATCH": "0",
    "SGLANG_DISAGGREGATION_FORCE_QUERY_PREFILL_DP_RANK": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_CONNECT_TIMEOUT": "1800",
    "ASCEND_USE_FIA": "1",
}

DECODE_ENVS = {
    "SGLANG_SET_CPU_AFFINITY": "1",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "HCCL_BUFFSIZE": "800",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "SGLANG_NPU_PROFILING": "0",
    "SGLANG_NPU_PROFILING_STAGE": "prefill",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "32",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "3584",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "3600",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "3600",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "0",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_BF16_DISPATCH": "0",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_CONNECT_TIMEOUT": "1800",
    "SGLANG_PROFILE_WITH_STACK": "True",
    "ASCEND_USE_FIA": "1",
}

PREFILL_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    8,
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--chunked-prefill-size",
    8192,
    "--trust-remote-code",
    "--max-running-requests",
    64,
    "--mem-fraction-static",
    0.8,
    "--swa-full-tokens-ratio",
    0.3,
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
    "--disable-radix-cache",
    "--disable-cuda-graph",
    "--disable-piecewise-cuda-graph",
    "--dp-size",
    "2",
]

DECODE_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    16,
    "--nnodes",
    "1",
    "--node-rank",
    "0",
    "--trust-remote-code",
    "--max-running-requests",
    64,
    "--mem-fraction-static",
    0.8,
    "--swa-full-tokens-ratio",
    0.3,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    12,
    16,
    20,
    24,
    28,
    32,
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    3,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    4,
    "--enable-multi-layer-eagle",
    "--disable-radix-cache",
    "--dp-size",
    "2",
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "low_latency",
]

ROUTER_ARGS = [
    "--health-check-interval-secs",
    "3600",
    "--mini-lb",
]

MODEL_CONFIG = {
    "model_path": MIMO_V2_FLASH_MODEL_PATH,
    "prefill_args": PREFILL_ARGS,
    "decode_args": DECODE_ARGS,
    "prefill_envs": PREFILL_ENVS,
    "decode_envs": DECODE_ENVS,
    "router_args": ROUTER_ARGS,
    "router_envs": {},
}


class TestNPUMimo_v2_flash_1P1D_16p_In16k_Out1_TTFT_5s(
    TestNpuPerfMultiNodePdSepTestCaseBase
):
    """Test NPU performance for mimo_v2_flash 1P+1D 16p: input_len=16000, output_len=1, 0 cache, TTFT=5s"""

    model_config = MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    dataset_name = "random"
    max_concurrency = 64
    num_prompts = 128
    request_rate = 0.4
    input_len = 16000
    output_len = 1
    random_range_ratio = 1
    seed = 1
    ttft = 5000

    def test_npu_mimo_v2_flash_1p1d_16p_in16k_out1_ttft_5s(self):
        """Run NPU performance test for 1P+1D 16p with 16k input, 1 output, 0 cache, TTFT=5s"""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
