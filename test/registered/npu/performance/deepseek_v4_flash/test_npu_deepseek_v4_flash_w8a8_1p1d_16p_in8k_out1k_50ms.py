import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
    TestNpuPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

# Prefill node environment variables for DSV4-Flash PD-Sep deployment.
DEEPSEEK_V4_FLASH_W8A8_1P1D_PREFILL_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    # deepep
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "DEEPEP_HCCL_BUFFSIZE": "2048",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "35",
    # war barrier
    "SGLANG_ENABLE_WAR_BARRIER": "1",
    "SGLANG_FORCE_COARSE_WAR_BARRIER": "1",
    # skip gpu branch
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    # PD disagg
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "60",
    # MTP
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

# Decode node environment variables for DSV4-Flash PD-Sep deployment.
DEEPSEEK_V4_FLASH_W8A8_1P1D_DECODE_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    # deepep
    "HCCL_BUFFSIZE": "1200",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "8",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
    # skip gpu branch
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    # MTP
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
}

# Prefill node launch arguments for DSV4-Flash PD-Sep.
DEEPSEEK_V4_FLASH_W8A8_1P1D_PREFILL_ARGS = [
    "--page-size",
    128,
    "--tp-size",
    16,
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--watchdog-timeout",
    9000,
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
    "--disaggregation-bootstrap-port",
    8998,
    "--mem-fraction-static",
    0.68,
    "--prefill-max-requests",
    6,
    "--max-prefill-tokens",
    80000,
    "--chunked-prefill-size",
    131072,
    "--max-running-requests",
    112,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "normal",
    "--quantization",
    "modelslim",
    "--enable-dp-lm-head",
    "--kv-cache-dtype",
    "bfloat16",
    "--disable-cuda-graph",
    "--disable-radix-cache",
    "--load-balance-method",
    "round_robin",
    "--ep-dispatch-algorithm",
    "static",
    "--init-expert-location",
    "/root/.cache/modelscope/hub/models/hot_map/pd_prefill_0720.pt",
]

# Decode node launch arguments for DSV4-Flash PD-Sep.
DEEPSEEK_V4_FLASH_W8A8_1P1D_DECODE_ARGS = [
    "--page-size",
    128,
    "--tp-size",
    16,
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.75,
    "--prefill-max-requests",
    1,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    -1,
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
    "--max-running-requests",
    896,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--enable-dp-lm-head",
    "--kv-cache-dtype",
    "bfloat16",
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    16,
    24,
    36,
    40,
    48,
    # MTP (EAGLE) configuration.
    "--speculative-algorithm",
    "EAGLE",
    "--speculative-num-steps",
    2,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    3,
]

# Model config for DSV4-Flash W8A8 1P+1D PD-Sep deployment.
DEEPSEEK_V4_FLASH_W8A8_1P1D_MODEL_CONFIG = {
    "model_path": DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
    "prefill_args": DEEPSEEK_V4_FLASH_W8A8_1P1D_PREFILL_ARGS,
    "decode_args": DEEPSEEK_V4_FLASH_W8A8_1P1D_DECODE_ARGS,
    "prefill_envs": DEEPSEEK_V4_FLASH_W8A8_1P1D_PREFILL_ENVS,
    "decode_envs": DEEPSEEK_V4_FLASH_W8A8_1P1D_DECODE_ENVS,
    "router_args": ["--policy", "cache_aware"],
    "router_envs": {},
}


class TestNPUDeepSeekV4FlashW8A81P1D16PIn8kOut1k50ms(
    TestNpuPerfMultiNodePdSepTestCaseBase
):
    """Test NPU performance for DeepSeek-V4-Flash W8A8 PD-Sep 1P+1D 16p in8k out1k."""

    model_config = DEEPSEEK_V4_FLASH_W8A8_1P1D_MODEL_CONFIG
    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    dataset_name = "random"
    input_len = 8000
    output_len = 1000
    num_prompts = 2400
    max_concurrency = 800
    random_range_ratio = 1
    warmup_requests = 16
    request_rate = float("inf")
    seed = 1
    tpot = 50
    output_token_throughput = 7046

    def test_npu_deepseek_v4_flash_w8a8_1p1d_16p_in8k_out1k_50ms(self):
        """Run NPU performance test for DeepSeek-V4-Flash W8A8 1P+1D 16p in8k out1k."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
