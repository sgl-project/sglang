import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

# Environment variables shared by prefill/decode nodes, ported from
# scripts_shell/pd/flash_1p1d/dsv4_flash_pd.sh.
# FORCE_DRAFT_MODEL_NON_QUANT is intentionally NOT set: the bundled DSPARK
# draft weights are modelslim-quantized.
DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_COMMON_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "USE_NPU_MOE_GATING_TOP_K": "1",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "60",
    # skip gpu branch
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "SGLANG_DSV4_FP4_EXPERTS": "False",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    # MTP (DSPARK)
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

# Prefill node environment variables for DSV4-Flash-0731 PD-Sep deployment.
DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_PREFILL_ENVS = {
    **DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_COMMON_ENVS,
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "8192",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "8",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "HCCL_BUFFSIZE": "2048",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    "SGLANG_ENABLE_WAR_BARRIER": "1",
    "SGLANG_FORCE_COARSE_WAR_BARRIER": "1",
    # send cached prefix to decode early for radix-cache hits
    "SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX": "1",
}

# Decode node environment variables for DSV4-Flash-0731 PD-Sep deployment.
DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_DECODE_ENVS = {
    **DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_COMMON_ENVS,
    "HCCL_BUFFSIZE": "1200",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "8",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "SGLANG_NPU_USE_MULTI_STREAM": "0",
    "SGLANG_NPU_SPLIT_SHARED_EXPERT_OVERLAP": "1",
    # dspark correctness-first setup
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FAST_KERNEL": "0",
    "SGLANG_DSPARK_FAST_SAMPLING": "0",
    "SGLANG_DSPARK_ENABLE_MULTI_STREAM": "0",
    "SGLANG_DSPARK_QUANT_AUDIT": "1",
    "SGLANG_DSPARK_QUANT_AUDIT_STRICT": "0",
}

# Prefill node (1 node x 16 NPUs, TP16 DP8) launch arguments.
# Radix cache is intentionally ENABLED on prefill (no --disable-radix-cache).
DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_PREFILL_ARGS = [
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
    0.62,
    "--prefill-max-requests",
    6,
    "--max-prefill-tokens",
    140000,
    "--chunked-prefill-size",
    65536,
    "--max-running-requests",
    128,
    "--dp-size",
    8,
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
    "--context-length",
    140000,
    "--disable-cuda-graph",
    "--load-balance-method",
    "round_robin",
]

# Decode node (1 node x 16 NPUs, TP16 DP16) launch arguments.
DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_DECODE_ARGS = [
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
    0.7,
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
    "--max-running-requests",
    256,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "low_latency",
    "--quantization",
    "modelslim",
    "--enable-dp-lm-head",
    "--kv-cache-dtype",
    "bfloat16",
    "--context-length",
    140000,
    "--load-balance-method",
    "round_robin",
    "--cuda-graph-bs-decode",
    1,
    2,
    3,
    4,
    5,
    6,
    # DSPARK speculative decoding with the bundled W8A8 draft.
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
    "--speculative-draft-model-quantization",
    "modelslim",
    "--speculative-draft-attention-backend",
    "ascend",
    "--speculative-num-draft-tokens",
    6,
    # Radix cache enabled on decode for the cache-hit scenario.
    "--disaggregation-decode-enable-radix-cache",
]

# Model config for DSV4-Flash-0731 W8A8 1P+1D PD-Sep deployment.
DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_MODEL_CONFIG = {
    "model_path": DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
    "prefill_args": DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_PREFILL_ARGS,
    "decode_args": DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_DECODE_ARGS,
    "prefill_envs": DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_PREFILL_ENVS,
    "decode_envs": DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_DECODE_ENVS,
    "router_args": ["--policy", "cache_aware"],
    "router_envs": {},
}

# Generation config for Think High mode (thinking=true, reasoning_effort=high).
DEEPSEEK_V4_FLASH_0731_W8A8_GENERATION_CONFIG_HIGH = {
    "max_tokens": 125000,
    "top_p": 1,
    "temperature": 1,
    "n": 1,
    "extra_body": {
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}
    },
}


class TestNPUDeepSeekV4Flash0731W8A8PDSEPGPQAHigh(
    TestNpuAccuracyMultiNodePdSepTestCaseBase
):
    """Test NPU accuracy for DSV4-Flash-0731 W8A8 PD-Sep GPQA High mode.

    Requirement: DSV4_Flash_Radix_Cache_0 (step 2, 1P1D PD separation with
    radix cache enabled).
    """

    model_config = DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_MODEL_CONFIG
    # TODO: calibrate the baseline on the first successful run
    # (reference: Flash W8A8 GPQA Diamond baseline is 0.874).
    accuracy = 0.85
    datasets = ["gpqa_diamond"]
    generation_config = DEEPSEEK_V4_FLASH_0731_W8A8_GENERATION_CONFIG_HIGH
    eval_batch_size = 128

    def test_npu_deepseek_v4_flash_0731_w8a8_pd_sep_gpqa_high(self):
        """Run NPU accuracy test for DSV4-Flash-0731 PD-Sep GPQA High mode."""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
