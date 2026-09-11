import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="nightly-acc-16-npu-a3",
    nightly=True,
)

register_npu_ci(
    est_time=7200,
    suite="nightly-acc-16-npu-a3-cann910",
    nightly=True,
)

# Environment variables for DSV4-Flash-0731 single-node PD-mix deployment,
# ported from scripts_shell/dspark-with-radix-cache/dsv4_flash_single.sh.
# FORCE_DRAFT_MODEL_NON_QUANT is intentionally NOT set: the bundled DSPARK
# draft weights are modelslim-quantized.
DEEPSEEK_V4_FLASH_0731_W8A8_8P_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_BUFFSIZE": "1200",
    # deepep
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "16",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "256",
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
    # dspark correctness-first setup
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FAST_KERNEL": "0",
    "SGLANG_DSPARK_FAST_SAMPLING": "0",
    "SGLANG_DSPARK_ENABLE_MULTI_STREAM": "0",
    "SGLANG_DSPARK_QUANT_AUDIT": "1",
    "SGLANG_DSPARK_QUANT_AUDIT_STRICT": "0",
}

# Server launch arguments for DSV4-Flash-0731 W8A8 single-node 16p PD-mix.
# Radix cache is intentionally ENABLED (no --disable-radix-cache).
DEEPSEEK_V4_FLASH_0731_W8A8_8P_OTHER_ARGS = [
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
    0.65,
    "--prefill-max-requests",
    16,
    "--max-prefill-tokens",
    204800,
    "--chunked-prefill-size",
    65536,
    "--max-running-requests",
    160,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--quantization",
    "modelslim",
    "--kv-cache-dtype",
    "bfloat16",
    "--context-length",
    140000,
    "--cuda-graph-bs",
    1,
    2,
    4,
    8,
    10,
    "--skip-server-warmup",
    "--disable-piecewise-cuda-graph",
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
]

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


class TestNPUDeepSeekV4Flash0731W8A88PGPQAHigh(TestNpuAccuracyTestCaseBase):
    """Test NPU accuracy for DeepSeek-V4-Flash-0731 W8A8 16p GPQA High mode.

    Requirement: DSV4_Flash_Radix_Cache_1 (step 2, single-node, radix on).
    """

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    model = DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH
    other_args = DEEPSEEK_V4_FLASH_0731_W8A8_8P_OTHER_ARGS
    envs = DEEPSEEK_V4_FLASH_0731_W8A8_8P_ENVS
    # TODO: calibrate the baseline on the first successful run
    # (reference: Flash W8A8 GPQA Diamond baseline is 0.874).
    accuracy = 0.85
    datasets = ["gpqa_diamond"]
    few_shot_num = 0
    generation_config = DEEPSEEK_V4_FLASH_0731_W8A8_GENERATION_CONFIG_HIGH
    eval_batch_size = 128
    stream = True
    timeout = 6000
    seed = 1

    def test_npu_deepseek_v4_flash_0731_w8a8_8p_gpqa_high(self):
        """Run NPU accuracy test for DSV4-Flash-0731 W8A8 16p GPQA High mode."""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
