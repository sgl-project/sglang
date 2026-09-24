import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    BENCHMARK_TOOL_DEFAULT,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH, TestNpuPerformanceTestCaseBase, AISBENCHMARK_DATASET_DEFAULT,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=4800,
    suite="nightly-acc-16-npu-a3",
    nightly=True,
)

# Environment variables for DSV4-Pro-0813 single-node PD-mix deployment,
# ported from scripts_shell/dspark-with-radix-cache/dsv4_pro_2mix.sh.
DEEPSEEK_V4_PRO_W4A8_8P_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_CONNECT_TIMEOUT": "300",
    "HCCL_EXEC_TIMEOUT": "68",
    "ACL_DEVICE_SYNC_TIMEOUT": "60",
    "DEEPEP_HCCL_BUFFSIZE": "1536",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "30",
    # skip gpu branch
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "FORCE_DRAFT_MODEL_NON_QUANT": "1",
    "SGLANG_DSV4_FP4_EXPERTS": "True",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_OPT_FP8_WO_A_GEMM": "0",
    "TRANSFORMERS_VERBOSITY": "error",
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

# Server launch arguments for DSV4-Pro W4A8 single-node 16p PD-mix.
# Aligned with dsv4_pro_2mix-without-dspark.sh (radix cache disabled).
DEEPSEEK_V4_PRO_W4A8_8P_OTHER_ARGS = [
    "--tp-size",
    16,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--watchdog-timeout",
    9000,
    "--max-running-requests",
    64,
    "--mem-fraction-static",
    0.8,
    "--quantization",
    "modelslim",
    "--chunked-prefill-size",
    65536,
    "--kv-cache-dtype",
    "auto",
    "--moe-dense-tp-size",
    1,
    "--cuda-graph-bs-decode",
    1,
    4,
    "--load-balance-method",
    "round_robin",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--enable-metrics",
    "--disable-radix-cache",
    # DSPARK speculative decoding with the bundled draft weights.
    "--speculative-algorithm",
    "DSPARK",
    "--speculative-draft-model-path",
    DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH,
    "--speculative-draft-model-quantization",
    "modelslim",
    "--speculative-draft-attention-backend",
    "ascend",
    "--speculative-num-draft-tokens",
    6,
]

# Generation config for Think High mode (thinking=true, reasoning_effort=high).
DEEPSEEK_V4_PRO_W4A8_GENERATION_CONFIG_HIGH = {
    "max_tokens": 125000,
    "top_p": 1,
    "temperature": 1,
    "n": 1,
    "extra_body": {
        "chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}
    },
}


class TestNPUDeepSeekV4ProW4A88PIn128kOut1kPrefix90(
    TestNpuPerformanceTestCaseBase
):
    """Test NPU performance for DeepSeek-V4-Pro W4A8 16p in128k out1k prefix90.

    Requirement: DSV4_Pro_Radix_Cache_1 (step 4, single-node 128k input with
    90% radix-cache hit rate). The shared-prefix dataset makes 90% of each
    input length a repeated prefix, so radix cache hits should reduce TTFT
    noticeably compared with the random-input test above.
    """

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = DEEPSEEK_V4_PRO_0813_W4A8_MODEL_PATH
    other_args = DEEPSEEK_V4_PRO_W4A8_8P_OTHER_ARGS
    envs = DEEPSEEK_V4_PRO_W4A8_8P_ENVS
    dataset_name = "generated-shared-prefix"
    warmup_requests = 0
    max_concurrency = 32
    num_prompts = 32
    repeat_rate = 0.9
    input_len = 131072
    output_len = 1024
    random_range_ratio = 1
    seed = 1
    temperature = 0.6
    top_p = 0.95
    request_rate = float("inf")
    max_attempts = 3
    pop_sglang_is_in_ci_for_gsp = True

    def test_npu_deepseek_v4_pro_w4a8_8p_in128k_out1k_prefix90(self):
        """Run NPU perf test for DeepSeek-V4-Pro W4A8 16p in128k prefix90."""
        self.run_throughput()

if __name__ == "__main__":
    unittest.main()