import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_V4_FLASH_W8A8_MTP_MODEL_PATH,
    TestNpuPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

# Environment variables for DSV4-Flash single-node PD-mix deployment.
DEEPSEEK_V4_FLASH_W8A8_8P_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    # deepep
    "HCCL_BUFFSIZE": "1000",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "16",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
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
    # MTP (EAGLE) related envs
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
}

# Server launch arguments for DSV4-Flash W8A8 single-node 8p PD-mix.
DEEPSEEK_V4_FLASH_W8A8_8P_OTHER_ARGS = [
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
    "--prefill-max-requests",
    2,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    -1,
    "--max-running-requests",
    160,
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
    10,
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


class TestNPUDeepSeekV4FlashW8A88PIn8kOut1k50ms(TestNpuPerformanceTestCaseBase):
    """Test NPU performance for DeepSeek-V4-Flash W8A8 8p in8k out1k."""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = DEEPSEEK_V4_FLASH_W8A8_MTP_MODEL_PATH
    other_args = DEEPSEEK_V4_FLASH_W8A8_8P_OTHER_ARGS
    envs = DEEPSEEK_V4_FLASH_W8A8_8P_ENVS
    dataset_name = "random"
    input_len = 8000
    output_len = 1000
    num_prompts = 160
    max_concurrency = 160
    random_range_ratio = 1
    warmup_requests = 0
    request_rate = float("inf")
    seed = 1
    tpot = 50
    output_token_throughput = 1708

    def test_npu_deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms(self):
        """Run NPU performance test for DeepSeek-V4-Flash W8A8 8p in8k out1k."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
