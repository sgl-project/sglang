import unittest

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK_DATASET_DEFAULT,
    BENCHMARK_TOOL_DEFAULT,
    DEEPSEEK_V4_FLASH_W8A8_MTP_MODEL_PATH,
    TestNpuPerformanceTestCaseBase,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=1800, suite="nightly-perf-16-npu-a3", nightly=True)

# Environment variables for DSV4-Flash single-node PD-mix deployment.
DEEPSEEK_V4_FLASH_W8A8_8P_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "USE_NPU_MOE_GATING_TOP_K": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    # deepep
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    # zbal
    "HCCL_BUFFSIZE": "8",
    "SGLANG_ZBAL_LOCAL_MEM_SIZE": "61000",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    "ZBAL_NPU_ALLOC_CONF": "use_vmm_for_static_memory:True",
    "SGLANG_ZBAL_BOOTSTRAP_URL": "tcp://127.0.0.1:24669",
    "ZBAL_ENABLE_GRAPH": "1",
    # dsv4
    "IS_DEEPSEEK_V4": "1",
    "SGLANG_DEBUG_LAYER_NORM": "1",
    "SGLANG_DEBUG_FWD_INPUT": "1",
    "USE_FUSED_HC_PRE_ASCENDC": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL": "1",
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
    # mtp
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_PROFILING": "0",
    "SGLANG_DEBUG_MTP_VERIFY": "0",
    "SGLANG_DEBUG_MTP_VERIFY_LIMIT": "8",
    "SGLANG_DEBUG_MTP_VERIFY_ROWS": "4",
    "SGLANG_DISABLE_DRAFT_EXTEND_GRAPH": "1",
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
    "--prefill-max-requests",
    160,
    "--attention-backend",
    "dsv4",
    "--watchdog-timeout",
    9000,
    "--mem-fraction-static",
    0.7,
    "--chunked-prefill-size",
    131072,
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
    "auto",
    "--skip-server-warmup",
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
    "--ep-size",
    16,
    "--disable-radix-cache",
]


class TestNPUDeepSeekV4FlashW8A88PIn8kOut1k50ms(TestNpuPerformanceTestCaseBase):
    """Test NPU performance for DeepSeek-V4-Flash W8A8 8p in8k out1k."""

    benchmark_tool = BENCHMARK_TOOL_DEFAULT
    dataset_type = AISBENCHMARK_DATASET_DEFAULT
    model = DEEPSEEK_V4_FLASH_W8A8_MTP_MODEL_PATH
    other_args = DEEPSEEK_V4_FLASH_W8A8_8P_OTHER_ARGS
    envs = DEEPSEEK_V4_FLASH_W8A8_8P_ENVS
    dataset_name = "random"
    dataset_path = "/root/.cache/modelscope/hub/datasets/gsm8k_deepseekv4/cache0_8000/formal_run1_160_8000_cache0.json"
    input_len = 8000
    output_len = 1000
    num_prompts = 160
    max_concurrency = 160
    random_range_ratio = 1
    warmup_requests = 0
    request_rate = float("inf")
    seed = 1
    tpot = 50
    max_attempts = 3
    output_token_throughput = 2825

    def test_npu_deepseek_v4_flash_w8a8_8p_in8k_out1k_50ms(self):
        """Run NPU performance test for DeepSeek-V4-Flash W8A8 8p in8k out1k."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
