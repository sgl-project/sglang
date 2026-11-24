import importlib.util
import unittest
from pathlib import Path

from sglang.test.ascend.e2e.test_npu_performance_utils import (
    AISBENCHMARK,
    AISBENCHMARK_DATASET_GSM8K,
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
    TestNpuPerfMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import (
    append_ld_library_path,
    write_memfabric_tent_conf,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
)

MOONCAKE_INSTALLED = importlib.util.find_spec("mooncake") is not None
DISAGGREGATION_TRANSFER_BACKEND = "mooncake" if MOONCAKE_INSTALLED else "ascend"

# Mooncake + TENT + MemFabric, only used when the Mooncake package is present.
MOONCAKE_ENVS = {
    "MC_USE_TENT": "1",
    "MF_PROTOCOL": "device_sdma",
    "MF_TRANSFER_TIMEOUT": "10000",
    "ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE": "true",
}

# Common environment variables for DSV4-Flash PD-Sep 128k1k deployment
DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_COMMON_ENVS = {
    "PYTHONUNBUFFERED": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
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
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "USE_NPU_MOE_GATING_TOP_K": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_ENABLE_MULTI_STREAM": "0",
    "SGLANG_DSPARK_QUANT_AUDIT": "1",
    "SGLANG_DSPARK_QUANT_AUDIT_STRICT": "0",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1800",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1800",
    "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER": "1",
    "SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX": "1",
}

# Prefill node environment variables for DSV4-Flash PD-Sep 128k1k deployment
DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_PREFILL_ENVS = {
    **DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_COMMON_ENVS,
    # deepep
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "DEEPEP_HCCL_BUFFSIZE": "2600",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "8192",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "1",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    # war barrier
    "SGLANG_ENABLE_WAR_BARRIER": "1",
    "SGLANG_FORCE_COARSE_WAR_BARRIER": "1",
    "SGLANG_PROFILE_WITH_STACK": "False",
}

# Decode node environment variables for DSV4-Flash PD-Sep 128k1k deployment
DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_DECODE_ENVS = {
    **DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_COMMON_ENVS,
    # deepep
    "HCCL_BUFFSIZE": "200",
    "DEEPEP_HCCL_BUFFSIZE": "1200",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "8",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "2048",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "SGLANG_NPU_SPLIT_SHARED_EXPERT_OVERLAP": "1",
}

# DSPARK speculative decoding arguments shared by the prefill and decode nodes:
# the PD prefill worker prunes the draft to context-KV injection.
DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_SPECULATIVE_ARGS = [
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
    "--speculative-dspark-block-size",
    5,
]

# Prefill node launch arguments for DSV4-Flash PD-Sep 128k1k
DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_PREFILL_ARGS = [
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
    DISAGGREGATION_TRANSFER_BACKEND,
    "--disaggregation-bootstrap-port",
    18998,
    "--mem-fraction-static",
    0.62,
    "--prefill-max-requests",
    256,
    "--max-prefill-tokens",
    70000,
    "--max-running-requests",
    256,
    "--tokenizer-worker-num",
    16,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "normal",
    "--quantization",
    "modelslim",
    "--enable-dp-lm-head",
    "--kv-cache-dtype",
    "bfloat16",
    "--disable-decode-cuda-graph",
    "--disable-prefill-cuda-graph",
    "--load-balance-method",
    "round_robin",
    "--enable-metrics",
    "--enable-cache-report",
    "--chunked-prefill-size",
    32768,
    "--init-expert-location",
    str(Path(__file__).parent / "expert-layout-ais-replicas16.json"),
    "--ep-dispatch-algorithm",
    "dynamic",
    "--ep-num-redundant-experts",
    16,
    "--attn-dp-size",
    16,
    "--optimistic-prefill-attempts",
    1,
    "--json-model-override-args",
    '{"use_index_cache":true,"index_topk_freq":2}',
    "--expert-distribution-recorder-mode",
    "stat",
    "--expert-distribution-recorder-buffer-size",
    512,
    *DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_SPECULATIVE_ARGS,
]

# Decode node launch arguments for DSV4-Flash PD-Sep 128k1k
DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_DECODE_ARGS = [
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
    0.71,
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    DISAGGREGATION_TRANSFER_BACKEND,
    "--max-running-requests",
    256,
    "--attn-dp-size",
    16,
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "low_latency",
    "--quantization",
    "modelslim",
    "--enable-dp-lm-head",
    "--kv-cache-dtype",
    "bfloat16",
    "--load-balance-method",
    "round_robin",
    "--enable-metrics",
    "--enable-cache-report",
    "--swa-full-tokens-ratio",
    0.05,
    "--cuda-graph-bs-decode",
    1,
    2,
    3,
    4,
    6,
    7,
    8,
    10,
    "--json-model-override-args",
    '{"use_index_cache":true,"index_topk_freq":2}',
    "--tokenizer-worker-num",
    16,
    *DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_SPECULATIVE_ARGS,
]

# Model config for DSV4-Flash W8A8 1P+1D PD-Sep 128k1k deployment
DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_MODEL_CONFIG = {
    "model_path": DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
    "prefill_args": DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_PREFILL_ARGS,
    "decode_args": DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_DECODE_ARGS,
    "prefill_envs": DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_PREFILL_ENVS,
    "decode_envs": DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_DECODE_ENVS,
    "router_args": [],
    "router_envs": {},
}


class TestNPUDeepSeekV4FlashW8A81P1D16PIn128kOut1k20msAisBench(
    TestNpuPerfMultiNodePdSepTestCaseBase
):
    """Test NPU performance for DeepSeek-V4-Flash W8A8 PD-Sep 1P+1D 16p in128k out1k prefix90."""

    model_config = DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_MODEL_CONFIG
    # max_attempts counts total runs, so 3 means one initial run + two retries.
    max_attempts = 3
    benchmark_tool = AISBENCHMARK
    dataset_type = AISBENCHMARK_DATASET_GSM8K
    dataset_path = None
    input_len = 131072
    output_len = 1024
    num_prompts = 90
    max_concurrency = 54
    repeat_rate = 0.9
    request_rate = 3
    dp = 16
    tpot = 20
    output_token_throughput = 964.864
    ttft = 7000

    @classmethod
    def setUpClass(cls):
        common_envs = dict(DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_COMMON_ENVS)
        if MOONCAKE_INSTALLED:
            common_envs.update(MOONCAKE_ENVS)
            common_envs["MC_TENT_CONF"] = write_memfabric_tent_conf()
            common_envs["LD_LIBRARY_PATH"] = append_ld_library_path(
                ("memfabric_hybrid", "lib"), ("mooncake", None)
            )
        cls.model_config = {
            **DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_MODEL_CONFIG,
            "prefill_args": list(DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_PREFILL_ARGS),
            "decode_args": list(DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_DECODE_ARGS),
            "prefill_envs": {
                **common_envs,
                **DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_PREFILL_ENVS,
            },
            "decode_envs": {
                **common_envs,
                **DEEPSEEK_V4_FLASH_W8A8_1P1D_128K_DECODE_ENVS,
            },
        }
        super().setUpClass()

    def test_npu_deepseek_v4_flash_w8a8_1p1d_16p_in128k_out1k_prefix90_20ms(self):
        """Run NPU performance test for DeepSeek-V4-Flash W8A8 1P+1D 16p in128k out1k prefix90."""
        self.run_throughput()


if __name__ == "__main__":
    unittest.main()
