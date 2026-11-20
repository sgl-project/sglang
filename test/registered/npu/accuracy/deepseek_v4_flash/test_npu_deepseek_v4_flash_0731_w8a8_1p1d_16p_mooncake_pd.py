import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_multi_node_utils import NIC_NAME
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
    assert_metrics,
    run_bench_serving,
)
from sglang.test.ascend.test_ascend_utils import (
    append_ld_library_path,
    write_memfabric_tent_conf,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3800,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_COMMON_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": NIC_NAME,
    "GLOO_SOCKET_IFNAME": NIC_NAME,
    "HCCL_OP_EXPANSION_MODE": "AIV",
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
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1800",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1800",
    "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER": "1",
}

# Prefill node environment variables for DSV4-Flash-0731 PD-Sep deployment.
DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_PREFILL_ENVS = {
    **DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_COMMON_ENVS,
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "128",
    "DEEPEP_HCCL_BUFFSIZE": "2600",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "8192",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "8",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK": "0",
    "SGLANG_ENABLE_WAR_BARRIER": "0",
    "SGLANG_FORCE_COARSE_WAR_BARRIER": "0",
    "SGLANG_PROFILE_WITH_STACK": "False",
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
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "SGLANG_NPU_SPLIT_SHARED_EXPERT_OVERLAP": "1",
    "SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX": "1",
}

# Prefill node launch arguments.
# TP 16 without PP: the Mooncake transfer backend validates the prefill/decode
# KV entry layout positionally, and the NPU DSV4 pool publishes several entries
# per layer, so a PP-sliced prefill list no longer lines up with the decode
# one (KVTransferError: PD KV layout mismatch). The decode node is TP 16 too,
# so 16 TP ranks keep the same device count.
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
    "mooncake",
    "--mem-fraction-static",
    0.62,
    "--prefill-max-requests",
    256,
    "--max-prefill-tokens",
    70000,
    "--max-running-requests",
    256,
    "--chunked-prefill-size",
    131072,
    "--attn-dp-size",
    8,
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
    "--enable-hierarchical-cache",
    "--hicache-ratio",
    0.5,
    "--hicache-io-backend",
    "kernel_ascend",
    "--hicache-mem-layout",
    "page_first",
    "--hicache-write-policy",
    "write_through",
    "--enable-cache-report",
    "--max-total-tokens",
    140000,
]

# Decode node launch arguments.
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
    0.71,
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "mooncake",
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
    "--cuda-graph-bs-decode",
    1,
    2,
    3,
    4,
    5,
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


class TestNPUDeepSeekV4Flash0731W8A8PDSEPGPQAHigh(
    TestNpuAccuracyMultiNodePdSepTestCaseBase
):
    """Test NPU accuracy for DeepSeek-V4-Flash W8A8 16p 1P1D GPQA."""

    # (reference: Flash W8A8 GPQA Diamond baseline is 0.874).
    accuracy = 0.874
    datasets = ["gpqa_diamond"]
    # A single attempt: gsm8k's 3-retry budget would exceed the step timeout.
    max_retries = 1
    # Weekday nightly overrides this to gsm8k (baseline is provisional).
    dataset_baselines = {
        "gsm8k": 0.90,
    }
    generation_config = DEEPSEEK_V4_FLASH_0731_W8A8_GENERATION_CONFIG_HIGH
    eval_batch_size = 128
    # Use a subset of GPQA Diamond (198 questions) to cut the step runtime.
    limit = 16
    min_hit_rate_pcts = {"host": 0}  # percent: L2 host hit rate must exceed 0%

    @classmethod
    def setUpClass(cls):
        # The Mooncake disaggregation transfer backend runs on both the prefill
        # and the decode node, so both get the TENT config and the
        # memfabric/mooncake library path of this pod. L3 HiCache storage is
        # disabled: only the L2 host pool is used.
        mooncake_envs = {
            # TENT transport over the NPU device path.
            "MC_USE_TENT": "1",
            "MF_PROTOCOL": "device_sdma",
            "MF_TRANSFER_TIMEOUT": "10000",
            "ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE": "true",
            "MC_TENT_CONF": write_memfabric_tent_conf(),
            "LD_LIBRARY_PATH": append_ld_library_path(
                ("memfabric_hybrid", "lib"), ("mooncake", None)
            ),
        }
        cls.model_config = {
            "model_path": DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
            "prefill_args": list(DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_PREFILL_ARGS),
            "decode_args": list(DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_DECODE_ARGS),
            "prefill_envs": {
                **DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_PREFILL_ENVS,
                **mooncake_envs,
            },
            "decode_envs": {
                **DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_DECODE_ENVS,
                **mooncake_envs,
            },
            # DSV4-Flash GPQA High streams up to 125k tokens per request, which
            # takes over an hour; the default 1800s router timeout cuts those
            # responses off mid-stream ("incomplete chunked read" in evalscope).
            "router_args": ["--request-timeout-secs", "10800"],
            "router_envs": {},
        }
        super().setUpClass()

    def test_npu_deepseek_v4_flash_0731_w8a8_pd_sep_gpqa_high(self):
        """Run NPU accuracy test for DSV4-Flash-0731 PD-Sep GPQA High mode."""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
