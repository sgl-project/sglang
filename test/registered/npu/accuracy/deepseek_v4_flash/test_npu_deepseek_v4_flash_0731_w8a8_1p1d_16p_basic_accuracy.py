import importlib.util
import unittest
from pathlib import Path

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyMultiNodePdSepTestCaseBase,
)
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
)

# This case runs PD separation over the Mooncake transfer engine. Fail fast
# instead of silently degrading to the ascend backend: the launch args, the
# TENT envs and the "same draft model on both nodes" state-wire requirement
# below all assume Mooncake.
if importlib.util.find_spec("mooncake") is None:
    raise RuntimeError(
        "DeepSeek-V4-Flash basic accuracy case requires the Mooncake package "
        "(Mooncake + TENT PD transfer backend)."
    )

DISAGGREGATION_TRANSFER_BACKEND = "mooncake"

# Mooncake + TENT + MemFabric.
MOONCAKE_ENVS = {
    "MC_USE_TENT": "1",
    "MF_PROTOCOL": "device_sdma",
    "MF_TRANSFER_TIMEOUT": "10000",
    "ENABLE_ASCEND_TRANSFER_WITH_MOONCAKE": "true",
}

# The deployment args mirror the 128k1k performance case so accuracy shares
# its expert layout and EP settings; the expert layout file lives next to it.
EXPERT_LAYOUT_PATH = (
    Path(__file__).resolve().parents[2]
    / "performance"
    / "deepseek_v4_flash"
    / "expert-layout-ais-replicas16.json"
)

DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_COMMON_ENVS = {
    "PYTHONUNBUFFERED": "1",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
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
    "USE_NPU_MOE_GATING_TOP_K": "1",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FAST_SAMPLING": "0",
    "SGLANG_DSPARK_ENABLE_MULTI_STREAM": "0",
    "SGLANG_DSPARK_QUANT_AUDIT": "1",
    "SGLANG_DSPARK_QUANT_AUDIT_STRICT": "0",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "1800",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "1800",
    "SGLANG_DISAGGREGATION_ALL_CP_RANKS_TRANSFER": "1",
    # send cached prefix to decode early for radix-cache hits
    "SGLANG_DISAGG_PREFILL_EARLY_SEND_CACHED_PREFIX": "1",
}

# Prefill node environment variables for DSV4-Flash-0731 PD-Sep deployment.
DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_PREFILL_ENVS = {
    **DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_COMMON_ENVS,
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

# Decode node environment variables for DSV4-Flash-0731 PD-Sep deployment.
DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_DECODE_ENVS = {
    **DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_COMMON_ENVS,
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

# DSpark speculative decoding. Both nodes must run the same draft model: the PD
# state wire check compares the registered state sub-pools by position, and a
# draft model only on decode adds a second SWA pool there (prefill 4 entries vs
# decode 5), which the mooncake transfer backend rejects.
DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_SPEC_ARGS = [
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

# Prefill node launch arguments. No --pp-size / --enable-prefill-cp: a single
# TP16 stage covers the 16 NPUs of the node. The hierarchical-cache block is
# kept because this file also asserts the L2 host hit rate.
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
    DISAGGREGATION_TRANSFER_BACKEND,
    "--mem-fraction-static",
    0.62,
    "--prefill-max-requests",
    256,
    "--max-prefill-tokens",
    70000,
    "--max-running-requests",
    256,
    "--dp-size",
    16,
    "--enable-dp-attention",
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
    "--disable-cuda-graph",
    "--load-balance-method",
    "round_robin",
    "--enable-metrics",
    "--enable-cache-report",
    "--chunked-prefill-size",
    32768,
    "--init-expert-location",
    str(EXPERT_LAYOUT_PATH),
    "--ep-dispatch-algorithm",
    "dynamic",
    "--ep-num-redundant-experts",
    16,
    "--optimistic-prefill-attempts",
    1,
    "--json-model-override-args",
    '{"use_index_cache":true,"index_topk_freq":2}',
    "--expert-distribution-recorder-mode",
    "stat",
    "--expert-distribution-recorder-buffer-size",
    512,
    # L2 host hit-rate assertions need HiCache on the prefill side.
    "--enable-hierarchical-cache",
    "--hicache-io-backend",
    "kernel_ascend",
    "--hicache-ratio",
    1.5,
    "--hicache-write-policy",
    "write_through",
    "--max-total-tokens",
    140000,
    *DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_SPEC_ARGS,
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
    DISAGGREGATION_TRANSFER_BACKEND,
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
    *DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_SPEC_ARGS,
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


class TestNPUDeepSeekV4FlashPDSepBasicAccuracy(
    TestNpuAccuracyMultiNodePdSepTestCaseBase
):
    """Test NPU accuracy for DeepSeek-V4-Flash W8A8 16p 1P1D Basic GPQA."""

    model_config = None
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
    eval_batch_size = 54
    # TEMPORARY smoke-run cap (the default 100000 means the whole dataset).
    # A partial dataset no longer justifies the fixed 5/198 fluctuation, so the
    # 0.874 baseline check can fail on sampling noise; restore this before the
    # real release run.
    limit = 20
    min_hit_rate_pcts = {"host": 0}  # percent: L2 host hit rate must exceed 0%

    @classmethod
    def setUpClass(cls):
        common_envs = dict(DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_COMMON_ENVS)
        common_envs.update(MOONCAKE_ENVS)
        common_envs["MC_TENT_CONF"] = write_memfabric_tent_conf()
        common_envs["LD_LIBRARY_PATH"] = append_ld_library_path(
            ("memfabric_hybrid", "lib"), ("mooncake", None)
        )
        cls.model_config = {
            "model_path": DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
            "prefill_args": list(DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_PREFILL_ARGS),
            "decode_args": list(DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_DECODE_ARGS),
            "prefill_envs": {
                **common_envs,
                **DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_PREFILL_ENVS,
            },
            "decode_envs": {
                **common_envs,
                **DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_DECODE_ENVS,
            },
            "router_args": ["--policy", "round_robin"],
            "router_envs": {},
        }
        super().setUpClass()

    def test_npu_deepseek_v4_flash_accuracy(self):
        """Run NPU accuracy test for DSV4-Flash-0731 PD-Sep GPQA High mode."""
        self.run_accuracy()

    def test_npu_deepseek_v4_flash_prefix_cache_hit_rate(self):
        """Run the 32k GSP bench; assert_metrics checks the host hit rate"""
        # 3 groups x 32 prompts share a 28800-token system prompt (32k * 0.9)
        # plus a 3200-token question, which exceeds L1 so evictions land in L2.
        metrics = run_bench_serving(
            host=self.host,
            port=self.port,
            model_path=self.model_config.get("model_path"),
            dataset_name="generated-shared-prefix",
            input_len=32000,
            output_len=1024,
            max_concurrency=32,
            gsp_num_groups=3,
            gsp_prompts_per_group=32,
            seed=100,
            warmup_requests=16,
            cache_report=True,
        )
        assert_metrics(self, metrics)


if __name__ == "__main__":
    unittest.main()
