import unittest
from pathlib import Path

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3800,
    suite="",
    nightly=True,
)

# Transfer through Ascend's native backend even when mooncake is installed:
# mooncake's PD registration rejected this DSPARK deployment's KV layout in
# earlier runs ("PD KV layout mismatch ... prefill=4096 decode=32768"), while
# Ascend transfers served every request. This is the only deviation from the
# 128k1k performance deployment below.
DISAGGREGATION_TRANSFER_BACKEND = "ascend"

# Expert layout shared with the 128k1k performance case (single copy lives
# there); the accuracy tree does not carry its own.
DEEPSEEK_V4_FLASH_EXPERT_LAYOUT = str(
    Path(__file__).resolve().parents[2]
    / "performance"
    / "deepseek_v4_flash"
    / "expert-layout-ais-replicas16.json"
)

# Environment variables / launch args mirror
# test/registered/npu/performance/deepseek_v4_flash/
# test_npu_deepseek_v4_flash_w8a8_1p1d_16p_in128k_out1k_prefix90_20ms.py
DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_COMMON_ENVS = {
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

# DSPARK speculative decoding arguments shared by the prefill and decode nodes:
# the PD prefill worker prunes the draft to context-KV injection, and both peers
# must register the same target + draft state sub-pools.
DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_SPECULATIVE_ARGS = [
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

# Prefill node launch arguments.
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
    # --disaggregation-bootstrap-port is injected by the multi-node harness
    # (BOOTSTRAP_INIT_PORT + pod index), so it is not repeated here.
    "--mem-fraction-static",
    0.68,
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
    DEEPSEEK_V4_FLASH_EXPERT_LAYOUT,
    "--ep-dispatch-algorithm",
    "dynamic",
    "--ep-num-redundant-experts",
    16,
    "--attn-dp-size",
    4,
    "--enable-prefill-cp",
    "--cp-strategy",
    "zigzag",
    "--optimistic-prefill-attempts",
    1,
    "--json-model-override-args",
    '{"use_index_cache":true,"index_topk_freq":2}',
    "--expert-distribution-recorder-mode",
    "stat",
    "--expert-distribution-recorder-buffer-size",
    512,
    # The DeepSWE agent (mini-swe-agent) drives the model through OpenAI
    # tool calls; without these parsers the engines return plain text and the
    # agent gives up after three format errors. The perf case does not need them.
    "--tool-call-parser",
    "deepseekv4",
    "--reasoning-parser",
    "deepseek-v4",
    *DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_SPECULATIVE_ARGS,
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
    "--tool-call-parser",
    "deepseekv4",
    "--reasoning-parser",
    "deepseek-v4",
    *DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_SPECULATIVE_ARGS,
]

# Model config for DSV4-Flash-0731 W8A8 1P+1D PD-Sep deployment.
DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_MODEL_CONFIG = {
    "model_path": DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
    "prefill_args": DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_PREFILL_ARGS,
    "decode_args": DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_DECODE_ARGS,
    "prefill_envs": DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_PREFILL_ENVS,
    "decode_envs": DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_DECODE_ENVS,
    "router_args": [],
    "router_envs": {},
}

# DeepSWE is driven by evalscope's `deep_swe` adapter, whose agent harness
# (datacurve-pier) shells out to Docker. The ARM NPU pods that serve the model
# cannot run it (no Docker daemon, amd64-only task images), so
# TestNpuAccuracyMultiNodePdSepTestCaseBase delegates the evaluation to a
# short-lived amd64 pod (kubernetes.io/arch=amd64) that provides DinD. This
# case therefore only owns the deployment config plus the eval parameters.
DEEPSWE_DATASET_ARGS = {
    "deep_swe": {
        "extra_params": {
            "pier_agent_kwargs": {
                # LiteLLM builds "openai/<model>" and routes it to
                # OPENAI_BASE_URL (the router's /v1).
                "model_class": "litellm",
                "provider": "openai",
            }
        }
    }
}


class TestNPUDeepSeekV4Flash0731W8A8PDSEPDeepSWE(
    TestNpuAccuracyMultiNodePdSepTestCaseBase
):
    """DeepSWE accuracy for DeepSeek-V4-Flash W8A8 16p 1P1D PD-Sep."""

    model_config = None
    datasets = ["deep_swe"]
    dataset_args = DEEPSWE_DATASET_ARGS
    # One Docker environment per task.
    eval_batch_size = 1
    # Smoke budget: raise to 113 for the full DeepSWE suite once the baseline
    # is calibrated and the workflow timeout is sized for it.
    limit = 1
    # dataset-level timeout forwarded to evalscope (task verifier 1800s + agent
    # 5400s, plus image pulls).
    timeout = 7200
    # A single attempt: retrying the whole agent suite would blow the step
    # budget.
    max_retries = 1
    # Placeholder: no calibrated baseline yet. The measured accuracy is still
    # recorded in metrics.json; once the first real run lands, set this to the
    # measured value to gate on it.
    accuracy = None

    @classmethod
    def setUpClass(cls):
        common_envs = dict(DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_COMMON_ENVS)
        cls.model_config = {
            **DEEPSEEK_V4_FLASH_0731_W8A8_PD_SEP_MODEL_CONFIG,
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
            # The router is launched with --no-tokenizer (workers tokenize
            # themselves), which cache_aware routing rejects, so keep the
            # default policy. With one prefill and one decode worker the choice
            # does not affect routing anyway.
            "router_args": [],
            "router_envs": {},
        }
        super().setUpClass()

    def test_npu_deepseek_v4_flash_0731_w8a8_pd_sep_deepswe(self):
        """Run DeepSWE on the amd64 eval pod against this PD-Sep deployment."""
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
