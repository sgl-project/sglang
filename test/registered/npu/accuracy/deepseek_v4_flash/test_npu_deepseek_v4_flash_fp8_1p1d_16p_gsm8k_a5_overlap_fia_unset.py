import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    EVALSCOPE,
    TestNpuAccuracyMultiNodePdSepTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_dual_pass import record_pass1
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=7200,
    suite="nightly-acc-4-npu-a5",
    nightly=True,
)

CASE_ID = "test_npu_deepseek_v4_flash_fp8_1p1d_16p_gsm8k_a5_overlap"

# Pass 1: ASCEND_USE_FIA left unset (image default), CPU seq_lens mirror kept (1).
_PREFILL_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "1",
    "SGLANG_SET_CPU_AFFINITY": "1",
    "TASK_QUEUE_ENABLE": "2",
    "INF_NAN_MODE_FORCE_DISABLE": "1",
    "SGLANG_DEFAULT_THINKING": "1",
    "SGLANG_DSV4_REASONING_EFFORT": "max",
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "SGLANG_DISAGGREGATION_WAITING_TIMEOUT": "600",
    "DEEPEP_HCCL_BUFFSIZE": "2560",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "IS_DEEPSEEK_V4": "1",
    "USE_FUSED_HC_PRE_ASCENDC": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR": "1",
    "SGLANG_DSV4_NPU_FUSED_COMPRESSOR_PREFILL": "1",
    "SGLANG_OPT_USE_OVERLAP_STORE_CACHE": "False",
    "SGLANG_DSV4_FP4_EXPERTS": "True",
    "SGLANG_OPT_FUSE_WQA_WKV": "0",
    "SGLANG_OPT_BF16_FP32_GEMM_ALGO": "torch",
    "SGLANG_OPT_USE_FUSED_HASH_TOPK": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_PRE": "False",
    "SGLANG_OPT_DEEPGEMM_HC_PRENORM": "False",
    "SGLANG_OPT_USE_TILELANG_MHC_POST": "False",
    "SGLANG_OPT_FP8_WO_A_GEMM": "False",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "SGLANG_NPU_USE_MULTI_STREAM": "1",
    "USE_NPU_MOE_GATING_TOP_K": "1",
    "ASCEND_LAUNCH_BLOCKING": "1",
}

_DECODE_ENVS = {
    **_PREFILL_ENVS,
    "TASK_QUEUE_ENABLE": "0",
}

_PREFILL_ARGS = [
    "--disaggregation-mode",
    "prefill",
    "--disaggregation-transfer-backend",
    "ascend",
    "--attention-backend",
    "dsv4",
    "--device",
    "npu",
    "--trust-remote-code",
    "--tp-size",
    16,
    "--pp-size",
    1,
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "normal",
    "--quantization",
    "fp8",
    "--mem-fraction-static",
    0.78,
    "--page-size",
    64,
    "--max-running-requests",
    16,
    "--chunked-prefill-size",
    8192,
    "--max-prefill-tokens",
    300000,
    "--watchdog-timeout",
    9000,
]

_DECODE_ARGS = [
    "--disaggregation-mode",
    "decode",
    "--disaggregation-transfer-backend",
    "ascend",
    "--attention-backend",
    "dsv4",
    "--device",
    "npu",
    "--trust-remote-code",
    "--tp-size",
    16,
    "--dp-size",
    16,
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "low_latency",
    "--quantization",
    "fp8",
    "--mem-fraction-static",
    0.75,
    "--page-size",
    64,
    "--max-running-requests",
    32,
    "--context-length",
    180000,
    "--cuda-graph-bs-decode",
    1,
    2,
    4,
    8,
    "--watchdog-timeout",
    9000,
]

MODEL_CONFIG = {
    "model_path": DEEPSEEK_V4_FLASH_DEFAULT_MODEL_PATH,
    "prefill_args": _PREFILL_ARGS,
    "decode_args": _DECODE_ARGS,
    "prefill_envs": _PREFILL_ENVS,
    "decode_envs": _DECODE_ENVS,
    "router_args": ["--policy", "round_robin"],
    "router_envs": {},
}


class TestNPUDeepSeekV4FlashFP8_1P1D_16PGSM8KA5OverlapFiaUnset(
    TestNpuAccuracyMultiNodePdSepTestCaseBase
):
    """Pass 1: SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=1, ASCEND_USE_FIA unset."""

    benchmark_tool = EVALSCOPE
    model_config = MODEL_CONFIG
    accuracy = 0.9
    datasets = ["gsm8k"]
    limit = 100
    generation_config = {"max_tokens": 2048, "temperature": 0.01}
    eval_batch_size = 16

    def test_npu_deepseek_v4_flash_fp8_1p1d_4p_gsm8k_a5_overlap_fia_unset(self):
        """Run pass-1 GSM8K and record the baseline accuracy."""
        self.run_accuracy()
        record_pass1(CASE_ID, self.last_metrics)


if __name__ == "__main__":
    unittest.main()
