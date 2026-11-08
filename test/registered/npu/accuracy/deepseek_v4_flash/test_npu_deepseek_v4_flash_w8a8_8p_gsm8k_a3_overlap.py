import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    EVALSCOPE,
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_dual_pass import assert_no_regression, record_pass1
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=7200, suite="base-c-test-acc-16-npu-a3")

register_npu_ci(
    est_time=7200,
    suite="nightly-acc-16-npu-a3",
    nightly=True,
)

CASE_ID = "test_npu_deepseek_v4_flash_w8a8_8p_gsm8k_a3_overlap"

# Based on the DSV4-w8a8 launch scripts in PR #46805.
DEEPSEEK_V4_FLASH_W8A8_16P_ENVS = {
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "0",
    "ASCEND_USE_FIA": "0",
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
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "SGLANG_ENABLE_WAR_BARRIER": "1",
    "SGLANG_FORCE_COARSE_WAR_BARRIER": "1",
    "DEEPEP_HCCL_BUFFSIZE": "2500",
    "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "96",
    "SGLANG_RAGGED_VERIFY_MODE": "static",
    "SGLANG_DSPARK_FAST_KERNEL": "0",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
    "SGLANG_DSPARK_USE_GREEDY_DRAFT": "1",
}

# Pass 1: ASCEND_USE_FIA left unset (image default), CPU seq_lens mirror kept (1).
DEEPSEEK_V4_FLASH_W8A8_16P_PASS1_ENVS = {
    **{k: v for k, v in DEEPSEEK_V4_FLASH_W8A8_16P_ENVS.items() if k != "ASCEND_USE_FIA"},
    "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "1",
}

DEEPSEEK_V4_FLASH_W8A8_16P_OTHER_ARGS = [
    "--page-size",
    128,
    "--tp-size",
    8,
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "dsv4",
    "--mem-fraction-static",
    0.68,
    "--prefill-max-requests",
    192,
    "--max-prefill-tokens",
    80000,
    "--chunked-prefill-size",
    65536,
    "--max-running-requests",
    192,
    "--dp-size",
    4,
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
    "--load-balance-method",
    "round_robin",
    "--cuda-graph-bs-decode",
    1,
    2,
    4,
    8,
]


class DeepSeekV4FlashW8A8_8PGSM8KA3OverlapBase(TestNpuAccuracyTestCaseBase):
    """Shared config for the two-pass (FIA unset vs FIA=0) GSM8K comparison."""

    benchmark_tool = EVALSCOPE
    model = DEEPSEEK_V4_FLASH_0731_W8A8_MODEL_PATH
    other_args = DEEPSEEK_V4_FLASH_W8A8_16P_OTHER_ARGS
    accuracy = 0.9
    datasets = ["gsm8k"]
    limit = 100
    generation_config = {"max_tokens": 2048, "temperature": 0.01}
    eval_batch_size = 16


class TestNPUDeepSeekV4FlashW8A8_8PGSM8KA3OverlapPass1(DeepSeekV4FlashW8A8_8PGSM8KA3OverlapBase):
    """Pass 1: SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=1, ASCEND_USE_FIA unset."""

    envs = DEEPSEEK_V4_FLASH_W8A8_16P_PASS1_ENVS

    def test_npu_deepseek_v4_flash_w8a8_16p_gsm8k_a3_overlap_pass1(self):
        """Run pass-1 GSM8K and record the baseline accuracy."""
        self.run_accuracy()
        record_pass1(CASE_ID, self.last_metrics)


class TestNPUDeepSeekV4FlashW8A8_8PGSM8KA3OverlapPass2(DeepSeekV4FlashW8A8_8PGSM8KA3OverlapBase):
    """Pass 2: SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=0, ASCEND_USE_FIA=0."""

    envs = DEEPSEEK_V4_FLASH_W8A8_16P_ENVS

    def test_npu_deepseek_v4_flash_w8a8_16p_gsm8k_a3_overlap_pass2(self):
        """Run pass-2 GSM8K and assert accuracy within 1% of pass 1."""
        self.run_accuracy()
        assert_no_regression(self, CASE_ID, self.last_metrics)


if __name__ == "__main__":
    unittest.main()
