import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    EVALSCOPE,
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_dual_pass import assert_no_regression, record_pass1
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    GLM_5_2_W8A8C8_MXFP8_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=7200,
    suite="nightly-acc-4-npu-a5",
    nightly=True,
)

CASE_ID = "test_npu_glm_5_2_w8a8c8_mxfp8_8p_gsm8k_a5_overlap"

# Based on the GLM-5.2-w4a8 launch scripts in PR #45949.
GLM_5_2_W4A8_4P_ENVS = {
    "STREAMS_PER_DEVICE": "32",
    "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "0",
    "ASCEND_USE_FIA": "0",
    "SGLANG_ENABLE_SPEC_V2": "1",
    "SGLANG_ENABLE_OVERLAP_PLAN_STREAM": "1",
    "HCCL_BUFFSIZE": "500",
    "DEEPEP_HCCL_BUFFSIZE": "2500",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "TRANSFORMERS_VERBOSITY": "error",
    "DEEPEP_NORMAL_LONG_SEQ_ROUND": "72",
    "DEEPEP_NORMAL_LONG_SEQ_PER_ROUND_TOKENS": "1024",
    "DEEPEP_NORMAL_COMBINE_ENABLE_LONG_SEQ": "1",
    "SGLANG_SCHEDULER_DECREASE_PREFILL_IDLE": "1",
    "SGLANG_PREFILL_DELAYER_MAX_DELAY_PASSES": "100",
    "DEEP_NORMAL_MODE_USE_INT8_QUANT": "1",
    "DEEPEP_HYBRID_DEPLOYMENT": "1",
}

# Pass 1: ASCEND_USE_FIA left unset (image default), CPU seq_lens mirror kept (1).
GLM_5_2_W4A8_4P_PASS1_ENVS = {
    **{k: v for k, v in GLM_5_2_W4A8_4P_ENVS.items() if k != "ASCEND_USE_FIA"},
    "SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS": "1",
}

GLM_5_2_W4A8_4P_OTHER_ARGS = [
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--tp-size",
    8,
    "--dp-size",
    4,
    "--enable-dp-attention",
    "--enable-dp-lm-head",
    "--trust-remote-code",
    "--mem-fraction-static",
    0.8,
    "--max-running-requests",
    64,
    "--quantization",
    "modelslim",
    "--moe-a2a-backend",
    "deepep",
    "--deepep-mode",
    "auto",
    "--load-balance-method",
    "round_robin",
    "--speculative-algorithm",
    "NEXTN",
    "--speculative-num-steps",
    4,
    "--speculative-eagle-topk",
    1,
    "--speculative-num-draft-tokens",
    5,
    "--cuda-graph-bs-decode",
    8,
    16,
    24,
    32,
    "--enable-metrics",
]


class Glm52W4A8_4PGSM8KA5OverlapBase(TestNpuAccuracyTestCaseBase):
    """Shared config for the two-pass (FIA unset vs FIA=0) GSM8K comparison."""

    benchmark_tool = EVALSCOPE
    model = GLM_5_2_W8A8C8_MXFP8_MODEL_PATH
    other_args = GLM_5_2_W4A8_4P_OTHER_ARGS
    accuracy = 0.9
    datasets = ["gsm8k"]
    limit = 100
    generation_config = {"max_tokens": 2048, "temperature": 0.01}
    eval_batch_size = 16


class TestNPUGLM5_2W4A8_4PGSM8KA5OverlapPass1(Glm52W4A8_4PGSM8KA5OverlapBase):
    """Pass 1: SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=1, ASCEND_USE_FIA unset."""

    envs = GLM_5_2_W4A8_4P_PASS1_ENVS

    def test_npu_glm_5_2_w4a8_4p_gsm8k_a5_overlap_pass1(self):
        """Run pass-1 GSM8K and record the baseline accuracy."""
        self.run_accuracy()
        record_pass1(CASE_ID, self.last_metrics)


class TestNPUGLM5_2W4A8_4PGSM8KA5OverlapPass2(Glm52W4A8_4PGSM8KA5OverlapBase):
    """Pass 2: SGLANG_NPU_ATTN_BACKEND_NEEDS_CPU_SEQ_LENS=0, ASCEND_USE_FIA=0."""

    envs = GLM_5_2_W4A8_4P_ENVS

    def test_npu_glm_5_2_w4a8_4p_gsm8k_a5_overlap_pass2(self):
        """Run pass-2 GSM8K and assert accuracy within 1% of pass 1."""
        self.run_accuracy()
        assert_no_regression(self, CASE_ID, self.last_metrics)


if __name__ == "__main__":
    unittest.main()
