import os
import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_VL_8B_THINKING_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=12000, suite="full-acc-test-2-npu-a3", nightly=True)

# The PR pipeline (pr-test-npu.yml) runs on both `pull_request` and a daily
# `schedule`, so match the workflow file instead of the event name. Every other
# caller (nightly, full test) still gets the full run.
_is_pr_pipeline = "pr-test-npu.yml" in os.environ.get("GITHUB_WORKFLOW_REF", "")

ENVS = {
    "SGLANG_DISAGGREGATION_BOOTSTRAP_TIMEOUT": "600",
    "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    "HCCL_SOCKET_IFNAME": "lo",
    "GLOO_SOCKET_IFNAME": "lo",
    "HCCL_OP_EXPANSION_MODE": "AIV",
    "DEEPEP_HCCL_BUFFSIZE": "2000",
}

OTHER_ARGS = [
    "--trust-remote-code",
    "--attention-backend",
    "ascend",
    "--device",
    "npu",
    "--max-running-requests",
    64,
    "--max-prefill-tokens",
    16384,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    -1,
    "--tp-size",
    2,
    "--mem-fraction-static",
    0.8,
    "--dtype",
    "bfloat16",
    "--reasoning-parser",
    "qwen3-thinking",
    "--tool-call-parser",
    "qwen",
    "--enable-multimodal",
    "--mm-attention-backend",
    "ascend_attn",
]


class TestQwen3_VL_8B_Thinking_MMMU(TestNpuAccuracyTestCaseBase):
    model = QWEN3_VL_8B_THINKING_MODEL_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    accuracy = 0.64 if _is_pr_pipeline else 0.7011
    datasets = ["mmmu"]
    few_shot_num = 0
    limit = 5 if _is_pr_pipeline else 100000
    generation_config = {"max_tokens": 65536}
    eval_batch_size = 64

    def test_mmmu(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
