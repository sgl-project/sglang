import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.e2e.test_npu_performance_utils import (
    QWEN3_VL_30B_A3B_THINKING_MODEL_PATH,
)
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="performance testcase",
)

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
    128,
    "--disable-radix-cache",
    "--chunked-prefill-size",
    -1,
    "--max-prefill-tokens",
    12800,
    "--prefill-max-requests",
    10,
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


class TestQwen3(TestNpuAccuracyTestCaseBase):
    model = QWEN3_VL_30B_A3B_THINKING_MODEL_PATH
    envs = ENVS
    other_args = OTHER_ARGS
    accuracy = 0.7167
    datasets = ["mmmu"]
    few_shot_num = 0
    generation_config = {"max_tokens": 65536}
    eval_batch_size = 64

    def test_mmmu(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
