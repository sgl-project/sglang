import unittest

from sglang.test.ascend.e2e.test_npu_accuracy_utils import (
    TestNpuAccuracyTestCaseBase,
)
from sglang.test.ascend.test_ascend_utils import DEEPSEEK_V3_2_EXP_W8A8_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=3600,
    suite="",
    nightly=True,
    disabled="accuracy testcase",
)

OTHER_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.9",
    "--attention-backend",
    "ascend",
    "--disable-cuda-graph",
    "--tp-size",
    "16",
    "--quantization",
    "modelslim",
    "--disable-radix-cache",
    "--reasoning-parser",
    "deepseek-v3",
    "--tool-call-parser",
    "deepseekv32",
]


class TestNPUDeepSeek_V3_2_8P_AIME2025(TestNpuAccuracyTestCaseBase):

    model = DEEPSEEK_V3_2_EXP_W8A8_WEIGHTS_PATH
    other_args = OTHER_ARGS
    accuracy = 0.936
    datasets = ["gsm8k"]
    few_shot_num = 0
    generation_config = {"max_tokens": 65536, "temperature": 1.0}
    eval_batch_size = 64

    def test_aime2025(self):
        self.run_accuracy()


if __name__ == "__main__":
    unittest.main()
