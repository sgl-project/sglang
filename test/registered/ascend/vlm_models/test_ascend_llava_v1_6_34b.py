import unittest

from sglang.test.ascend.test_ascend_utils import LLAVA_V1_6_34B_WEIGHTS_PATH, LLAVA_V1_6_34B_TOKENIZER_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestLlava(TestVLMModels):
    """Testcase: Verify that the inference accuracy of the AI-ModelScope/llava-v1.6-34b model on the MMMU dataset is no less than 0.2.

    [Test Category] Model
    [Test Target] AI-ModelScope/llava-v1.6-34b
    """

    model = LLAVA_V1_6_34B_WEIGHTS_PATH
    mmmu_accuracy = 0.2
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        4,
        "--max-running-requests",
        2048,
        "--mem-fraction-static",
        "0.7",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--mm-per-request-timeout",
        60,
        "--enable-multimodal",
        "--tokenizer-path",
        LLAVA_V1_6_34B_TOKENIZER_PATH,
    ]

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
