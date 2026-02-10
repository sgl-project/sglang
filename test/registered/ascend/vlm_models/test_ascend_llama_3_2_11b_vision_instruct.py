import unittest

from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_11B_VISION_INSTRUCT_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)


class TestLlama3211BVisionInstruct(TestVLMModels):
    """Testcase: Verify that the inference accuracy of the LLM-Research/Llama-3.2-11B-Vision-Instruct model on the MMMU dataset is no less than 0.2.

    [Test Category] Model
    [Test Target] LLM-Research/Llama-3.2-11B-Vision-Instruct
    """

    model = LLAMA_3_2_11B_VISION_INSTRUCT_WEIGHTS_PATH
    mmmu_accuracy = 0.2
    other_args = [
        "--trust-remote-code",
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--disable-radix-cache",
    ]

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
