import unittest

from sglang.test.ascend.test_ascend_utils import (
    MISTRAL_SMALL_3_1_24B_INSTRUCT_2503_WEIGHTS_PATH,
)
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestMistralModels(TestVLMModels):
    """Testcase: Verify that the inference accuracy of the mistralai/Mistral-Small-3.1-24B-Instruct-2503 model on the MMMU dataset is no less than 0.2.

    [Test Category] Model
    [Test Target] mistralai/Mistral-Small-3.1-24B-Instruct-2503
    """

    model = MISTRAL_SMALL_3_1_24B_INSTRUCT_2503_WEIGHTS_PATH
    mmmu_accuracy = 0.2

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
