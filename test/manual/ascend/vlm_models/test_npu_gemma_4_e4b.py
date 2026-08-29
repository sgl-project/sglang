import unittest

from sglang.test.ascend.test_ascend_utils import GEMMA_4_E4B_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels


class TestGemma4E4B(TestVLMModels):
    """Testcase: Verify that the inference accuracy of the google/gemma-4-E4B-it model on the MMMU dataset is no less than 0.30.

    [Test Category] Model
    [Test Target] google/gemma-4-E4B-it
    """

    model = GEMMA_4_E4B_WEIGHTS_PATH
    mmmu_accuracy = 0.30

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
