import unittest

from sglang.test.ascend.test_ascend_utils import GEMMA_3_4B_IT_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestGemma34bModels(TestVLMModels):
    """Testcase: Verify that the inference accuracy of the google/gemma-3-4b-it model on the MMMU dataset is no less than 0.2.

    [Test Category] Model
    [Test Target] google/gemma-3-4b-it
    """

    model = GEMMA_3_4B_IT_WEIGHTS_PATH
    mmmu_accuracy = 0.2

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
