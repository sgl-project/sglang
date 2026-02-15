import unittest

from sglang.test.ascend.test_ascend_utils import KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestKimiVLA3BInstruct(TestVLMModels):
    """Testcase: Verify that the inference accuracy of the Kimi/Kimi-VL-A3B-Instruct model on the MMMU dataset is no less than 0.2.

    [Test Category] Model
    [Test Target] Kimi/Kimi-VL-A3B-Instruct
    """

    model = KIMI_VL_A3B_INSTRUCT_WEIGHTS_PATH
    mmmu_accuracy = 0.2

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
