import unittest

from sglang.test.ascend.test_ascend_utils import QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestQwen25VL3B(TestVLMModels):
    """Testcase: Verify that the inference accuracy of the Qwen/Qwen2.5-VL-3B-Instruct model on the MMMU dataset is no less than 0.2.

    [Test Category] Model
    [Test Target] Qwen/Qwen2.5-VL-3B-Instruct
    """

    model = QWEN2_5_VL_3B_INSTRUCT_WEIGHTS_PATH
    mmmu_accuracy = 0.2

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
