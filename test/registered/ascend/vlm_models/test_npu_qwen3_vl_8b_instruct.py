import unittest

from sglang.test.ascend.test_ascend_utils import QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestQwen3VL8B(TestVLMModels):
    """Testcase: Verify that the inference accuracy of the Qwen/Qwen3-VL-8B-Instruct model on the MMMU dataset is no less than 0.2.

    [Test Category] Model
    [Test Target] Qwen/Qwen3-VL-8B-Instruct
    """

    model = QWEN3_VL_8B_INSTRUCT_WEIGHTS_PATH
    mmmu_accuracy = 0.2

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
