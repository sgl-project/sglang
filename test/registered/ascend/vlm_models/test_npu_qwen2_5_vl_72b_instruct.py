import unittest

from sglang.test.ascend.test_ascend_utils import QWEN2_5_VL_72B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)


class TestQwen25VL72B(TestVLMModels):
    """Testcase: Verify that the inference accuracy of the Qwen/Qwen2.5-VL-72B-Instruct model on the MMMU dataset is no less than 0.2.

    [Test Category] Model
    [Test Target] Qwen/Qwen2.5-VL-72B-Instruct
    """

    model = QWEN2_5_VL_72B_INSTRUCT_WEIGHTS_PATH
    mmmu_accuracy = 0.2
    other_args = [
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "32",
        "--enable-multimodal",
        "--mem-fraction-static",
        0.6,
        "--log-level",
        "info",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        8,
    ]

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
