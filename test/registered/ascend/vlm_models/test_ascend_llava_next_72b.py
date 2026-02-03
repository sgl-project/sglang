import unittest

from sglang.test.ascend.test_ascend_utils import LLAVA_NEXT_72B_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestLlavaNext72B(TestVLMModels):
    """Testcase: Verify that the inference accuracy of the lmms-lab/llava-next-72b model on the MMMU dataset is no less than 0.2.

    [Test Category] Model
    [Test Target] lmms-lab/llava-next-72b
    """

    model = LLAVA_NEXT_72B_WEIGHTS_PATH
    mmmu_accuracy = 0.2
    other_args = [
        "--trust-remote-code",
        "--tp-size",
        4,
        "--mem-fraction-static",
        "0.8",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--enable-multimodal",
    ]

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
