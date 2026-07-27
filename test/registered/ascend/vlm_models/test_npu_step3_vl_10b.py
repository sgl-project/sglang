import unittest

from sglang.test.ascend.test_ascend_utils import STEP3_VL_10B_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=400,
    suite="full-2-npu-a3",
    nightly=True,
)


class TestDeepseekVl2(TestVLMModels):
    """Testcase: Verify that the inference accuracy of the stepfun-ai/Step3-VL-10B model on the MMMU dataset is no less than 0.69.

    [Test Category] Model
    [Test Target] stepfun-ai/Step3-VL-10B
    """

    model = STEP3_VL_10B_WEIGHTS_PATH
    mmmu_accuracy = 0.69
    other_args = [
        "--trust-remote-code",
        "--disable-radix-cache",
        "--chunked-prefill-size",
        -1,
        "--tp-size",
        2,
        "--mem-fraction-static",
        0.8,
        "--dtype",
        "bfloat16",
        "--enable-multimodal",
    ]
    max_tokens = 32768

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
