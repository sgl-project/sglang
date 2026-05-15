import unittest

from sglang.test.ascend.test_ascend_utils import DOTS_OCR_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(
    est_time=400,
    suite="nightly-1-npu-a3",
    nightly=True,
    disabled="https://github.com/Ascend/sglang/issues/81",
)


class TestDeepseekVl2(TestVLMModels):
    """Testcase: Verify that the inference accuracy of the rednote-hilab/dots.ocr model on the MMMU dataset is no less than 0.2.

    [Test Category] Model
    [Test Target] rednote-hilab/dots.ocr
    """

    model = DOTS_OCR_WEIGHTS_PATH
    mmmu_accuracy = 0.2
    other_args = [
        "--max-prefill-tokens",
        "40960",
        "--chunked-prefill-size",
        "40960",
        "--attention-backend",
        "ascend",
        "--mem-fraction-static",
        "0.7",
        "--mm-attention-backend",
        "ascend_attn",
        "--max-running-requests",
        "80",
        "--disable-radix-cache",
        "--cuda-graph-bs",
        1,
        4,
        8,
        16,
        24,
        32,
        40,
        60,
        80,
        "--enable-multimodal",
        "--sampling-backend",
        "ascend",
    ]

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
