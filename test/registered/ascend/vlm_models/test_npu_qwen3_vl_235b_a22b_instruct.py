import unittest

from sglang.test.ascend.test_ascend_utils import (
    QWEN3_VL_235B_A22B_INSTRUCT_WEIGHTS_PATH,
)
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-16-npu-a3", nightly=True)


class TestQwen3VL235BA22B(TestVLMModels):
    """Testcase: Verify that the inference accuracy of the Qwen/Qwen3-VL-235B-A22B-Instruct model on the MMMU dataset is no less than 0.2.

    [Test Category] Model
    [Test Target] Qwen/Qwen3-VL-235B-A22B-Instruct
    """

    model = QWEN3_VL_235B_A22B_INSTRUCT_WEIGHTS_PATH
    mmmu_accuracy = 0.2
    other_args = [
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "32",
        "--enable-multimodal",
        "--mem-fraction-static",
        0.8,
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        16,
    ]
    timeout_for_server_launch = 3000

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
