import unittest

from sglang.test.ascend.test_ascend_utils import GEMMA_3_4B_IT_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestAscendMMAttentionBackend(TestVLMModels):
    """Testcase: Verify MMMU dataset accuracy ≥0.2 for google/gemma-3-4b-it with --mm-attention-backend ascend_attn.

    [Test Category] Parameter
    [Test Target] --mm-attention-backend
    """

    model = GEMMA_3_4B_IT_WEIGHTS_PATH
    mmmu_accuracy = 0.2
    other_args = [
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "32",
        "--enable-multimodal",
        "--mem-fraction-static",
        0.35,
        "--log-level",
        "info",
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--tp-size",
        4,
        "--mm-attention-backend",
        "ascend_attn",
    ]

    def test_mmmu(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
