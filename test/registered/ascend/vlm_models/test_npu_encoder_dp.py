import unittest
from types import SimpleNamespace

from sglang.test.ascend.test_ascend_utils import QWEN2_5_VL_72B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=500, suite="nightly-4-gpu", nightly=True)
register_amd_ci(est_time=500, suite="nightly-amd-4-gpu", nightly=True)

MODELS = [
    SimpleNamespace(model=QWEN2_5_VL_72B_INSTRUCT_WEIGHTS_PATH, mmmu_accuracy=0.55),
]


class TestVLMEncoderDP(TestVLMModels):
    model = QWEN2_5_VL_72B_INSTRUCT_WEIGHTS_PATH
    mmmu_accuracy = 0.55
    # --cuda-graph-max-bs 32 last-wins over the kit's default 64.
    other_args = [
        "--mm-enable-dp-encoder",
        "--tp-size",
        "4",
        "--cuda-graph-max-bs-decode",
        "32",
    ]

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
