import unittest

from sglang.test.ascend.vlm_utils import TestVLMModels
from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-8-npu-a3", nightly=True)


class TestGLM4Models(TestVLMModels):
    model = "/root/.cache/modelscope/hub/models/ZhipuAI/GLM-4.5V"
    mmmu_accuracy = 0.2
    other_args = [
        "--trust-remote-code",
        "--cuda-graph-max-bs",
        "32",
        "--enable-multimodal",
        "--mem-fraction-static",
        0.7,
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
