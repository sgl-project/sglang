import unittest

from test_vlm_utils import TestVLMModels

from sglang.test.ci.ci_register import register_npu_ci

register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)


class TestGemmaModels(TestVLMModels):
    model = "/root/.cache/modelscope/hub/models/openbmb/MiniCPM-V-2_6"
    mmmu_accuracy = 0.2

    def test_vlm_mmmu_benchmark(self):
        self._run_vlm_mmmu_test()


if __name__ == "__main__":
    unittest.main()
