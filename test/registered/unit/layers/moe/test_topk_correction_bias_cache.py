import unittest

import torch

from sglang.srt.layers.moe.topk import TopKConfig
from sglang.test.ci.ci_register import (
    register_amd_ci,
    register_cpu_ci,
    register_cuda_ci,
)
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=7, suite="base-a-test-cpu")
register_cuda_ci(est_time=10, stage="base-a", runner_config="1-gpu-small")
register_amd_ci(est_time=5, stage="stage-a", runner_config="1-gpu-small-amd")


class CorrectionBiasCacheTestMixin:
    device: str

    def test_lazy_cache_uses_loaded_value_and_reuses_pointer(self):
        correction_bias = torch.empty(8, dtype=torch.bfloat16, device=self.device)
        config = TopKConfig(top_k=2, correction_bias=correction_bias)

        loaded_value = torch.linspace(-1, 1, 8, dtype=torch.float32).to(
            device=self.device, dtype=torch.bfloat16
        )
        with torch.no_grad():
            correction_bias.copy_(loaded_value)

        converted = config.correction_bias_for_dtype(torch.float32)
        converted_again = config.correction_bias_for_dtype(torch.float32)

        torch.testing.assert_close(converted, loaded_value.float())
        self.assertEqual(converted.data_ptr(), converted_again.data_ptr())
        self.assertNotEqual(converted.data_ptr(), correction_bias.data_ptr())

    def test_cache_refreshes_if_weight_is_reloaded(self):
        correction_bias = torch.zeros(8, dtype=torch.bfloat16, device=self.device)
        config = TopKConfig(top_k=2, correction_bias=correction_bias)
        converted = config.correction_bias_for_dtype(torch.float32)

        reloaded_value = torch.arange(8, dtype=torch.float32).to(
            device=self.device, dtype=torch.bfloat16
        )
        with torch.no_grad():
            correction_bias.copy_(reloaded_value)
        converted_after_reload = config.correction_bias_for_dtype(torch.float32)

        torch.testing.assert_close(converted_after_reload, reloaded_value.float())
        self.assertNotEqual(converted.data_ptr(), converted_after_reload.data_ptr())

    def test_matching_dtype_returns_original_tensor(self):
        correction_bias = torch.randn(8, dtype=torch.float32, device=self.device)
        config = TopKConfig(top_k=2, correction_bias=correction_bias)

        result = config.correction_bias_for_dtype(torch.float32)

        self.assertEqual(result.data_ptr(), correction_bias.data_ptr())


class TestCorrectionBiasCacheCPU(CorrectionBiasCacheTestMixin, CustomTestCase):
    device = "cpu"


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestCorrectionBiasCacheGPU(CorrectionBiasCacheTestMixin, CustomTestCase):
    device = "cuda"


if __name__ == "__main__":
    unittest.main()
