import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _dtypes_reaching_aiter(bias_dtype, device):
    """(logits, bias) dtypes the aiter router receives from select_experts."""
    seen = {}

    def fake_aiter_biased_grouped_topk(gating_output, correction_bias, w, ids, *args):
        seen["dtypes"] = (gating_output.dtype, correction_bias.dtype)
        w.zero_()
        ids.zero_()

    config = TopKConfig(
        top_k=2,
        use_grouped_topk=True,
        topk_group=1,
        num_expert_group=1,
        correction_bias=torch.randn(16, dtype=bias_dtype, device=device),
    )
    with (
        patch.object(topk_module, "_use_aiter", True),
        patch.object(topk_module, "_is_cuda", False),
        patch.object(
            topk_module, "biased_grouped_topk", topk_module.biased_grouped_topk_gpu
        ),
        patch.object(
            topk_module,
            "aiter_biased_grouped_topk",
            fake_aiter_biased_grouped_topk,
            create=True,
        ),
    ):
        select_experts(
            hidden_states=torch.randn(4, 8, dtype=torch.bfloat16, device=device),
            router_logits=torch.randn(4, 16, dtype=torch.bfloat16, device=device),
            topk_config=config,
        )
    return seen["dtypes"]


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

    def test_fp32_bias_routes_in_fp32_on_aiter(self):
        """bf16 logits must not drag an fp32 bias down to bf16 before aiter."""
        self.assertEqual(
            _dtypes_reaching_aiter(torch.float32, self.device),
            (torch.float32, torch.float32),
        )

    def test_other_bias_routes_in_logits_dtype_on_aiter(self):
        self.assertEqual(
            _dtypes_reaching_aiter(torch.float16, self.device),
            (torch.bfloat16, torch.bfloat16),
        )


class TestCorrectionBiasCacheCPU(CorrectionBiasCacheTestMixin, CustomTestCase):
    device = "cpu"


if __name__ == "__main__":
    unittest.main()
