import unittest
from unittest.mock import patch

import torch

from sglang.kernels.ops.moe import moe_route_radix4
from sglang.srt.layers.moe import topk as topk_module
from sglang.srt.layers.moe.topk import TopKConfig, select_experts
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


def _routing_inputs(bias_dtype, device, *, num_tokens=4, radix4=False):
    seen = {}

    def aiter_kernel(logits, bias, weights, ids, *args):
        seen.update(route="aiter", dtypes=(logits.dtype, bias.dtype))
        weights.zero_()
        ids.zero_()

    def radix4_kernel(logits, bias, topk, *args):
        seen.update(route="radix4", dtypes=(logits.dtype, bias.dtype), bias=bias)
        return (
            torch.zeros((num_tokens, topk), device=device),
            torch.zeros((num_tokens, topk), dtype=torch.int32, device=device),
        )

    num_experts, topk = (896, 16) if radix4 else (16, 2)
    config = TopKConfig(
        top_k=topk,
        use_grouped_topk=True,
        topk_group=1,
        num_expert_group=1,
        correction_bias=torch.randn(num_experts, dtype=bias_dtype, device=device),
    )
    with (
        get_context().override_server_args(),
        topk_module.envs.SGLANG_OPT_USE_JIT_KERNEL_GROUPED_TOPK.override(False),
        topk_module.envs.SGLANG_K3_RADIX4_TOPK.override(radix4),
        patch.object(topk_module, "_use_aiter", True),
        patch.object(topk_module, "_is_cuda", False),
        patch.object(
            topk_module, "biased_grouped_topk", topk_module.biased_grouped_topk_gpu
        ),
        patch.object(
            topk_module, "aiter_biased_grouped_topk", aiter_kernel, create=True
        ),
        patch.object(moe_route_radix4, "available", return_value=True),
        patch.object(moe_route_radix4, "route_radix4", radix4_kernel),
    ):
        select_experts(
            hidden_states=torch.zeros(
                num_tokens, 8, dtype=torch.bfloat16, device=device
            ),
            router_logits=torch.zeros(
                num_tokens, num_experts, dtype=torch.bfloat16, device=device
            ),
            topk_config=config,
        )
    seen["config"] = config
    return seen


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
            _routing_inputs(torch.float32, self.device)["dtypes"],
            (torch.float32, torch.float32),
        )

    def test_other_bias_routes_in_logits_dtype_on_aiter(self):
        self.assertEqual(
            _routing_inputs(torch.float16, self.device)["dtypes"],
            (torch.bfloat16, torch.bfloat16),
        )

    def test_radix4_routes_fp32_bias_in_logits_dtype(self):
        """radix4 takes the cached bf16 bias with bf16 logits, so nothing is cast
        per call; past its batch limit aiter still gets the fp32 bias."""
        cases = (
            (1024, "radix4", (torch.bfloat16, torch.bfloat16)),
            (1025, "aiter", (torch.float32, torch.float32)),
        )
        for num_tokens, route, dtypes in cases:
            with self.subTest(num_tokens=num_tokens, route=route):
                seen = _routing_inputs(
                    torch.float32, self.device, num_tokens=num_tokens, radix4=True
                )
                self.assertEqual(seen["route"], route)
                self.assertEqual(seen["dtypes"], dtypes)
                if route == "radix4":
                    cached = seen["config"].correction_bias_for_dtype(torch.bfloat16)
                    self.assertIs(seen["bias"], cached)


class TestCorrectionBiasCacheCPU(CorrectionBiasCacheTestMixin, CustomTestCase):
    device = "cpu"


if __name__ == "__main__":
    unittest.main()
