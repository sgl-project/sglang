"""On ROCm without aiter, RMSNorm and GemmaRMSNorm with a residual must call vLLM's
fused_add_rms_norm with its in-place (input, residual, weight, epsilon) signature
and match forward_native."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers import layernorm
from sglang.srt.layers.layernorm import GemmaRMSNorm, RMSNorm
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

HIDDEN = 64


def _vllm_fused_add_rms_norm(
    input: torch.Tensor, residual: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    # Reference for vllm._custom_ops.fused_add_rms_norm: residual += input, then
    # input = rms_norm(residual) * weight, both in place.
    residual.add_(input)
    r = residual.float()
    normed = r * torch.rsqrt(r.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    input.copy_((normed * weight.float()).to(input.dtype))


def _vllm_rms_norm(
    out: torch.Tensor, input: torch.Tensor, weight: torch.Tensor, epsilon: float
) -> None:
    x = input.float()
    normed = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon)
    out.copy_((normed * weight.float()).to(out.dtype))


class TestLayerNormHipVllm(CustomTestCase):
    def setUp(self):
        self.patches = [
            patch.object(layernorm, "_use_aiter", False),
            patch.object(layernorm, "_has_vllm_rms_norm", True),
            patch.object(
                layernorm, "fused_add_rms_norm", _vllm_fused_add_rms_norm, create=True
            ),
            patch.object(layernorm, "rms_norm", _vllm_rms_norm, create=True),
        ]
        for p in self.patches:
            p.start()
        torch.manual_seed(0)

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()

    def _check(self, layer, num_tokens, post):
        x = torch.randn(num_tokens, HIDDEN)
        residual = torch.randn(num_tokens, HIDDEN)
        post_add = torch.randn(num_tokens, HIDDEN) if post else None
        ref_out, ref_residual = layer.forward_native(
            x.clone(), residual.clone(), None if post_add is None else post_add.clone()
        )
        out, out_residual = layer.forward_hip(x.clone(), residual.clone(), post_add)
        torch.testing.assert_close(out, ref_out)
        torch.testing.assert_close(out_residual, ref_residual)

    def test_rms_norm_with_residual(self):
        layer = RMSNorm(HIDDEN, eps=1e-6)
        with torch.no_grad():
            layer.weight.normal_(mean=1.0, std=0.1)
        for num_tokens in (0, 7):
            with self.subTest(num_tokens=num_tokens):
                self._check(layer, num_tokens=num_tokens, post=False)

    def test_gemma_rms_norm_with_residual(self):
        layer = GemmaRMSNorm(HIDDEN, eps=1e-6)
        layer.weight.weight_loader(layer.weight, torch.randn(HIDDEN) * 0.1)
        for num_tokens, post in ((0, False), (7, False), (7, True)):
            with self.subTest(num_tokens=num_tokens, post=post):
                self._check(layer, num_tokens=num_tokens, post=post)


if __name__ == "__main__":
    unittest.main()
