"""Regression test: top-k renormalization must not emit NaN for degenerate tokens.

With sigmoid scoring plus a selection bias (DeepSeek noaux_tc-style gates),
experts are selected by `sigmoid(logits) + bias` but weighted by the raw
sigmoid. A token whose router logits are all deeply negative (< ~-88) has
every gathered sigmoid weight underflow to exactly 0.0, so a bare
`weights / weights.sum()` renormalization computes 0/0 = NaN and poisons the
token's whole output row (observed in production as '!'-spam / NaN logits,
see sgl-project/sglang#30989). The DeepSeek reference gate guards this with
`sum + 1e-20` (modeling_deepseek.py), as does flashinfer's trtllm routing
(flashinfer-ai/flashinfer#3803); these torch implementations must match.
"""

import unittest

import torch

from sglang.srt.layers.moe.topk import (
    biased_grouped_topk_impl,
    biased_topk_impl,
    fused_topk_torch_native,
    grouped_topk_gpu,
    kimi_k2_biased_topk_impl,
)
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b", runner_config="1-gpu-large")
register_amd_ci(est_time=60, stage="stage-b", runner_config="1-gpu-small-amd")

torch.manual_seed(1234)

NUM_EXPERTS = 256
TOPK = 8
HIDDEN = 64


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestTopkRenormalizeDegenerate(CustomTestCase):
    DEVICE = "cuda"

    def _inputs(self, num_tokens=4, degenerate_rows=(0,), dtype=torch.float32):
        """Rows in `degenerate_rows` get all-very-negative logits so every
        selected expert's sigmoid weight underflows to exactly zero. In
        float16 the sigmoid already underflows below logits of about -18."""
        hidden = torch.randn(num_tokens, HIDDEN, device=self.DEVICE, dtype=dtype)
        logits = torch.randn(num_tokens, NUM_EXPERTS, device=self.DEVICE)
        for r in degenerate_rows:
            logits[r] = -100.0 + torch.rand(NUM_EXPERTS, device=self.DEVICE)
        bias = 11.2 + torch.rand(NUM_EXPERTS, device=self.DEVICE) * 0.1
        return hidden, logits.to(dtype), bias.to(dtype)

    def _check(self, topk_weights, name):
        self.assertFalse(
            torch.isnan(topk_weights).any().item(),
            f"{name}: renormalized top-k weights contain NaN",
        )
        self.assertTrue(
            torch.isfinite(topk_weights).all().item(),
            f"{name}: renormalized top-k weights are not finite",
        )
        # healthy rows (>=1) must still renormalize to 1
        row_sums = topk_weights.float().sum(dim=-1)
        self.assertTrue(
            torch.allclose(row_sums[1:], torch.ones_like(row_sums[1:]), atol=1e-3),
            f"{name}: healthy rows no longer sum to 1: {row_sums.tolist()}",
        )

    def test_biased_topk_impl(self):
        hidden, logits, bias = self._inputs()
        weights, _ = biased_topk_impl(hidden, logits, bias, topk=TOPK, renormalize=True)
        self._check(weights, "biased_topk_impl")

    def test_biased_topk_impl_fp16(self):
        # the 1e-20 epsilon underflows to zero in float16; the renormalization
        # must run in float32 for the guard to hold (fp16 sigmoid already
        # underflows below logits of about -18)
        hidden, logits, bias = self._inputs(dtype=torch.float16)
        weights, _ = biased_topk_impl(hidden, logits, bias, topk=TOPK, renormalize=True)
        self._check(weights, "biased_topk_impl[fp16]")

    def test_fused_topk_torch_native_sigmoid_bias_fp16(self):
        hidden, logits, bias = self._inputs(dtype=torch.float16)
        weights, _ = fused_topk_torch_native(
            hidden,
            logits,
            topk=TOPK,
            renormalize=True,
            correction_bias=bias,
            scoring_func="sigmoid",
        )
        self._check(weights, "fused_topk_torch_native[fp16]")

    def test_biased_grouped_topk_impl(self):
        hidden, logits, bias = self._inputs()
        weights, _ = biased_grouped_topk_impl(
            hidden,
            logits,
            bias,
            topk=TOPK,
            renormalize=True,
            num_expert_group=8,
            topk_group=4,
        )
        self._check(weights, "biased_grouped_topk_impl")

    def test_kimi_k2_biased_topk_impl(self):
        hidden, logits, bias = self._inputs()
        weights, _ = kimi_k2_biased_topk_impl(
            hidden, logits, bias, topk=TOPK, renormalize=True
        )
        self._check(weights, "kimi_k2_biased_topk_impl")

    def test_fused_topk_torch_native_sigmoid_bias(self):
        hidden, logits, bias = self._inputs()
        weights, _ = fused_topk_torch_native(
            hidden,
            logits,
            topk=TOPK,
            renormalize=True,
            correction_bias=bias,
            scoring_func="sigmoid",
        )
        self._check(weights, "fused_topk_torch_native")

    def test_grouped_topk_gpu_sigmoid(self):
        hidden, logits, _ = self._inputs()
        weights, _ = grouped_topk_gpu(
            hidden,
            logits,
            topk=TOPK,
            renormalize=True,
            num_expert_group=8,
            topk_group=4,
            scoring_func="sigmoid",
        )
        self._check(weights, "grouped_topk_gpu")


if __name__ == "__main__":
    unittest.main()
