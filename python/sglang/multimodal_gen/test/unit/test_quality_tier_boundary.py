# SPDX-License-Identifier: Apache-2.0
"""The tier boundary holds for the shapes the request-gated fusions actually use.

`_QUALITY_FUSION_HANDLERS` assigns each fusion a tier from what it does to the
numbers. This pins that reading to measurement: the transformations the
lossless-tier fusions perform pass the admission gate, and the ones this
repository classifies as approximate fail it, on activations shaped like a
DiT's -- outlier channels included, because they are what decides whether a
reassociated reduction stays within the reference's error.

Measured on real FLUX.1-dev MLP activations (4096x3072 bf16, max/p99 = 8.1)
the two groups separate by three to six orders of magnitude, so these verdicts
do not sit near the 1.5x threshold.
"""

import sys
import unittest

import pytest
import torch

from sglang.multimodal_gen.test.quality_tier_admission import (
    assert_error_no_worse_than_reference,
)


def _dit_activations(rows: int, cols: int) -> torch.Tensor:
    """BF16 activations with the outlier channels a DiT block carries."""
    generator = torch.Generator(device="cuda").manual_seed(0)
    x = torch.randn(rows, cols, device="cuda", dtype=torch.float32, generator=generator)
    x[:, ::97] *= 8.0
    return x.to(torch.bfloat16)


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestLosslessTierTransformations(unittest.TestCase):
    """What the lossless-tier fusions do stays inside the gate."""

    def test_a_gemm_epilogue_fusion_is_admitted(self):
        # fused linear+GELU: the cublasLt epilogue rounds once where the
        # reference rounds the GEMM output and then the GELU output.
        x = _dit_activations(1024, 1024)
        weight = _dit_activations(2048, 1024)
        bias = _dit_activations(1, 2048)[0]

        reference = torch.nn.functional.gelu(
            x.double() @ weight.double().t() + bias.double(), approximate="tanh"
        )
        baseline = torch.nn.functional.gelu(
            torch.nn.functional.linear(x, weight, bias), approximate="tanh"
        )
        candidate = torch._addmm_activation(bias, x, weight.t(), use_gelu=True)

        assert_error_no_worse_than_reference(
            reference_fp64=reference,
            baseline=baseline,
            candidate=candidate,
            label="fused linear+GELU",
        )

    def test_a_reassociated_reduction_is_admitted(self):
        x = _dit_activations(512, 2048).float()

        assert_error_no_worse_than_reference(
            reference_fp64=x.double().sum(dim=-1),
            baseline=x.sum(dim=-1),
            candidate=x.reshape(x.shape[0], -1, 2).sum(-1).sum(-1),
            label="reassociated fp32 reduction",
        )

    def test_the_gated_residual_rounding_is_admitted(self):
        # Helios / LingBot: the reference multiplies the FP32 gate and update
        # and rounds once at the end; the fused kernel needs one dtype, so it
        # rounds both to BF16 first. One rounding moves earlier -- which the
        # taxonomy can only call equivalent if it measures that way.
        residual = _dit_activations(1024, 1024)
        update = _dit_activations(1024, 1024).float()
        gate = torch.rand(1024, 1, device="cuda") * 0.2

        assert_error_no_worse_than_reference(
            reference_fp64=residual.double() + update.double() * gate.double(),
            baseline=residual + (update * gate).to(residual.dtype),
            candidate=residual + (update.to(residual.dtype) * gate.to(residual.dtype)),
            label="per-token gated residual",
        )


@unittest.skipUnless(torch.cuda.is_available(), "needs CUDA")
class TestApproximateTransformationsAreRejected(unittest.TestCase):
    """What the high-tier fusions do cannot pass as a rounding difference."""

    def _assert_rejected(self, label, reference_fp64, baseline, candidate):
        with self.assertRaisesRegex(AssertionError, "not lossless-tier"):
            assert_error_no_worse_than_reference(
                reference_fp64=reference_fp64,
                baseline=baseline,
                candidate=candidate,
                label=label,
            )

    def test_norm_statistics_rounded_through_bf16(self):
        # Ideogram's BF16-native gate RMSNorm against F.rms_norm's fp32 stats.
        x = _dit_activations(512, 2048)
        self._assert_rejected(
            "bf16 norm statistics",
            x.double().pow(2).mean(dim=-1),
            x.float().pow(2).mean(dim=-1),
            x.pow(2).mean(dim=-1).float(),
        )

    def test_a_gemm_fed_bf16_where_the_reference_promotes_to_fp32(self):
        # SANA-Video's BF16-input linear attention.
        a = _dit_activations(512, 1024)
        b = _dit_activations(512, 1024).t().contiguous()
        self._assert_rejected(
            "BF16-input attention GEMM",
            a.double() @ b.double(),
            a.float() @ b.float(),
            (a @ b).float(),
        )

    def test_quantizing_before_the_reference_intermediate_exists(self):
        # FLUX.2's NVFP4 FC1+SwiGLU+quant.
        x = _dit_activations(512, 2048)
        scale = x.abs().amax(dim=-1, keepdim=True).float() / 448.0
        quantized = (x.float() / scale).clamp(-448, 448).to(
            torch.float8_e4m3fn
        ).float() * scale
        self._assert_rejected(
            "quantized FC2 input",
            torch.nn.functional.silu(x.double()),
            torch.nn.functional.silu(x.float()),
            torch.nn.functional.silu(quantized),
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
