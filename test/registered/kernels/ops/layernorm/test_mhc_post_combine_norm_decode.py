"""Fused decode mHC post + combine + RMSNorm (+ MXFP8) against the unfused kernels."""

import sys

import pytest
import torch

from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or not get_platform().is_blackwell,
    reason="the MXFP8 reference quantizer is Blackwell-only",
)

HIDDEN = 5120
EPS = 1e-6


def _inputs(m: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    x = torch.randn((m, HIDDEN), device="cuda", dtype=torch.bfloat16, generator=g)
    residual = (
        torch.randn((m, 4, HIDDEN), device="cuda", dtype=torch.bfloat16, generator=g)
        * 4
    )
    post = torch.rand((m, 4), device="cuda", generator=g) * 2
    comb = torch.softmax(
        torch.randn((m, 4, 4), device="cuda", generator=g), -1
    ).contiguous()
    pre = torch.rand((m, 4), device="cuda", generator=g)
    weight = (torch.rand((HIDDEN,), device="cuda", generator=g) + 0.5).to(
        torch.bfloat16
    )
    return x, residual, post, comb, pre, weight


@pytest.mark.parametrize("m", [1, 6, 48, 128, 129, 198, 336, 384])
@pytest.mark.parametrize("threads", [160, 320, 640])
@pytest.mark.parametrize("quant", [False, True])
def test_matches_unfused(m: int, threads: int, quant: bool):
    from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm
    from sglang.kernels.ops.layernorm.mhc_post_combine_norm_decode import (
        mhc_post_combine_norm_decode,
    )
    from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    x, residual, post, comb, pre, weight = _inputs(m, m * 7 + threads)
    updated_ref = mhc_post_split_h(x, residual, post, comb)
    y_ref = hc_combine_norm(updated_ref.flatten(1), pre, weight, EPS)

    updated, y, q, sf = mhc_post_combine_norm_decode(
        x, residual, post, comb, pre, weight, EPS, quant=quant, threads=threads
    )
    assert torch.equal(updated, updated_ref)
    # Only the RMS statistic's summation order differs from the Triton kernel.
    torch.testing.assert_close(y.float(), y_ref.float(), rtol=1e-2, atol=1e-2)
    assert (y != y_ref).float().mean().item() < 1e-2
    if quant:
        q_ref, sf_ref = flashinfer_mxfp8_quantize(y, True, 32)
        assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8))
        assert torch.equal(sf, sf_ref.reshape(-1))
    else:
        assert q is None and sf is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
