"""Fused decode mHC post + combine + RMSNorm (+ MXFP8) against the unfused kernels."""

import sys

import pytest
import torch

from sglang.srt.runtime_context import get_platform
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.version.cuda is None
    or not get_platform().is_blackwell,
    reason="the MXFP8 reference quantizer is Blackwell-only",
)

HIDDEN = 5120
EPS = 1e-6


def _inputs(m: int):
    g = torch.Generator(device="cuda").manual_seed(m)
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


def _transition_rows():
    from sglang.kernels.ops.layernorm.mhc_post_combine_norm_decode import (
        MAX_ROWS,
        _rows_per_wave,
    )

    rows = {1, MAX_ROWS}
    for threads in (640, 320):
        b = _rows_per_wave(torch.cuda.current_device(), threads)
        rows |= {b, b + 1}
    return sorted(r for r in rows if r <= MAX_ROWS)


@pytest.mark.parametrize("quant", [False, True])
def test_matches_unfused_across_thread_transitions(quant: bool):
    from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm
    from sglang.kernels.ops.layernorm.mhc_post_combine_norm_decode import (
        mhc_post_combine_norm_decode,
    )
    from sglang.kernels.ops.layernorm.mhc_post_split_h import mhc_post_split_h
    from sglang.srt.layers.quantization.fp8_utils import flashinfer_mxfp8_quantize

    for m in _transition_rows():
        x, residual, post, comb, pre, weight = _inputs(m)
        updated_ref = mhc_post_split_h(x, residual, post, comb)
        y_ref = hc_combine_norm(updated_ref.flatten(1), pre, weight, EPS)

        updated, y, q, sf = mhc_post_combine_norm_decode(
            x, residual, post, comb, pre, weight, EPS, quant=quant
        )
        assert torch.equal(updated, updated_ref), m
        torch.testing.assert_close(y.float(), y_ref.float(), rtol=1e-2, atol=1e-2)
        assert (y != y_ref).float().mean().item() < 1e-2, m
        if quant:
            q_ref, sf_ref = flashinfer_mxfp8_quantize(y, True, 32)
            assert torch.equal(q.view(torch.uint8), q_ref.view(torch.uint8)), m
            assert torch.equal(sf, sf_ref.reshape(-1)), m
        else:
            assert q is None and sf is None


def test_gate_skips_batch_invariant_and_large_batches(monkeypatch):
    import sglang.srt.batch_invariant_ops as bio
    from sglang.kernels.ops.layernorm.mhc_post_combine_norm_decode import MAX_ROWS
    from sglang.srt.models.deepseek_v4_mhc import (
        HcConfig,
        _can_use_decode_post_combine,
    )

    cfg = HcConfig(4, 20, 1e-6, 1e-6, HIDDEN, True, False)
    monkeypatch.setattr(bio, "is_batch_invariant_mode_enabled", lambda: False)
    assert _can_use_decode_post_combine(cfg, MAX_ROWS)
    assert not _can_use_decode_post_combine(cfg, MAX_ROWS + 1)
    assert not _can_use_decode_post_combine(cfg, 0)
    monkeypatch.setattr(bio, "is_batch_invariant_mode_enabled", lambda: True)
    assert not _can_use_decode_post_combine(cfg, 1)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
