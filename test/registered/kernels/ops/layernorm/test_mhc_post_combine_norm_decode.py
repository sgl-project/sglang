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


def _transition_rows(quant: bool):
    from sglang.kernels.ops.layernorm.mhc_post_combine_norm_decode import (
        MAX_ROWS,
        SPLIT2_MAX_ROWS,
        SPLIT4_MAX_ROWS,
        _num_sms,
        _rows_per_wave,
    )

    device = torch.cuda.current_device()
    rows = {1, MAX_ROWS}
    for b in (SPLIT4_MAX_ROWS, SPLIT2_MAX_ROWS, _num_sms(device)):
        rows |= {b, b + 1}
    for threads in (640, 320, 160):
        b = _rows_per_wave(device, threads, quant)
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

    for m in _transition_rows(quant):
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


@pytest.mark.parametrize("block_m", [64, 128])
@pytest.mark.parametrize("m", [1, 37, 64, 65, 180, 257])
def test_moe_input_matches_mega_moe_pre_dispatch(m: int, block_m: int):
    from sglang.kernels.ops.layernorm.mhc_post_combine_norm_decode import (
        mhc_post_combine_norm_decode,
    )
    from sglang.kernels.ops.moe.dsv4 import mega_moe_pre_dispatch

    def buffers():
        x = torch.zeros((m, HIDDEN), device="cuda", dtype=torch.float8_e4m3fn)
        sf = torch.zeros((m, HIDDEN // 128), device="cuda", dtype=torch.int32)
        shared_rows = (m + block_m - 1) // block_m * ((block_m + 127) // 128 * 128)
        shared_sf = torch.zeros(
            (HIDDEN // 128, shared_rows), device="cuda", dtype=torch.int32
        ).t()
        return x, sf, shared_sf

    x, residual, post, comb, pre, weight = _inputs(m)
    moe_x, moe_sf, moe_shared_sf = buffers()
    updated, y, _, _ = mhc_post_combine_norm_decode(
        x,
        residual,
        post,
        comb,
        pre,
        weight,
        EPS,
        moe_input=(moe_x, moe_sf, moe_shared_sf, block_m),
    )
    updated_ref, y_ref, _, _ = mhc_post_combine_norm_decode(
        x, residual, post, comb, pre, weight, EPS
    )
    assert torch.equal(updated, updated_ref) and torch.equal(y, y_ref)

    ref_x, ref_sf, ref_shared_sf = buffers()
    top_k = 6
    mega_moe_pre_dispatch(
        y_ref,
        torch.empty((m, top_k), device="cuda", dtype=torch.int32),
        torch.empty((m, top_k), device="cuda", dtype=torch.float32),
        ref_x,
        ref_sf,
        torch.empty((m, top_k), device="cuda", dtype=torch.int64),
        torch.empty((m, top_k), device="cuda", dtype=torch.float32),
        shared_x_sf=ref_shared_sf,
        shared_block_m=block_m,
        copy_topk=False,
    )
    assert torch.equal(moe_x.view(torch.uint8), ref_x.view(torch.uint8))
    assert torch.equal(moe_sf, ref_sf)
    assert torch.equal(moe_shared_sf, ref_shared_sf)


def test_gate_skips_batch_invariant_and_large_batches(monkeypatch):
    import sglang.srt.models.deepseek_v4_mhc as mhc
    from sglang.kernels.ops.layernorm.mhc_post_combine_norm_decode import MAX_ROWS
    from sglang.srt.models.deepseek_v4_mhc import (
        HcConfig,
        _can_use_decode_post_combine,
    )

    cfg = HcConfig(4, 20, 1e-6, 1e-6, HIDDEN, True, False)
    monkeypatch.setattr(mhc, "is_batch_invariant_mode_enabled", lambda: False)
    assert _can_use_decode_post_combine(cfg, MAX_ROWS)
    assert not _can_use_decode_post_combine(cfg, MAX_ROWS + 1)
    assert not _can_use_decode_post_combine(cfg, 0)
    monkeypatch.setattr(mhc, "is_batch_invariant_mode_enabled", lambda: True)
    assert not _can_use_decode_post_combine(cfg, 1)


def test_mega_mhc_follows_local_extend_mode():
    from types import SimpleNamespace

    from sglang.srt.model_executor.forward_mode import ForwardMode
    from sglang.srt.models.deepseek_v4_mhc import is_local_extend

    def batch(mode):
        return SimpleNamespace(forward_mode=mode)

    assert is_local_extend(batch(ForwardMode.EXTEND))
    assert not is_local_extend(batch(ForwardMode.TARGET_VERIFY))
    assert not is_local_extend(batch(ForwardMode.DECODE))
    assert not is_local_extend(None)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
