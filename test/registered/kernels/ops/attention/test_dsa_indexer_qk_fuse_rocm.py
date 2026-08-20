"""Correctness tests for the ROCm/aiter DSA indexer Q/K fusion.

``indexer_qk_rope_quant_and_cache`` replaces five launches per layer -- Q rope,
K LayerNorm, K rope, fp8 quant of both, and the index-K cache write -- plus the
head-gate scale. Every case here pins that single kernel against the unfused
ROCm path it replaces, because the two must stay interchangeable: the k-only
decode fast path still writes the cache the unfused way, and decode reads back
whatever either wrote.
"""

from __future__ import annotations

import pytest
import torch

from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci

register_amd_ci(est_time=45, suite="jit-kernel-unit-test-amd")

HEAD_DIM = 128
N_HEADS = 32
ROPE_DIM = 64
HALF = ROPE_DIM // 2
PAGE_SIZE = 64
BYTES_PER_TOKEN = HEAD_DIM + 4  # 128 fp8 + 4-byte fp32 scale
CACHE_STRIDE = BYTES_PER_TOKEN
EPS = 1e-5
MAX_POS = 8192
BLOCK_SIZE = 128
SCALE_FMT = "ue8m0"
WEIGHTS_SCALE = HEAD_DIM**-0.5 * N_HEADS**-0.5


def _skip_if_unavailable():
    if not is_hip():
        pytest.skip("aiter indexer Q/K fusion is ROCm-specific")
    if not torch.cuda.is_available():
        pytest.skip("GPU required")
    pytest.importorskip("aiter")
    from aiter.ops import cache as aiter_cache

    if not hasattr(aiter_cache, "indexer_qk_rope_quant_and_cache"):
        pytest.skip("aiter lacks indexer_qk_rope_quant_and_cache")


def _aiter_ops():
    from aiter.ops.cache import (
        indexer_k_quant_and_cache,
        indexer_qk_rope_quant_and_cache,
    )

    return indexer_qk_rope_quant_and_cache, indexer_k_quant_and_cache


def _fp8_dtype():
    from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype

    return fp8_dtype


def _make_inputs(B, n_heads=N_HEADS, seed=0, strided=False):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    cos = torch.randn(MAX_POS, HALF, dtype=torch.bfloat16, device=dev, generator=g)
    sin = torch.randn(MAX_POS, HALF, dtype=torch.bfloat16, device=dev, generator=g)
    positions = torch.randint(0, 4096, (B,), device=dev, dtype=torch.int64, generator=g)
    q = torch.randn(B, n_heads, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=g)
    # The real inputs are slices of one wk_weights_proj GEMM output, so they are
    # strided; the fused path passes them through without a contiguous copy.
    kw = torch.randn(
        B, HEAD_DIM + n_heads, dtype=torch.bfloat16, device=dev, generator=g
    )
    key, weights_raw = kw[:, :HEAD_DIM], kw[:, HEAD_DIM:]
    if not strided:
        key, weights_raw = key.contiguous(), weights_raw.contiguous()
    # The kernel requires fp32 norm params (cache_kernels.cu), which is why the
    # Indexer builds k_norm in fp32 whenever fusion is on.
    norm_weight = torch.randn(HEAD_DIM, dtype=torch.float32, device=dev, generator=g)
    norm_bias = torch.randn(HEAD_DIM, dtype=torch.float32, device=dev, generator=g)
    return cos, sin, positions, q, key, weights_raw, norm_weight, norm_bias


def _make_cache(B, seed=0):
    g = torch.Generator(device="cuda").manual_seed(seed + 991)
    loc = torch.randperm(B * 4, device="cuda", generator=g)[:B].to(torch.int64)
    num_pages = int(loc.max().item()) // PAGE_SIZE + 2
    buf = torch.zeros(
        num_pages, CACHE_STRIDE * PAGE_SIZE, dtype=torch.uint8, device="cuda"
    )
    return buf, loc, num_pages


def _cache_view(buf, page_size=PAGE_SIZE):
    return buf.view(-1, page_size, CACHE_STRIDE).view(_fp8_dtype())


def _rope_interleaved(x, cos_p, sin_p):
    """Interleaved (indexer_rope_interleave) rope on the leading ROPE_DIM dims."""
    x = x.clone()
    xr = x[..., 0:ROPE_DIM:2].clone()
    xi = x[..., 1:ROPE_DIM:2].clone()
    x[..., 0:ROPE_DIM:2] = xr * cos_p - xi * sin_p
    x[..., 1:ROPE_DIM:2] = xr * sin_p + xi * cos_p
    return x


def _run_fused(inputs, buf, loc, preshuffle, page_size=PAGE_SIZE):
    fused, _ = _aiter_ops()
    cos, sin, positions, q, key, weights_raw, norm_weight, norm_bias = inputs
    q_fp8 = torch.empty(q.shape, dtype=_fp8_dtype(), device=q.device)
    weights = torch.empty(
        (q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device
    )
    fused(
        q,
        q_fp8,
        weights_raw,
        weights,
        key,
        _cache_view(buf, page_size),
        loc,
        norm_weight,
        norm_bias,
        positions,
        cos,
        sin,
        EPS,
        BLOCK_SIZE,
        SCALE_FMT,
        WEIGHTS_SCALE,
        preshuffle=preshuffle,
        is_neox=False,
    )
    torch.cuda.synchronize()
    return q_fp8, weights


def _unfused_k_cache(inputs, loc, npages, preshuffle, page_size):
    """What the k-only decode path writes: bf16 LayerNorm+rope, then a quant kernel."""
    _, k_quant_and_cache = _aiter_ops()
    cos, sin, positions, _, key, _, norm_weight, norm_bias = inputs
    normed = torch.nn.functional.layer_norm(
        key.float(), (HEAD_DIM,), weight=norm_weight, bias=norm_bias, eps=EPS
    )
    key_bf16 = _rope_interleaved(
        normed, cos[positions].float(), sin[positions].float()
    ).to(torch.bfloat16)
    buf = torch.zeros(
        npages, CACHE_STRIDE * page_size, dtype=torch.uint8, device="cuda"
    )
    k_quant_and_cache(
        key_bf16,
        buf.view(-1, page_size, CACHE_STRIDE).view(_fp8_dtype()),
        loc,
        BLOCK_SIZE,
        SCALE_FMT,
        preshuffle=preshuffle,
    )
    torch.cuda.synchronize()
    return buf


@pytest.mark.parametrize("preshuffle", [False, True])
def test_k_cache_agrees_with_the_unfused_writer(preshuffle):
    """Both writers stay live -- the k-only decode path uses the unfused one --
    and decode reads back whichever wrote the token, so they must agree.

    Not byte-identical by construction: the unfused path rounds to bf16 before
    quantizing, the fused kernel goes fp32 -> fp8 in one step, so a value near an
    fp8 midpoint can land on either neighbour.
    """
    _skip_if_unavailable()
    B = 41
    inputs = _make_inputs(B)
    buf_fused, loc, npages = _make_cache(B)
    _run_fused(inputs, buf_fused, loc, preshuffle)
    buf_unfused = _unfused_k_cache(inputs, loc, npages, preshuffle, PAGE_SIZE)

    differing = int((buf_fused != buf_unfused).sum())
    assert differing < 0.01 * buf_fused.numel(), (
        f"{differing}/{buf_fused.numel()} bytes differ; double rounding alone "
        "moves far fewer than 1%"
    )


def test_k_cache_is_within_one_fp8_step_of_the_unfused_writer():
    """The flat page_size=1 layout is the one this test can decode, so the exact
    numeric bound lives here and the paged layouts get the byte-fraction check."""
    _skip_if_unavailable()
    B = 41
    page_size = 1
    inputs = _make_inputs(B, seed=11)
    loc = torch.arange(B, device="cuda", dtype=torch.int64)
    npages = B + 2
    buf_fused = torch.zeros(
        npages, CACHE_STRIDE * page_size, dtype=torch.uint8, device="cuda"
    )
    _run_fused(inputs, buf_fused, loc, preshuffle=False, page_size=page_size)
    buf_unfused = _unfused_k_cache(inputs, loc, npages, False, page_size)

    def decode(buf):
        v = buf.view(-1, CACHE_STRIDE)[:B]
        payload = v[:, :HEAD_DIM].contiguous().view(_fp8_dtype()).float()
        scale = v[:, HEAD_DIM:].contiguous().view(torch.float32)
        return payload * scale, scale

    fused, scale_fused = decode(buf_fused)
    unfused, scale_unfused = decode(buf_unfused)

    # A differing ue8m0 exponent would rescale a whole token, which is a bigger
    # claim than double rounding can make.
    assert torch.equal(scale_fused, scale_unfused)
    # Adjacent fp8-e4m3 codes are at most 1/8 apart in relative terms (3 mantissa
    # bits, worst case at the bottom of a binade); one scale step covers zero.
    assert (
        (fused - unfused).abs() <= 0.125 * unfused.abs() + scale_unfused
    ).all(), f"max deviation {(fused - unfused).abs().max().item()}"


def test_q_and_head_gate_match_unfused():
    """q_fp8 and the folded head gate must match rope + act_quant + _scale_head_gates."""
    _skip_if_unavailable()
    from sglang.kernels.ops.attention.dsa.tilelang_kernel import act_quant

    B = 37
    inputs = _make_inputs(B, seed=3)
    cos, sin, positions, q, _, weights_raw, _, _ = inputs
    buf, loc, _ = _make_cache(B, seed=3)
    q_fp8, weights = _run_fused(inputs, buf, loc, preshuffle=True)

    cp = cos[positions].float()[:, None, :]
    sp = sin[positions].float()[:, None, :]
    q_roped = _rope_interleaved(q.float(), cp, sp).to(torch.bfloat16)
    q_fp8_ref, q_scale_ref = act_quant(q_roped, BLOCK_SIZE, SCALE_FMT)
    torch.cuda.synchronize()

    # ue8m0 rounds the scale to a power of two, so both paths must land on the
    # identical scale; only the fp8 payload may differ by a rounding step.
    weights_ref = weights_raw.float() * WEIGHTS_SCALE * q_scale_ref.squeeze(-1)
    torch.testing.assert_close(weights, weights_ref, atol=0, rtol=1e-6)

    deq = q_fp8.float() * q_scale_ref
    deq_ref = q_fp8_ref.float() * q_scale_ref
    # fp8-e4m3 has 3 mantissa bits: one rounding step is 1/16 relative.
    err = (deq - deq_ref).abs()
    assert (
        err <= 0.0625 * deq_ref.abs() + q_scale_ref
    ).all(), f"max fp8 mismatch {err.max().item()}"


def test_strided_inputs_match_contiguous():
    """The no-copy path: key/weights_raw as wk_weights_proj slices, not copies."""
    _skip_if_unavailable()
    B = 29
    strided = _make_inputs(B, seed=7, strided=True)
    contig = _make_inputs(B, seed=7, strided=False)
    assert not strided[4].is_contiguous() and not strided[5].is_contiguous()

    buf_a, loc, _ = _make_cache(B, seed=7)
    buf_b = torch.zeros_like(buf_a)
    q_fp8_a, w_a = _run_fused(strided, buf_a, loc, preshuffle=True)
    q_fp8_b, w_b = _run_fused(contig, buf_b, loc, preshuffle=True)

    assert torch.equal(buf_a, buf_b)
    assert torch.equal(q_fp8_a, q_fp8_b)
    assert torch.equal(w_a, w_b)


def test_cos_sin_view_tracks_a_replaced_rope_cache():
    """aiter keeps cos/sin as [max_position, 1, 1, dim/2]; the kernel needs 2-D.

    Read live, not cached at __init__, so a rope module whose buffers were
    replaced (a grown cache) is picked up.
    """
    from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer

    class DummyRotary:
        pass

    indexer = Indexer.__new__(Indexer)
    indexer.rotary_emb = DummyRotary()
    old_cos = torch.randn(16, 1, 1, HALF, dtype=torch.bfloat16)
    indexer.rotary_emb.cos_cache = old_cos
    indexer.rotary_emb.sin_cache = torch.randn(16, 1, 1, HALF, dtype=torch.bfloat16)
    cos, sin = indexer._aiter_indexer_cos_sin()
    assert cos.shape == (16, HALF) and sin.shape == (16, HALF)
    assert cos.data_ptr() == old_cos.data_ptr()

    grown = torch.randn(128, 1, 1, HALF, dtype=torch.bfloat16)
    indexer.rotary_emb.cos_cache = grown
    indexer.rotary_emb.sin_cache = torch.randn(128, 1, 1, HALF, dtype=torch.bfloat16)
    cos, _ = indexer._aiter_indexer_cos_sin()
    assert cos.shape == (128, HALF) and cos.data_ptr() == grown.data_ptr()
