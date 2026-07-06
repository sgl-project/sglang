"""Correctness tests for the DeepSeek-V3.2 DSA indexer fused kernels.

Covers:
  - fused_q_indexer_rope_first_quant      (Q: rope-first + fp8 quant + head-gate fold)
  - fused_k_indexer_norm_rope             (K: LayerNorm + rope-first -> bf16)
  - fused_k_indexer_norm_rope_store       (K: the above + fp8 quant + paged index-k cache write)

The store kernel is checked for byte-exact equivalence against the un-fused path
(bf16 K kernel + standalone fused_store_index_k_cache), so it needs no fp8
reference. The Q/K math kernels are checked against torch references. Strided
inputs (the non-contiguous wk_weights_proj slices) are checked to match
contiguous inputs (the no-copy path).
"""

from __future__ import annotations

import pytest
import torch

from sglang.jit_kernel.dsv4 import fused_q_indexer_rope_first_quant
from sglang.jit_kernel.dsv32 import (
    fused_k_indexer_norm_rope,
    fused_k_indexer_norm_rope_store,
)
from sglang.jit_kernel.fused_store_index_cache import (
    can_use_dsa_fused_store,
    fused_store_index_k_cache,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_cuda_ci

_is_hip = is_hip()

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=90, suite="nightly-kernel-1-gpu", nightly=True)

HEAD_DIM = 128
ROPE_DIM = 64
HALF = ROPE_DIM // 2
FP8_MAX = 448.0
PAGE_SIZE = 64
BYTES_PER_TOKEN = HEAD_DIM + 4  # 128 fp8 + 4-byte fp32 scale
EPS = 1e-6
MAX_POS = 8192


def _skip_if_unavailable():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if _is_hip:
        pytest.skip("Indexer fused kernels are CUDA-specific")


def _make_inputs(B, seed=0, pos_dtype=torch.int32):
    g = torch.Generator(device="cuda").manual_seed(seed)
    dev = "cuda"
    cos = torch.randn(MAX_POS, HALF, device=dev, generator=g)
    sin = torch.randn(MAX_POS, HALF, device=dev, generator=g)
    cos_sin_cache = torch.cat((cos, sin), dim=-1)
    positions = torch.randint(0, 4096, (B,), device=dev, dtype=pos_dtype, generator=g)
    return cos, sin, cos_sin_cache, positions


def _rope_first(x, cos_p, sin_p):
    """Interleaved complex rope on the leading ROPE_DIM dims (kRopeFirst)."""
    x = x.clone()
    xr = x[..., 0:ROPE_DIM:2].clone()
    xi = x[..., 1:ROPE_DIM:2].clone()
    x[..., 0:ROPE_DIM:2] = xr * cos_p - xi * sin_p
    x[..., 1:ROPE_DIM:2] = xr * sin_p + xi * cos_p
    return x


def _rope_first_neox(x, cos_p, sin_p):
    """NeoX rope on the leading ROPE_DIM dims: dim i pairs with i + HALF."""
    x = x.clone()
    x1 = x[..., 0:HALF].clone()
    x2 = x[..., HALF:ROPE_DIM].clone()
    x[..., 0:HALF] = x1 * cos_p - x2 * sin_p
    x[..., HALF:ROPE_DIM] = x2 * cos_p + x1 * sin_p
    return x


def _rope_ref(x, cos_p, sin_p, is_neox):
    return (
        _rope_first_neox(x, cos_p, sin_p) if is_neox else _rope_first(x, cos_p, sin_p)
    )


# ----------------------------------------------------------------------------
# K kernel (-> bf16): LayerNorm + rope-first
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("is_neox", [False, True])
def test_k_norm_rope_matches_reference(is_neox):
    _skip_if_unavailable()
    dev = "cuda"
    B = 37
    cos, sin, cos_sin_cache, positions = _make_inputs(B)
    key = torch.randn(B, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    weight = torch.randn(HEAD_DIM, dtype=torch.float32, device=dev)
    bias = torch.randn(HEAD_DIM, dtype=torch.float32, device=dev)

    out = fused_k_indexer_norm_rope(
        key, weight, bias, EPS, cos_sin_cache, positions, is_neox=is_neox
    )
    torch.cuda.synchronize()

    normed = torch.nn.functional.layer_norm(
        key.float(), (HEAD_DIM,), weight=weight, bias=bias, eps=EPS
    )
    cp, sp = cos[positions.long()], sin[positions.long()]
    ref = _rope_ref(normed, cp, sp, is_neox)

    torch.testing.assert_close(out.float(), ref, atol=0.06, rtol=0.0)


# ----------------------------------------------------------------------------
# K kernel + fused store == bf16 K kernel + standalone store (byte-exact).
# Also covers the strided (non-contiguous wk slice) no-copy input path.
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("strided", [False, True])
@pytest.mark.parametrize("is_neox", [False, True])
def test_k_store_matches_unfused(strided, is_neox):
    _skip_if_unavailable()
    if not can_use_dsa_fused_store(torch.bfloat16, torch.int64, PAGE_SIZE):
        pytest.skip("fused store JIT unavailable")
    dev = "cuda"
    B, n_heads = 41, 64
    cos, sin, cos_sin_cache, positions = _make_inputs(B)
    weight = torch.randn(HEAD_DIM, dtype=torch.float32, device=dev)
    bias = torch.randn(HEAD_DIM, dtype=torch.float32, device=dev)

    if strided:
        # Mimic the fused wk_weights_proj GEMM output: key is kw[:, :head_dim].
        kw = torch.randn(B, HEAD_DIM + n_heads, dtype=torch.bfloat16, device=dev)
        key = kw[:, :HEAD_DIM]
        assert not key.is_contiguous()
    else:
        key = torch.randn(B, HEAD_DIM, dtype=torch.bfloat16, device=dev)

    loc = torch.randperm(B * 4, device=dev)[:B].to(torch.int64)
    num_pages = int(loc.max().item()) // PAGE_SIZE + 2
    buf_ref = torch.zeros(
        num_pages, BYTES_PER_TOKEN * PAGE_SIZE, dtype=torch.uint8, device=dev
    )
    buf_fused = torch.zeros_like(buf_ref)

    key_bf16 = fused_k_indexer_norm_rope(
        key, weight, bias, EPS, cos_sin_cache, positions, is_neox=is_neox
    )
    fused_store_index_k_cache(key_bf16, buf_ref, loc, PAGE_SIZE)

    fused_k_indexer_norm_rope_store(
        key,
        buf_fused,
        loc,
        weight,
        bias,
        EPS,
        cos_sin_cache,
        positions,
        PAGE_SIZE,
        is_neox=is_neox,
    )
    torch.cuda.synchronize()

    assert torch.equal(buf_ref, buf_fused)


# ----------------------------------------------------------------------------
# Q kernel: rope-first + fp8 quant + head-gate fold
# ----------------------------------------------------------------------------
@pytest.mark.parametrize("pos_dtype", [torch.int32, torch.int64])
@pytest.mark.parametrize("is_neox", [False, True])
def test_q_rope_quant_matches_reference(pos_dtype, is_neox):
    _skip_if_unavailable()
    dev = "cuda"
    B, n_heads = 37, 64
    cos, sin, cos_sin_cache, positions = _make_inputs(B, pos_dtype=pos_dtype)
    q = torch.randn(B, n_heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    weight = torch.randn(B, n_heads, dtype=torch.bfloat16, device=dev)
    weight_scale = 0.137

    q_fp8, weights_out = fused_q_indexer_rope_first_quant(
        q, weight, weight_scale, cos_sin_cache, positions, is_neox=is_neox
    )
    torch.cuda.synchronize()

    cp = cos[positions.long()][:, None, :]
    sp = sin[positions.long()][:, None, :]
    ref = _rope_ref(q.float(), cp, sp, is_neox)  # [B, n_heads, 128]
    amax = ref.abs().amax(dim=-1, keepdim=True)
    scale = torch.clamp(amax, min=1e-4) / FP8_MAX

    # weights_out[b,h] = weight * weight_scale * scale
    w_ref = weight.float() * weight_scale * scale.squeeze(-1)
    torch.testing.assert_close(weights_out.squeeze(-1), w_ref, atol=1e-3, rtol=1e-3)

    # dequantized q should match the rope result within fp8-e4m3 precision:
    # round-to-nearest with 3 mantissa bits => <= 1/16 relative error, plus one
    # scale step at the bottom of the range.
    deq = q_fp8.float() * scale
    err = (deq - ref).abs()
    assert (
        err <= 0.0625 * ref.abs() + scale
    ).all(), f"max fp8 dequant error {err.max().item()}"


# ----------------------------------------------------------------------------
# Strided weight (the wk_weights_proj slice) matches contiguous for the Q kernel
# ----------------------------------------------------------------------------
def test_q_strided_weight_matches_contiguous():
    _skip_if_unavailable()
    dev = "cuda"
    B, n_heads = 29, 64
    cos, sin, cos_sin_cache, positions = _make_inputs(B)
    q = torch.randn(B, n_heads, HEAD_DIM, dtype=torch.bfloat16, device=dev)
    # weights_raw = kw[:, head_dim:] is a non-contiguous slice.
    kw = torch.randn(B, HEAD_DIM + n_heads, dtype=torch.bfloat16, device=dev)
    w_strided = kw[:, HEAD_DIM:]
    w_contig = w_strided.contiguous()
    assert not w_strided.is_contiguous()

    a_fp8, a_w = fused_q_indexer_rope_first_quant(
        q, w_strided, 0.137, cos_sin_cache, positions
    )
    b_fp8, b_w = fused_q_indexer_rope_first_quant(
        q, w_contig, 0.137, cos_sin_cache, positions
    )
    torch.cuda.synchronize()
    assert torch.equal(a_fp8, b_fp8)
    assert torch.equal(a_w, b_w)


# ----------------------------------------------------------------------------
# NeoX top-k checksum: the fused kernels' selected indices must match the
# rotary_emb-based eager reference under is_neox_style=True. This is the guard
# for the DeepSeek-V3.2 accuracy regression: an interleave-only rotation
# corrupts the indexer scores and collapses the top-k overlap to ~1/3.
# ----------------------------------------------------------------------------
def test_neox_topk_matches_rotary_emb_reference():
    _skip_if_unavailable()
    from sglang.srt.layers.rotary_embedding.utils import apply_rotary_emb

    dev = "cuda"
    B, n_heads, n_keys, topk = 64, 4, 512, 64
    g = torch.Generator(device=dev).manual_seed(7)
    # Real rope frequencies (not random cos/sin): convention bugs shear real
    # rotations; random tables can mask the pairing structure.
    inv_freq = 1.0 / (
        10000.0 ** (torch.arange(0, ROPE_DIM, 2, dtype=torch.float) / ROPE_DIM)
    )
    t = torch.arange(MAX_POS, dtype=torch.float)
    freqs = torch.einsum("i,j->ij", t, inv_freq)
    cos_sin_cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).to(dev)

    def eager_neox(x, positions):
        cos, sin = cos_sin_cache.index_select(0, positions).chunk(2, dim=-1)
        x_rot = x[..., :ROPE_DIM].float()
        squeeze = x_rot.dim() == 2
        if squeeze:
            x_rot = x_rot.unsqueeze(1)
        out = apply_rotary_emb(x_rot, cos, sin, True)
        if squeeze:
            out = out.squeeze(1)
        return torch.cat((out, x[..., ROPE_DIM:].float()), dim=-1)

    q = torch.randn(B, n_heads, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=g)
    w = torch.ones(B, n_heads, dtype=torch.bfloat16, device=dev)
    q_pos = torch.randint(1, MAX_POS, (B,), dtype=torch.int64, device=dev, generator=g)
    keys = torch.randn(n_keys, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=g)
    k_pos = torch.arange(n_keys, dtype=torch.int64, device=dev)
    k_weight = torch.randn(HEAD_DIM, dtype=torch.float32, device=dev, generator=g)
    k_bias = torch.randn(HEAD_DIM, dtype=torch.float32, device=dev, generator=g)

    # Fused path (dequantized q so both sides rank in the same scale).
    q_fp8, weights_out = fused_q_indexer_rope_first_quant(
        q, w, 1.0, cos_sin_cache, q_pos, is_neox=True
    )
    q_fused = (q_fp8.float() * weights_out.view(B, n_heads, 1))[:, 0, :]
    k_fused = fused_k_indexer_norm_rope(
        keys, k_weight, k_bias, EPS, cos_sin_cache, k_pos, is_neox=True
    ).float()

    # Eager reference: LayerNorm + apply_rotary_emb(is_neox_style=True), the
    # same math the non-fused indexer path executes via self.rotary_emb.
    normed = torch.nn.functional.layer_norm(
        keys.float(), (HEAD_DIM,), weight=k_weight, bias=k_bias, eps=EPS
    )
    k_eager = eager_neox(normed, k_pos)
    q_eager = eager_neox(q, q_pos)[:, 0, :]

    top_fused = (q_fused @ k_fused.T).topk(topk, dim=-1).indices
    top_eager = (q_eager @ k_eager.T).topk(topk, dim=-1).indices
    overlaps = torch.tensor(
        [
            len(set(top_fused[i].tolist()) & set(top_eager[i].tolist()))
            for i in range(B)
        ],
        dtype=torch.float,
    )
    # fp8 quantization of q can flip near-tied boundary picks; a convention
    # bug collapses overlap to ~topk/3, so these bounds have a wide margin.
    assert overlaps.min() >= topk - 8, f"min overlap {overlaps.min()}/{topk}"
    assert overlaps.mean() >= topk - 4, f"mean overlap {overlaps.mean()}/{topk}"


def test_indexer_uses_replaced_rope_cache_for_fused_kernels():
    _skip_if_unavailable()
    from sglang.srt.layers.attention.dsa.dsa_indexer import Indexer

    dev = "cuda"
    B, n_heads = 7, 64
    old_len = 16
    new_len = 128
    g = torch.Generator(device=dev).manual_seed(123)
    old_cache = torch.randn(old_len, ROPE_DIM, dtype=torch.float32, device=dev)
    cos = torch.randn(new_len, HALF, dtype=torch.float32, device=dev, generator=g)
    sin = torch.randn(new_len, HALF, dtype=torch.float32, device=dev, generator=g)
    grown_cache = torch.cat((cos, sin), dim=-1)
    positions = torch.arange(old_len, old_len + B, device=dev, dtype=torch.int32)

    class DummyRotary:
        pass

    rotary_emb = DummyRotary()
    rotary_emb.cos_sin_cache = old_cache
    indexer = Indexer.__new__(Indexer)
    indexer.rotary_emb = rotary_emb
    assert indexer._indexer_cos_sin_cache.data_ptr() == old_cache.data_ptr()

    rotary_emb.cos_sin_cache = grown_cache
    assert indexer._indexer_cos_sin_cache.data_ptr() == grown_cache.data_ptr()

    key = torch.randn(B, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=g)
    k_weight = torch.randn(HEAD_DIM, dtype=torch.float32, device=dev, generator=g)
    k_bias = torch.randn(HEAD_DIM, dtype=torch.float32, device=dev, generator=g)
    k_out = fused_k_indexer_norm_rope(
        key, k_weight, k_bias, EPS, indexer._indexer_cos_sin_cache, positions
    )

    normed = torch.nn.functional.layer_norm(
        key.float(), (HEAD_DIM,), weight=k_weight, bias=k_bias, eps=EPS
    )
    k_ref = _rope_first(normed, cos[positions.long()], sin[positions.long()])
    torch.testing.assert_close(k_out.float(), k_ref, atol=0.06, rtol=0.0)

    q = torch.randn(B, n_heads, HEAD_DIM, dtype=torch.bfloat16, device=dev, generator=g)
    q_weight = torch.randn(B, n_heads, dtype=torch.bfloat16, device=dev, generator=g)
    weight_scale = 0.137
    q_fp8, weights_out = fused_q_indexer_rope_first_quant(
        q, q_weight, weight_scale, indexer._indexer_cos_sin_cache, positions
    )
    torch.cuda.synchronize()

    q_ref = _rope_first(
        q.float(), cos[positions.long()][:, None, :], sin[positions.long()][:, None, :]
    )
    scale = torch.clamp(q_ref.abs().amax(dim=-1, keepdim=True), min=1e-4) / FP8_MAX
    torch.testing.assert_close(
        weights_out.squeeze(-1),
        q_weight.float() * weight_scale * scale.squeeze(-1),
        atol=1e-3,
        rtol=1e-3,
    )
    assert ((q_fp8.float() * scale - q_ref).abs() <= 0.0625 * q_ref.abs() + scale).all()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
