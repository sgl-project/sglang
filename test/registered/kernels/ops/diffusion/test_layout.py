"""``diffusion.layout``: data-movement kernels.

Every kernel in this domain only moves values (plus zero fill, plus at most
one same-order add), so each one is *bitwise* identical to the aten chain it
replaces.  That makes ``torch.equal`` -- not ``assert_close`` -- the right
assertion throughout this file; a tolerance here would hide a real bug.

Covered: USP output head merge, Ulysses destination-major QKV pack, varlen
pack/scatter, causal Conv3d cat+pad (CUDA and Triton), and the Wan causal-VAE
cache kernels.
"""

import sys
from unittest.mock import patch

import pytest
import torch
import torch.nn.functional as F

from sglang.kernels.jit.utils import get_ci_test_range
from sglang.kernels.ops.attention.flash_attention import flash_attn_varlen_func
from sglang.kernels.ops.diffusion import (
    build_inv_indices,
    can_use_usp_merge_heads,
    cat_pad_channels_last_3d,
    dup_up3d_add,
)
from sglang.kernels.ops.diffusion import (
    fused_causal_conv3d_cat_pad as fused_causal_conv3d_cat_pad_triton,
)
from sglang.kernels.ops.diffusion import (
    fused_causal_conv3d_cat_pad_cuda,
    fused_pack_qkv,
    fused_pack_segmented_qkv,
    fused_scatter_to_padded,
    pack_qkv_destination_major,
    usp_merge_heads,
)
from sglang.multimodal_gen.runtime.layers.attention.backends import (
    flash_attn as _fa_backend,
)
from sglang.multimodal_gen.runtime.layers.attention.layer import build_varlen_mask_meta
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=110, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
# Nightly is not redundant: it sets SGLANG_JIT_KERNEL_RUN_FULL_TESTS=1, which
# expands the get_ci_test_range sweeps below.
register_cuda_ci(est_time=20, stage="nightly", runner_config="1-gpu-large")
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")
register_amd_ci(est_time=15, suite="nightly-amd-kernel-1-gpu", nightly=True)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

DEVICE = "cuda"


def _cl3d(shape, dtype):
    return torch.randn(shape, device=DEVICE, dtype=dtype).contiguous(
        memory_format=torch.channels_last_3d
    )


# ---------------------------------------------------------------------------
# USP output head merge
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "world,seq,batch,h_local,head_dim",
    [
        (4, 7936, 1, 14, 128),  # H3 768p production shape (Ulysses 4)
        (2, 64, 3, 4, 64),  # batched
        (4, 33, 2, 4, 100),  # scalar fallback inside the CUDA kernel
    ],
)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.skipif(
    bool(torch.version.hip),
    reason="the USP merge-heads JIT fast path is CUDA-only by design -- "
    "can_use_usp_merge_heads() returns False under HIP, and the aten fallback "
    "it degrades to is covered by the unsupported-inputs test below",
)
def test_usp_merge_heads_bitwise(dtype, world, seq, batch, h_local, head_dim):
    generator = torch.Generator(device=DEVICE).manual_seed(4321)
    x = torch.randn(
        world,
        seq,
        batch,
        h_local,
        head_dim,
        dtype=dtype,
        device=DEVICE,
        generator=generator,
    )
    assert can_use_usp_merge_heads(x)
    out = usp_merge_heads(x)
    ref = x.permute(2, 1, 0, 3, 4).contiguous()
    assert out.shape == ref.shape
    assert torch.equal(out, ref)


def test_usp_merge_heads_unsupported_inputs_use_exact_fallback():
    # The wrapper degrades to the aten permute for anything the fast path
    # rejects -- a wrong rank, a transposed view, an empty leading dim, or a
    # ROCm build -- so callers never need their own guard.
    x = torch.randn(2, 4, 1, 4, 64, dtype=torch.bfloat16, device=DEVICE)
    for value in (x.transpose(0, 1), x[:0], x[0]):
        assert not can_use_usp_merge_heads(value)
        if value.dim() == 5:
            assert torch.equal(
                usp_merge_heads(value), value.permute(2, 1, 0, 3, 4).contiguous()
            )

    with patch.object(torch.version, "hip", "6.3"):
        assert not can_use_usp_merge_heads(x)
        assert torch.equal(usp_merge_heads(x), x.permute(2, 1, 0, 3, 4).contiguous())


# ---------------------------------------------------------------------------
# Ulysses destination-major QKV pack
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_pack_qkv_destination_major_is_bit_exact(dtype):
    torch.manual_seed(0)
    rows, world_size, global_heads, head_size = 17, 4, 12, 64
    q, k, v = (
        torch.randn(rows, global_heads, head_size, device=DEVICE, dtype=dtype)
        for _ in range(3)
    )

    local_heads = global_heads // world_size
    expected = torch.empty(
        world_size, rows, local_heads, 3 * head_size, device=DEVICE, dtype=dtype
    )
    for index, tensor in enumerate((q, k, v)):
        shards = tensor.view(rows, world_size, local_heads, head_size).permute(
            1, 0, 2, 3
        )
        expected[..., index * head_size : (index + 1) * head_size].copy_(shards)

    assert torch.equal(pack_qkv_destination_major(q, k, v, world_size), expected)


def test_pack_qkv_destination_major_validates_inputs():
    q = torch.empty(2, 4, 8, device=DEVICE, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match="same 3D shape"):
        pack_qkv_destination_major(q, q[:, :-1], q, 2)
    with pytest.raises(ValueError, match="divide global_heads"):
        pack_qkv_destination_major(q, q, q, 3)
    with pytest.raises(ValueError, match="expected shape"):
        pack_qkv_destination_major(q, q, q, 2, out=torch.empty_like(q))


# ---------------------------------------------------------------------------
# Varlen pack / scatter
# ---------------------------------------------------------------------------

VARLEN_DTYPES = get_ci_test_range([torch.bfloat16, torch.float16], [torch.bfloat16])
# (name, bs, s_txt, s_img, num_heads, head_dim, valid_txt_lens)
VARLEN_SHAPES = get_ci_test_range(
    [
        ("small_c2", 2, 64, 128, 4, 64, [32, 48]),
        ("prod_c2", 2, 256, 1024, 24, 128, [128, 200]),
        ("all_valid_b1", 1, 64, 128, 4, 64, [64]),
        ("all_valid_b4", 4, 64, 128, 4, 64, [64, 64, 64, 64]),
        ("c8_prod", 8, 256, 4096, 24, 128, [128, 200, 256, 100, 50, 256, 256, 50]),
        # one batch with zero valid text tokens (image side still valid)
        ("zero_txt_one_batch", 2, 64, 128, 4, 64, [0, 32]),
        # bs=1 with no text validity (only image rows packed)
        ("bs1_zero_txt", 1, 64, 128, 4, 64, [0]),
    ],
    [
        ("small_c2", 2, 64, 128, 4, 64, [32, 48]),
        ("prod_c2", 2, 256, 1024, 24, 128, [128, 200]),
        ("all_valid_b4", 4, 64, 128, 4, 64, [64, 64, 64, 64]),
    ],
)


def _build_mask(bs, s_txt, s_img, valid_txt_lens):
    mask = torch.zeros(bs, s_txt + s_img, dtype=torch.bool, device=DEVICE)
    for b, vt in enumerate(valid_txt_lens):
        mask[b, :vt] = True
        mask[b, s_txt:] = True
    return mask


def _build_meta(mask):
    bs, seq = mask.shape
    indices = mask.reshape(-1).nonzero(as_tuple=False).flatten()
    return indices, build_inv_indices(indices, bs * seq)


@pytest.mark.parametrize("dtype", VARLEN_DTYPES)
@pytest.mark.parametrize("shape", VARLEN_SHAPES, ids=lambda s: s[0])
def test_varlen_pack_matches_index_select(dtype, shape):
    _, bs, s_txt, s_img, num_heads, head_dim, valid_txt_lens = shape
    torch.manual_seed(0)
    s = s_txt + s_img
    indices, _ = _build_meta(_build_mask(bs, s_txt, s_img, valid_txt_lens))

    q, k, v = (
        torch.randn(bs, s, num_heads, head_dim, dtype=dtype, device=DEVICE)
        for _ in range(3)
    )
    fused = fused_pack_qkv(q, k, v, indices)
    for got, src in zip(fused, (q, k, v), strict=True):
        want = src.reshape(bs * s, num_heads, head_dim).index_select(0, indices)
        assert torch.equal(got, want)


@pytest.mark.parametrize("dtype", VARLEN_DTYPES)
@pytest.mark.parametrize("shape", VARLEN_SHAPES, ids=lambda s: s[0])
def test_varlen_segmented_pack_matches_materialized_joint(dtype, shape):
    _, bs, s_txt, s_img, num_heads, head_dim, valid_txt_lens = shape
    torch.manual_seed(42)
    indices, _ = _build_meta(_build_mask(bs, s_txt, s_img, valid_txt_lens))
    txt_qkv = tuple(
        torch.randn(bs, s_txt, num_heads, head_dim, dtype=dtype, device=DEVICE)
        for _ in range(3)
    )
    img_qkv = tuple(
        torch.randn(bs, s_img, num_heads, head_dim, dtype=dtype, device=DEVICE)
        for _ in range(3)
    )

    got = fused_pack_segmented_qkv(*txt_qkv, *img_qkv, indices)
    for actual, txt, img in zip(got, txt_qkv, img_qkv, strict=True):
        joint = torch.cat([txt, img], dim=1)
        expected = joint.flatten(0, 1).index_select(0, indices)
        assert torch.equal(actual, expected)


@pytest.mark.parametrize("dtype", VARLEN_DTYPES)
@pytest.mark.parametrize("shape", VARLEN_SHAPES, ids=lambda s: s[0])
def test_varlen_scatter_matches_index_copy(dtype, shape):
    _, bs, s_txt, s_img, num_heads, head_dim, valid_txt_lens = shape
    torch.manual_seed(1)
    s = s_txt + s_img
    mask = _build_mask(bs, s_txt, s_img, valid_txt_lens)
    indices, inv_indices = _build_meta(mask)

    out_unpad = torch.randn(
        indices.shape[0], num_heads, head_dim, dtype=dtype, device=DEVICE
    )
    flat = torch.zeros(bs * s, num_heads, head_dim, dtype=dtype, device=DEVICE)
    flat.index_copy_(0, indices, out_unpad)
    out_ref = flat.view(bs, s, num_heads, head_dim)

    out_fused = fused_scatter_to_padded(out_unpad, inv_indices, bs, s)
    assert torch.equal(out_ref, out_fused)
    invalid = ~mask
    if invalid.any():
        # Padding rows must be exactly zero, not merely small.
        assert out_fused[invalid].abs().max().item() == 0.0


def test_varlen_pack_handles_non_contiguous_input():
    # Q/K/V arrive as (B, H, S, D) permutes from attention; the helper must
    # make them contiguous itself rather than reading the wrong strides.
    torch.manual_seed(2)
    bs, s_txt, s_img, num_heads, head_dim = 2, 64, 128, 4, 64
    indices, _ = _build_meta(_build_mask(bs, s_txt, s_img, [32, 48]))

    pre = torch.randn(
        bs, num_heads, s_txt + s_img, head_dim, dtype=torch.bfloat16, device=DEVICE
    )
    q, k, v = (torch.randn_like(pre).permute(0, 2, 1, 3) for _ in range(3))
    assert not q.is_contiguous()

    fused = fused_pack_qkv(q, k, v, indices)
    for got, src in zip(fused, (q, k, v), strict=True):
        want = src.contiguous().flatten(0, 1).index_select(0, indices)
        assert torch.equal(got, want)


def test_build_inv_indices_matches_manual():
    torch.manual_seed(3)
    bs, s = 2, 32
    mask = torch.bernoulli(torch.full((bs, s), 0.6, device=DEVICE)).to(torch.bool)
    indices = mask.reshape(-1).nonzero(as_tuple=False).flatten()

    manual = torch.full((bs * s,), -1, dtype=torch.int32, device=DEVICE)
    if indices.numel():
        manual[indices.long()] = torch.arange(
            indices.numel(), dtype=torch.int32, device=DEVICE
        )
    assert torch.equal(build_inv_indices(indices, bs * s), manual)


def test_varlen_empty_valid_set_handled():
    # An all-False mask is reachable (a request whose text side is fully
    # masked): pack must return empty tensors and scatter an all-zero dense
    # output rather than launching a degenerate grid.
    bs, s, num_heads, head_dim = 2, 16, 4, 64
    indices = torch.zeros(bs, s, dtype=torch.bool, device=DEVICE).reshape(-1).nonzero()
    indices = indices.flatten()
    inv_indices = build_inv_indices(indices, bs * s)
    assert indices.numel() == 0

    q = torch.randn(bs, s, num_heads, head_dim, dtype=torch.bfloat16, device=DEVICE)
    unpad = fused_pack_qkv(q, q.clone(), q.clone(), indices)
    assert all(t.shape == (0, num_heads, head_dim) for t in unpad)

    out_padded = fused_scatter_to_padded(unpad[0], inv_indices, bs, s)
    assert out_padded.shape == (bs, s, num_heads, head_dim)
    assert out_padded.abs().max().item() == 0.0


# The kernels above are unit-tested against index_select/index_copy_; this
# section drives them through the production USPAttention masked branch, where
# a wrong index layout would produce plausible-looking attention output rather
# than an obvious mismatch.


def _sdpa_with_key_mask(q, k, v, key_mask, softmax_scale):
    """Reference: SDPA with a ``[B, S]`` key mask broadcast to ``[B, 1, 1, S]``."""
    q_ = q.transpose(1, 2)
    k_ = k.transpose(1, 2)
    v_ = v.transpose(1, 2)
    mask = key_mask.to(dtype=q.dtype)[:, None, None, :]
    mask = (mask - 1.0) * torch.finfo(q.dtype).max
    out = F.scaled_dot_product_attention(
        q_,
        k_,
        v_,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=False,
        scale=softmax_scale,
    )
    return out.transpose(1, 2)


def _varlen_path(q, k, v, key_mask, softmax_scale):
    """Production varlen path matching USPAttention.forward masked branch."""
    bs, seq = q.shape[0], q.shape[1]
    meta = build_varlen_mask_meta(key_mask)
    indices = meta["indices"]
    if indices.shape[0] == 0:
        return torch.zeros_like(q)
    q_unpad, k_unpad, v_unpad = fused_pack_qkv(q, k, v, indices)
    try:
        out_unpad = flash_attn_varlen_func(
            q=q_unpad,
            k=k_unpad,
            v=v_unpad,
            cu_seqlens_q=meta["cu_seqlens"],
            cu_seqlens_k=meta["cu_seqlens"],
            max_seqlen_q=meta["max_seqlen"],
            max_seqlen_k=meta["max_seqlen"],
            softmax_scale=softmax_scale,
            causal=False,
            ver=_fa_backend.fa_ver,
        )
    except ImportError as exc:  # pragma: no cover - image-dependent
        # ``flash_attn_varlen_func`` resolves its backend lazily, so an image
        # without the selected FlashAttention build raises here rather than at
        # import.  This file also runs on the B200 lane (for the causal-Conv3d
        # section), which ships no ``flash_attn`` -- skip only this end-to-end
        # comparison there; the pack/scatter kernels themselves are covered
        # unit-wise above on every lane.
        pytest.skip(f"FlashAttention varlen v{_fa_backend.fa_ver} unavailable: {exc}")
    return fused_scatter_to_padded(out_unpad, meta["inv_indices"], bs, seq)


@pytest.mark.parametrize("dtype", VARLEN_DTYPES)
def test_fa_dense_scheduler_matches_single_sequence_varlen(dtype):
    torch.manual_seed(7)
    batch_size, seq, num_heads, head_dim = 1, 256, 4, 128
    q, k, v = (
        torch.randn(
            batch_size,
            seq,
            num_heads,
            head_dim,
            dtype=dtype,
            device=DEVICE,
        )
        for _ in range(3)
    )
    cu_seqlens = torch.tensor([0, seq], dtype=torch.int32, device=DEVICE)
    kwargs = dict(
        max_seqlen_q=seq,
        max_seqlen_k=seq,
        softmax_scale=head_dim**-0.5,
        causal=False,
        ver=_fa_backend.fa_ver,
    )
    try:
        varlen = flash_attn_varlen_func(
            q.flatten(0, 1),
            k.flatten(0, 1),
            v.flatten(0, 1),
            cu_seqlens,
            cu_seqlens,
            **kwargs,
        ).view_as(q)
        dense = flash_attn_varlen_func(q, k, v, None, None, **kwargs)
    except ImportError as exc:  # pragma: no cover - image-dependent
        pytest.skip(f"FlashAttention unavailable: {exc}")

    torch.testing.assert_close(dense, varlen, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dtype", VARLEN_DTYPES)
@pytest.mark.parametrize("shape", VARLEN_SHAPES, ids=lambda s: s[0])
def test_varlen_path_matches_sdpa_on_valid_rows(dtype, shape):
    """Valid rows: varlen output ≈ SDPA output within FA tolerance."""
    _, bs, s_txt, s_img, num_heads, head_dim, valid_txt_lens = shape
    torch.manual_seed(0)
    s = s_txt + s_img
    softmax_scale = head_dim**-0.5
    mask = _build_mask(bs, s_txt, s_img, valid_txt_lens)
    q = torch.randn(bs, s, num_heads, head_dim, dtype=dtype, device=DEVICE)
    k = torch.randn(bs, s, num_heads, head_dim, dtype=dtype, device=DEVICE)
    v = torch.randn(bs, s, num_heads, head_dim, dtype=dtype, device=DEVICE)

    out_sdpa = _sdpa_with_key_mask(q, k, v, mask, softmax_scale)
    out_varlen = _varlen_path(q, k, v, mask, softmax_scale)

    valid = mask[..., None, None].expand_as(out_sdpa)
    rtol = 1e-2 if dtype == torch.bfloat16 else 5e-3
    atol = 5e-2 if dtype == torch.bfloat16 else 1e-2
    torch.testing.assert_close(
        out_sdpa[valid],
        out_varlen[valid],
        rtol=rtol,
        atol=atol,
    )


@pytest.mark.parametrize("dtype", VARLEN_DTYPES)
@pytest.mark.parametrize("shape", VARLEN_SHAPES, ids=lambda s: s[0])
def test_varlen_path_zeros_masked_rows(dtype, shape):
    """Masked rows: varlen path produces exact zeros (documented contract)."""
    _, bs, s_txt, s_img, num_heads, head_dim, valid_txt_lens = shape
    torch.manual_seed(1)
    s = s_txt + s_img
    softmax_scale = head_dim**-0.5
    mask = _build_mask(bs, s_txt, s_img, valid_txt_lens)
    q = torch.randn(bs, s, num_heads, head_dim, dtype=dtype, device=DEVICE)
    k = torch.randn(bs, s, num_heads, head_dim, dtype=dtype, device=DEVICE)
    v = torch.randn(bs, s, num_heads, head_dim, dtype=dtype, device=DEVICE)

    out_varlen = _varlen_path(q, k, v, mask, softmax_scale)

    invalid = ~mask
    if invalid.any():
        assert (out_varlen[invalid] == 0).all(), "masked rows must be zero-filled"


# ---------------------------------------------------------------------------
# Causal Conv3d cat + pad (CUDA JIT vs Triton)
# ---------------------------------------------------------------------------

CONV3D_CASES = get_ci_test_range(
    [
        (1024, 1, 30, 52, 1),
        (1024, 1, 30, 52, 2),
        (1024, 2, 60, 104, 1),
        (1024, 2, 60, 104, 2),
        (512, 4, 120, 208, 1),
        (512, 4, 120, 208, 2),
        (256, 4, 240, 416, 1),
        (256, 4, 240, 416, 2),
    ],
    [(1024, 1, 30, 52, 1), (512, 4, 120, 208, 2)],
)


def _conv3d_inputs(channels, t_size, h_size, w_size, cache_t):
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(channels * 1009 + t_size * 251 + h_size + cache_t)
    x = torch.randn(
        (1, channels, t_size, h_size, w_size),
        device=DEVICE,
        dtype=torch.bfloat16,
        generator=generator,
    )
    cache_x = torch.randn(
        (1, channels, cache_t, h_size, w_size),
        device=DEVICE,
        dtype=torch.bfloat16,
        generator=generator,
    )
    return x, cache_x, (1, 1, 1, 1, cache_t, 0)


@pytest.mark.parametrize("channels,t_size,h_size,w_size,cache_t", CONV3D_CASES)
def test_causal_conv3d_cat_pad_cuda_matches_triton(
    channels, t_size, h_size, w_size, cache_t
):
    x, cache_x, padding = _conv3d_inputs(channels, t_size, h_size, w_size, cache_t)
    actual = fused_causal_conv3d_cat_pad_cuda(x, cache_x, padding)
    expected = fused_causal_conv3d_cat_pad_triton(x, cache_x, padding)
    assert torch.equal(actual, expected)


def test_causal_conv3d_cat_pad_torch_compile():
    # The CUDA path is a registered custom op, so a fullgraph compile must not
    # graph-break on it.
    x, cache_x, padding = _conv3d_inputs(1024, 1, 30, 52, 1)

    @torch.compile(fullgraph=True)
    def fn(x, cache_x):
        return fused_causal_conv3d_cat_pad_cuda(x, cache_x, padding)

    assert torch.equal(
        fn(x, cache_x), fused_causal_conv3d_cat_pad_triton(x, cache_x, padding)
    )


# ---------------------------------------------------------------------------
# Wan causal VAE cache kernels
# ---------------------------------------------------------------------------


def _ref_cat_pad(x, cache, padding):
    p = list(padding)
    if cache is not None:
        x = torch.cat([cache, x], dim=2)
        p[4] -= cache.shape[2]
    if any(p):
        x = F.pad(x, p)
    return x.contiguous(memory_format=torch.channels_last_3d)


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "c,t,h,w,cache_t,pads",
    [
        (96, 1, 10, 14, 0, (1, 1, 1, 1, 2, 0)),  # first chunk, zero-fill front
        (96, 1, 10, 14, 1, (1, 1, 1, 1, 2, 0)),  # legacy 1-frame cache
        (96, 1, 10, 14, 2, (1, 1, 1, 1, 2, 0)),  # steady state k3 conv
        (64, 1, 10, 14, 2, (0, 0, 0, 0, 2, 0)),  # time_conv (temporal only)
        (48, 4, 10, 14, 2, (1, 1, 1, 1, 2, 0)),  # encoder-style T=4 chunk
    ],
)
def test_cat_pad_channels_last_3d_bitwise(dtype, c, t, h, w, cache_t, pads):
    torch.cuda.manual_seed(0)
    x = _cl3d((1, c, t, h, w), dtype)
    cache = None
    if cache_t:
        # Strided interior view: caches may arrive as non-contiguous slices.
        ph, pw = pads[2], pads[0]
        buf = _cl3d((1, c, cache_t, h + 2 * ph, w + 2 * pw), dtype)
        cache = buf[:, :, :, ph : ph + h, pw : pw + w]
    ref = _ref_cat_pad(x, cache, pads)

    out = cat_pad_channels_last_3d(x, cache, pads)
    assert out is not None and out.shape == ref.shape
    assert out.is_contiguous(memory_format=torch.channels_last_3d)
    assert torch.equal(out, ref)

    # Dual-output mode: the same pass also emits the compact feature cache
    # (unpadded interior of the last frames), bitwise equal to the slice.
    pair = cat_pad_channels_last_3d(x, cache, pads, keep_cache_t=2)
    assert pair is not None
    out2, keep = pair
    assert torch.equal(out2, ref)
    ph, pw = pads[2], pads[0]
    keep_t = min(2, ref.shape[2])
    want = ref[:, :, ref.shape[2] - keep_t :, ph : ph + h, pw : pw + w]
    assert keep.shape == want.shape
    assert keep.is_contiguous(memory_format=torch.channels_last_3d)
    assert torch.equal(keep, want)


@torch.no_grad()
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "c_in,c_out,t,h,w,ft,fs,drop",
    [
        (128, 64, 1, 10, 14, 2, 2, False),
        (128, 64, 1, 10, 14, 2, 2, True),  # first_chunk slicing
        (64, 32, 2, 10, 14, 1, 2, False),
    ],
)
def test_dup_up3d_add_bitwise(dtype, c_in, c_out, t, h, w, ft, fs, drop):
    torch.cuda.manual_seed(0)
    repeats = c_out * ft * fs * fs // c_in
    src = _cl3d((1, c_in, t, h, w), dtype)
    t_out = t * ft - (ft - 1 if drop else 0)
    # Main arm as a permuted view, like the WanResample 2D output.
    main = torch.randn(
        (1, t_out, c_out, h * fs, w * fs), device=DEVICE, dtype=dtype
    ).permute(0, 2, 1, 3, 4)

    dup = src.repeat_interleave(repeats, dim=1)
    dup = dup.view(1, c_out, ft, fs, fs, t, h, w)
    dup = dup.permute(0, 1, 5, 2, 6, 3, 7, 4).contiguous()
    dup = dup.view(1, c_out, t * ft, h * fs, w * fs)
    if drop:
        dup = dup[:, :, ft - 1 :, :, :]
    ref = main + dup

    out = dup_up3d_add(main, src, ft, fs, repeats, drop)
    assert out is not None and out.shape == ref.shape
    # Layout must match the aten add output exactly (downstream reductions
    # are layout-sensitive), and every value must be bitwise identical.
    assert out.stride() == ref.stride()
    assert torch.equal(out, ref)


@torch.no_grad()
@pytest.mark.parametrize("pads_temporal_only", [False, True])
def test_wan_cached_conv_chunk_loop_bitwise(pads_temporal_only):
    """The fused conv-input/compact-cache scheme must reproduce the original
    clone/cat bookkeeping bitwise across a chunked decode, including the
    first-chunk zero fill and the "Rep" marker start used by WanResample."""
    from sglang.multimodal_gen.runtime.models.vaes import wanvae
    from sglang.multimodal_gen.runtime.models.vaes.wanvae import (
        CACHE_T,
        WanCausalConv3d,
        _cache_payload,
        _run_cached_causal_conv,
    )

    torch.cuda.manual_seed(0)
    c = 64
    if pads_temporal_only:
        conv = WanCausalConv3d(c, 2 * c, (3, 1, 1), padding=(1, 0, 0))
    else:
        conv = WanCausalConv3d(c, c, 3, padding=1)
    conv = conv.to(device=DEVICE, dtype=torch.float32)
    conv.weight.data = conv.weight.data.contiguous(memory_format=torch.channels_last_3d)
    chunks = [_cl3d((1, c, 1, 10, 14), torch.float32) for _ in range(4)]

    def run(force_fallback, start):
        cache = [start]
        outs = []
        orig = wanvae.cat_pad_channels_last_3d
        if force_fallback:
            wanvae.cat_pad_channels_last_3d = None
        try:
            for x in chunks:
                outs.append(_run_cached_causal_conv(conv, x, cache, 0))
        finally:
            wanvae.cat_pad_channels_last_3d = orig
        return outs, cache[0]

    for start in (None, "Rep"):
        fused_outs, fused_cache = run(False, start)
        ref_outs, ref_cache = run(True, start)
        for got, want in zip(fused_outs, ref_outs, strict=True):
            assert torch.equal(got, want)
        got_payload = _cache_payload(fused_cache)
        assert got_payload is not None and got_payload.shape[2] == CACHE_T
        # Reference cache holds the last CACHE_T unpadded frames.
        assert torch.equal(got_payload, ref_cache[:, :, -CACHE_T:])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
