"""Fused Q per-head RMSNorm + KV RMSNorm + RoPE + FP8 nope quant + paged SWA store.

Single Triton kernel replacing the 2-kernel path:
  1. fused_reduce_qk_norm_rope_swa_write (norm + RoPE)
  2. store_cache -> fused_store_cache (FP8 quant + paged scatter)

Grid: (cdiv(M, BLOCK_SIZE_M), num_local_heads + 1).
  pid_h < num_local_heads: Q head programs (split-K reduce + norm + RoPE)
  pid_h == num_local_heads: KV program (norm + RoPE + FP8 quant nope + paged scatter)
"""

from functools import lru_cache
from typing import Optional

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.dsv4.unified_kv_kernels.layout import (
    check_two_pool_pair,
)
from sglang.kernels.ops.quantization.fp8_kernel import is_fp8_fnuz
from sglang.srt.utils import is_gfx95_supported

_fp8_fnuz = is_fp8_fnuz()

# The two-pool fp8 store defers to aiter: its kernel already emits the exact
# 512 B nope row (448 fp8 + 14 dup e8m0 + pad) that the v4 asm attention reader
# expects, so the Triton kernel below stays bf16/legacy-packed only.
try:
    from aiter.ops.fused_qk_norm_rope_cache_quant import fused_qk_norm_rope_group_quant

    _HAS_AITER_OP = True
except ImportError:
    fused_qk_norm_rope_group_quant = None
    _HAS_AITER_OP = False

_HAS_GROUP_QUANT = _HAS_AITER_OP and is_gfx95_supported()


# ---------------------------------------------------------------------------
# Triton JIT helpers
# ---------------------------------------------------------------------------


@triton.jit
def _batched_rmsnorm(row, weight, n_cols, epsilon):
    row_norm = tl.sum(row * row, axis=-1)
    norm_factor = tl.math.rsqrt((row_norm / n_cols) + epsilon)
    if weight is not None:
        return row * norm_factor[:, None] * weight[None, :]
    return row * norm_factor[:, None]


@triton.jit
def _gptj_rotate(x, mask, BM: tl.constexpr, BD: tl.constexpr, BDH: tl.constexpr):
    x_rot = tl.where(mask, x, -x)
    x_rot = tl.reshape(x_rot, (BM, BDH, 2))
    x_rot = tl.flip(x_rot, 2)
    return tl.reshape(x_rot, (BM, BD))


@triton.jit
def _batched_rope(
    x_pe, cos, sin, d_pe_offs, BM: tl.constexpr, BD: tl.constexpr, BDH: tl.constexpr
):
    mask = (d_pe_offs % 2 == 0)[None, :]
    x_rot = _gptj_rotate(x_pe, mask, BM, BD, BDH)
    return x_pe * cos + x_rot * sin


# ---------------------------------------------------------------------------
# Main kernel
# ---------------------------------------------------------------------------


@triton.jit
def _fused_qk_norm_rope_store_kernel(
    q_in_ptr,
    q_out_ptr,
    kv_ptr,
    q_norm_weight_ptr,
    kv_norm_weight_ptr,
    positions_ptr,
    cos_ptr,
    sin_ptr,
    swa_cache_ptr,
    swa_loc_ptr,
    M,
    q_in_splitk_stride,
    q_in_m_stride,
    q_in_d_stride,
    stride_qm,
    stride_qh,
    stride_qd,
    stride_kv_m,
    stride_kv_d,
    cos_stride_t,
    cos_stride_d,
    swa_cache_stride_page,
    q_eps,
    kv_eps,
    BLOCK_SIZE_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    ROPE_DIM: tl.constexpr,
    NUM_LOCAL_HEADS: tl.constexpr,
    NUM_SPLITK: tl.constexpr,
    HAS_SWA_STORE: tl.constexpr,
    DIM_NOPE: tl.constexpr,
    TILE_SIZE: tl.constexpr,
    NUM_NOPE_TILES: tl.constexpr,
    FP8_MIN: tl.constexpr,
    FP8_MAX: tl.constexpr,
    BYTES_PER_TOKEN: tl.constexpr,
    SWA_PAGE_SIZE: tl.constexpr,
    BF16_STORE: tl.constexpr,
    IS_FNUZ: tl.constexpr,
):
    pid_m = tl.program_id(0).to(tl.int64)
    pid_h = tl.program_id(1).to(tl.int64)
    NOPE_DIM: tl.constexpr = HEAD_DIM - ROPE_DIM
    NUM_PE_CHUNKS: tl.constexpr = HEAD_DIM // ROPE_DIM

    m_offs = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M).to(tl.int64)
    m_mask = m_offs < M

    offs_d_full = tl.arange(0, HEAD_DIM)
    nope_d_mask = offs_d_full < NOPE_DIM

    d_pe_offs = tl.arange(0, ROPE_DIM).to(tl.int64)
    d_cos_offs = d_pe_offs // 2

    # ===== Q path =====
    if pid_h < NUM_LOCAL_HEADS:
        head_id = pid_h.to(tl.int32)
        offs_n = head_id * HEAD_DIM + offs_d_full

        splitk_offs = tl.arange(0, NUM_SPLITK).to(tl.int64)
        q_ptrs = (
            q_in_ptr
            + splitk_offs[:, None, None] * q_in_splitk_stride
            + m_offs[None, :, None] * q_in_m_stride
            + offs_n[None, None, :] * q_in_d_stride
        )
        q_tile = tl.load(q_ptrs, mask=m_mask[None, :, None], other=0.0).to(tl.float32)
        q_acc = tl.sum(q_tile, axis=0)

        if q_norm_weight_ptr is not None:
            w_q = tl.load(q_norm_weight_ptr + offs_d_full).to(tl.float32)
        else:
            w_q = None
        q_normed = _batched_rmsnorm(q_acc, w_q, HEAD_DIM, q_eps)

        q_base = q_out_ptr + m_offs[:, None] * stride_qm + pid_h * stride_qh
        tl.store(
            q_base + offs_d_full[None, :] * stride_qd,
            q_normed.to(q_out_ptr.dtype.element_ty),
            mask=m_mask[:, None] & nope_d_mask[None, :],
        )

        q_pe = tl.where((offs_d_full >= NOPE_DIM)[None, :], q_normed, 0.0)
        q_pe = tl.reshape(q_pe, (BLOCK_SIZE_M, NUM_PE_CHUNKS, ROPE_DIM))
        q_pe = tl.sum(q_pe, axis=1)

        pos = tl.load(positions_ptr + m_offs, mask=m_mask, other=0)
        cos_o = pos[:, None] * cos_stride_t + d_cos_offs[None, :] * cos_stride_d
        cos = tl.load(cos_ptr + cos_o, mask=m_mask[:, None], other=0)
        sin = tl.load(sin_ptr + cos_o, mask=m_mask[:, None], other=0)

        q_pe = _batched_rope(
            q_pe, cos, sin, d_pe_offs, BLOCK_SIZE_M, ROPE_DIM, ROPE_DIM // 2
        )
        tl.store(
            q_base + (NOPE_DIM + d_pe_offs[None, :]) * stride_qd,
            q_pe.to(q_out_ptr.dtype.element_ty),
            mask=m_mask[:, None],
        )
        return

    # ===== KV path =====
    src_id = m_offs.to(tl.int32)
    src_mask = m_mask

    pos = tl.load(positions_ptr + src_id, mask=src_mask, other=0)
    cos_o = pos[:, None] * cos_stride_t + d_cos_offs[None, :] * cos_stride_d
    cos = tl.load(cos_ptr + cos_o, mask=src_mask[:, None], other=0)
    sin = tl.load(sin_ptr + cos_o, mask=src_mask[:, None], other=0)

    kv_base = kv_ptr + src_id[:, None].to(tl.int64) * stride_kv_m
    kv_full_ptrs = kv_base + offs_d_full[None, :] * stride_kv_d

    kv_full = tl.load(kv_full_ptrs, mask=src_mask[:, None], other=0.0).to(tl.float32)

    if kv_norm_weight_ptr is not None:
        w_kv = tl.load(kv_norm_weight_ptr + offs_d_full).to(tl.float32)
    else:
        w_kv = None
    kv_normed = _batched_rmsnorm(kv_full, w_kv, HEAD_DIM, kv_eps)

    tl.store(
        kv_full_ptrs,
        kv_normed.to(kv_ptr.dtype.element_ty),
        mask=src_mask[:, None] & nope_d_mask[None, :],
    )

    kv_pe = tl.where((offs_d_full >= NOPE_DIM)[None, :], kv_normed, 0.0)
    kv_pe = tl.reshape(kv_pe, (BLOCK_SIZE_M, NUM_PE_CHUNKS, ROPE_DIM))
    kv_pe = tl.sum(kv_pe, axis=1)

    kv_pe = _batched_rope(
        kv_pe, cos, sin, d_pe_offs, BLOCK_SIZE_M, ROPE_DIM, ROPE_DIM // 2
    )
    tl.store(
        kv_base + (NOPE_DIM + d_pe_offs[None, :]) * stride_kv_d,
        kv_pe.to(kv_ptr.dtype.element_ty),
        mask=src_mask[:, None],
    )

    # ===== Paged SWA store: FP8 quant nope + BF16 rope + scales =====
    # Layout within a page (matches fused_store_flashmla_cache CUDA kernel):
    #   Values region: [page_size tokens * 576 bytes/token]
    #     Per token: 448 bytes FP8 nope + 128 bytes BF16 rope
    #   Scales region: [page_size tokens * 8 bytes/token]
    #     Per token: 7 scale bytes + 1 pad byte
    # Total per page before padding: page_size * 584
    VALUE_STRIDE: tl.constexpr = DIM_NOPE + ROPE_DIM * 2
    SCALE_BYTES: tl.constexpr = NUM_NOPE_TILES + 1

    if HAS_SWA_STORE and BF16_STORE:
        # unified_kv unified_kv: write the whole head_dim as plain bf16 into a
        # [num_slots, head_dim] bf16 cache at row=loc (no fp8 / no scale).
        loc = tl.load(swa_loc_ptr + src_id, mask=src_mask, other=0)
        row_base = loc.to(tl.int64)[:, None] * swa_cache_stride_page
        # nope
        tl.store(
            swa_cache_ptr + row_base + offs_d_full[None, :],
            kv_normed.to(swa_cache_ptr.dtype.element_ty),
            mask=src_mask[:, None] & nope_d_mask[None, :],
        )
        # pe
        tl.store(
            swa_cache_ptr + row_base + (NOPE_DIM + d_pe_offs[None, :]),
            kv_pe.to(swa_cache_ptr.dtype.element_ty),
            mask=src_mask[:, None],
        )
    elif HAS_SWA_STORE:
        loc = tl.load(swa_loc_ptr + src_id, mask=src_mask, other=0)
        page_id = loc // SWA_PAGE_SIZE
        page_off = loc % SWA_PAGE_SIZE
        page_base = page_id.to(tl.int64) * swa_cache_stride_page
        value_base = page_base + page_off.to(tl.int64) * VALUE_STRIDE
        scale_base = (
            page_base
            + SWA_PAGE_SIZE * VALUE_STRIDE
            + page_off.to(tl.int64) * SCALE_BYTES
        )

        EPS: tl.constexpr = 1e-8
        nope_tile_offs = tl.arange(0, TILE_SIZE)

        for tile_i in tl.static_range(NUM_NOPE_TILES):
            tile_start = tile_i * TILE_SIZE
            tile_data = tl.load(
                kv_ptr
                + src_id[:, None].to(tl.int64) * stride_kv_m
                + (tile_start + nope_tile_offs[None, :]) * stride_kv_d,
                mask=src_mask[:, None],
                other=0.0,
            ).to(tl.float32)

            abs_max = tl.max(tl.abs(tile_data), axis=-1)
            abs_max_c = tl.maximum(abs_max, EPS)
            scale_f = abs_max_c / FP8_MAX
            log2_s = tl.log2(scale_f)
            ceil_log2 = tl.math.ceil(log2_s)
            scale_pow2 = tl.exp2(ceil_log2)
            inv_scale = 1.0 / scale_pow2
            x_scaled = tile_data * inv_scale[:, None]
            x_fp8 = tl.clamp(x_scaled, FP8_MIN, FP8_MAX)

            # Encode with the SAME fp8 type the decode bitcasts (read
            # _KV_FP8_TY = float8e4b8 for e4m3fnuz / float8e4nv otherwise).
            # An implicit/fn cast mis-encodes under fnuz (exponent bias 7 vs
            # fnuz bias 8), so the fnuz read decodes every element 2x too small.
            if IS_FNUZ:
                x_fp8_cast = x_fp8.to(tl.float8e4b8)
            else:
                x_fp8_cast = x_fp8.to(tl.float8e4nv)
            x_fp8_bytes = x_fp8_cast.to(tl.uint8, bitcast=True)
            fp8_byte_offs = value_base[:, None] + tile_start + nope_tile_offs[None, :]
            tl.store(
                swa_cache_ptr + fp8_byte_offs,
                x_fp8_bytes,
                mask=src_mask[:, None],
            )

            scale_uint8 = (ceil_log2.to(tl.int32) + 127).to(tl.uint8)
            tl.store(
                swa_cache_ptr + scale_base + tile_i,
                scale_uint8,
                mask=src_mask,
            )

        rope_data = kv_pe.to(tl.bfloat16)
        rope_offs = tl.arange(0, ROPE_DIM)
        rope_byte_base = value_base[:, None] + DIM_NOPE + rope_offs[None, :] * 2
        rope_data_as_i16 = rope_data.to(tl.int16, bitcast=True)
        lo = (rope_data_as_i16 & 0xFF).to(tl.uint8)
        hi = ((rope_data_as_i16 >> 8) & 0xFF).to(tl.uint8)
        tl.store(swa_cache_ptr + rope_byte_base, lo, mask=src_mask[:, None])
        tl.store(swa_cache_ptr + rope_byte_base + 1, hi, mask=src_mask[:, None])


# ---------------------------------------------------------------------------
# Python wrapper
# ---------------------------------------------------------------------------


@lru_cache(maxsize=None)
def _token_identity_map(num_tokens: int, device: torch.device) -> torch.Tensor:
    # Cached because every layer asks for it on every decode step and the answer
    # only depends on the token count. Never evict: a captured cuda graph holds
    # this address, and the default capture list has ~36 distinct batch sizes, so
    # a bounded cache would free a live graph's buffer back into the graph pool.
    return torch.arange(num_tokens, dtype=torch.int32, device=device)


def _fp8_2buff_store(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_norm_weight: Optional[torch.Tensor],
    kv_norm_weight: torch.Tensor,
    rms_eps: float,
    rope_head_dim: int,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_out: torch.Tensor,
    swa_cache: Optional[torch.Tensor],
    swa_rope_cache: Optional[torch.Tensor],
    swa_loc: Optional[torch.Tensor],
    k_nope_out: Optional[torch.Tensor],
    k_rope_out: Optional[torch.Tensor],
) -> torch.Tensor:
    if not _HAS_GROUP_QUANT:
        # is_unified_kv_fp8() already falls back to bf16 off gfx95, so reaching here
        # on gfx95 means the installed aiter predates the op
        raise RuntimeError(
            "fp8 two-pool unified_kv needs aiter's fused_qk_norm_rope_group_quant: "
            f"aiter exports it={_HAS_AITER_OP}, gfx95={is_gfx95_supported()}"
        )

    assert (
        q.dim() == 2
    ), f"aiter takes q as [T, H, D] with no split-K reduce, got {tuple(q.shape)}"
    assert (
        cos_cache.shape[-1] * 2 == rope_head_dim
    ), f"rot_dim from cos_cache ({cos_cache.shape[-1] * 2}) != {rope_head_dim}"
    # int64 is what the kernel indexes with; a silent .to() here would copy every
    # call and hide a caller that changed dtype.
    assert (
        positions.dtype == torch.int64
    ), f"positions must be int64, got {positions.dtype}"

    batch_id = None
    has_swa = swa_cache is not None
    if has_swa:
        # Same coupling as store_swa_into_unified: one row index addresses both
        # pools, and aiter bounds-checks neither -- a short rope pool aborts the
        # process with nothing on stderr, so the pair is checked there first.
        assert swa_loc is not None, "fp8 SWA store needs swa_loc alongside the pools"
        check_two_pool_pair(
            swa_cache,
            swa_rope_cache,
            rope_width=rope_head_dim,
            rope_dtype=kv.dtype,
        )
        # aiter rejects the SWA write without a token->seq map even in dest-row
        # mode, where all it does with it is drop tokens whose id is negative
        # (CG pad). The ring row itself comes from swa_loc, and stale tokens are
        # dropped on positions < 0, so identity is the map decode wants: one
        # token per sequence, nothing masked. A caller with several tokens per
        # sequence would have to pass its own.
        batch_id = _token_identity_map(kv.shape[0], kv.device)

    # aiter has no stride arguments, so it can only write a packed q_out. With
    # attn_tp_size > 1 the caller hands us a slice of a head-padded buffer
    # ([T, 64, D] sliced to [T, n_local_heads, D]), which is strided unless the
    # padding happened to be zero -- stage through a packed buffer then.
    # contiguous_format is explicit: empty_like's default would copy q_out's
    # strides for any input that is dense, and a strided staging buffer would
    # silently misplace the heads.
    q_dst = (
        q_out
        if q_out.is_contiguous()
        else torch.empty_like(q_out, memory_format=torch.contiguous_format)
    )

    # The packed K pair lands in the caller's buffers when it passed them (verify
    # and prefill hand those to store_swa_into_unified); decode only wants the
    # fused ring write, so it lets aiter allocate them and drops them.
    q_packed, _, _, _ = fused_qk_norm_rope_group_quant(
        q.view(q_dst.shape),
        kv,
        kv_norm_weight,
        positions,
        cos_cache,
        sin_cache,
        rms_eps,
        is_neox=False,
        q_nope_scale_buff=q_dst,
        k_nope_scale_buff=k_nope_out,
        k_rope_buff=k_rope_out,
        q_weight=q_norm_weight,
        quant_group_size=64,
        scale_dtype="e8m0",
        swa_nope_scale_buff=swa_cache,
        swa_rope_buff=swa_rope_cache,
        swa_dest_row=swa_loc,
        batch_id_per_token=batch_id,
    )
    if q_dst is not q_out:
        q_out.copy_(q_packed)
        return q_out
    return q_packed


def fused_qk_norm_rope_swa_store(
    q: torch.Tensor,
    kv: torch.Tensor,
    q_norm_weight: Optional[torch.Tensor],
    kv_norm_weight: Optional[torch.Tensor],
    q_rms_eps: float,
    kv_rms_eps: float,
    rope_head_dim: int,
    cos_cache: torch.Tensor,
    sin_cache: torch.Tensor,
    positions: torch.Tensor,
    swa_cache: Optional[torch.Tensor] = None,
    swa_loc: Optional[torch.Tensor] = None,
    swa_page_size: int = 128,
    q_out: Optional[torch.Tensor] = None,
    dtype: torch.dtype = torch.bfloat16,
    bf16_store: bool = False,
    fp8_2buff: bool = False,
    swa_rope_cache: Optional[torch.Tensor] = None,
    k_nope_out: Optional[torch.Tensor] = None,
    k_rope_out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Fused Q norm + KV norm + RoPE + optional SWA store.

    Args:
        q: [M, N] or [splitk, M, N] where N = num_local_heads * head_dim
        kv: [M, head_dim=512] mutated in-place (norm + RoPE)
        swa_cache: paged SWA KV pool buffer [num_pages, bytes_per_page] uint8 OR a plain [num_slots, head_dim] bf16 cache
        swa_loc: [M] int32 pre-translated paged indices
        swa_page_size: tokens per SWA page (default 128)
        bf16_store: write the whole head_dim as plain bf16 at swa_cache[swa_loc]
        fp8_2buff: two-pool fp8 unified_kv. Delegates to aiter; ``swa_cache`` is
            the fp8 nope pool and ``swa_rope_cache`` the bf16 rope pool, both
            addressed by ``swa_loc``. Q stays bf16. Unlike the Triton path this
            leaves ``kv`` untouched -- the normed + RoPE'd K comes back packed in
            ``k_nope_out`` / ``k_rope_out`` when the caller supplies them.
    """
    head_dim = kv.shape[1]

    if q.dim() == 3:
        num_splitk, M, N = q.shape
        q_in_splitk_stride = q.stride(0)
        q_in_m_stride = q.stride(1)
        q_in_d_stride = q.stride(2)
    else:
        M, N = q.shape
        num_splitk = 1
        q_in_splitk_stride = 0
        q_in_m_stride = q.stride(0)
        q_in_d_stride = q.stride(1)

    num_local_heads = N // head_dim

    if q_out is None:
        q_out = torch.empty(
            (M, num_local_heads, head_dim), dtype=dtype, device=q.device
        )

    if fp8_2buff:
        assert not bf16_store, "fp8_2buff and bf16_store are different stores"
        assert (
            q_rms_eps == kv_rms_eps
        ), f"aiter norms Q and K with one eps, got {q_rms_eps} / {kv_rms_eps}"
        return _fp8_2buff_store(
            q,
            kv,
            q_norm_weight,
            kv_norm_weight,
            kv_rms_eps,
            rope_head_dim,
            cos_cache,
            sin_cache,
            positions,
            q_out,
            swa_cache,
            swa_rope_cache,
            swa_loc,
            k_nope_out,
            k_rope_out,
        )

    HAS_SWA_STORE = swa_cache is not None and swa_loc is not None

    dim_nope = 448
    dim_rope = 64
    tile_size = 64
    num_nope_tiles = dim_nope // tile_size
    scale_pad = 1
    bytes_per_token = dim_nope + dim_rope * 2 + num_nope_tiles + scale_pad

    if _fp8_fnuz:
        fp8_info = torch.finfo(torch.float8_e4m3fnuz)
    else:
        fp8_info = torch.finfo(torch.float8_e4m3fn)

    BLOCK_SIZE_M = min(4, triton.next_power_of_2(M)) if M < 4 else 4
    num_warps = 4

    grid = (triton.cdiv(M, BLOCK_SIZE_M), num_local_heads + 1)
    _fused_qk_norm_rope_store_kernel[grid](
        q,
        q_out,
        kv,
        q_norm_weight,
        kv_norm_weight,
        positions,
        cos_cache,
        sin_cache,
        swa_cache if HAS_SWA_STORE else None,
        swa_loc if HAS_SWA_STORE else None,
        M,
        q_in_splitk_stride,
        q_in_m_stride,
        q_in_d_stride,
        q_out.stride(0),
        q_out.stride(1),
        q_out.stride(2),
        kv.stride(0),
        kv.stride(1),
        cos_cache.stride(0),
        cos_cache.stride(-1),
        swa_cache.stride(0) if HAS_SWA_STORE else 0,
        q_rms_eps,
        kv_rms_eps,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        HEAD_DIM=head_dim,
        ROPE_DIM=rope_head_dim,
        NUM_LOCAL_HEADS=num_local_heads,
        NUM_SPLITK=num_splitk,
        HAS_SWA_STORE=HAS_SWA_STORE,
        DIM_NOPE=dim_nope,
        TILE_SIZE=tile_size,
        NUM_NOPE_TILES=num_nope_tiles,
        FP8_MIN=fp8_info.min,
        FP8_MAX=fp8_info.max,
        BYTES_PER_TOKEN=bytes_per_token,
        SWA_PAGE_SIZE=swa_page_size,
        BF16_STORE=bf16_store,
        IS_FNUZ=_fp8_fnuz,
        num_warps=num_warps,
    )
    return q_out
