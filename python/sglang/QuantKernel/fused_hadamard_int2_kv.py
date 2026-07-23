"""
Fused Hadamard (FWHT) + int2 KV cache write (Triton).

Packs 4 x 2-bit values per byte (4 levels, scale = range / 3). Rotation is
mandatory for int2.

``hadamard_order`` must be a power of two with ``2 <= hadamard_order <=
MAX_HADAMARD_ORDER`` and ``head_dim % hadamard_order == 0``.
"""

from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import triton
import triton.language as tl

from sglang.srt.mem_cache.kv_quant_kernels import (
    _get_num_scale_groups,
    _is_power_of_two,
    _quantized_set_kv_int2_kernel,
    quantized_set_kv_int2_triton,
)

# Upper bound for a single FWHT block (power-of-two only).
MAX_HADAMARD_ORDER = 4096


def _validate_hadamard_order_impl(hadamard_order: int, head_dim: int) -> None:
    if hadamard_order < 2:
        raise ValueError(f"hadamard_order must be >= 2, got {hadamard_order}")
    if hadamard_order & (hadamard_order - 1):
        raise ValueError(f"hadamard_order must be a power of two, got {hadamard_order}")
    if hadamard_order > MAX_HADAMARD_ORDER:
        raise ValueError(
            f"hadamard_order must be <= {MAX_HADAMARD_ORDER} (FWHT segment size), got {hadamard_order}"
        )
    if head_dim % hadamard_order:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by hadamard_order ({hadamard_order})"
        )


@triton.jit
def _fwht_blocked_segments_tensor(x, head_dim_: tl.constexpr, LOG: tl.constexpr):
    """FWHT on each contiguous block of size ``2**LOG`` tiling ``head_dim_``."""
    i = tl.arange(0, head_dim_)
    for s in tl.static_range(0, LOG):
        stride = 1 << s
        partner = i ^ stride
        lo = tl.minimum(i, partner)
        hi = tl.maximum(i, partner)
        u0 = tl.gather(x, lo, 0)
        v0 = tl.gather(x, hi, 0)
        x = tl.where(i == lo, u0 + v0, u0 - v0)
    return x


def validate_hadamard_order_for_kv_fuse_int2(
    hadamard_order: int, head_dim: int
) -> None:
    """Raise ``ValueError`` if ``hadamard_order`` / ``head_dim`` are invalid."""
    _validate_hadamard_order_impl(hadamard_order, head_dim)


def _make_fused_kernel_int2(head_dim: int, hadamard_order: int):
    _validate_hadamard_order_impl(hadamard_order, head_dim)
    log_n = int(math.log2(hadamard_order))
    block_quarter = triton.next_power_of_2(head_dim // 4)
    pre_scale = 1.0 / math.sqrt(float(hadamard_order))

    @triton.jit
    def _fused_hadamard_int2_set_kv_kernel(
        input_ptr,
        loc_ptr,
        cache_ptr,
        scales_zeros_ptr,
        num_tokens,
        num_heads,
        head_dim_: tl.constexpr,
        input_stride_token,
        input_stride_head,
        input_stride_dim,
        cache_stride_loc,
        cache_stride_head,
        cache_stride_dim,
        sz_stride_loc,
        sz_stride_head,
        sz_stride_dim,
        LOG: tl.constexpr,
        PRE_SCALE: tl.constexpr,
        BLOCK_QUARTER: tl.constexpr,
        HP_OFFSET: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        if token_idx >= num_tokens or head_idx >= num_heads:
            return

        cache_loc = tl.load(loc_ptr + token_idx)
        # Early-return for HP-tier slots in the mixed pool: skip the FWHT +
        # pack + min/max work entirely. Each program is one (token, head)
        # so this is uniform across the program's lanes.
        if HP_OFFSET >= 0 and cache_loc >= HP_OFFSET:
            return
        active = (HP_OFFSET < 0) | (cache_loc < HP_OFFSET)

        # ---- Hadamard rotation (always on for int2) ----
        dim_full = tl.arange(0, head_dim_)
        input_off_base = token_idx * input_stride_token + head_idx * input_stride_head
        x_bf16 = tl.load(
            input_ptr + input_off_base + dim_full * input_stride_dim,
            mask=dim_full < head_dim_,
            other=0.0,
        ).to(tl.float32)
        x_scaled = x_bf16 * PRE_SCALE
        acc = _fwht_blocked_segments_tensor(x_scaled, head_dim_, LOG)

        # ---- INT2 quantization (4 quarters) ----
        quarter_dim = head_dim_ // 4
        dim_offsets = tl.arange(0, BLOCK_QUARTER)
        dim_mask = dim_offsets < quarter_dim

        safe_off0 = tl.where(dim_mask, dim_offsets, 0)
        safe_off1 = tl.where(dim_mask, dim_offsets + quarter_dim, 0)
        safe_off2 = tl.where(dim_mask, dim_offsets + 2 * quarter_dim, 0)
        safe_off3 = tl.where(dim_mask, dim_offsets + 3 * quarter_dim, 0)

        # bf16 round-trip to match unfused numerics
        vals0 = (
            tl.where(dim_mask, tl.gather(acc, safe_off0, 0), 0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        vals1 = (
            tl.where(dim_mask, tl.gather(acc, safe_off1, 0), 0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        vals2 = (
            tl.where(dim_mask, tl.gather(acc, safe_off2, 0), 0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )
        vals3 = (
            tl.where(dim_mask, tl.gather(acc, safe_off3, 0), 0.0)
            .to(tl.bfloat16)
            .to(tl.float32)
        )

        # Per-head min/max across all 4 quarters
        val_min = tl.minimum(
            tl.minimum(tl.min(vals0, axis=0), tl.min(vals1, axis=0)),
            tl.minimum(tl.min(vals2, axis=0), tl.min(vals3, axis=0)),
        )
        val_max = tl.maximum(
            tl.maximum(tl.max(vals0, axis=0), tl.max(vals1, axis=0)),
            tl.maximum(tl.max(vals2, axis=0), tl.max(vals3, axis=0)),
        )
        val_range = tl.maximum(val_max - val_min, 1e-8)
        scale = val_range / 3.0
        zero = -val_min / scale

        q0 = (vals0 / scale + zero + 0.5).to(tl.uint8)
        q1 = (vals1 / scale + zero + 0.5).to(tl.uint8)
        q2 = (vals2 / scale + zero + 0.5).to(tl.uint8)
        q3 = (vals3 / scale + zero + 0.5).to(tl.uint8)
        packed = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)

        cache_offset = (
            cache_loc * cache_stride_loc
            + head_idx * cache_stride_head
            + dim_offsets * cache_stride_dim
        )
        tl.store(cache_ptr + cache_offset, packed, mask=active & dim_mask)
        sz_offset_base = cache_loc * sz_stride_loc + head_idx * sz_stride_head
        tl.store(
            scales_zeros_ptr + sz_offset_base + 0 * sz_stride_dim, scale, mask=active
        )
        tl.store(
            scales_zeros_ptr + sz_offset_base + 1 * sz_stride_dim, zero, mask=active
        )

    return _fused_hadamard_int2_set_kv_kernel, {
        "head_dim_": head_dim,
        "LOG": log_n,
        "PRE_SCALE": pre_scale,
        "BLOCK_QUARTER": block_quarter,
    }


def _make_fused_grouped_kernel_int2(
    head_dim: int, hadamard_order: int, num_groups: int
):
    _validate_hadamard_order_impl(hadamard_order, head_dim)
    if head_dim % num_groups != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by num_groups ({num_groups})"
        )
    group_size = head_dim // num_groups
    log_n = int(math.log2(hadamard_order))
    block_quarter = triton.next_power_of_2(head_dim // 4)
    pre_scale = 1.0 / math.sqrt(float(hadamard_order))

    @triton.jit
    def _fused_hadamard_int2_grouped_set_kv_kernel(
        input_ptr,
        loc_ptr,
        cache_ptr,
        scales_zeros_ptr,
        num_tokens,
        num_heads,
        head_dim_: tl.constexpr,
        input_stride_token,
        input_stride_head,
        input_stride_dim,
        cache_stride_loc,
        cache_stride_head,
        cache_stride_dim,
        sz_stride_loc,
        sz_stride_head,
        sz_stride_dim,
        LOG: tl.constexpr,
        PRE_SCALE: tl.constexpr,
        BLOCK_QUARTER: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        HP_OFFSET: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        if token_idx >= num_tokens or head_idx >= num_heads:
            return

        cache_loc = tl.load(loc_ptr + token_idx)
        # Early-return for HP-tier slots — see note in the single-group
        # variant ``_fused_hadamard_int2_set_kv_kernel``.
        if HP_OFFSET >= 0 and cache_loc >= HP_OFFSET:
            return
        active = (HP_OFFSET < 0) | (cache_loc < HP_OFFSET)

        dim_full = tl.arange(0, head_dim_)
        input_off_base = token_idx * input_stride_token + head_idx * input_stride_head
        x_bf16 = tl.load(
            input_ptr + input_off_base + dim_full * input_stride_dim,
            mask=dim_full < head_dim_,
            other=0.0,
        ).to(tl.float32)
        acc = _fwht_blocked_segments_tensor(x_bf16 * PRE_SCALE, head_dim_, LOG)

        # Match the standalone hadamard -> bf16 tensor -> triton quantize path.
        acc = acc.to(tl.bfloat16).to(tl.float32)
        grouped = tl.reshape(acc, (NUM_GROUPS, GROUP_SIZE))
        val_min = tl.min(grouped, axis=1)
        val_max = tl.max(grouped, axis=1)
        scale = tl.maximum(val_max - val_min, 1e-8) / 3.0
        zero = tl.math.div_rn(-val_min, scale)

        quarter_dim = head_dim_ // 4
        dim_offsets = tl.arange(0, BLOCK_QUARTER)
        dim_mask = dim_offsets < quarter_dim

        safe_off0 = tl.where(dim_mask, dim_offsets, 0)
        safe_off1 = tl.where(dim_mask, dim_offsets + quarter_dim, 0)
        safe_off2 = tl.where(dim_mask, dim_offsets + 2 * quarter_dim, 0)
        safe_off3 = tl.where(dim_mask, dim_offsets + 3 * quarter_dim, 0)

        vals0 = tl.where(dim_mask, tl.gather(acc, safe_off0, 0), 0.0)
        vals1 = tl.where(dim_mask, tl.gather(acc, safe_off1, 0), 0.0)
        vals2 = tl.where(dim_mask, tl.gather(acc, safe_off2, 0), 0.0)
        vals3 = tl.where(dim_mask, tl.gather(acc, safe_off3, 0), 0.0)

        g0 = safe_off0 // GROUP_SIZE
        g1 = safe_off1 // GROUP_SIZE
        g2 = safe_off2 // GROUP_SIZE
        g3 = safe_off3 // GROUP_SIZE

        s0 = tl.gather(scale, g0, 0)
        s1 = tl.gather(scale, g1, 0)
        s2 = tl.gather(scale, g2, 0)
        s3 = tl.gather(scale, g3, 0)
        z0 = tl.gather(zero, g0, 0)
        z1 = tl.gather(zero, g1, 0)
        z2 = tl.gather(zero, g2, 0)
        z3 = tl.gather(zero, g3, 0)

        q0 = tl.minimum(tl.maximum(tl.math.div_rn(vals0, s0) + z0 + 0.5, 0.0), 3.0).to(
            tl.uint8
        )
        q1 = tl.minimum(tl.maximum(tl.math.div_rn(vals1, s1) + z1 + 0.5, 0.0), 3.0).to(
            tl.uint8
        )
        q2 = tl.minimum(tl.maximum(tl.math.div_rn(vals2, s2) + z2 + 0.5, 0.0), 3.0).to(
            tl.uint8
        )
        q3 = tl.minimum(tl.maximum(tl.math.div_rn(vals3, s3) + z3 + 0.5, 0.0), 3.0).to(
            tl.uint8
        )
        packed = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)

        cache_offset = (
            cache_loc * cache_stride_loc
            + head_idx * cache_stride_head
            + dim_offsets * cache_stride_dim
        )
        tl.store(cache_ptr + cache_offset, packed, mask=active & dim_mask)

        group_ids = tl.arange(0, NUM_GROUPS)
        sz_offset_base = cache_loc * sz_stride_loc + head_idx * sz_stride_head
        tl.store(
            scales_zeros_ptr + sz_offset_base + (group_ids * 2) * sz_stride_dim,
            scale,
            mask=active,
        )
        tl.store(
            scales_zeros_ptr + sz_offset_base + (group_ids * 2 + 1) * sz_stride_dim,
            zero,
            mask=active,
        )

    return _fused_hadamard_int2_grouped_set_kv_kernel, {
        "head_dim_": head_dim,
        "LOG": log_n,
        "PRE_SCALE": pre_scale,
        "BLOCK_QUARTER": block_quarter,
        "NUM_GROUPS": num_groups,
        "GROUP_SIZE": group_size,
    }


_KERNEL_CACHE_INT2: Dict[Tuple[int, int, int], Tuple] = {}
_KERNEL_REV_INT2 = 1
_GROUPED_KERNEL_CACHE_INT2: Dict[Tuple[int, int, int, int], Tuple] = {}
_GROUPED_KERNEL_REV_INT2 = 1
_PRETRANSFORMED_GROUPED_KERNEL_CACHE_INT2: Dict[Tuple[int, int, int], Tuple] = {}
_PRETRANSFORMED_GROUPED_KERNEL_REV_INT2 = 1


def _get_kernel_int2(head_dim: int, hadamard_order: int):
    k = (head_dim, hadamard_order, _KERNEL_REV_INT2)
    if k not in _KERNEL_CACHE_INT2:
        fn, cfg = _make_fused_kernel_int2(head_dim, hadamard_order)
        _KERNEL_CACHE_INT2[k] = (fn, cfg)
    return _KERNEL_CACHE_INT2[k]


def can_fuse_hadamard_grouped_int2(
    head_dim: int, scales_zeros_buffer: torch.Tensor
) -> bool:
    num_groups = _get_num_scale_groups(scales_zeros_buffer)
    if num_groups == 1:
        return True
    if head_dim % num_groups != 0:
        return False
    group_size = head_dim // num_groups
    return _is_power_of_two(num_groups) and _is_power_of_two(group_size)


def _get_grouped_kernel_int2(head_dim: int, hadamard_order: int, num_groups: int):
    k = (head_dim, hadamard_order, num_groups, _GROUPED_KERNEL_REV_INT2)
    if k not in _GROUPED_KERNEL_CACHE_INT2:
        fn, cfg = _make_fused_grouped_kernel_int2(head_dim, hadamard_order, num_groups)
        _GROUPED_KERNEL_CACHE_INT2[k] = (fn, cfg)
    return _GROUPED_KERNEL_CACHE_INT2[k]


def _make_pretransformed_grouped_kernel_int2(head_dim: int, num_groups: int):
    if head_dim % num_groups != 0:
        raise ValueError(
            f"head_dim ({head_dim}) must be divisible by num_groups ({num_groups})"
        )
    group_size = head_dim // num_groups
    block_quarter = triton.next_power_of_2(head_dim // 4)

    @triton.jit
    def _pretransformed_grouped_int2_set_kv_kernel(
        input_ptr,
        loc_ptr,
        cache_ptr,
        scales_zeros_ptr,
        num_tokens,
        num_heads,
        head_dim_: tl.constexpr,
        input_stride_token,
        input_stride_head,
        input_stride_dim,
        cache_stride_loc,
        cache_stride_head,
        cache_stride_dim,
        sz_stride_loc,
        sz_stride_head,
        sz_stride_dim,
        BLOCK_QUARTER: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        HP_OFFSET: tl.constexpr,
    ):
        token_idx = tl.program_id(0)
        head_idx = tl.program_id(1)
        if token_idx >= num_tokens or head_idx >= num_heads:
            return

        cache_loc = tl.load(loc_ptr + token_idx)
        # Early-return for HP-tier slots — see note in the fused
        # ``_fused_hadamard_int2_set_kv_kernel`` variant.
        if HP_OFFSET >= 0 and cache_loc >= HP_OFFSET:
            return
        active = (HP_OFFSET < 0) | (cache_loc < HP_OFFSET)

        dim_full = tl.arange(0, head_dim_)
        input_off_base = token_idx * input_stride_token + head_idx * input_stride_head
        acc = tl.load(
            input_ptr + input_off_base + dim_full * input_stride_dim,
            mask=dim_full < head_dim_,
            other=0.0,
        ).to(tl.float32)

        grouped = tl.reshape(acc, (NUM_GROUPS, GROUP_SIZE))
        val_min = tl.min(grouped, axis=1)
        val_max = tl.max(grouped, axis=1)
        scale = tl.maximum(val_max - val_min, 1e-8) / 3.0
        zero = tl.math.div_rn(-val_min, scale)

        quarter_dim = head_dim_ // 4
        dim_offsets = tl.arange(0, BLOCK_QUARTER)
        dim_mask = dim_offsets < quarter_dim

        safe_off0 = tl.where(dim_mask, dim_offsets, 0)
        safe_off1 = tl.where(dim_mask, dim_offsets + quarter_dim, 0)
        safe_off2 = tl.where(dim_mask, dim_offsets + 2 * quarter_dim, 0)
        safe_off3 = tl.where(dim_mask, dim_offsets + 3 * quarter_dim, 0)

        vals0 = tl.where(dim_mask, tl.gather(acc, safe_off0, 0), 0.0)
        vals1 = tl.where(dim_mask, tl.gather(acc, safe_off1, 0), 0.0)
        vals2 = tl.where(dim_mask, tl.gather(acc, safe_off2, 0), 0.0)
        vals3 = tl.where(dim_mask, tl.gather(acc, safe_off3, 0), 0.0)

        g0 = safe_off0 // GROUP_SIZE
        g1 = safe_off1 // GROUP_SIZE
        g2 = safe_off2 // GROUP_SIZE
        g3 = safe_off3 // GROUP_SIZE

        s0 = tl.gather(scale, g0, 0)
        s1 = tl.gather(scale, g1, 0)
        s2 = tl.gather(scale, g2, 0)
        s3 = tl.gather(scale, g3, 0)
        z0 = tl.gather(zero, g0, 0)
        z1 = tl.gather(zero, g1, 0)
        z2 = tl.gather(zero, g2, 0)
        z3 = tl.gather(zero, g3, 0)

        q0 = (tl.math.div_rn(vals0, s0) + z0 + 0.5).to(tl.uint8)
        q1 = (tl.math.div_rn(vals1, s1) + z1 + 0.5).to(tl.uint8)
        q2 = (tl.math.div_rn(vals2, s2) + z2 + 0.5).to(tl.uint8)
        q3 = (tl.math.div_rn(vals3, s3) + z3 + 0.5).to(tl.uint8)
        packed = q0 | (q1 << 2) | (q2 << 4) | (q3 << 6)

        cache_offset = (
            cache_loc * cache_stride_loc
            + head_idx * cache_stride_head
            + dim_offsets * cache_stride_dim
        )
        tl.store(cache_ptr + cache_offset, packed, mask=active & dim_mask)

        group_ids = tl.arange(0, NUM_GROUPS)
        sz_offset_base = cache_loc * sz_stride_loc + head_idx * sz_stride_head
        tl.store(
            scales_zeros_ptr + sz_offset_base + (group_ids * 2) * sz_stride_dim,
            scale,
            mask=active,
        )
        tl.store(
            scales_zeros_ptr + sz_offset_base + (group_ids * 2 + 1) * sz_stride_dim,
            zero,
            mask=active,
        )

    return _pretransformed_grouped_int2_set_kv_kernel, {
        "head_dim_": head_dim,
        "BLOCK_QUARTER": block_quarter,
        "NUM_GROUPS": num_groups,
        "GROUP_SIZE": group_size,
    }


def _get_pretransformed_grouped_kernel_int2(head_dim: int, num_groups: int):
    k = (head_dim, num_groups, _PRETRANSFORMED_GROUPED_KERNEL_REV_INT2)
    if k not in _PRETRANSFORMED_GROUPED_KERNEL_CACHE_INT2:
        fn, cfg = _make_pretransformed_grouped_kernel_int2(head_dim, num_groups)
        _PRETRANSFORMED_GROUPED_KERNEL_CACHE_INT2[k] = (fn, cfg)
    return _PRETRANSFORMED_GROUPED_KERNEL_CACHE_INT2[k]


def quantized_set_kv_int2_hadamard_fused_triton(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    k_cache_buffer: torch.Tensor,
    v_cache_buffer: torch.Tensor,
    k_scales_zeros_buffer: torch.Tensor,
    v_scales_zeros_buffer: torch.Tensor,
    hadamard_order: int,
    hp_global_offset=None,
) -> None:
    """Fused Hadamard + int2 pack for both K and V (rotation always on)."""
    num_tokens, num_heads, head_dim = cache_k.shape
    assert cache_v.shape == cache_k.shape
    assert head_dim % 4 == 0
    _validate_hadamard_order_impl(hadamard_order, head_dim)

    fused_grid = (num_tokens, num_heads)

    for data, buf, sz_buf in [
        (cache_k, k_cache_buffer, k_scales_zeros_buffer),
        (cache_v, v_cache_buffer, v_scales_zeros_buffer),
    ]:
        num_groups = _get_num_scale_groups(sz_buf)
        if num_groups == 1:
            kernel, cfg = _get_kernel_int2(head_dim, hadamard_order)
            kernel[fused_grid](
                data,
                loc,
                buf,
                sz_buf,
                num_tokens,
                num_heads,
                cfg["head_dim_"],
                data.stride(0),
                data.stride(1),
                data.stride(2),
                buf.stride(0),
                buf.stride(1),
                buf.stride(2),
                sz_buf.stride(0),
                sz_buf.stride(1),
                sz_buf.stride(2),
                LOG=cfg["LOG"],
                PRE_SCALE=cfg["PRE_SCALE"],
                BLOCK_QUARTER=cfg["BLOCK_QUARTER"],
                HP_OFFSET=-1 if hp_global_offset is None else int(hp_global_offset),
                num_warps=1,
                num_stages=1,
            )
            continue

        grouped_kernel, grouped_cfg = _get_grouped_kernel_int2(
            head_dim, hadamard_order, num_groups
        )
        grouped_kernel[fused_grid](
            data,
            loc,
            buf,
            sz_buf,
            num_tokens,
            num_heads,
            grouped_cfg["head_dim_"],
            data.stride(0),
            data.stride(1),
            data.stride(2),
            buf.stride(0),
            buf.stride(1),
            buf.stride(2),
            sz_buf.stride(0),
            sz_buf.stride(1),
            sz_buf.stride(2),
            LOG=grouped_cfg["LOG"],
            PRE_SCALE=grouped_cfg["PRE_SCALE"],
            BLOCK_QUARTER=grouped_cfg["BLOCK_QUARTER"],
            NUM_GROUPS=grouped_cfg["NUM_GROUPS"],
            GROUP_SIZE=grouped_cfg["GROUP_SIZE"],
            HP_OFFSET=-1 if hp_global_offset is None else int(hp_global_offset),
            num_warps=1,
            num_stages=1,
        )


def quantized_set_kv_int2_pretransformed_triton(
    cache_k: torch.Tensor,
    cache_v: torch.Tensor,
    loc: torch.Tensor,
    k_cache_buffer: torch.Tensor,
    v_cache_buffer: torch.Tensor,
    k_scales_zeros_buffer: torch.Tensor,
    v_scales_zeros_buffer: torch.Tensor,
    hp_global_offset=None,
) -> None:
    """Quantize/write K/V when the caller already applied int2 hadamard."""
    num_tokens, num_heads, head_dim = cache_k.shape
    assert cache_v.shape == cache_k.shape
    assert head_dim % 4 == 0

    if not (
        can_fuse_hadamard_grouped_int2(head_dim, k_scales_zeros_buffer)
        and can_fuse_hadamard_grouped_int2(head_dim, v_scales_zeros_buffer)
    ):
        quantized_set_kv_int2_triton(
            cache_k,
            cache_v,
            loc,
            k_cache_buffer,
            v_cache_buffer,
            k_scales_zeros_buffer,
            v_scales_zeros_buffer,
            hp_global_offset=hp_global_offset,
        )
        return

    grid = (num_tokens, num_heads)
    for data, buf, sz_buf in [
        (cache_k, k_cache_buffer, k_scales_zeros_buffer),
        (cache_v, v_cache_buffer, v_scales_zeros_buffer),
    ]:
        num_groups = _get_num_scale_groups(sz_buf)
        if num_groups == 1:
            block_size_dim = triton.next_power_of_2(head_dim // 4)
            _quantized_set_kv_int2_kernel[grid](
                data,
                loc,
                buf,
                sz_buf,
                num_tokens,
                num_heads,
                head_dim,
                data.stride(0),
                data.stride(1),
                data.stride(2),
                buf.stride(0),
                buf.stride(1),
                buf.stride(2),
                sz_buf.stride(0),
                sz_buf.stride(1),
                sz_buf.stride(2),
                HP_OFFSET=-1 if hp_global_offset is None else int(hp_global_offset),
                BLOCK_SIZE_DIM=block_size_dim,
                num_warps=1,
                num_stages=1,
            )
            continue

        kernel, cfg = _get_pretransformed_grouped_kernel_int2(head_dim, num_groups)
        kernel[grid](
            data,
            loc,
            buf,
            sz_buf,
            num_tokens,
            num_heads,
            cfg["head_dim_"],
            data.stride(0),
            data.stride(1),
            data.stride(2),
            buf.stride(0),
            buf.stride(1),
            buf.stride(2),
            sz_buf.stride(0),
            sz_buf.stride(1),
            sz_buf.stride(2),
            BLOCK_QUARTER=cfg["BLOCK_QUARTER"],
            NUM_GROUPS=cfg["NUM_GROUPS"],
            GROUP_SIZE=cfg["GROUP_SIZE"],
            HP_OFFSET=-1 if hp_global_offset is None else int(hp_global_offset),
            num_warps=1,
            num_stages=1,
        )
