import torch
import triton
import triton.language as tl

from sglang.srt.utils import is_cpu

_is_cpu = is_cpu()

if _is_cpu:
    from sgl_kernel import copy_all_layer_kv_cache_cpu


@triton.jit
def set_kv_buffer_prefix_valid_tiled(
    src_k_ptr,
    src_v_ptr,
    dst_k_ptr,
    dst_v_ptr,
    loc_2d_ptr,
    commit_len_ptr,
    src_k_row_stride,
    src_v_row_stride,
    dst_k_row_stride,
    dst_v_row_stride,
    block_size,
    ROW_BYTES: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    bid = tl.program_id(0)
    row = tl.program_id(1)
    tid = tl.program_id(2)

    commit_len = tl.load(commit_len_ptr + bid)
    if row >= commit_len:
        return

    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < ROW_BYTES
    tl.multiple_of(byte_off, 16)

    loc = tl.load(loc_2d_ptr + bid * block_size + row)
    src_row = bid * block_size + row

    src_k_ptr = tl.cast(src_k_ptr, tl.pointer_type(tl.uint8))
    src_v_ptr = tl.cast(src_v_ptr, tl.pointer_type(tl.uint8))
    dst_k_ptr = tl.cast(dst_k_ptr, tl.pointer_type(tl.uint8))
    dst_v_ptr = tl.cast(dst_v_ptr, tl.pointer_type(tl.uint8))

    src_k_row_ptr = src_k_ptr + src_row * src_k_row_stride + byte_off
    src_v_row_ptr = src_v_ptr + src_row * src_v_row_stride + byte_off
    dst_k_row_ptr = dst_k_ptr + loc * dst_k_row_stride + byte_off
    dst_v_row_ptr = dst_v_ptr + loc * dst_v_row_stride + byte_off

    k_val = tl.load(src_k_row_ptr, mask=mask_byte, other=0)
    v_val = tl.load(src_v_row_ptr, mask=mask_byte, other=0)
    tl.store(dst_k_row_ptr, k_val, mask=mask_byte)
    tl.store(dst_v_row_ptr, v_val, mask=mask_byte)


@triton.jit
def _saturate_to_fp8_range(val, FP8_MAX: tl.constexpr, DTYPE: tl.constexpr):
    """Clamp into the FP8 range, leaving NaN untouched.

    Mirrors ``saturate_to_fp8_range`` on the eager side. Comparisons against
    NaN are false, so a NaN input falls through both ``tl.where`` arms and
    still converts to the FP8 NaN encoding.

    Each arm casts back to ``DTYPE`` because the FP8_MAX literal is FP32 and
    would otherwise widen the result. Keeping the value in the source dtype
    matters: the final cast to FP8 must start from the same type it did before
    this clamp existed, or it can lower to a different conversion with its own
    tie-breaking in the FP8 subnormal range. The limit itself is exact in
    BF16/FP16/FP32, so the round trip loses nothing.
    """
    val = tl.where(val > FP8_MAX, FP8_MAX, val).to(DTYPE)
    return tl.where(val < -FP8_MAX, -FP8_MAX, val).to(DTYPE)


@triton.jit
def _quantize_kv_fp8(
    val,
    scale,
    SCALE_IS_TENSOR: tl.constexpr,
    SRC_DTYPE: tl.constexpr,
    DST_DTYPE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    val = val.to(tl.float32)
    if SCALE_IS_TENSOR:
        # Tensor division rounds the divisor first and uses correctly rounded division.
        val = tl.div_rn(val, scale.to(SRC_DTYPE).to(tl.float32))
    else:
        # Eager host-scalar division multiplies by a rounded FP32 reciprocal.
        val = val * tl.div_rn(1.0, scale)
    val = val.to(SRC_DTYPE)
    return _saturate_to_fp8_range(val, FP8_MAX, SRC_DTYPE).to(DST_DTYPE)


@triton.jit
def set_kv_buffer_prefix_valid_tiled_fp8(
    src_k_ptr,
    src_v_ptr,
    dst_k_ptr,
    dst_v_ptr,
    loc_2d_ptr,
    commit_len_ptr,
    k_scale,
    v_scale,
    src_k_row_stride,
    src_v_row_stride,
    dst_k_row_stride,
    dst_v_row_stride,
    block_size,
    ROW_ELEMS: tl.constexpr,
    ELEMS_PER_TILE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    K_SCALE_IS_TENSOR: tl.constexpr = False,
    V_SCALE_IS_TENSOR: tl.constexpr = False,
):
    bid = tl.program_id(0)
    row = tl.program_id(1)
    tid = tl.program_id(2)

    commit_len = tl.load(commit_len_ptr + bid)
    if row >= commit_len:
        return

    elem_off = tid * ELEMS_PER_TILE + tl.arange(0, ELEMS_PER_TILE)
    mask_elem = elem_off < ROW_ELEMS

    loc = tl.load(loc_2d_ptr + bid * block_size + row)
    src_row = bid * block_size + row

    src_k_row_ptr = src_k_ptr + src_row * src_k_row_stride + elem_off
    src_v_row_ptr = src_v_ptr + src_row * src_v_row_stride + elem_off
    dst_k_row_ptr = dst_k_ptr + loc * dst_k_row_stride + elem_off
    dst_v_row_ptr = dst_v_ptr + loc * dst_v_row_stride + elem_off

    k_val = _quantize_kv_fp8(
        tl.load(src_k_row_ptr, mask=mask_elem, other=0),
        k_scale,
        K_SCALE_IS_TENSOR,
        src_k_ptr.dtype.element_ty,
        dst_k_ptr.dtype.element_ty,
        FP8_MAX,
    )
    v_val = _quantize_kv_fp8(
        tl.load(src_v_row_ptr, mask=mask_elem, other=0),
        v_scale,
        V_SCALE_IS_TENSOR,
        src_v_ptr.dtype.element_ty,
        dst_v_ptr.dtype.element_ty,
        FP8_MAX,
    )

    tl.store(dst_k_row_ptr, k_val, mask=mask_elem)
    tl.store(dst_v_row_ptr, v_val, mask=mask_elem)


@triton.jit
def set_kv_buffer_tiled_fp8(
    src_k_ptr,
    src_v_ptr,
    dst_k_ptr,
    dst_v_ptr,
    loc_ptr,
    k_scale,
    v_scale,
    src_k_row_stride,
    src_v_row_stride,
    dst_k_row_stride,
    dst_v_row_stride,
    size_limit,
    ROW_ELEMS: tl.constexpr,
    ELEMS_PER_TILE: tl.constexpr,
    FP8_MAX: tl.constexpr,
    K_SCALE_IS_PTR: tl.constexpr = False,
    V_SCALE_IS_PTR: tl.constexpr = False,
):
    row = tl.program_id(0).to(tl.int64)
    loc = tl.load(loc_ptr + row).to(tl.int64)
    tl.device_assert((loc >= 0) & (loc < size_limit), "KV slot out of range")
    # Slot 0 is the reserved padding slot (store_cache's reserved_skip_index):
    # padded tokens point there, and their source rows may be garbage.
    if loc == 0:
        return

    elem_off = tl.program_id(1) * ELEMS_PER_TILE + tl.arange(0, ELEMS_PER_TILE)
    mask_elem = elem_off < ROW_ELEMS

    # A device scale is read here, so a captured CUDA graph needs no host copy.
    if K_SCALE_IS_PTR:
        k_scale = tl.load(k_scale)
    if V_SCALE_IS_PTR:
        v_scale = tl.load(v_scale)

    k_val = _quantize_kv_fp8(
        tl.load(src_k_ptr + row * src_k_row_stride + elem_off, mask=mask_elem, other=0),
        k_scale,
        K_SCALE_IS_PTR,
        src_k_ptr.dtype.element_ty,
        dst_k_ptr.dtype.element_ty,
        FP8_MAX,
    )
    v_val = _quantize_kv_fp8(
        tl.load(src_v_ptr + row * src_v_row_stride + elem_off, mask=mask_elem, other=0),
        v_scale,
        V_SCALE_IS_PTR,
        src_v_ptr.dtype.element_ty,
        dst_v_ptr.dtype.element_ty,
        FP8_MAX,
    )

    tl.store(dst_k_ptr + loc * dst_k_row_stride + elem_off, k_val, mask=mask_elem)
    tl.store(dst_v_ptr + loc * dst_v_row_stride + elem_off, v_val, mask=mask_elem)


@triton.jit
def copy_all_layer_kv_cache_tiled(
    data_ptrs,
    strides,
    tgt_loc_ptr,
    src_loc_ptr,
    num_locs,
    num_locs_upper: tl.constexpr,
    BYTES_PER_TILE: tl.constexpr,
):
    """2D tiled kernel. Safe for in-place copy."""
    bid = tl.program_id(0)
    tid = tl.program_id(1)

    stride = tl.load(strides + bid)
    base_ptr = tl.load(data_ptrs + bid)
    base_ptr = tl.cast(base_ptr, tl.pointer_type(tl.uint8))

    byte_off = tid * BYTES_PER_TILE + tl.arange(0, BYTES_PER_TILE)
    mask_byte = byte_off < stride
    tl.multiple_of(byte_off, 16)

    loc_idx = tl.arange(0, num_locs_upper)
    mask_loc = loc_idx < num_locs

    src = tl.load(src_loc_ptr + loc_idx, mask=mask_loc, other=0)
    tgt = tl.load(tgt_loc_ptr + loc_idx, mask=mask_loc, other=0)

    src_ptr = base_ptr + src[:, None] * stride + byte_off[None, :]
    tgt_ptr = base_ptr + tgt[:, None] * stride + byte_off[None, :]

    mask = mask_loc[:, None] & mask_byte[None, :]
    vals = tl.load(src_ptr, mask=mask)
    tl.store(tgt_ptr, vals, mask=mask)


def copy_all_layer_kv_cache_func(
    data_ptrs: torch.Tensor,
    strides: torch.Tensor,
    tgt_loc: torch.Tensor,
    src_loc: torch.Tensor,
    num_locs: int,
    num_locs_upper: int,
    kv_copy_config: dict,
):
    if _is_cpu:
        copy_all_layer_kv_cache_cpu(
            data_ptrs,
            strides,
            tgt_loc[:num_locs],
            src_loc[:num_locs],
        )
        return
    grid = (data_ptrs.numel(), kv_copy_config["byte_tiles"])
    copy_all_layer_kv_cache_tiled[grid](
        data_ptrs,
        strides,
        tgt_loc,
        src_loc,
        num_locs,
        num_locs_upper,
        BYTES_PER_TILE=kv_copy_config["bytes_per_tile"],
        num_warps=kv_copy_config["num_warps"],
        num_stages=2,
    )
