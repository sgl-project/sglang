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
    value,
    scale,
    SCALE_IS_TENSOR: tl.constexpr,
    SRC_DTYPE: tl.constexpr,
    DST_DTYPE: tl.constexpr,
    FP8_MAX: tl.constexpr,
):
    # Preserve eager div_'s scale-kind and source-dtype rounding before conversion.
    value = value.to(tl.float32)
    if SCALE_IS_TENSOR:
        scale = scale.to(SRC_DTYPE).to(tl.float32)
        value = tl.div_rn(value, scale)
    else:
        value = value * tl.div_rn(1.0, scale)
    value = value.to(SRC_DTYPE)
    return _saturate_to_fp8_range(value, FP8_MAX, SRC_DTYPE).to(DST_DTYPE)


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
def _store_cache_fp8(
    k,
    v,
    k_cache,
    v_cache,
    indices,
    k_scale,
    v_scale,
    k_stride,
    v_stride,
    k_cache_stride,
    v_cache_stride,
    index_stride,
    K_WIDTH: tl.constexpr,
    V_WIDTH: tl.constexpr,
    K_SCALE_IS_TENSOR: tl.constexpr,
    V_SCALE_IS_TENSOR: tl.constexpr,
    FP8_MAX: tl.constexpr,
    SIZE_LIMIT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    slot = tl.load(indices + row * index_stride).to(tl.int64)
    # Slot zero is the ordinary writer's reserved CUDA-graph padding slot.
    valid = (slot > 0) & (slot < SIZE_LIMIT)
    tl.device_assert((slot >= 0) & (slot < SIZE_LIMIT), "KV slot out of bounds")
    offsets = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    if K_SCALE_IS_TENSOR:
        k_scale = tl.load(k_scale)
    if V_SCALE_IS_TENSOR:
        v_scale = tl.load(v_scale)
    kk = _quantize_kv_fp8(
        tl.load(k + row * k_stride + offsets, (offsets < K_WIDTH) & valid, 0),
        k_scale,
        K_SCALE_IS_TENSOR,
        k.dtype.element_ty,
        k_cache.dtype.element_ty,
        FP8_MAX,
    )
    vv = _quantize_kv_fp8(
        tl.load(v + row * v_stride + offsets, (offsets < V_WIDTH) & valid, 0),
        v_scale,
        V_SCALE_IS_TENSOR,
        v.dtype.element_ty,
        v_cache.dtype.element_ty,
        FP8_MAX,
    )
    tl.store(k_cache + slot * k_cache_stride + offsets, kk, (offsets < K_WIDTH) & valid)
    tl.store(v_cache + slot * v_cache_stride + offsets, vv, (offsets < V_WIDTH) & valid)


def store_cache_fp8(k, v, k_cache, v_cache, indices, k_scale, v_scale):
    """Store E4M3 NHD rows on CUDA SM89+, using prefix-commit numerics.

    Inputs are not mutated. Sources have contiguous head/dimension axes;
    token strides may differ. Scales
    are host numbers, None, or scalar FP32 device tensors. Slot zero is reserved.
    The pool owns layout and scale eligibility; other forms use its eager writer.
    """
    if indices.numel() == 0:
        return
    k_width = k.shape[-2] * k.shape[-1]
    v_width = v.shape[-2] * v.shape[-1]
    block = 256
    _store_cache_fp8[(indices.numel(), triton.cdiv(max(k_width, v_width), block))](
        k,
        v,
        k_cache,
        v_cache,
        indices,
        1.0 if k_scale is None else k_scale,
        1.0 if v_scale is None else v_scale,
        k.stride(0),
        v.stride(0),
        k_cache.stride(0),
        v_cache.stride(0),
        indices.stride(0),
        K_WIDTH=k_width,
        V_WIDTH=v_width,
        K_SCALE_IS_TENSOR=isinstance(k_scale, torch.Tensor),
        V_SCALE_IS_TENSOR=isinstance(v_scale, torch.Tensor),
        FP8_MAX=torch.finfo(k_cache.dtype).max,
        SIZE_LIMIT=k_cache.shape[0],
        BLOCK=block,
        debug=True,
    )


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
