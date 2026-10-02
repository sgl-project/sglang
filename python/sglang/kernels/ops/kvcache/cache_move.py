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
