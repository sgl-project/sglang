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


@triton.jit
def store_k_slots_kernel(
    k_buffer_ptr,
    src_ptr,
    loc_ptr,
    stride_dst_slot,
    stride_src_row,
    ROW_DIM: tl.constexpr,  # head_num * head_dim
    BLOCK: tl.constexpr,
):
    """Writes ``k_buffer[loc[i]] = src[i]``, one program per (token, row block).

    Grid ``(N, ceil(ROW_DIM / BLOCK))``. CUDA-graph safe: no host branching on tensor
    values, no ``.item()``.
    """
    pid_n = tl.program_id(0)
    pid_b = tl.program_id(1)

    loc = tl.load(loc_ptr + pid_n).to(tl.int64)
    # Negative slot = skip, matching reshape_and_cache_flash. Note ATen advanced
    # indexing would wrap instead, so a fallback caller is not equivalent here.
    if loc < 0:
        return

    off = pid_b * BLOCK + tl.arange(0, BLOCK)
    mask = off < ROW_DIM
    src = tl.load(src_ptr + pid_n * stride_src_row + off, mask=mask)
    tl.store(k_buffer_ptr + loc * stride_dst_slot + off, src, mask=mask)


def store_k_slots(k_buffer: torch.Tensor, src: torch.Tensor, loc: torch.Tensor) -> None:
    """Scatter ``src[i]`` into slot-major ``k_buffer[loc[i]]`` in place, one launch.

    Negative ``loc`` entries are skipped. The trailing ``(head_num, head_dim)`` dims must
    be contiguous, so the kernel can treat them as one flat axis.
    """
    if loc.numel() == 0:
        return
    assert k_buffer.ndim == src.ndim == 3, (
        f"store_k_slots: k_buffer/src must be 3-D, got {k_buffer.ndim}/{src.ndim}"
    )
    assert k_buffer.dtype == src.dtype, (
        f"store_k_slots: dtype mismatch: {k_buffer.dtype} vs {src.dtype}"
    )
    assert k_buffer.shape[1:] == src.shape[1:], (
        f"store_k_slots: row shape mismatch: {tuple(k_buffer.shape)} vs "
        f"{tuple(src.shape)}"
    )
    assert src.shape[0] == loc.numel(), (
        f"store_k_slots: src/loc batch mismatch: {src.shape[0]} vs {loc.numel()}"
    )
    for name, t in (("k_buffer", k_buffer), ("src", src)):
        assert t.stride(-1) == 1 and t.stride(-2) == t.shape[-1], (
            f"store_k_slots: {name} trailing dims must be contiguous; "
            f"got stride={t.stride()}, shape={tuple(t.shape)}"
        )

    ROW_DIM = k_buffer.shape[1] * k_buffer.shape[2]
    BLOCK = 128
    grid = (loc.numel(), triton.cdiv(ROW_DIM, BLOCK))
    store_k_slots_kernel[grid](
        k_buffer,
        src,
        loc,
        k_buffer.stride(0),
        src.stride(0),
        ROW_DIM=ROW_DIM,
        BLOCK=BLOCK,
        num_warps=4,
    )
