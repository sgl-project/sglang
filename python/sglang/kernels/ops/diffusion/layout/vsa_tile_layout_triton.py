"""Pack 4x4x4 VSA tiles directly into contiguous BHSD storage.

Partial tiles contain their valid tokens first, followed by zeros, matching
the reference gather/scatter (not geometric zero padding inside each tile).
"""

import math

import torch
import triton
import triton.language as tl


@triton.jit
def _vsa_tile_to_bhsd_kernel(
    x_ptr,
    out_ptr,
    stride_batch: tl.constexpr,
    stride_seq: tl.constexpr,
    stride_head: tl.constexpr,
    stride_dim: tl.constexpr,
    frames: tl.constexpr,
    height: tl.constexpr,
    width: tl.constexpr,
    num_heads: tl.constexpr,
    head_dim: tl.constexpr,
    padded_seq_len: tl.constexpr,
    BLOCK_SEQ: tl.constexpr,
):
    seq = tl.program_id(0) * BLOCK_SEQ + tl.arange(0, BLOCK_SEQ)
    batch_head = tl.program_id(1)
    batch, head = batch_head // num_heads, batch_head % num_heads
    dim = tl.arange(0, head_dim)

    tile_index, local_index = seq // 64, seq % 64
    height_tiles, width_tiles = tl.cdiv(height, 4), tl.cdiv(width, 4)
    tile_start_t = (tile_index // (height_tiles * width_tiles)) * 4
    tile_start_h = ((tile_index // width_tiles) % height_tiles) * 4
    tile_start_w = (tile_index % width_tiles) * 4
    valid_frames = tl.minimum(frames - tile_start_t, 4)
    valid_height = tl.minimum(height - tile_start_h, 4)
    valid_width = tl.minimum(width - tile_start_w, 4)

    # Partial tiles pack valid tokens first, with padding at the end of the tile.
    local_t = local_index // (valid_height * valid_width)
    local_h = (local_index // valid_width) % valid_height
    local_w = local_index % valid_width
    src_t = tile_start_t + local_t
    src_h = tile_start_h + local_h
    src_w = tile_start_w + local_w
    src_index = (src_t * height + src_h) * width + src_w
    valid_token = (seq < padded_seq_len) & (
        local_index < valid_frames * valid_height * valid_width
    )

    src_offset = (
        batch.to(tl.int64) * stride_batch
        + src_index[:, None].to(tl.int64) * stride_seq
        + head.to(tl.int64) * stride_head
        + dim[None, :].to(tl.int64) * stride_dim
    )
    value = tl.load(
        x_ptr + src_offset,
        mask=valid_token[:, None] & (dim[None, :] < head_dim),
        other=0,
    )
    dst_offset = (batch_head.to(tl.int64) * padded_seq_len + seq[:, None]) * head_dim
    dst_offset += dim[None, :]
    tl.store(
        out_ptr + dst_offset,
        value,
        mask=(seq[:, None] < padded_seq_len) & (dim[None, :] < head_dim),
    )


def vsa_tile_to_bhsd(
    x: torch.Tensor,
    dit_seq_shape: tuple[int, int, int],
    out: torch.Tensor | None = None,
) -> torch.Tensor:
    """Copy BSHD input into tiled BHSD output; all padding is written each call.

    Supports strided CUDA FP16/BF16/FP32 inputs and head dimensions 64/128.
    This inference-only operation does not compute gradients. ``out``, when
    supplied, must be contiguous and must not alias ``x``.
    """
    if x.ndim != 4 or min(x.shape) <= 0:
        raise ValueError("x must be a nonempty 4D BSHD tensor")
    if not x.is_cuda or torch.version.hip is not None:
        raise ValueError("x must be on a CUDA device; ROCm is not supported")
    if x.dtype not in (torch.float16, torch.bfloat16, torch.float32) or x.requires_grad:
        raise ValueError("x must be FP16, BF16 or FP32 without gradients")
    if x.shape[-1] not in (64, 128):
        raise ValueError("head dimension must be 64 or 128")
    if len(dit_seq_shape) != 3 or min(dit_seq_shape) <= 0:
        raise ValueError("dit_seq_shape must contain three positive dimensions")
    if x.shape[1] != math.prod(dit_seq_shape):
        raise ValueError("sequence length must equal the product of dit_seq_shape")

    batch, _, num_heads, head_dim = x.shape
    frames, height, width = dit_seq_shape
    padded_seq_len = math.prod(triton.cdiv(n, 4) for n in dit_seq_shape) * 64
    shape = (batch, num_heads, padded_seq_len, head_dim)
    if out is None:
        out = torch.empty(shape, dtype=x.dtype, device=x.device)
    else:
        if out.shape != shape or not out.is_contiguous():
            raise ValueError(
                "out must be contiguous BHSD storage with the expected shape"
            )
        if out.dtype != x.dtype or out.device != x.device:
            raise ValueError("out must share the input dtype and device")
        if (
            out.requires_grad
            or out.untyped_storage().data_ptr() == x.untyped_storage().data_ptr()
        ):
            raise ValueError("out must not require gradients or share storage with x")

    block_seq = 32
    grid = (triton.cdiv(padded_seq_len, block_seq), batch * num_heads)
    with torch.cuda.device(x.device):
        _vsa_tile_to_bhsd_kernel[grid](
            x,
            out,
            *x.stride(),
            frames,
            height,
            width,
            num_heads,
            head_dim,
            padded_seq_len,
            BLOCK_SEQ=block_seq,
            num_warps=4,
        )
    return out
