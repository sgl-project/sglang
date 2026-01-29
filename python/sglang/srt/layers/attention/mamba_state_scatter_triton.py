"""
Fused Triton kernel for Mamba state scatter operations.

This kernel replaces the expensive advanced indexing operations in
`update_mamba_state_after_mtp_verify` with a single fused gather-scatter kernel,
avoiding multiple `index_elementwise_kernel` launches.
"""

import os

import torch
import triton
import triton.language as tl


@triton.jit
def _fused_mamba_state_scatter_kernel(
    # Source tensor (intermediate cache)
    src_ptr,
    # Destination tensor (mamba states)
    dst_ptr,
    # Index arrays
    dst_indices_ptr,  # [N] destination cache line indices
    src_indices_ptr,  # [N] source request indices
    step_indices_ptr,  # [N] source step indices
    # Dimensions
    num_valid,  # N - number of valid requests
    num_layers,  # L - number of mamba layers
    elem_per_entry: tl.constexpr,  # E - elements per (layer, request) entry
    # Source strides (intermediate cache: [L, S, D, ...])
    src_layer_stride,
    src_req_stride,
    src_step_stride,
    # Destination strides (mamba states: [L, C, ...])
    dst_layer_stride,
    dst_req_stride,
    # Bounds (sizes of indexed dimensions)
    src_req_size,  # S - src.shape[1]
    src_step_size,  # D - src.shape[2]
    dst_req_size,  # C - dst.shape[1]
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused gather-scatter kernel for mamba state updates.
    
    For each valid request i:
        dst[layer, dst_indices[i], :] = src[layer, src_indices[i], step_indices[i], :]
    
    Grid: (num_valid * num_layers, ceil(elem_per_entry / BLOCK_SIZE))
    """
    # Program ID encodes (valid_idx * num_layers + layer_idx, block_idx)
    pid_entry = tl.program_id(0)
    pid_block = tl.program_id(1)
    
    valid_idx = pid_entry // num_layers
    # Avoid `%` which is usually slower than mul-sub
    layer_idx = pid_entry - valid_idx * num_layers
    
    # Load indices for this valid request
    dst_idx = tl.load(dst_indices_ptr + valid_idx)
    src_idx = tl.load(src_indices_ptr + valid_idx)
    step_idx = tl.load(step_indices_ptr + valid_idx)
    
    # Match PyTorch indexing semantics for negative indices:
    # valid indices are in [-size, size-1] and negative values wrap from the end.
    dst_idx = tl.where(dst_idx < 0, dst_idx + dst_req_size, dst_idx)
    src_idx = tl.where(src_idx < 0, src_idx + src_req_size, src_idx)
    step_idx = tl.where(step_idx < 0, step_idx + src_step_size, step_idx)

    # Bounds check to avoid illegal memory access.
    # If any index is out of range, mask out all loads/stores for this entry.
    in_bounds = (
        (dst_idx >= 0)
        & (dst_idx < dst_req_size)
        & (src_idx >= 0)
        & (src_idx < src_req_size)
        & (step_idx >= 0)
        & (step_idx < src_step_size)
    )

    # Compute base offsets
    src_offset = (
        layer_idx * src_layer_stride
        + src_idx * src_req_stride
        + step_idx * src_step_stride
    )
    dst_offset = layer_idx * dst_layer_stride + dst_idx * dst_req_stride
    
    # Compute element range for this block
    start = pid_block * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < elem_per_entry) & in_bounds
    
    # Load from source and store to destination
    data = tl.load(src_ptr + src_offset + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offset + offsets, data, mask=mask)


def fused_mamba_state_scatter(
    dst: torch.Tensor,  # [num_layers, cache_size, *state_shape]
    src: torch.Tensor,  # [num_layers, spec_size, draft_tokens, *state_shape]
    dst_indices: torch.Tensor,  # [N] int64
    src_indices: torch.Tensor,  # [N] int64
    step_indices: torch.Tensor,  # [N] int64
):
    """
    Fused gather-scatter for mamba state updates.
    
    Equivalent to:
        for i in range(N):
            dst[:, dst_indices[i], :] = src[:, src_indices[i], step_indices[i], :]
    
    But with a single fused kernel instead of advanced indexing.
    """
    if dst_indices.numel() == 0:
        return

    if dst.device != src.device:
        raise ValueError(f"dst and src must be on the same device. {dst.device=} {src.device=}")
    if not dst.is_cuda or not src.is_cuda:
        raise ValueError("fused_mamba_state_scatter only supports CUDA tensors.")
    if dst.ndim < 2 or src.ndim < 3:
        raise ValueError(f"Unexpected tensor ranks: {dst.ndim=} {src.ndim=}")
    if dst.shape[0] != src.shape[0]:
        raise ValueError(
            f"Layer dimension mismatch: {dst.shape[0]=} vs {src.shape[0]=}"
        )
    if dst.shape[2:] != src.shape[3:]:
        raise ValueError(
            f"Trailing dims mismatch: {dst.shape[2:]=} vs {src.shape[3:]=}"
        )
    if dst_indices.ndim != 1 or src_indices.ndim != 1 or step_indices.ndim != 1:
        raise ValueError(
            f"indices must be 1D: {dst_indices.shape=} {src_indices.shape=} {step_indices.shape=}"
        )
    
    num_valid = dst_indices.shape[0]
    num_layers = dst.shape[0]
    src_req_size = src.shape[1]
    src_step_size = src.shape[2]
    dst_req_size = dst.shape[1]
    
    # Compute elements per entry (flatten trailing dimensions)
    # dst: [L, C, d1, d2, ...] -> elem_per_entry = d1 * d2 * ...
    # src: [L, S, D, d1, d2, ...] -> same trailing dims
    elem_per_entry = dst[0, 0].numel()
    
    # Get strides (in elements, not bytes)
    # For contiguous tensors, stride is product of trailing dimensions
    src_layer_stride = src.stride(0)
    src_req_stride = src.stride(1)
    src_step_stride = src.stride(2)
    dst_layer_stride = dst.stride(0)
    dst_req_stride = dst.stride(1)
    
    # Ensure indices are int64 and contiguous
    dst_indices = dst_indices.contiguous()
    src_indices = src_indices.contiguous()
    step_indices = step_indices.contiguous()

    # Optional debug bounds check (fail fast with a clear error instead of
    # corrupting CUDA state and failing later).
    if os.environ.get("SGLANG_DEBUG_MAMBA_STATE_SCATTER_BOUNDS", "0") != "0":
        if (dst_indices < -dst_req_size).any() or (dst_indices >= dst_req_size).any():
            raise ValueError(
                "dst_indices out of bounds for dst.shape[1]. "
                f"{dst_req_size=} {dst_indices.min().item()=} {dst_indices.max().item()=}"
            )
        if (src_indices < -src_req_size).any() or (src_indices >= src_req_size).any():
            raise ValueError(
                "src_indices out of bounds for src.shape[1]. "
                f"{src_req_size=} {src_indices.min().item()=} {src_indices.max().item()=}"
            )
        if (step_indices < -src_step_size).any() or (step_indices >= src_step_size).any():
            raise ValueError(
                "step_indices out of bounds for src.shape[2]. "
                f"{src_step_size=} {step_indices.min().item()=} {step_indices.max().item()=}"
            )
    
    # Ensure tensors are contiguous in trailing dimensions
    if not dst.is_contiguous():
        raise ValueError("dst tensor must be contiguous")
    if not src.is_contiguous():
        raise ValueError("src tensor must be contiguous")
    
    # Block size for copying elements
    BLOCK_SIZE = 1024
    
    # Grid: (num_valid * num_layers, ceil(elem_per_entry / BLOCK_SIZE))
    grid = (num_valid * num_layers, triton.cdiv(elem_per_entry, BLOCK_SIZE))
    
    _fused_mamba_state_scatter_kernel[grid](
        src,
        dst,
        dst_indices,
        src_indices,
        step_indices,
        num_valid,
        num_layers,
        elem_per_entry,
        src_layer_stride,
        src_req_stride,
        src_step_stride,
        dst_layer_stride,
        dst_req_stride,
        src_req_size,
        src_step_size,
        dst_req_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def fused_mamba_state_scatter_with_dtype_conversion(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_indices: torch.Tensor,
    src_indices: torch.Tensor,
    step_indices: torch.Tensor,
):
    """
    Fused gather-scatter with automatic dtype conversion if needed.
    
    If src and dst have different dtypes, converts on the fly.
    """
    if dst_indices.numel() == 0:
        return
    
    # If dtypes match, use the optimized kernel directly
    if src.dtype == dst.dtype:
        fused_mamba_state_scatter(dst, src, dst_indices, src_indices, step_indices)
    else:
        # For dtype mismatch, we need a different approach
        # Use the kernel but with dtype conversion
        # For now, fall back to a slightly less optimal path that still avoids
        # the worst advanced indexing patterns
        _fused_mamba_state_scatter_with_cast(
            dst, src, dst_indices, src_indices, step_indices
        )


@triton.jit
def _fused_mamba_state_scatter_cast_kernel(
    # Source tensor (intermediate cache) - may have different dtype
    src_ptr,
    # Destination tensor (mamba states)
    dst_ptr,
    # Index arrays
    dst_indices_ptr,
    src_indices_ptr,
    step_indices_ptr,
    # Dimensions
    num_valid,
    num_layers,
    elem_per_entry: tl.constexpr,
    # Strides
    src_layer_stride,
    src_req_stride,
    src_step_stride,
    dst_layer_stride,
    dst_req_stride,
    src_req_size,
    src_step_size,
    dst_req_size,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """Same as _fused_mamba_state_scatter_kernel but handles dtype conversion."""
    pid_entry = tl.program_id(0)
    pid_block = tl.program_id(1)
    
    valid_idx = pid_entry // num_layers
    layer_idx = pid_entry - valid_idx * num_layers
    
    dst_idx = tl.load(dst_indices_ptr + valid_idx)
    src_idx = tl.load(src_indices_ptr + valid_idx)
    step_idx = tl.load(step_indices_ptr + valid_idx)

    dst_idx = tl.where(dst_idx < 0, dst_idx + dst_req_size, dst_idx)
    src_idx = tl.where(src_idx < 0, src_idx + src_req_size, src_idx)
    step_idx = tl.where(step_idx < 0, step_idx + src_step_size, step_idx)

    in_bounds = (
        (dst_idx >= 0)
        & (dst_idx < dst_req_size)
        & (src_idx >= 0)
        & (src_idx < src_req_size)
        & (step_idx >= 0)
        & (step_idx < src_step_size)
    )
    
    src_offset = (
        layer_idx * src_layer_stride
        + src_idx * src_req_stride
        + step_idx * src_step_stride
    )
    dst_offset = layer_idx * dst_layer_stride + dst_idx * dst_req_stride
    
    start = pid_block * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = (offsets < elem_per_entry) & in_bounds
    
    # Load, cast implicitly via tl.store's handling, and store
    data = tl.load(src_ptr + src_offset + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offset + offsets, data, mask=mask)


def _fused_mamba_state_scatter_with_cast(
    dst: torch.Tensor,
    src: torch.Tensor,
    dst_indices: torch.Tensor,
    src_indices: torch.Tensor,
    step_indices: torch.Tensor,
):
    """Internal function for scatter with dtype conversion."""
    num_valid = dst_indices.shape[0]
    num_layers = dst.shape[0]
    elem_per_entry = dst[0, 0].numel()
    src_req_size = src.shape[1]
    src_step_size = src.shape[2]
    dst_req_size = dst.shape[1]
    
    src_layer_stride = src.stride(0)
    src_req_stride = src.stride(1)
    src_step_stride = src.stride(2)
    dst_layer_stride = dst.stride(0)
    dst_req_stride = dst.stride(1)
    
    dst_indices = dst_indices.contiguous()
    src_indices = src_indices.contiguous()
    step_indices = step_indices.contiguous()
    
    BLOCK_SIZE = 1024
    grid = (num_valid * num_layers, triton.cdiv(elem_per_entry, BLOCK_SIZE))
    
    _fused_mamba_state_scatter_cast_kernel[grid](
        src,
        dst,
        dst_indices,
        src_indices,
        step_indices,
        num_valid,
        num_layers,
        elem_per_entry,
        src_layer_stride,
        src_req_stride,
        src_step_stride,
        dst_layer_stride,
        dst_req_stride,
        src_req_size,
        src_step_size,
        dst_req_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
