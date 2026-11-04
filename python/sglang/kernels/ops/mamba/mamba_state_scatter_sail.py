"""SAIL (PPU/HGGC) variants of the mamba state-scatter Triton kernels.

The HGGC Triton backend does not lower the combined ``&`` bounds check of
``mamba_state_scatter_triton`` into real branches and lets masked-out
``tl.load`` lanes touch out-of-range addresses, so on SAIL the checks are
written as sequential early-returns and the masked loads carry an explicit
``other``. Keeping that variant here leaves the CUDA kernels in
``mamba_state_scatter_triton`` untouched.

Original integration: ``fix(mamba kernels): make mamba state scatter Triton
kernels safe on PPU/HGGC`` (thmyl), see also ``fix(ppu): restore GLM DSA
startup path``.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.mamba.mamba_state_scatter_triton import (
    _prepare_src_rows,
    _require_entry_contiguous_dst,
)


@triton.jit
def _fused_mamba_state_scatter_with_mask_kernel(
    src_ptr,
    dst_ptr,
    # Raw index arrays (before index_select)
    dst_indices_raw_ptr,  # [total_requests] - state_indices_tensor
    step_indices_raw_ptr,  # [total_requests] - last_correct_step_indices or mamba_steps_to_track
    src_indices_raw_ptr,  # [total_requests] - verify scratch row for each request
    elem_per_entry: tl.constexpr,
    src_layer_stride,
    src_req_stride,
    src_step_stride,
    dst_layer_stride,
    dst_req_stride,
    src_req_size,
    src_step_size,
    dst_req_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Fused gather-scatter kernel with built-in masking.

    This kernel fuses the index_select operations by:
    1. Iterating over all requests (pid_req from 0 to total_requests-1)
    2. Checking if step_indices_raw[pid_req] >= 0 (valid mask)
    3. If valid, performing the scatter:
       dst[l, dst_indices_raw[pid_req], :] = src[l, pid_req, step_indices_raw[pid_req], :]

    Grid: (total_requests, num_layers, ceil(elem_per_entry / BLOCK_SIZE))
    """
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1).to(tl.int64)
    pid_block = tl.program_id(2).to(tl.int64)

    # Load step index to check validity (step >= 0 means valid)
    step_idx = tl.load(step_indices_raw_ptr + pid_req).to(tl.int64)

    # Early exit if this request is not valid (step < 0)
    if step_idx < 0:
        return

    # Load destination index
    dst_idx = tl.load(dst_indices_raw_ptr + pid_req).to(tl.int64)

    src_idx = tl.load(src_indices_raw_ptr + pid_req).to(tl.int64)

    # Bounds check to avoid illegal memory access. Sequential early-returns are
    # required here: HGGC does not lower the combined & form into branches.
    if dst_idx < 0 or dst_idx >= dst_req_size:
        return
    if src_idx < 0 or src_idx >= src_req_size:
        return
    if step_idx < 0 or step_idx >= src_step_size:
        return

    # Compute base offsets
    src_offset = (
        pid_layer * src_layer_stride
        + src_idx * src_req_stride
        + step_idx * src_step_stride
    )
    dst_offset = pid_layer * dst_layer_stride + dst_idx * dst_req_stride

    # Compute element range for this block
    start = pid_block * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < elem_per_entry

    # Load from source and store to destination. other=0.0 keeps masked-out
    # lanes off an out-of-range address on HGGC.
    data = tl.load(src_ptr + src_offset + offsets, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_offset + offsets, data, mask=mask)


def fused_mamba_state_scatter_with_mask(
    dst: torch.Tensor,  # [num_layers, cache_size, *state_shape]
    src: torch.Tensor,  # [num_layers, spec_size, draft_tokens, *state_shape]
    dst_indices_raw: torch.Tensor,  # [total_requests] - raw indices (e.g., state_indices_tensor)
    step_indices_raw: torch.Tensor,  # [total_requests] - raw step indices (step >= 0 means valid)
    src_indices_raw: torch.Tensor | None = None,  # [total_requests] - scratch rows
):
    """SAIL variant of :func:`fused_mamba_state_scatter_with_mask`.

    Same contract and validation as the CUDA implementation; only the kernel's
    bounds-check lowering differs.
    """
    total_requests = step_indices_raw.shape[0]
    if total_requests == 0:
        return

    if dst.device != src.device:
        raise ValueError(
            f"dst and src must be on the same device. {dst.device=} {src.device=}"
        )
    if not (dst.is_cuda or dst.is_xpu) or dst.device != src.device:
        raise ValueError(
            "fused_mamba_state_scatter_with_mask only supports CUDA/XPU tensors."
        )
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
    if dst_indices_raw.ndim != 1 or step_indices_raw.ndim != 1:
        raise ValueError(
            f"indices must be 1D: {dst_indices_raw.shape=} {step_indices_raw.shape=}"
        )
    if dst_indices_raw.shape[0] != step_indices_raw.shape[0]:
        raise ValueError(
            f"indices length mismatch: {dst_indices_raw.shape[0]=} vs {step_indices_raw.shape[0]=}"
        )

    num_layers = dst.shape[0]
    src_req_size = src.shape[1]
    src_step_size = src.shape[2]
    dst_req_size = dst.shape[1]

    # Flatten trailing dimensions: number of elements per (layer, cache_line) entry.
    elem_per_entry = dst.numel() // (dst.shape[0] * dst.shape[1])

    # Get strides (in elements, not bytes)
    src_layer_stride = src.stride(0)
    src_req_stride = src.stride(1)
    src_step_stride = src.stride(2)
    dst_layer_stride = dst.stride(0)
    dst_req_stride = dst.stride(1)

    # Ensure indices are int32 and contiguous
    dst_indices_raw = dst_indices_raw.to(torch.int32).contiguous()
    step_indices_raw = step_indices_raw.to(torch.int32).contiguous()
    src_indices_raw = _prepare_src_rows(src_indices_raw, step_indices_raw)

    _require_entry_contiguous_dst(dst, 2, "fused_mamba_state_scatter_with_mask")
    if not src.is_contiguous():
        raise ValueError("src tensor must be contiguous")

    # Block size for copying elements
    BLOCK_SIZE = 1024

    # Grid over all requests - invalid ones will early-exit in the kernel
    grid = (total_requests, num_layers, triton.cdiv(elem_per_entry, BLOCK_SIZE))

    _fused_mamba_state_scatter_with_mask_kernel[grid](
        src,
        dst,
        dst_indices_raw,
        step_indices_raw,
        src_indices_raw,
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


@triton.jit
def _fused_conv_window_scatter_with_mask_kernel(
    src_ptr,
    dst_ptr,
    dst_indices_raw_ptr,  # [total_requests]
    step_indices_raw_ptr,  # [total_requests], entry >= 0 means valid
    src_indices_raw_ptr,  # [total_requests]
    elem_per_entry: tl.constexpr,  # dim * (K-1)
    KM1: tl.constexpr,  # K-1 (conv window width)
    src_layer_stride,
    src_req_stride,
    src_step_stride,
    src_dim_stride,
    src_win_stride,
    dst_layer_stride,
    dst_req_stride,
    src_req_size,
    src_step_size,
    dst_req_size,
    BLOCK_SIZE: tl.constexpr,
):
    """Scatter accepted conv windows from the deduplicated sliding-window source.

    Unlike ``_fused_mamba_state_scatter_with_mask_kernel`` (which flat-copies a
    contiguous per-step state row), the source here is an *overlapping* view: the
    deduplicated layout keeps one shared ``[dim, D+K-2]`` buffer per (layer, slot)
    and step ``t``'s window is the slice ``shared[:, t:t+K-1]``. That window is
    non-contiguous, so we index every ``(dim, win)`` element through the view's
    strides (``src_step_stride`` / ``src_dim_stride`` / ``src_win_stride``). The
    destination conv-state row stays contiguous in ``(dim, K-1)`` order.
    """
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1).to(tl.int64)
    pid_block = tl.program_id(2).to(tl.int64)

    step_idx = tl.load(step_indices_raw_ptr + pid_req).to(tl.int64)
    if step_idx < 0:
        return

    dst_idx = tl.load(dst_indices_raw_ptr + pid_req).to(tl.int64)
    src_idx = tl.load(src_indices_raw_ptr + pid_req).to(tl.int64)

    # Sequential early-returns are required here: HGGC does not lower the
    # combined & form into branches.
    if dst_idx < 0 or dst_idx >= dst_req_size:
        return
    if src_idx < 0 or src_idx >= src_req_size:
        return
    if step_idx >= src_step_size:
        return

    start = pid_block * BLOCK_SIZE
    e = start + tl.arange(0, BLOCK_SIZE)
    mask = e < elem_per_entry

    # Decode the flat (dim, K-1)-row element index into (dim, win) coordinates.
    d = e // KM1
    w = e % KM1

    src_off = (
        pid_layer * src_layer_stride
        + src_idx * src_req_stride
        + step_idx * src_step_stride
        + d * src_dim_stride
        + w * src_win_stride
    )
    # dst window is contiguous in (dim, K-1) order -> flat element index `e`.
    dst_off = pid_layer * dst_layer_stride + dst_idx * dst_req_stride + e

    data = tl.load(src_ptr + src_off, mask=mask, other=0.0)
    tl.store(dst_ptr + dst_off, data, mask=mask)


def fused_conv_window_scatter_with_mask(
    dst: torch.Tensor,  # conv_states [num_layers, cache_size, dim, K-1] (contiguous)
    src: torch.Tensor,  # deduped conv-window view [num_layers, spec_size, draft_tokens, dim, K-1]
    dst_indices_raw: torch.Tensor,  # [total_requests]
    step_indices_raw: torch.Tensor,  # [total_requests], entry >= 0 means valid
    src_indices_raw: torch.Tensor | None = None,  # [total_requests]
):
    """SAIL variant of :func:`fused_conv_window_scatter_with_mask`.

    ``src`` is the deduplicated sliding-window conv-intermediate cache: an
    overlapping ``as_strided`` view over a shared ``[..., dim, D+K-2]`` buffer,
    so its per-step windows are intentionally non-contiguous. This kernel indexes
    ``(dim, win)`` elements through the view's strides instead of flat-copying.
    ``dst`` (the real conv-state pool) is the usual contiguous
    ``[layers, cache, dim, K-1]``.
    """
    total_requests = step_indices_raw.shape[0]
    if total_requests == 0 or dst.numel() == 0:
        return

    if not (dst.is_cuda or dst.is_xpu) or dst.device != src.device:
        raise ValueError(
            "fused_conv_window_scatter_with_mask requires dst and src to be "
            f"CUDA/XPU tensors on the same device ({dst.device=}, {src.device=})."
        )
    if dst.ndim != 4 or src.ndim != 5:
        raise ValueError(f"Unexpected ranks: {dst.ndim=} (want 4) {src.ndim=} (want 5)")
    if dst.shape[0] != src.shape[0]:
        raise ValueError(f"Layer dim mismatch: {dst.shape[0]=} vs {src.shape[0]=}")
    if dst.shape[2:] != src.shape[3:]:
        raise ValueError(f"Window dims mismatch: {dst.shape[2:]=} vs {src.shape[3:]=}")
    if dst_indices_raw.ndim != 1 or step_indices_raw.ndim != 1:
        raise ValueError(
            f"indices must be 1D: {dst_indices_raw.shape=} {step_indices_raw.shape=}"
        )
    if dst_indices_raw.shape[0] != step_indices_raw.shape[0]:
        raise ValueError(
            f"indices length mismatch: {dst_indices_raw.shape[0]=} vs {step_indices_raw.shape[0]=}"
        )

    num_layers = dst.shape[0]
    dim = dst.shape[2]
    km1 = dst.shape[3]
    elem_per_entry = dim * km1

    src_req_size = src.shape[1]
    src_step_size = src.shape[2]
    dst_req_size = dst.shape[1]

    # `src` is an intentionally non-contiguous (overlapping) view indexed per
    # dim through its real strides, so we do NOT assert src contiguity here
    # (unlike the dense scatter).
    _require_entry_contiguous_dst(dst, 2, "fused_conv_window_scatter_with_mask")

    dst_indices_raw = dst_indices_raw.to(torch.int32).contiguous()
    step_indices_raw = step_indices_raw.to(torch.int32).contiguous()
    src_indices_raw = _prepare_src_rows(src_indices_raw, step_indices_raw)

    BLOCK_SIZE = 1024
    grid = (total_requests, num_layers, triton.cdiv(elem_per_entry, BLOCK_SIZE))

    _fused_conv_window_scatter_with_mask_kernel[grid](
        src,
        dst,
        dst_indices_raw,
        step_indices_raw,
        src_indices_raw,
        elem_per_entry,
        km1,
        src.stride(0),
        src.stride(1),
        src.stride(2),
        src.stride(3),
        src.stride(4),
        dst.stride(0),
        dst.stride(1),
        src_req_size,
        src_step_size,
        dst_req_size,
        BLOCK_SIZE=BLOCK_SIZE,
    )
