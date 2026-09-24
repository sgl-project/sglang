"""
Fused Triton kernel for Mamba state scatter operations.

This kernel replaces the expensive advanced indexing operations in
`update_mamba_state_after_mtp_verify` with a single fused gather-scatter kernel,
avoiding multiple `index_elementwise_kernel` launches.
"""

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.memory.ptr_table import make_ptr_table


def _require_entry_contiguous_dst(
    dst: torch.Tensor, entry_start_dim: int, fn_name: str
) -> None:
    """dst layout contract: the kernels index through the real layer/slot
    strides (int64) plus a FLAT element offset within one (layer, slot)
    entry — layer/slot strides may be arbitrary (envelope-strided unified
    pool views), but the trailing entry dims must be contiguous.
    """
    expected = 1
    for i in range(dst.ndim - 1, entry_start_dim - 1, -1):
        if dst.shape[i] != 1 and dst.stride(i) != expected:
            raise ValueError(
                f"{fn_name}: dst entry dims (dims {entry_start_dim}.."
                f"{dst.ndim - 1}) must be contiguous; got "
                f"shape={tuple(dst.shape)} strides={tuple(dst.stride())}"
            )
        expected *= dst.shape[i]


@triton.jit
def track_mamba_state_if_needed_kernel(
    conv_states_ptr,
    ssm_states_ptr,
    cache_indices_ptr,
    mamba_track_mask_ptr,
    mamba_track_indices_ptr,
    conv_state_stride_0,  # stride for first dimension (batch/pool index)
    ssm_state_stride_0,  # stride for first dimension (batch/pool index)
    conv_state_numel_per_row: tl.constexpr,  # total elements per row
    ssm_state_numel_per_row: tl.constexpr,  # total elements per row
    BLOCK_SIZE: tl.constexpr,
    check_freed_slots: tl.constexpr,  # only the shared/unified KV pool emits -1
):
    """
    Track conv_states and ssm_states rows based on track mask.

    This kernel replaces a Python loop that copies state tensors for mamba attention.
    For each batch element, if the track mask is True, it copies the entire row from
    the source index (cache_indices[i]) to the destination index (mamba_track_indices[i]).

    Grid: (batch_size,)
    Each block handles one batch element, using multiple threads to copy data in parallel.
    """
    batch_idx = tl.program_id(0)

    # Load the copy mask for this batch element
    track_mask = tl.load(mamba_track_mask_ptr + batch_idx)

    # Early exit if we don't need to track
    if not track_mask:
        return

    # Cast indices to int64 before they multiply the row stride. The
    # page-granularity envelope layout makes the conv/ssm row stride large
    # (stride_0 = entry_bytes / itemsize), so an int32 `idx * stride_0` can
    # overflow for moderately large idx and wrap to an illegal address. int64 is
    # harmless for the small-stride (per-layer) case.
    src_idx = tl.load(cache_indices_ptr + batch_idx).to(tl.int64)
    dst_idx = tl.load(mamba_track_indices_ptr + batch_idx).to(tl.int64)

    # Skip freed slots (-1): `state_ptr + (-1)*stride` would fault. Only the unified
    # pool emits -1 tombstones (from the v2p translate); compiled out for static.
    if check_freed_slots:
        if src_idx < 0 or dst_idx < 0:
            return

    # Copy conv_states
    # Each thread handles BLOCK_SIZE elements
    for offset in range(0, conv_state_numel_per_row, BLOCK_SIZE):
        element_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = element_indices < conv_state_numel_per_row

        src_ptr = conv_states_ptr + src_idx * conv_state_stride_0 + element_indices
        dst_ptr = conv_states_ptr + dst_idx * conv_state_stride_0 + element_indices

        data = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_ptr, data, mask=mask)

    # Copy ssm_states
    for offset in range(0, ssm_state_numel_per_row, BLOCK_SIZE):
        element_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = element_indices < ssm_state_numel_per_row

        src_ptr = ssm_states_ptr + src_idx * ssm_state_stride_0 + element_indices
        dst_ptr = ssm_states_ptr + dst_idx * ssm_state_stride_0 + element_indices

        data = tl.load(src_ptr, mask=mask, other=0.0)
        tl.store(dst_ptr, data, mask=mask)


def track_mamba_states_if_needed(
    conv_states: torch.Tensor,
    ssm_states: torch.Tensor,
    cache_indices: torch.Tensor,
    mamba_track_mask: torch.Tensor,
    mamba_track_indices: torch.Tensor,
    batch_size: int,
    check_freed_slots: bool = False,
):
    """
    Track mamba states using Triton kernel for better performance.

    Args:
        conv_states: Convolution states tensor [pool_size, ...]
        ssm_states: SSM states tensor [pool_size, ...]
        cache_indices: Source indices for each batch element [batch_size]
        mamba_track_mask: Boolean mask indicating which elements to track [batch_size]
        mamba_track_indices: Indices to track for each batch element [batch_size]
        batch_size: Number of batch elements
    """
    conv_state_numel_per_row = conv_states[0].numel()
    ssm_state_numel_per_row = ssm_states[0].numel()

    # Choose BLOCK_SIZE based on the size of the data
    BLOCK_SIZE = 1024

    # Launch kernel with batch_size blocks
    grid = (batch_size,)
    track_mamba_state_if_needed_kernel[grid](
        conv_states,
        ssm_states,
        cache_indices,
        mamba_track_mask,
        mamba_track_indices,
        conv_states.stride(0),
        ssm_states.stride(0),
        conv_state_numel_per_row,
        ssm_state_numel_per_row,
        BLOCK_SIZE,
        check_freed_slots,
    )


@triton.jit
def _fused_mamba_state_scatter_multi_kernel(
    src_ptrs,
    dst_ptrs,
    dst_indices_ptr,
    step_indices_ptr,
    dst_indices_stride,
    step_indices_stride,
    LAYERS: tl.constexpr,
    ELEMENTS: tl.constexpr,
    BLOCKS: tl.constexpr,
    ENTRY_LAYOUTS: tl.constexpr,
    SRC_STRIDES: tl.constexpr,
    DST_STRIDES: tl.constexpr,
    SRC_SIZES: tl.constexpr,
    DST_SIZES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    track_indices_ptr,
    track_steps_ptr,
    track_indices_stride,
    track_steps_stride,
    BS: tl.constexpr,
    HAS_TRACK: tl.constexpr,
    TRACK_BLOCK: tl.constexpr,
):
    req = tl.program_id(0).to(tl.int64)
    tile = tl.program_id(1).to(tl.int64)
    active_dst = tl.load(dst_indices_ptr + req * dst_indices_stride).to(tl.int64)
    active_step = tl.load(step_indices_ptr + req * step_indices_stride).to(tl.int64)
    if HAS_TRACK:
        track_dst = tl.load(track_indices_ptr + req * track_indices_stride).to(tl.int64)
        track_step = tl.load(track_steps_ptr + req * track_steps_stride).to(tl.int64)
        rows = tl.arange(0, TRACK_BLOCK)
        all_track_dst = tl.load(
            track_indices_ptr + rows * track_indices_stride, rows < BS, other=-1
        ).to(tl.int64)
        all_track_step = tl.load(
            track_steps_ptr + rows * track_steps_stride, rows < BS, other=-1
        ).to(tl.int64)
    first_tile = 0
    for i in tl.static_range(len(LAYERS)):
        if (tile >= first_tile) & (tile < first_tile + LAYERS[i] * BLOCKS[i]):
            overwritten = False
            if HAS_TRACK:
                valid_track = (
                    (rows < BS)
                    & (rows < SRC_SIZES[i][0])
                    & (all_track_step >= 0)
                    & (all_track_step < SRC_SIZES[i][1])
                )
                overwritten = (
                    tl.sum(
                        (valid_track & (all_track_dst == active_dst)).to(tl.int32), 0
                    )
                    > 0
                )
            for commit in tl.static_range(2 if HAS_TRACK else 1):
                dst_idx = active_dst
                step = active_step
                enabled = not overwritten
                if commit == 1:
                    dst_idx = track_dst
                    step = track_step
                    enabled = True
                if (
                    enabled
                    & (dst_idx >= 0)
                    & (dst_idx < DST_SIZES[i])
                    & (req < SRC_SIZES[i][0])
                    & (step >= 0)
                    & (step < SRC_SIZES[i][1])
                ):
                    local_tile = tile - first_tile
                    local_layer = local_tile // BLOCKS[i]
                    block = local_tile % BLOCKS[i]
                    offsets = block * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
                    if ENTRY_LAYOUTS[i][0] > 0:
                        source_offsets = (
                            offsets // ENTRY_LAYOUTS[i][0] * ENTRY_LAYOUTS[i][1]
                            + offsets % ENTRY_LAYOUTS[i][0] * ENTRY_LAYOUTS[i][2]
                        )
                    else:
                        source_offsets = offsets
                    src_offset = (
                        local_layer * SRC_STRIDES[i][0]
                        + req * SRC_STRIDES[i][1]
                        + step * SRC_STRIDES[i][2]
                    )
                    dst_offset = (
                        local_layer * DST_STRIDES[i][0] + dst_idx * DST_STRIDES[i][1]
                    )
                    values = tl.load(
                        src_ptrs[i] + src_offset + source_offsets,
                        mask=offsets < ELEMENTS[i],
                    )
                    tl.store(
                        dst_ptrs[i] + dst_offset + offsets,
                        values,
                        mask=offsets < ELEMENTS[i],
                    )
        first_tile += LAYERS[i] * BLOCKS[i]


def prepare_mamba_state_scatter_multi(state_pairs):
    device = state_pairs[0][0].device
    for dst, src in state_pairs:
        if dst.device != device or src.device != device:
            raise ValueError("states and indices must be on the same CUDA device")
        if dst.ndim < 2 or src.ndim < 3:
            raise ValueError("unexpected state ranks")
        if dst.shape[0] != src.shape[0] or dst.shape[2:] != src.shape[3:]:
            raise ValueError("state layer and trailing dimensions must match")
        _require_entry_contiguous_dst(dst, 2, "fused_mamba_state_scatter_multi")
        if not src.is_contiguous() and src.ndim != 5:
            raise ValueError("src entries must be contiguous or two-dimensional")
    layers = tuple(dst.shape[0] for dst, _ in state_pairs)
    elements = tuple(
        dst.numel() // (dst.shape[0] * dst.shape[1]) for dst, _ in state_pairs
    )
    blocks = tuple(triton.cdiv(n, 1024) for n in elements)
    entry_layouts = tuple(
        (
            (0, 0, 0)
            if src.is_contiguous()
            else (src.shape[-1], src.stride(-2), src.stride(-1))
        )
        for _, src in state_pairs
    )
    return (
        device,
        (
            tuple(src for _, src in state_pairs),
            tuple(dst for dst, _ in state_pairs),
        ),
        (
            layers,
            elements,
            blocks,
            entry_layouts,
            tuple(src.stride()[:3] for _, src in state_pairs),
            tuple(dst.stride()[:2] for dst, _ in state_pairs),
            tuple(src.shape[1:3] for _, src in state_pairs),
            tuple(dst.shape[1] for dst, _ in state_pairs),
        ),
        sum(n * b for n, b in zip(layers, blocks)),
    )


def fused_mamba_state_scatter_multi(
    state_pairs,
    dst_indices,
    step_indices,
    track_indices=None,
    track_steps=None,
    *,
    _metadata=None,
):
    if not state_pairs or step_indices.numel() == 0:
        return
    if dst_indices.ndim != 1 or step_indices.ndim != 1:
        raise ValueError("indices must be 1D")
    if dst_indices.shape != step_indices.shape:
        raise ValueError("indices must have matching shapes")
    if (track_indices is None) != (track_steps is None):
        raise ValueError("track indices and steps must be supplied together")
    indices_to_check = (dst_indices, step_indices)
    if track_indices is not None:
        if (
            track_indices.shape != step_indices.shape
            or track_steps.shape != step_indices.shape
        ):
            raise ValueError("track indices must have matching shapes")
        indices_to_check += (track_indices, track_steps)
    for indices in indices_to_check:
        if indices.dtype not in (torch.int32, torch.int64):
            raise ValueError("indices must have int32 or int64 dtype")
        if not indices.is_cuda or indices.device != step_indices.device:
            raise ValueError("indices must be on the same CUDA device")
    metadata = _metadata or prepare_mamba_state_scatter_multi(state_pairs)
    device, pointers, constants, tiles = metadata
    if device != step_indices.device:
        raise ValueError("states and indices must be on the same CUDA device")
    _fused_mamba_state_scatter_multi_kernel[(step_indices.numel(), tiles)](
        *pointers,
        dst_indices,
        step_indices,
        dst_indices.stride(0),
        step_indices.stride(0),
        *constants,
        BLOCK_SIZE=1024,
        track_indices_ptr=track_indices,
        track_steps_ptr=track_steps,
        track_indices_stride=(
            track_indices.stride(0) if track_indices is not None else 0
        ),
        track_steps_stride=track_steps.stride(0) if track_steps is not None else 0,
        BS=step_indices.numel(),
        HAS_TRACK=track_indices is not None,
        TRACK_BLOCK=triton.next_power_of_2(step_indices.numel()),
    )


@triton.jit
def _fused_mamba_state_scatter_with_mask_kernel(
    src_ptr,
    dst_ptr,
    # Raw index arrays (before index_select)
    dst_indices_raw_ptr,  # [total_requests] - state_indices_tensor
    step_indices_raw_ptr,  # [total_requests] - last_correct_step_indices or mamba_steps_to_track
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
    """
    Fused gather-scatter kernel with built-in masking.

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

    # Source index is just the request index itself
    src_idx = pid_req

    # Bounds check to avoid illegal memory access
    if not (
        (dst_idx >= 0)
        & (dst_idx < dst_req_size)
        & (src_idx >= 0)
        & (src_idx < src_req_size)
        & (step_idx < src_step_size)
    ):
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

    # Load from source and store to destination
    data = tl.load(src_ptr + src_offset + offsets, mask=mask)
    tl.store(dst_ptr + dst_offset + offsets, data, mask=mask)


def fused_mamba_state_scatter_with_mask(
    dst: torch.Tensor,  # [num_layers, cache_size, *state_shape]
    src: torch.Tensor,  # [num_layers, spec_size, draft_tokens, *state_shape]
    dst_indices_raw: torch.Tensor,  # [total_requests] - raw indices (e.g., state_indices_tensor)
    step_indices_raw: torch.Tensor,  # [total_requests] - raw step indices (step >= 0 means valid)
):
    """
    Fully fused gather-scatter with built-in masking for mamba state updates.

    This function fuses the following operations into a single kernel:
    1. valid_mask = step_indices_raw >= 0
    2. valid_indices = valid_mask.nonzero()
    3. dst_indices = dst_indices_raw[valid_indices]  (index_select)
    4. step_indices = step_indices_raw[valid_indices]  (index_select)
    5. for each valid i: dst[:, dst_indices[i], :] = src[:, i, step_indices[i], :]

    Args:
        dst: Destination tensor [num_layers, cache_size, *state_shape]
        src: Source tensor [num_layers, spec_size, draft_tokens, *state_shape]
        dst_indices_raw: Raw destination indices for all requests [total_requests]
        step_indices_raw: Raw step indices; entry >= 0 means valid [total_requests]
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

    # Ensure index buffers are contiguous.
    if dst_indices_raw.dtype not in (torch.int32, torch.int64):
        dst_indices_raw = dst_indices_raw.to(torch.int32)
    if step_indices_raw.dtype not in (torch.int32, torch.int64):
        step_indices_raw = step_indices_raw.to(torch.int32)
    dst_indices_raw = dst_indices_raw.contiguous()
    step_indices_raw = step_indices_raw.contiguous()

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
    src_idx = pid_req

    if not (
        (dst_idx >= 0)
        & (dst_idx < dst_req_size)
        & (src_idx < src_req_size)
        & (step_idx < src_step_size)
    ):
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
):
    """Conv-window variant of :func:`fused_mamba_state_scatter_with_mask`.

    ``src`` is the deduplicated sliding-window conv-intermediate cache: an
    overlapping ``as_strided`` view over a shared ``[..., dim, D+K-2]`` buffer,
    so its per-step windows are intentionally non-contiguous. This kernel indexes
    ``(dim, win)`` elements through the view's strides instead of flat-copying.
    ``dst`` (the real conv-state pool) is the usual contiguous
    ``[layers, cache, dim, K-1]``.
    """
    total_requests = step_indices_raw.shape[0]
    if total_requests == 0:
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

    BLOCK_SIZE = 1024
    grid = (total_requests, num_layers, triton.cdiv(elem_per_entry, BLOCK_SIZE))

    _fused_conv_window_scatter_with_mask_kernel[grid](
        src,
        dst,
        dst_indices_raw,
        step_indices_raw,
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


_CONV_MULTI_MAX_TYPES = 8
_CONV_MULTI_META_COLS = 12
_conv_multi_meta_cache: dict = {}


@triton.jit
def _fused_conv_window_scatter_multi_kernel(
    meta_ptr,  # int64 [num_types, 12]: src_ptr, dst_ptr, elem, s_l, s_r, s_s, s_d, s_w, d_l, d_r, block_start, last_axis
    idx1_ptr,
    step1_ptr,
    idx2_ptr,
    step2_ptr,
    n1,
    src_req_size,
    src_step_size,
    dst_req_size,
    NUM_TYPES: tl.constexpr,
    META_COLS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid_req = tl.program_id(0)
    pid_layer = tl.program_id(1).to(tl.int64)
    pid_block = tl.program_id(2).to(tl.int64)

    is1 = pid_req < n1
    is2 = pid_req >= n1
    off1 = pid_req
    off2 = pid_req - n1
    s1 = tl.load(step1_ptr + off1, mask=is1, other=-1).to(tl.int64)
    s2 = tl.load(step2_ptr + off2, mask=is2, other=-1).to(tl.int64)
    step_idx = tl.where(is2, s2, s1)
    if step_idx < 0:
        return
    d1 = tl.load(idx1_ptr + off1, mask=is1, other=-1).to(tl.int64)
    d2 = tl.load(idx2_ptr + off2, mask=is2, other=-1).to(tl.int64)
    dst_idx = tl.where(is2, d2, d1)
    src_idx = tl.where(is2, off2, off1).to(tl.int64)

    if not (
        (dst_idx >= 0)
        & (dst_idx < dst_req_size)
        & (src_idx < src_req_size)
        & (step_idx < src_step_size)
    ):
        return

    for t in tl.static_range(NUM_TYPES):
        block_start = tl.load(meta_ptr + t * META_COLS + 10)
        block_end = tl.load(
            meta_ptr + (t + 1) * META_COLS + 10,
            mask=t + 1 < NUM_TYPES,
            other=2147483647,
        )
        if (pid_block >= block_start) & (pid_block < block_end):
            src_ptr = tl.load(meta_ptr + t * META_COLS + 0).to(
                tl.pointer_type(tl.bfloat16)
            )
            dst_ptr = tl.load(meta_ptr + t * META_COLS + 1).to(
                tl.pointer_type(tl.bfloat16)
            )
            elem_per_entry = tl.load(meta_ptr + t * META_COLS + 2)
            src_layer_stride = tl.load(meta_ptr + t * META_COLS + 3)
            src_req_stride = tl.load(meta_ptr + t * META_COLS + 4)
            src_step_stride = tl.load(meta_ptr + t * META_COLS + 5)
            src_dim_stride = tl.load(meta_ptr + t * META_COLS + 6)
            src_win_stride = tl.load(meta_ptr + t * META_COLS + 7)
            dst_layer_stride = tl.load(meta_ptr + t * META_COLS + 8)
            dst_req_stride = tl.load(meta_ptr + t * META_COLS + 9)
            last_axis = tl.load(meta_ptr + t * META_COLS + 11)

            start = (pid_block - block_start) * BLOCK_SIZE
            e = start + tl.arange(0, BLOCK_SIZE)
            mask = e < elem_per_entry
            d = e // last_axis
            w = e % last_axis
            src_off = (
                pid_layer * src_layer_stride
                + src_idx * src_req_stride
                + step_idx * src_step_stride
                + d * src_dim_stride
                + w * src_win_stride
            )
            dst_off = pid_layer * dst_layer_stride + dst_idx * dst_req_stride + e
            data = tl.load(src_ptr + src_off, mask=mask, other=0.0)
            tl.store(dst_ptr + dst_off, data, mask=mask)


def _conv_multi_build_meta(pairs, block_size: int):
    rows = []
    block_start = 0
    for dst, src in pairs:
        elem = dst.shape[2] * dst.shape[3]
        rows.append(
            [
                src.data_ptr(),
                dst.data_ptr(),
                elem,
                src.stride(0),
                src.stride(1),
                src.stride(2),
                src.stride(3),
                src.stride(4),
                dst.stride(0),
                dst.stride(1),
                block_start,
                dst.shape[3],
            ]
        )
        block_start += triton.cdiv(elem, block_size)
    meta = make_ptr_table(rows, device=pairs[0][0].device)
    return meta, block_start


def _conv_multi_eligible(pairs) -> bool:
    if not (0 < len(pairs) <= _CONV_MULTI_MAX_TYPES):
        return False
    layers = pairs[0][0].shape[0]
    for dst, src in pairs:
        if dst.dtype != torch.bfloat16 or src.dtype != torch.bfloat16:
            return False
        if dst.ndim != 4 or src.ndim != 5:
            return False
        if dst.shape[0] != layers:
            return False
        if src.shape[0] != layers or src.shape[3:] != dst.shape[2:]:
            return False
        if not dst.is_contiguous():
            return False
        if src.shape[1:3] != pairs[0][1].shape[1:3]:
            return False
        if dst.shape[1] != pairs[0][0].shape[1]:
            return False
    return True


def fused_conv_window_scatter_multi(
    pairs,
    dst_indices_raw: torch.Tensor,
    step_indices_raw: torch.Tensor,
    dst_indices2_raw: torch.Tensor | None = None,
    step_indices2_raw: torch.Tensor | None = None,
) -> None:
    """Single-launch variant of ``fused_conv_window_scatter_with_mask`` over
    multiple (dst, src) conv-type pairs and up to two request-index sets (the
    accept commit plus the optional interval-crossing track set)."""
    n1 = step_indices_raw.shape[0]
    n2 = 0 if step_indices2_raw is None else step_indices2_raw.shape[0]
    if n1 + n2 == 0:
        return

    BLOCK_SIZE = 1024
    key = tuple(
        (dst.data_ptr(), src.data_ptr()) + tuple(src.stride()) + tuple(dst.shape)
        for dst, src in pairs
    )
    cached = _conv_multi_meta_cache.get(key)
    if cached is None:
        cached = _conv_multi_build_meta(pairs, BLOCK_SIZE)
        _conv_multi_meta_cache.clear()
        _conv_multi_meta_cache[key] = cached
    meta, total_blocks = cached

    idx1 = (
        dst_indices_raw
        if dst_indices_raw.is_contiguous()
        else dst_indices_raw.contiguous()
    )
    st1 = (
        step_indices_raw
        if step_indices_raw.is_contiguous()
        else step_indices_raw.contiguous()
    )
    if n2 > 0:
        idx2 = (
            dst_indices2_raw
            if dst_indices2_raw.is_contiguous()
            else dst_indices2_raw.contiguous()
        )
        st2 = (
            step_indices2_raw
            if step_indices2_raw.is_contiguous()
            else step_indices2_raw.contiguous()
        )
    else:
        idx2, st2 = idx1, st1

    dst0, src0 = pairs[0]
    grid = (n1 + n2, dst0.shape[0], total_blocks)
    _fused_conv_window_scatter_multi_kernel[grid](
        meta,
        idx1,
        st1,
        idx2,
        st2,
        n1,
        src0.shape[1],
        src0.shape[2],
        dst0.shape[1],
        NUM_TYPES=len(pairs),
        META_COLS=_CONV_MULTI_META_COLS,
        BLOCK_SIZE=BLOCK_SIZE,
    )


def scatter_mamba_states_after_mtp_verify(
    mamba_caches,
    state_indices_tensor: torch.Tensor,
    last_correct_step_indices: torch.Tensor,
    mamba_track_indices: torch.Tensor | None,
    mamba_steps_to_track: torch.Tensor | None,
) -> None:
    """Scatter per-step verify states (ssm + all conv types) into the
    persistent caches, plus the interval-crossing track slots."""
    ssm_states = mamba_caches.temporal
    intermediate_state_cache = mamba_caches.intermediate_ssm

    if ssm_states.numel() > 0:
        fused_mamba_state_scatter_with_mask(
            ssm_states,
            intermediate_state_cache,
            state_indices_tensor,
            last_correct_step_indices,
        )
        if mamba_track_indices is not None:
            assert mamba_steps_to_track is not None
            fused_mamba_state_scatter_with_mask(
                ssm_states,
                intermediate_state_cache,
                mamba_track_indices,
                mamba_steps_to_track,
            )

    pairs = list(zip(mamba_caches.conv, mamba_caches.intermediate_conv_window))
    if not pairs:
        return
    if mamba_track_indices is not None:
        assert mamba_steps_to_track is not None
    if _conv_multi_eligible(pairs):
        fused_conv_window_scatter_multi(
            pairs,
            state_indices_tensor,
            last_correct_step_indices,
            mamba_track_indices,
            mamba_steps_to_track,
        )
        return
    for conv_states, intermediate_conv_window_cache in pairs:
        fused_conv_window_scatter_with_mask(
            conv_states,
            intermediate_conv_window_cache,
            state_indices_tensor,
            last_correct_step_indices,
        )
    if mamba_track_indices is not None:
        for conv_states, intermediate_conv_window_cache in pairs:
            fused_conv_window_scatter_with_mask(
                conv_states,
                intermediate_conv_window_cache,
                mamba_track_indices,
                mamba_steps_to_track,
            )


@triton.jit
def _fused_commit_track_indices_kernel(
    accept_index_ptr,
    accept_lens_ptr,
    seq_lens_ptr,
    last_correct_out_ptr,
    track_steps_out_ptr,
    dtn,
    accept_stride,
    interval,
    HAS_TRACK: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    al = tl.load(accept_lens_ptr + b).to(tl.int64)
    row = b * accept_stride
    base = b * dtn
    last = tl.load(accept_index_ptr + row + al - 1).to(tl.int64) - base
    tl.store(last_correct_out_ptr + b, last)
    if HAS_TRACK:
        pre = tl.load(seq_lens_ptr + b).to(tl.int64)
        post = pre + al
        cross = (pre // interval) != (post // interval)
        tp = (post // interval) * interval
        ti = tp - pre - 1
        ti = tl.where(ti < 0, 0, ti)
        cand = tl.load(accept_index_ptr + row + ti).to(tl.int64) - base
        tl.store(track_steps_out_ptr + b, tl.where(cross, cand, -1))


def fused_commit_track_indices(
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    seq_lens: torch.Tensor | None,
    draft_token_num: int,
    mamba_track_interval: int,
):
    """Single-launch replacement for the eager verify-commit index math;
    accept_index is [bs, tree_depth] but its values index bs * draft_token_num rows."""
    accept_index = accept_index.contiguous()
    bs = accept_lens.shape[0]
    last_correct_step_indices = torch.empty(
        bs, dtype=torch.int64, device=accept_lens.device
    )
    has_track = seq_lens is not None
    mamba_steps_to_track = (
        torch.empty(bs, dtype=torch.int64, device=accept_lens.device)
        if has_track
        else last_correct_step_indices
    )
    _fused_commit_track_indices_kernel[(bs,)](
        accept_index,
        accept_lens,
        seq_lens if has_track else accept_lens,
        last_correct_step_indices,
        mamba_steps_to_track,
        draft_token_num,
        accept_index.shape[1],
        mamba_track_interval,
        HAS_TRACK=has_track,
    )
    return last_correct_step_indices, (mamba_steps_to_track if has_track else None)


@triton.jit
def track_mamba_states_all_layers_kernel(
    conv_states_ptr,  # [num_layers, pool_size, ...] full conv pool
    ssm_states_ptr,  # [num_layers, pool_size, ...] full ssm pool
    cache_indices_ptr,
    mamba_track_mask_ptr,
    mamba_track_indices_ptr,
    conv_layer_stride,
    conv_row_stride,
    ssm_layer_stride,
    ssm_row_stride,
    batch_size,
    conv_state_numel_per_row: tl.constexpr,
    ssm_state_numel_per_row: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    check_freed_slots: tl.constexpr,
):
    """All-layers variant of track_mamba_state_if_needed_kernel: one launch
    covers every mamba layer (grid = num_layers * batch_size) instead of one
    launch per layer. The track mask / source / destination indices are shared
    across layers, so a single launch at the end of the step is equivalent to
    the per-layer launches (each layer's state is final by then)."""
    pid = tl.program_id(0)
    layer_idx = (pid // batch_size).to(tl.int64)
    batch_idx = pid % batch_size

    track_mask = tl.load(mamba_track_mask_ptr + batch_idx)
    if not track_mask:
        return

    src_idx = tl.load(cache_indices_ptr + batch_idx).to(tl.int64)
    dst_idx = tl.load(mamba_track_indices_ptr + batch_idx).to(tl.int64)
    if check_freed_slots:
        if src_idx < 0 or dst_idx < 0:
            return

    conv_base = conv_states_ptr + layer_idx * conv_layer_stride
    for offset in range(0, conv_state_numel_per_row, BLOCK_SIZE):
        element_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = element_indices < conv_state_numel_per_row
        data = tl.load(
            conv_base + src_idx * conv_row_stride + element_indices,
            mask=mask,
            other=0.0,
        )
        tl.store(
            conv_base + dst_idx * conv_row_stride + element_indices, data, mask=mask
        )

    ssm_base = ssm_states_ptr + layer_idx * ssm_layer_stride
    for offset in range(0, ssm_state_numel_per_row, BLOCK_SIZE):
        element_indices = offset + tl.arange(0, BLOCK_SIZE)
        mask = element_indices < ssm_state_numel_per_row
        data = tl.load(
            ssm_base + src_idx * ssm_row_stride + element_indices, mask=mask, other=0.0
        )
        tl.store(ssm_base + dst_idx * ssm_row_stride + element_indices, data, mask=mask)


def track_mamba_states_all_layers(
    conv_states_pool: torch.Tensor,
    ssm_states_pool: torch.Tensor,
    cache_indices: torch.Tensor,
    mamba_track_mask: torch.Tensor,
    mamba_track_indices: torch.Tensor,
    batch_size: int,
    check_freed_slots: bool = False,
):
    """Track conv/ssm states for ALL mamba layers in one launch.

    conv_states_pool / ssm_states_pool are the full [num_layers, pool_size,
    ...] pools; per-row copy semantics are identical to
    track_mamba_states_if_needed applied per layer.
    """
    num_layers = conv_states_pool.shape[0]
    conv_state_numel_per_row = conv_states_pool[0, 0].numel()
    ssm_state_numel_per_row = ssm_states_pool[0, 0].numel()

    BLOCK_SIZE = 1024
    grid = (num_layers * batch_size,)
    track_mamba_states_all_layers_kernel[grid](
        conv_states_pool,
        ssm_states_pool,
        cache_indices,
        mamba_track_mask,
        mamba_track_indices,
        conv_states_pool.stride(0),
        conv_states_pool.stride(1),
        ssm_states_pool.stride(0),
        ssm_states_pool.stride(1),
        batch_size,
        conv_state_numel_per_row,
        ssm_state_numel_per_row,
        BLOCK_SIZE,
        check_freed_slots,
    )


@triton.jit
def _gather_mamba_track_indices_kernel(
    mapping,
    requests,
    positions,
    output,
    N: tl.constexpr,
    ROWS: tl.constexpr,
    ROW_STRIDE: tl.constexpr,
    COL_STRIDE: tl.constexpr,
    REQ_STRIDE: tl.constexpr,
    POS_STRIDE: tl.constexpr,
    BLOCK: tl.constexpr,
    HAS_POSITIONS: tl.constexpr,
    UNIFORM_POSITION,
):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    req = tl.load(requests + i * REQ_STRIDE, i < N, other=0).to(tl.int64)
    req = tl.where(req < 0, req + ROWS, req)
    if HAS_POSITIONS:
        pos = tl.load(positions + i * POS_STRIDE, i < N, other=0).to(tl.int64)
    else:
        pos = tl.full((), 0, tl.int64) + UNIFORM_POSITION
    value = tl.load(mapping + req * ROW_STRIDE + pos * COL_STRIDE, i < N, other=0)
    tl.store(output + i, value, i < N)


def gather_mamba_track_indices(
    mapping, requests, positions=None, uniform_position=None
):
    n = requests.numel()
    if positions is not None:
        assert positions.numel() == n
    else:
        assert uniform_position is not None and 0 <= uniform_position < mapping.shape[1]
    output = torch.empty((n,), dtype=torch.int64, device=mapping.device)
    if n:
        _gather_mamba_track_indices_kernel[(triton.cdiv(n, 128),)](
            mapping,
            requests,
            positions,
            output,
            n,
            mapping.shape[0],
            mapping.stride(0),
            mapping.stride(1),
            requests.stride(0),
            positions.stride(0) if positions is not None else 0,
            BLOCK=128,
            HAS_POSITIONS=positions is not None,
            UNIFORM_POSITION=uniform_position if positions is None else 0,
        )
    return output
