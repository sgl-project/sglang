"""
Fused Triton kernel for Mamba state scatter operations.

This kernel replaces the expensive advanced indexing operations in
`update_mamba_state_after_mtp_verify` with a single fused gather-scatter kernel,
avoiding multiple `index_elementwise_kernel` launches.
"""

import torch
import triton
import triton.language as tl


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
    if not dst.is_cuda or not src.is_cuda:
        raise ValueError(
            "fused_mamba_state_scatter_with_mask only supports CUDA tensors."
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

    # Ensure tensors are contiguous
    if not dst.is_contiguous():
        raise ValueError("dst tensor must be contiguous")
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

    if not (dst.is_cuda and src.is_cuda and dst.device == src.device):
        raise ValueError(
            "fused_conv_window_scatter_with_mask requires dst and src to be CUDA "
            f"tensors on the same device ({dst.device=}, {src.device=})."
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

    # `dst` stays contiguous; `src` is an intentionally non-contiguous (overlapping)
    # view, so we do NOT assert src contiguity here (unlike the dense scatter).
    if not dst.is_contiguous():
        raise ValueError(
            "dst tensor in fused_conv_window_scatter_with_mask must be contiguous"
        )

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
    meta = torch.tensor(rows, dtype=torch.int64, device=pairs[0][0].device)
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
    interval,
    HAS_TRACK: tl.constexpr,
):
    b = tl.program_id(0).to(tl.int64)
    al = tl.load(accept_lens_ptr + b).to(tl.int64)
    base = b * dtn
    last = tl.load(accept_index_ptr + base + al - 1).to(tl.int64) - base
    tl.store(last_correct_out_ptr + b, last)
    if HAS_TRACK:
        pre = tl.load(seq_lens_ptr + b).to(tl.int64)
        post = pre + al
        cross = (pre // interval) != (post // interval)
        tp = (post // interval) * interval
        ti = tp - pre - 1
        ti = tl.where(ti < 0, 0, ti)
        cand = tl.load(accept_index_ptr + base + ti).to(tl.int64) - base
        tl.store(track_steps_out_ptr + b, tl.where(cross, cand, -1))


def fused_commit_track_indices(
    accept_index: torch.Tensor,
    accept_lens: torch.Tensor,
    seq_lens: torch.Tensor | None,
    draft_token_num: int,
    mamba_track_interval: int,
):
    """Single-launch replacement for the eager index math in
    ``commit_mamba_states_after_verify`` (index ranges, gathers, floordiv chain)."""
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
        mamba_track_interval,
        HAS_TRACK=has_track,
    )
    return last_correct_step_indices, (mamba_steps_to_track if has_track else None)
