from __future__ import annotations

import torch
import triton
import triton.language as tl

_QWEN4_NGRAM_SIZE = 3
_QWEN4_HEADS_PER_NGRAM = 8
_QWEN4_NGRAM_HEADS = 16


@triton.jit(do_not_specialize=["num_tokens", "num_reqs", "eos_token_id"])
def _qwen4_packed_ngram_hash_kernel(
    input_ids_ptr,
    query_start_loc_ptr,
    history_ptr,
    multipliers_ptr,
    vocab_sizes_ptr,
    offsets_ptr,
    output_ptr,
    num_tokens,
    num_reqs,
    eos_token_id,
    SEARCH_STEPS: tl.constexpr,
    HEADS_PER_NGRAM: tl.constexpr,
    NGRAM_HEADS: tl.constexpr,
):
    token = tl.program_id(0)

    # Find the last row start <= token. Using live prefix sums rather than a
    # captured token-to-row map keeps breakable CUDA-graph replay correct.
    lo = 0
    hi = num_reqs
    for _ in tl.static_range(SEARCH_STEPS):
        mid = (lo + hi + 1) // 2
        start = tl.load(query_start_loc_ptr + mid, mask=lo < hi, other=0)
        go_right = (lo < hi) & (start <= token)
        lo = tl.where(go_right, mid, lo)
        hi = tl.where((lo < hi) & ~go_right, mid - 1, hi)
    req = tl.minimum(lo, num_reqs - 1)
    seq_start = tl.load(query_start_loc_ptr + req)
    token_offset = token - seq_start

    current = tl.load(input_ids_ptr + token)
    previous_1 = tl.load(
        input_ids_ptr + token - 1,
        mask=token_offset >= 1,
        other=0,
    )
    previous_1 = tl.where(
        token_offset >= 1,
        previous_1,
        tl.load(history_ptr + req * 2 + 1),
    )
    previous_2 = tl.load(
        input_ids_ptr + token - 2,
        mask=token_offset >= 2,
        other=0,
    )
    previous_2 = tl.where(
        token_offset >= 2,
        previous_2,
        tl.where(
            token_offset == 1,
            tl.load(history_ptr + req * 2 + 1),
            tl.load(history_ptr + req * 2),
        ),
    )
    previous_2 = tl.where(
        (previous_1 == eos_token_id) | (previous_2 == eos_token_id),
        eos_token_id,
        previous_2,
    )

    multiplier_0 = tl.load(multipliers_ptr)
    multiplier_1 = tl.load(multipliers_ptr + 1)
    multiplier_2 = tl.load(multipliers_ptr + 2)
    mixed_2 = (current * multiplier_0) ^ (previous_1 * multiplier_1)
    mixed_3 = mixed_2 ^ (previous_2 * multiplier_2)

    head = tl.arange(0, NGRAM_HEADS)
    mixed = tl.where(head < HEADS_PER_NGRAM, mixed_2, mixed_3)
    vocab_size = tl.load(vocab_sizes_ptr + head)
    offset = tl.load(offsets_ptr + head)
    # Match torch.remainder for signed int64 values after two's-complement
    # multiply/XOR overflow. Triton's signed `%` may return a negative residue.
    remainder = mixed % vocab_size
    remainder = tl.where(remainder < 0, remainder + vocab_size, remainder)
    tl.store(output_ptr + token * NGRAM_HEADS + head, remainder + offset)


def can_fuse_qwen4_packed_ngram_hash(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    history: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> bool:
    """Return whether ordinary packed prefill can hash directly from live inputs."""

    requests = history.shape[0] if history.ndim == 2 else -1
    device = input_ids.device
    return (
        input_ids.is_cuda
        and input_ids.dtype == torch.long
        and input_ids.ndim == 1
        and input_ids.is_contiguous()
        and (requests > 0 or input_ids.numel() == 0)
        and query_start_loc.device == device
        and query_start_loc.dtype == torch.long
        and query_start_loc.shape == (requests + 1,)
        and query_start_loc.is_contiguous()
        and history.device == device
        and history.dtype == torch.long
        and history.shape == (requests, _QWEN4_NGRAM_SIZE - 1)
        and history.is_contiguous()
        and multipliers.device == device
        and multipliers.dtype == torch.long
        and multipliers.numel() == _QWEN4_NGRAM_SIZE
        and multipliers.is_contiguous()
        and vocab_sizes.device == device
        and vocab_sizes.dtype == torch.long
        and vocab_sizes.numel() == _QWEN4_NGRAM_HEADS
        and vocab_sizes.is_contiguous()
        and offsets.device == device
        and offsets.dtype == torch.long
        and offsets.numel() == _QWEN4_NGRAM_HEADS
        and offsets.is_contiguous()
    )


def fused_qwen4_packed_ngram_hash(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    history: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    """Generate packed prefill N-gram IDs directly in one kernel launch.

    ``query_start_loc`` must be a nondecreasing prefix sum beginning at zero
    and ending at ``input_ids.numel()``. Value checks intentionally stay on the
    device side of the API contract so dispatch never synchronizes the host.
    """

    if not can_fuse_qwen4_packed_ngram_hash(
        input_ids,
        query_start_loc,
        history,
        multipliers,
        vocab_sizes,
        offsets,
    ):
        raise ValueError("unsupported input for fused Qwen4 packed N-gram hash")
    output = torch.empty(
        (input_ids.shape[0], _QWEN4_NGRAM_HEADS),
        dtype=torch.long,
        device=input_ids.device,
    )
    if input_ids.shape[0]:
        requests = history.shape[0]
        search_steps = max(1, (requests + 1).bit_length())
        _qwen4_packed_ngram_hash_kernel[(input_ids.shape[0],)](
            input_ids,
            query_start_loc,
            history,
            multipliers,
            vocab_sizes,
            offsets,
            output,
            input_ids.shape[0],
            requests,
            eos_token_id,
            SEARCH_STEPS=search_steps,
            HEADS_PER_NGRAM=_QWEN4_HEADS_PER_NGRAM,
            NGRAM_HEADS=_QWEN4_NGRAM_HEADS,
            num_warps=1,
        )
    return output


@triton.jit
def _qwen4_packed_ngram_update_kernel(
    input_ids_ptr,
    query_start_loc_ptr,
    history_ptr,
    context_pool_ptr,
    state_indices_ptr,
    track_indices_ptr,
    track_offsets_ptr,
    CONTEXT_LEN: tl.constexpr,
    HAS_TRACK: tl.constexpr,
):
    req = tl.program_id(0)
    context_col = tl.arange(0, CONTEXT_LEN)
    seq_start = tl.load(query_start_loc_ptr + req)
    length = tl.load(query_start_loc_ptr + req + 1) - seq_start

    source_col = length - CONTEXT_LEN + context_col
    old_col = CONTEXT_LEN + source_col
    old_value = tl.load(
        history_ptr + req * CONTEXT_LEN + old_col,
        mask=source_col < 0,
        other=0,
    )
    token_value = tl.load(
        input_ids_ptr + seq_start + source_col,
        mask=source_col >= 0,
        other=0,
    )
    value = tl.where(source_col >= 0, token_value, old_value)
    state_index = tl.load(state_indices_ptr + req)
    tl.store(
        context_pool_ptr + state_index * CONTEXT_LEN + context_col,
        value,
        mask=(length > 0) & (state_index > 0),
    )

    if HAS_TRACK:
        track_index = tl.load(track_indices_ptr + req)
        track_offset = tl.load(track_offsets_ptr + req)
        track_source_col = track_offset - CONTEXT_LEN + context_col
        track_old_col = CONTEXT_LEN + track_source_col
        track_old_value = tl.load(
            history_ptr + req * CONTEXT_LEN + track_old_col,
            mask=track_source_col < 0,
            other=0,
        )
        track_token_value = tl.load(
            input_ids_ptr + seq_start + track_source_col,
            mask=track_source_col >= 0,
            other=0,
        )
        track_value = tl.where(
            track_source_col >= 0, track_token_value, track_old_value
        )
        tl.store(
            context_pool_ptr + track_index * CONTEXT_LEN + context_col,
            track_value,
            mask=(length > 0) & (track_index > 0),
        )


def fused_qwen4_packed_ngram_update(
    input_ids: torch.Tensor,
    query_start_loc: torch.Tensor,
    history: torch.Tensor,
    context_pool: torch.Tensor,
    state_indices: torch.Tensor,
    *,
    track_indices: torch.Tensor | None = None,
    track_offsets: torch.Tensor | None = None,
) -> None:
    """Advance packed N-gram history without materializing padded token rows."""

    requests = history.shape[0] if history.ndim == 2 else -1
    device = input_ids.device
    has_track = track_indices is not None or track_offsets is not None
    valid = (
        input_ids.is_cuda
        and input_ids.dtype == torch.long
        and input_ids.ndim == 1
        and input_ids.is_contiguous()
        and query_start_loc.device == device
        and query_start_loc.dtype == torch.long
        and query_start_loc.shape == (requests + 1,)
        and query_start_loc.is_contiguous()
        and history.device == device
        and history.dtype == torch.long
        and history.shape == (requests, _QWEN4_NGRAM_SIZE - 1)
        and history.is_contiguous()
        and context_pool.device == device
        and context_pool.dtype == torch.long
        and context_pool.ndim == 2
        and context_pool.shape[1] == _QWEN4_NGRAM_SIZE - 1
        and context_pool.is_contiguous()
        and state_indices.device == device
        and state_indices.dtype == torch.long
        and state_indices.shape == (requests,)
        and state_indices.is_contiguous()
    )
    if has_track:
        valid = valid and (
            track_indices is not None
            and track_offsets is not None
            and track_indices.device == device
            and track_offsets.device == device
            and track_indices.dtype == torch.long
            and track_offsets.dtype == torch.long
            and track_indices.shape == (requests,)
            and track_offsets.shape == (requests,)
            and track_indices.is_contiguous()
            and track_offsets.is_contiguous()
        )
    if not valid:
        raise ValueError("unsupported input for fused Qwen4 packed N-gram update")
    if requests:
        _qwen4_packed_ngram_update_kernel[(requests,)](
            input_ids,
            query_start_loc,
            history,
            context_pool,
            state_indices,
            track_indices if has_track else state_indices,
            track_offsets if has_track else state_indices,
            CONTEXT_LEN=_QWEN4_NGRAM_SIZE - 1,
            HAS_TRACK=has_track,
            num_warps=1,
        )


@triton.jit
def _qwen4_ngram_hash_kernel(
    contexts_ptr,
    multipliers_ptr,
    vocab_sizes_ptr,
    offsets_ptr,
    output_ptr,
    num_outputs,
    eos_token_id,
    NGRAM_SIZE: tl.constexpr,
    HEADS_PER_NGRAM: tl.constexpr,
    NGRAM_HEADS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    output_idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = output_idx < num_outputs
    token_idx = output_idx // NGRAM_HEADS
    head_idx = output_idx % NGRAM_HEADS
    context_base = token_idx * NGRAM_SIZE

    token_0 = tl.load(contexts_ptr + context_base, mask=mask, other=0)
    token_1 = tl.load(contexts_ptr + context_base + 1, mask=mask, other=0)
    token_2 = tl.load(contexts_ptr + context_base + 2, mask=mask, other=0)
    multiplier_0 = tl.load(multipliers_ptr)
    multiplier_1 = tl.load(multipliers_ptr + 1)
    multiplier_2 = tl.load(multipliers_ptr + 2)

    # Only the final position of each three-token context is materialized.
    previous_1 = tl.where(token_1 == eos_token_id, eos_token_id, token_1)
    previous_2 = tl.where(
        (token_0 == eos_token_id) | (token_1 == eos_token_id),
        eos_token_id,
        token_0,
    )
    mixed = (token_2 * multiplier_0) ^ (previous_1 * multiplier_1)
    mixed_3 = mixed ^ (previous_2 * multiplier_2)
    mixed = tl.where(head_idx < HEADS_PER_NGRAM, mixed, mixed_3)

    vocab_size = tl.load(vocab_sizes_ptr + head_idx, mask=mask, other=1)
    offset = tl.load(offsets_ptr + head_idx, mask=mask, other=0)
    tl.store(output_ptr + output_idx, mixed % vocab_size + offset, mask=mask)


def can_fuse_qwen4_ngram_hash(
    contexts: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
) -> bool:
    """Return whether inputs match the fixed Qwen4 PLE hash contract."""

    return (
        contexts.is_cuda
        and contexts.dtype == torch.long
        and contexts.dim() == 2
        and contexts.shape[1] == _QWEN4_NGRAM_SIZE
        and contexts.is_contiguous()
        and multipliers.is_cuda
        and multipliers.dtype == torch.long
        and multipliers.numel() == _QWEN4_NGRAM_SIZE
        and vocab_sizes.is_cuda
        and vocab_sizes.dtype == torch.long
        and vocab_sizes.numel() == _QWEN4_NGRAM_HEADS
        and offsets.is_cuda
        and offsets.dtype == torch.long
        and offsets.numel() == _QWEN4_NGRAM_HEADS
    )


def fused_qwen4_ngram_hash(
    contexts: torch.Tensor,
    multipliers: torch.Tensor,
    vocab_sizes: torch.Tensor,
    offsets: torch.Tensor,
    eos_token_id: int,
) -> torch.Tensor:
    """Return the 16 Qwen4 PLE N-gram IDs in one kernel launch."""

    if not can_fuse_qwen4_ngram_hash(contexts, multipliers, vocab_sizes, offsets):
        raise ValueError("unsupported input for fused Qwen4 PLE N-gram hash")
    output = torch.empty(
        (contexts.shape[0], _QWEN4_NGRAM_HEADS),
        dtype=torch.long,
        device=contexts.device,
    )
    num_outputs = output.numel()
    if num_outputs:
        block_size = 256
        _qwen4_ngram_hash_kernel[(triton.cdiv(num_outputs, block_size),)](
            contexts,
            multipliers,
            vocab_sizes,
            offsets,
            output,
            num_outputs,
            eos_token_id,
            NGRAM_SIZE=_QWEN4_NGRAM_SIZE,
            HEADS_PER_NGRAM=_QWEN4_HEADS_PER_NGRAM,
            NGRAM_HEADS=_QWEN4_NGRAM_HEADS,
            BLOCK_SIZE=block_size,
            num_warps=4,
        )
    return output
