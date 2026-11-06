"""Distributed QSA top-k candidate transport and deterministic merge."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.srt.distributed.device_communicators.triton_symm_mem_ag import (
    _blockwise_barrier,
    _multimem_st_128,
    _sync_threads,
    all_gather_inner,
)
from sglang.srt.distributed.utils import all_gather_single

_INTEGER_DTYPES = (
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
)


def _validate_topk(topk: int) -> None:
    if not isinstance(topk, int) or topk <= 0:
        raise ValueError(f"topk must be a positive integer, got {topk!r}")


def _validate_scores(scores: torch.Tensor, expected_width: int | None = None) -> None:
    if not isinstance(scores, torch.Tensor):
        raise TypeError(f"scores must be a tensor, got {type(scores)!r}")
    if scores.ndim != 2:
        raise ValueError(f"scores must have shape [rows, states], got {scores.shape}")
    if not scores.dtype.is_floating_point:
        raise TypeError(f"scores must have floating-point dtype, got {scores.dtype}")
    if expected_width is not None and scores.shape[1] != expected_width:
        raise ValueError(
            f"local score width must equal local shard size {expected_width}, "
            f"got {scores.shape[1]}"
        )


def all_gather_qsa_topk_candidates(
    candidate_scores: torch.Tensor,
    candidate_global_indices: torch.Tensor,
    *,
    group,
    static_buffers=None,
) -> torch.Tensor:
    """All-gather fixed-width candidates through a ``GroupCoordinator``."""

    _validate_scores(candidate_scores)
    if candidate_global_indices.shape != candidate_scores.shape:
        raise ValueError("candidate scores and indices must have matching shapes")
    if candidate_global_indices.dtype not in _INTEGER_DTYPES:
        raise TypeError("candidate indices must have integer dtype")
    if static_buffers is None:
        packed = torch.empty(
            (*candidate_scores.shape, 2),
            dtype=torch.float32,
            device=candidate_scores.device,
        )
        packed[..., 0].copy_(candidate_scores)
        packed[..., 1].view(torch.int32).copy_(candidate_global_indices)
    else:
        rows, width = candidate_scores.shape
        static_buffers = static_buffers.for_batch(rows)
        packed = static_buffers.candidate_transport
        packed[..., 0].fill_(float("-inf"))
        packed[..., 1].view(torch.int32).fill_(-1)
        packed[:rows, :width, 0].copy_(candidate_scores)
        packed[:rows, :width, 1].view(torch.int32).copy_(candidate_global_indices)
    with torch.profiler.record_function("qsa.candidate_gather"):
        if static_buffers is None:
            return group.all_gather(packed, dim=1)
        symm_state = getattr(static_buffers, "candidate_symm_state", None)
        if symm_state is not None:
            packed_bf16 = packed.view(torch.bfloat16).reshape(
                static_buffers.max_rows, -1
            )
            gathered_bf16 = all_gather_inner(
                symm_state,
                packed_bf16,
                tp_hidden_dim=packed_bf16.shape[1] * group.world_size,
                safe=False,
                skip_entry_sync=True,
            )
            return gathered_bf16.view(torch.float32).reshape(
                static_buffers.max_rows, group.world_size * width, 2
            )[:rows]
        recv = static_buffers.candidate_recv.view(
            group.world_size * static_buffers.max_rows,
            static_buffers.topk,
            2,
        )
        if not packed.is_cuda:
            group.all_gather_into_tensor(recv, packed)
        else:
            device_group = getattr(group, "device_group", None)
            if device_group is None:
                group.all_gather_into_tensor(recv, packed)
            else:
                work = all_gather_single(
                    recv,
                    packed,
                    group=device_group,
                    async_op=True,
                )
                work.block_current_stream()
        return static_buffers.candidate_recv[:, :rows, :width]


@triton.jit
def _fused_publish_qsa_topk_candidates_kernel(
    logits_ptr,
    local_indices_ptr,
    row_starts_ptr,
    logical_positions_ptr,
    multicast_ptr,
    signal_pad_ptr,
    logits_row_stride,
    logical_row_stride,
    rows,
    TOPK: tl.constexpr,
    RANK: tl.constexpr,
    WORLD_SIZE: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    """Materialize score/id pairs directly into the symmetric receive buffer."""
    # A rank may reach this layer in the next replay while a peer is still
    # merging the previous payload. Synchronize before overwriting the shared
    # fixed-address transport, then use the exit barrier for publication.
    _blockwise_barrier(signal_pad_ptr, RANK, WORLD_SIZE, sem="relaxed")
    _sync_threads()

    # One 128-bit multimem store publishes two {fp32 score, int32 id} records.
    chunks_per_row: tl.constexpr = TOPK // 2
    total_chunks = rows * chunks_per_row
    block_start = tl.program_id(0) * BLOCK_SIZE
    grid_stride: tl.constexpr = 4 * BLOCK_SIZE
    while block_start < total_chunks:
        chunk = block_start + tl.arange(0, BLOCK_SIZE)
        valid_chunk = chunk < total_chunks
        row = chunk // chunks_per_row
        first_col = (chunk % chunks_per_row) * 2
        first_offset = row * TOPK + first_col
        first_local = tl.load(
            local_indices_ptr + first_offset, mask=valid_chunk, other=-1
        ).to(tl.int32)
        second_local = tl.load(
            local_indices_ptr + first_offset + 1, mask=valid_chunk, other=-1
        ).to(tl.int32)
        row_start = tl.load(row_starts_ptr + row, mask=valid_chunk, other=0)
        first_valid = valid_chunk & (first_local >= 0)
        second_valid = valid_chunk & (second_local >= 0)
        first_score = tl.load(
            logits_ptr + row * logits_row_stride + row_start + first_local,
            mask=first_valid,
            other=-float("inf"),
        ).to(tl.float32)
        second_score = tl.load(
            logits_ptr + row * logits_row_stride + row_start + second_local,
            mask=second_valid,
            other=-float("inf"),
        ).to(tl.float32)
        first_id = tl.load(
            logical_positions_ptr + row * logical_row_stride + first_local,
            mask=first_valid,
            other=-1,
        ).to(tl.int32)
        second_id = tl.load(
            logical_positions_ptr + row * logical_row_stride + second_local,
            mask=second_valid,
            other=-1,
        ).to(tl.int32)
        first_id = tl.where(first_valid, first_id, -1)
        second_id = tl.where(second_valid, second_id, -1)

        # The symmetric buffer is bf16 storage viewed as uint32 words. Each
        # rank owns TOPK*2 words per row; each chunk writes four words.
        out_word = row * WORLD_SIZE * TOPK * 2 + RANK * TOPK * 2 + first_col * 2
        out_ptr = multicast_ptr.to(tl.int64).to(tl.pointer_type(tl.uint32)) + out_word
        _multimem_st_128(
            out_ptr,
            first_score.to(tl.uint32, bitcast=True),
            first_id.to(tl.uint32, bitcast=True),
            second_score.to(tl.uint32, bitcast=True),
            second_id.to(tl.uint32, bitcast=True),
            valid_chunk,
        )
        block_start += grid_stride

    _sync_threads()
    _blockwise_barrier(signal_pad_ptr, RANK, WORLD_SIZE, sem="acq_rel")


def fused_publish_and_merge_qsa_topk_candidates(
    logits: torch.Tensor,
    local_indices: torch.Tensor,
    row_starts: torch.Tensor,
    logical_block_positions: torch.Tensor | None,
    *,
    group,
    topk: int,
    static_buffers,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Publish local candidates with one no-pack multimem kernel, then merge."""
    if logical_block_positions is None:
        raise ValueError("fused candidate publish requires logical block positions")
    rows = local_indices.shape[0]
    static_buffers = static_buffers.for_batch(rows)
    state = static_buffers.candidate_symm_state
    if state is None:
        raise RuntimeError("fused candidate publish requires symmetric memory")
    if topk % 2:
        raise ValueError("fused candidate publish requires an even topk")
    if logits.dtype != torch.float32 or local_indices.dtype != torch.int32:
        raise TypeError("fused candidate publish requires fp32 logits and int32 ids")
    _fused_publish_qsa_topk_candidates_kernel[(4,)](
        logits,
        local_indices,
        row_starts,
        logical_block_positions,
        state.symm_mem_hdl.multicast_ptr,
        state.symm_mem_hdl.signal_pad_ptrs_dev,
        logits.stride(0),
        logical_block_positions.stride(0),
        rows,
        TOPK=topk,
        RANK=int(group.rank_in_group),
        WORLD_SIZE=int(group.world_size),
        BLOCK_SIZE=256,
        num_warps=8,
    )
    gathered = (
        state.comm_buff[:rows, : topk * 4 * group.world_size]
        .view(torch.float32)
        .reshape(rows, group.world_size * topk, 2)
    )
    scores, indices = merge_qsa_topk_candidates(gathered, topk=topk)
    static_buffers.candidate_scores[:rows, :topk].copy_(scores)
    static_buffers.candidate_indices[:rows, :topk].copy_(indices)
    return (
        static_buffers.candidate_scores[:rows, :topk],
        static_buffers.candidate_indices[:rows, :topk],
    )


def _candidate_order_keys(scores: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    """Order published candidates by score descending, then id ascending."""

    score_bits = scores.contiguous().view(torch.int32).to(torch.int64)
    score_bits &= 0xFFFFFFFF
    # IEEE floats become monotonically ordered unsigned integers. Canonicalize
    # signed zero and reject NaNs from selection without a device-to-host sync.
    score_bits = torch.where(scores == 0, 0, score_bits)
    negative = (score_bits & 0x80000000) != 0
    ordered_score = torch.where(
        negative, (~score_bits) & 0xFFFFFFFF, score_bits ^ 0x80000000
    )
    valid = (indices >= 0) & ~torch.isnan(scores)
    index_tiebreak = 0xFFFFFFFF - indices.to(torch.int64)
    keys = (ordered_score - 0x80000000) * 0x100000000 + index_tiebreak
    return keys.masked_fill(~valid, torch.iinfo(torch.int64).min)


@triton.jit
def _merge_qsa_topk_candidates_kernel(
    candidates_ptr,
    output_scores_ptr,
    output_indices_ptr,
    width: tl.constexpr,
    output_width: tl.constexpr,
    block_width: tl.constexpr,
):
    row = tl.program_id(0)
    offsets = tl.arange(0, block_width)
    input_offsets = (row * width + offsets) * 2
    in_bounds = offsets < width
    scores = tl.load(
        candidates_ptr + input_offsets, mask=in_bounds, other=-float("inf")
    )
    indices = tl.load(
        candidates_ptr + input_offsets + 1, mask=in_bounds, other=0xFFFFFFFF
    ).to(tl.int32, bitcast=True)
    valid = in_bounds & (indices >= 0) & (scores == scores)  # noqa: PLR0124

    score_bits = scores.to(tl.int32, bitcast=True).to(tl.int64) & 0xFFFFFFFF
    score_bits = tl.where(scores == 0.0, 0, score_bits)
    ordered_score = tl.where(
        (score_bits & 0x80000000) != 0,
        (~score_bits) & 0xFFFFFFFF,
        score_bits ^ 0x80000000,
    )
    keys = (ordered_score - 0x80000000) * 0x100000000 + (
        0xFFFFFFFF - indices.to(tl.int64)
    )
    keys = tl.where(valid, keys, -0x8000000000000000)
    keys = tl.sort(keys, dim=0, descending=True)
    selected = offsets < output_width
    selected_keys = tl.where(selected, keys, -0x8000000000000000)
    selected_valid = selected & (selected_keys != -0x8000000000000000)
    # Arithmetic shift preserves the exact high 32-bit score key. Integer
    # division can round negative packed keys through fp64 and flip one ulp.
    selected_ordered_score = (selected_keys >> 32) + 0x80000000
    selected_score_bits = tl.where(
        (selected_ordered_score & 0x80000000) != 0,
        selected_ordered_score ^ 0x80000000,
        (~selected_ordered_score) & 0xFFFFFFFF,
    )
    selected_scores = selected_score_bits.to(tl.int32).to(tl.float32, bitcast=True)
    selected_ids = 0xFFFFFFFF - (selected_keys & 0xFFFFFFFF)
    tl.store(
        output_scores_ptr + row * output_width + offsets,
        tl.where(selected_valid, selected_scores, -float("inf")),
        mask=selected,
    )
    tl.store(
        output_indices_ptr + row * output_width + offsets,
        tl.where(selected_valid, selected_ids, -1),
        mask=selected,
    )


def _merge_qsa_topk_candidates_triton(
    gathered_candidates: torch.Tensor,
    *,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    rows, width, _ = gathered_candidates.shape
    output_scores = torch.empty(
        (rows, topk), dtype=torch.float32, device=gathered_candidates.device
    )
    output_indices = torch.empty(
        (rows, topk), dtype=torch.int32, device=gathered_candidates.device
    )
    block_width = triton.next_power_of_2(width)
    _merge_qsa_topk_candidates_kernel[(rows,)](
        gathered_candidates,
        output_scores,
        output_indices,
        width=width,
        output_width=topk,
        block_width=block_width,
        num_warps=4 if block_width <= 128 else 8,
    )
    return output_scores, output_indices


def merge_qsa_topk_candidates(
    gathered_candidates: torch.Tensor,
    *,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the best published candidates entirely on their current device.

    Local selectors may return candidates in any order. Equal-score candidates
    already omitted by a local top-k cannot be reconstructed here; global ids
    only make the merge deterministic among candidates that were published.
    """

    _validate_topk(topk)
    if gathered_candidates.ndim == 4:
        if gathered_candidates.shape[-1] != 2:
            raise ValueError("candidate transport must end in score/index pairs")
        gathered_candidates = (
            gathered_candidates.permute(1, 0, 2, 3)
            .reshape(gathered_candidates.shape[1], -1, 2)
            .contiguous()
        )
    if gathered_candidates.ndim != 3 or gathered_candidates.shape[-1] != 2:
        raise ValueError(
            "gathered candidates must be [rows, width, 2] or "
            f"[ranks, rows, width, 2], got {tuple(gathered_candidates.shape)}"
        )
    if gathered_candidates.shape[1] < topk:
        raise ValueError(
            f"candidate width {gathered_candidates.shape[1]} is smaller than "
            f"topk {topk}"
        )
    if gathered_candidates.dtype != torch.float32:
        raise TypeError("candidate transport must use float32")

    with torch.profiler.record_function("qsa.candidate_merge"):
        if gathered_candidates.is_cuda:
            return _merge_qsa_topk_candidates_triton(gathered_candidates, topk=topk)
        return _merge_qsa_topk_candidates_impl(gathered_candidates, topk=topk)


def _merge_qsa_topk_candidates_impl(
    gathered_candidates: torch.Tensor,
    *,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    scores = gathered_candidates[..., 0]
    indices = gathered_candidates[..., 1].contiguous().view(torch.int32)
    valid = (indices >= 0) & ~torch.isnan(scores)

    keys = _candidate_order_keys(scores, indices)
    positions = torch.topk(keys, k=topk, dim=-1, largest=True, sorted=True).indices
    output_indices = indices.gather(-1, positions)
    output_scores = scores.gather(-1, positions)
    output_valid = valid.gather(-1, positions)
    return (
        output_scores.masked_fill(~output_valid, float("-inf")),
        output_indices.masked_fill(~output_valid, -1),
    )


def gather_and_merge_qsa_topk_candidates(
    candidate_scores: torch.Tensor,
    candidate_global_indices: torch.Tensor,
    *,
    group,
    topk: int,
    static_buffers=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one fixed-shape all-gather and merge the global candidates."""

    gathered = all_gather_qsa_topk_candidates(
        candidate_scores,
        candidate_global_indices,
        group=group,
        static_buffers=static_buffers,
    )
    scores, indices = merge_qsa_topk_candidates(gathered, topk=topk)
    if static_buffers is None:
        return scores, indices
    rows = candidate_scores.shape[0]
    static_buffers.candidate_scores[:rows, :topk].copy_(scores)
    static_buffers.candidate_indices[:rows, :topk].copy_(indices)
    return (
        static_buffers.candidate_scores[:rows, :topk],
        static_buffers.candidate_indices[:rows, :topk],
    )


__all__ = [
    "all_gather_qsa_topk_candidates",
    "fused_publish_and_merge_qsa_topk_candidates",
    "gather_and_merge_qsa_topk_candidates",
    "merge_qsa_topk_candidates",
]
