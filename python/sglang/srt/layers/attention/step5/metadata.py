"""Step5 per-forward metadata and backend-owned indexer state."""

from __future__ import annotations

import msgspec
import torch

from sglang.kernels.ops.attention.step5.indexer import (
    STEP5_HISTORY_BLOCKS,
    can_use_step5_topk_deepselect,
    deepselect_score_width,
)

# 512 MiB FP32 score budget, independent of packed prefill length.
STEP5_SCORE_WORKSPACE_ELEMENTS = 128 * 1024 * 1024


class Step5LayerIndexerState(msgspec.Struct, frozen=True, kw_only=True):
    compressed_index_keys: torch.Tensor
    index_key_ring: torch.Tensor
    compression_logits_ring: torch.Tensor
    index_positions: torch.Tensor


class Step5SelectionWorkspace(msgspec.Struct, frozen=True, kw_only=True):
    """Per-forward selection scratch, reused by every DSA layer in turn.

    packed/counts/lengths have one row per query; indexer_logits/ids hold one row tile.
    """

    packed: torch.Tensor
    counts: torch.Tensor
    lengths: torch.Tensor
    indexer_logits: torch.Tensor
    ids: torch.Tensor
    tiles: tuple[tuple[int, int], ...]
    use_deepselect: bool = False
    topk_lengths: torch.Tensor | None = None
    topk_ids: torch.Tensor | None = None


class Step5IndexerMetadata(msgspec.Struct, frozen=True, kw_only=True):
    """Indexer metadata for one forward; R = compression request rows, Q = query rows,
    G = maximum completed blocks per request."""

    # Borrowed references; storage and lifetime remain owned by the backend and pools.
    layer_states: dict[int, Step5LayerIndexerState]
    req_to_token: torch.Tensor

    # Compression access plan. Request rows need not equal query rows: prefill
    # packs a variable number of input tokens per request.
    request_slots: torch.Tensor  # [R], request-pool row, zero means padding
    input_start_positions: torch.Tensor  # [R], absolute logical token position
    input_start_rows: torch.Tensor  # [R], offset into this forward's packed K/Z
    input_lengths: torch.Tensor  # [R], number of input K/Z rows for the request
    # [G, R], flattened compressed-key address, NOT a physical KV page number;
    # ordinal g names block floor(input_start_positions / 8) + g.
    compressed_key_write_indices: torch.Tensor
    num_input_tokens: int  # capacity of this forward's packed K/Z, including padding

    # Input validity, shared by all layers; it does not cover layer-local ring tags.
    request_active: torch.Tensor  # bool [R], live request row, excluding slot 0
    request_mapping_valid: torch.Tensor  # bool [R], admission from input checks
    request_count_valid: torch.Tensor  # bool [1], live count fits request capacity

    # Query/scoring layout, complete at construction (no later replace/binding).
    query_request_slots: torch.Tensor  # [Q], request slot for each query
    query_positions: torch.Tensor  # [Q], absolute logical query positions
    score_width: int  # maximum logical block count to score per query

    # Per-layer scratch: _admit overwrites both buffers for every request row,
    # so they must not be cached as input validity.
    compression_enabled: torch.Tensor  # bool [R], input AND layer-ring admission
    compression_success: torch.Tensor  # bool [R], success (including inert padding)
    selection_workspace: Step5SelectionWorkspace


def allocate_selection_workspace(
    *,
    rows: int,
    score_width: int,
    max_score_width: int,
    row_to_req: torch.Tensor,
    positions: torch.Tensor,
    device: torch.device,
) -> Step5SelectionWorkspace:
    use_deepselect = False
    if score_width > STEP5_HISTORY_BLOCKS and can_use_step5_topk_deepselect(
        torch.device(device)
    ):
        padded = deepselect_score_width(score_width)
        if padded <= deepselect_score_width(max_score_width):
            score_width, use_deepselect = padded, True
    capacity = min(
        int(rows), STEP5_SCORE_WORKSPACE_ELEMENTS // max(int(score_width), 1)
    )
    indexer_logits = torch.empty(
        (capacity, score_width), dtype=torch.float32, device=device
    )
    topk_lengths = None
    if use_deepselect:
        topk_lengths = torch.empty(rows, dtype=torch.int32, device=device)
        topk_lengths.copy_(torch.where(row_to_req > 0, positions // 8, 0))
    return Step5SelectionWorkspace(
        packed=torch.empty((rows, 513), dtype=torch.int64, device=device),
        counts=torch.empty(rows, dtype=torch.int32, device=device),
        lengths=torch.empty(rows, dtype=torch.int32, device=device),
        indexer_logits=indexer_logits,
        ids=torch.empty((capacity, 513), dtype=torch.int32, device=device),
        use_deepselect=use_deepselect,
        topk_lengths=topk_lengths,
        topk_ids=(
            torch.empty(
                (capacity, STEP5_HISTORY_BLOCKS), dtype=torch.int32, device=device
            )
            if use_deepselect
            else None
        ),
        tiles=tuple(
            (start, min(start + capacity, rows))
            for start in range(0, rows, capacity or 1)
        ),
    )
