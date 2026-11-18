"""Step5 per-forward metadata and backend-owned indexer state."""

from __future__ import annotations

import msgspec
import torch

from sglang.kernels.ops.attention.step5.indexer import (
    STEP5_HISTORY_BLOCKS,
    can_use_step5_topk_deepselect,
    deepselect_score_width,
)

STEP5_SCORE_WORKSPACE_ELEMENTS = 128 * 1024 * 1024


def selection_tile_rows(rows: int, score_width: int) -> int:
    """Query rows per DSA selection tile: the FP32 score tile stays within the budget."""
    return min(int(rows), STEP5_SCORE_WORKSPACE_ELEMENTS // max(int(score_width), 1))


class Step5LayerIndexerState(msgspec.Struct, frozen=True, kw_only=True):
    compressed_index_keys: torch.Tensor
    index_key_ring: torch.Tensor
    compression_logits_ring: torch.Tensor
    index_positions: torch.Tensor


class Step5SelectionWorkspace(msgspec.Struct, frozen=True, kw_only=True):
    """Per-layer scoring/selection outputs, not immutable query layout.

    packed/counts/lengths have one row per query (lengths are causal KV lengths).
    indexer_logits/ids are bounded tile scratch; tiles are shape policy. Selection writes each consumed range before attention reads it.
    """

    packed: torch.Tensor
    counts: torch.Tensor
    lengths: torch.Tensor
    indexer_logits: torch.Tensor
    ids: torch.Tensor
    tiles: tuple[tuple[int, int], ...]
    use_deepselect: bool = False
    topk_lengths: torch.Tensor | None = None
    topk_lengths_ready: bool = False
    topk_ids: torch.Tensor | None = None


class Step5IndexerMetadata(msgspec.Struct, frozen=True, kw_only=True):
    """Complete indexer metadata, assembled before entering decoder layers.

    R = compression request rows, Q = packed query/token rows, G = maximum
    completed blocks per request. Plans/validity are computed once per forward
    (or by captured GPU operations on replay); scratch is rewritten per layer.
    K/Z rings, position tags and completed compressed keys belong to the backend.
    """

    # Borrowed references; storage and lifetime remain owned by the backend and pools.
    layer_states: dict[int, Step5LayerIndexerState]
    req_to_token: torch.Tensor

    # Compression access plan. Request rows need not equal query rows: prefill
    # packs a variable number of input tokens per request.
    request_slots: torch.Tensor  # [R], request-pool row, zero means padding
    input_start_positions: torch.Tensor  # [R], absolute logical token position
    input_start_rows: torch.Tensor  # [R], offset into this forward's packed K/Z
    input_lengths: torch.Tensor  # [R], number of input K/Z rows for the request
    # [G, R], flattened compressed-key address, NOT a physical KV page number.
    # Ordinal g names block floor(input_start_positions / 8) + g. Incomplete
    # destinations are unused; consumers test the block's causal completion.
    compressed_key_write_indices: torch.Tensor
    num_input_tokens: int  # capacity of this forward's packed K/Z, including padding

    # Input validity, immutable across layers. Padding is inactive regardless
    # of length. Mapping validity covers native slot bounds/duplicate ownership,
    # logical spans and FULL mappings; it does not validate layer-local ring tags.
    request_active: torch.Tensor  # bool [R], live request row, excluding slot 0
    request_mapping_valid: torch.Tensor  # bool [R], admission from input checks
    request_count_valid: torch.Tensor  # bool [1], live count fits request capacity

    # Query/scoring layout, complete at construction (no later replace/binding).
    query_request_slots: torch.Tensor  # [Q], request slot for each query
    query_positions: torch.Tensor  # [Q], absolute logical query positions
    score_width: int  # maximum logical block count to score per query

    # Per-layer scratch, owned by this eager forward or captured graph bucket.
    # _admit overwrites BOTH bool buffers for every request row, including padding,
    # before compression/ring writes. Do not cache these as input validity: ring
    # tags differ by layer. Padding succeeds without enabling any write.
    compression_enabled: torch.Tensor  # bool [R], input AND layer-ring admission
    compression_success: torch.Tensor  # bool [R], success (including inert padding)
    selection_workspace: Step5SelectionWorkspace


def allocate_selection_workspace(
    *,
    rows,
    score_width,
    device,
    max_score_width=None,
    row_to_req=None,
    positions=None,
):
    """Per-forward scratch shared by sequential DSA layers on the model stream.

    Selection/attention consume it before the next layer overwrites it. Each
    eager forward owns its own tensors; graph capture gives each bucket fixed
    addresses. It is not a global buffer shared with overlapping forwards.
    """
    if rows < 0 or not 1 <= score_width <= STEP5_SCORE_WORKSPACE_ELEMENTS:
        raise ValueError(
            "Step5 requires nonnegative rows and a score row fitting its workspace budget"
        )
    use_deepselect = False
    if score_width > STEP5_HISTORY_BLOCKS and can_use_step5_topk_deepselect(
        torch.device(device)
    ):
        padded = deepselect_score_width(score_width)
        if max_score_width is None or padded <= deepselect_score_width(max_score_width):
            score_width, use_deepselect = padded, True
    # 512 MiB FP32 score budget, independent of packed prefill length.
    capacity = selection_tile_rows(rows=rows, score_width=score_width)
    if rows > 0 and capacity == 0:
        raise ValueError(
            "Step5 selection tile cannot fit one score row: "
            f"rows={rows}, score_width={score_width}"
        )
    indexer_logits = torch.empty(
        (capacity, score_width), dtype=torch.float32, device=device
    )
    topk_lengths, topk_lengths_ready = None, False
    if use_deepselect:
        topk_lengths = torch.empty(rows, dtype=torch.int32, device=device)
        if row_to_req is not None and positions is not None:
            topk_lengths.copy_(torch.where(row_to_req > 0, positions // 8, 0))
            topk_lengths_ready = True
    return Step5SelectionWorkspace(
        packed=torch.empty((rows, 513), dtype=torch.int64, device=device),
        counts=torch.empty(rows, dtype=torch.int32, device=device),
        lengths=torch.empty(rows, dtype=torch.int32, device=device),
        indexer_logits=indexer_logits,
        ids=torch.empty((capacity, 513), dtype=torch.int32, device=device),
        use_deepselect=use_deepselect,
        topk_lengths=topk_lengths,
        topk_lengths_ready=topk_lengths_ready,
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
