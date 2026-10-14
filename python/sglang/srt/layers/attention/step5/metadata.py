"""Step5 per-forward metadata. Persistent cache storage belongs to the KV pool."""

from __future__ import annotations

import msgspec
import numpy as np
import torch

from sglang.kernels.ops.attention.step5 import options as step5_options

STEP5_SCORE_WORKSPACE_ELEMENTS = 128 * 1024 * 1024


def selection_tile_rows(rows: int, score_width: int) -> int:
    """Query rows per DSA selection tile: the FP32 score tile stays within the budget;
    step5_options.SELECTION_TILE_ROWS adds a fixed cap (0 = none)."""
    tile = min(int(rows), STEP5_SCORE_WORKSPACE_ELEMENTS // max(int(score_width), 1))
    # TODO(yhyang201): frozen SGLANG_OPT_STEP5_SELECTION_TILE_ROWS; prune dead paths.
    cap = step5_options.SELECTION_TILE_ROWS
    if cap < 0:
        raise ValueError("step5_options.SELECTION_TILE_ROWS must be nonnegative")
    if cap > 0:
        tile = min(tile, cap)
    return tile


def query_group_spans(lengths, query_group: int):
    """Query-group starts and ends in packed-token rows, plus the len(lengths) + 1
    request boundaries; all int64, in per-request order."""
    lengths = np.asarray(lengths, dtype=np.int64).reshape(-1)
    boundaries = np.zeros(lengths.size + 1, dtype=np.int64)
    np.cumsum(lengths, out=boundaries[1:])
    groups = (lengths + query_group - 1) // query_group
    request = np.repeat(np.arange(lengths.size), groups)
    first = np.repeat(np.cumsum(groups) - groups, groups)
    offsets = (np.arange(int(groups.sum())) - first) * query_group
    starts = boundaries[request] + offsets
    ends = np.minimum(starts + query_group, boundaries[request + 1])
    return starts, ends, boundaries


class Step5PrefillWorkspace(msgspec.Struct, frozen=True, kw_only=True):
    """Per-layer union/bitset scratch; only allocated for the prefill kernel.

    Query rows hold selected logical/physical regions; group rows hold their
    union and membership. The union builder initializes the buffers it reads
    each layer. Addresses are local to one forward, never shared globally.
    """

    logical_starts: torch.Tensor
    physical_regions: torch.Tensor
    union_physical: torch.Tensor
    union_logical: torch.Tensor
    membership: torch.Tensor
    union_counts: torch.Tensor
    bitset: torch.Tensor
    sm100_block_counts: torch.Tensor | None = None
    sm100_block_indices: torch.Tensor | None = None
    sm100_full_block_counts: torch.Tensor | None = None
    sm100_region_words: torch.Tensor | None = None
    sm100_token_limits: torch.Tensor | None = None


class Step5SelectionWorkspace(msgspec.Struct, frozen=True, kw_only=True):
    """Per-layer scoring/selection outputs, not immutable query layout.

    packed/counts/lengths have one row per query (lengths are causal KV lengths).
    region_scores/ids are bounded tile scratch; tiles/use_cute are shape/device
    policy. Selection writes each consumed range before attention reads it.
    """

    packed: torch.Tensor
    counts: torch.Tensor
    lengths: torch.Tensor
    region_scores: torch.Tensor
    ids: torch.Tensor | None  # portable selector only; SM10x packs directly
    tiles: tuple[tuple[int, int], ...]
    use_cute: bool
    scoring_workspace: object | None = None
    scoring_metadata_ready: bool = False
    use_deepselect: bool = False
    topk_lengths: torch.Tensor | None = None
    topk_lengths_ready: bool = False
    topk_ids: torch.Tensor | None = None
    prefill_scoring: dict | None = None


class Step5IndexerMetadata(msgspec.Struct, frozen=True, kw_only=True):
    """Complete indexer metadata, assembled before entering decoder layers.

    R = compression request rows, Q = packed query/token rows, G = maximum
    completed regions per request. Plans/validity are computed once per forward
    (or by captured GPU operations on replay); scratch is rewritten per layer.
    K/Z rings, position tags and completed compressed keys belong to the KV pool.
    """

    # Borrowed references; storage and lifetime remain owned by the pools.
    token_to_kv_pool: object
    req_to_token: torch.Tensor

    # Compression access plan. Request rows need not equal query rows: verify
    # packs multiple tokens per request, while prefill has variable input lengths.
    request_slots: torch.Tensor  # [R], request-pool row, zero means padding
    input_start_positions: torch.Tensor  # [R], absolute logical token position
    input_start_rows: torch.Tensor  # [R], offset into this forward's packed K/Z
    input_lengths: torch.Tensor  # [R], number of input K/Z rows for the request
    # [G, R], flattened compressed-key address, NOT a physical KV page number.
    # Ordinal g names region floor(input_start_positions / 8) + g. Incomplete
    # destinations are unused; consumers test the region's causal completion.
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
    score_width: int  # maximum logical region count to score per query
    score_group_rows: int  # rows sharing a scoring tile; does not change causality

    # Per-layer scratch, owned by this eager forward or captured graph bucket.
    # _admit overwrites BOTH bool buffers for every request row, including padding,
    # before compression/ring writes. Do not cache these as input validity: ring
    # tags differ by layer. Padding succeeds without enabling any write.
    compression_enabled: torch.Tensor  # bool [R], input AND layer-ring admission
    compression_success: torch.Tensor  # bool [R], success (including inert padding)
    selection_workspace: Step5SelectionWorkspace


class Step5AttentionMetadata(msgspec.Struct, frozen=True, kw_only=True):
    """Attention layout and scratch owned by one eager forward or graph bucket.

    Tensor contents can be updated on the owning model stream. Overlapping
    forwards must have separate objects and storage; none of this is global.
    """

    indexer_metadata: Step5IndexerMetadata
    # Layout: packed query boundaries, separate from request-pool slot indices.
    query_start_loc: torch.Tensor  # [request rows + 1], cumulative query counts
    group_start: torch.Tensor  # [groups], first query row (prefill/verify)
    group_end: torch.Tensor  # [groups], exclusive last query row
    region_bins: int  # power-of-two logical region capacity for union bitsets
    # Kernel family: target verification also uses per-row sparse decode.
    decode: bool
    valid_rows: torch.Tensor  # int32 [1], live QUERY rows, not request count
    # Shape/device-derived split schedule, fixed for this metadata's query layout.
    decode_dynamic_splits: bool  # one request, runtime split count from selection
    decode_split4_rows: int  # leading query rows assigned four partials (static)
    decode_split2_rows: int  # following query rows assigned two partials (static)
    decode_work_rows: int  # total partial-output rows (dynamic: 16 * query rows)
    # Per-layer execution scratch; the selected attention kernel/merge overwrites
    # its consumed range before the next reader. Storage follows this metadata.
    lse: (
        torch.Tensor | None
    )  # FP32 [query rows, query heads]; unused by candidate prefill
    partial: torch.Tensor | None
    partial_lse: torch.Tensor | None
    prefill: Step5PrefillWorkspace | None
