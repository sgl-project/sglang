"""Bound score workspace while retaining the native per-row selection contract."""

import torch

from sglang.kernels.ops.attention.step5 import options as step5_options
from sglang.kernels.ops.attention.step5.scoring import (
    can_use_step5_scoring_cute,
    step5_paged_scores,
)
from sglang.kernels.ops.attention.step5.scoring_prefill import (
    build_step5_prefill_scoring_plan,
    can_use_step5_scoring_prefill,
    step5_prefill_scores,
)
from sglang.kernels.ops.attention.step5.selection import (
    can_use_step5_selection_cute,
    step5_select_regions,
)
from sglang.kernels.ops.attention.step5.selection_deepselect import (
    STEP5_HISTORY_REGIONS,
    can_use_step5_selection_deepselect,
    deepselect_score_width,
    step5_select_regions_deepselect,
)
from sglang.kernels.ops.attention.step5.sparse_metadata import step5_pack_decode_regions
from sglang.srt.layers.attention.step5.metadata import (
    STEP5_SCORE_WORKSPACE_ELEMENTS,
    Step5SelectionWorkspace,
    selection_tile_rows,
)


def allocate_selection_workspace(
    *,
    rows,
    score_width,
    score_group_rows,
    device,
    prefill=False,
    max_score_width=None,
    prefill_spans=None,
    req_to_token=None,
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
    if (
        # TODO(yhyang201): frozen SGLANG_OPT_STEP5_TOPK_DEEPSELECT_DECODE; prune dead paths.
        (prefill or step5_options.TOPK_DEEPSELECT_DECODE)
        and score_width > STEP5_HISTORY_REGIONS
        and can_use_step5_selection_deepselect(torch.device(device))
    ):
        padded = deepselect_score_width(score_width)
        if max_score_width is None or padded <= deepselect_score_width(max_score_width):
            score_width, use_deepselect = padded, True
    # 512 MiB FP32 score budget, independent of packed prefill length.
    capacity = selection_tile_rows(rows=rows, score_width=score_width)
    if score_group_rows > 1 and capacity:
        capacity -= capacity % score_group_rows
    if rows > 0 and capacity == 0:
        raise ValueError(
            "Step5 selection tile cannot fit one score group: "
            f"rows={rows}, score_group_rows={score_group_rows}, "
            f"score_width={score_width}. Check "
            "step5_options.SELECTION_TILE_ROWS and the score workspace budget."
        )
    prefill_scoring = None
    if use_deepselect and prefill_spans is not None and req_to_token is not None:
        prefill_scoring = dict(spans=prefill_spans, req_to_token=req_to_token)
    score_rows = (capacity + 63) // 64 * 64 + 64 if prefill_scoring else capacity
    region_scores = torch.empty(
        (score_rows, score_width), dtype=torch.float32, device=device
    )
    use_cute = can_use_step5_selection_cute(region_scores)
    topk_lengths, topk_lengths_ready = None, False
    if use_deepselect:
        topk_lengths = torch.empty(rows, dtype=torch.int32, device=device)
        if row_to_req is not None and positions is not None:
            topk_lengths.copy_(torch.where(row_to_req > 0, positions // 8, 0))
            topk_lengths_ready = True
    scoring_workspace = None
    scoring_metadata_ready = False
    if capacity and can_use_step5_scoring_cute(region_scores.device):
        from sglang.kernels.ops.attention.step5.scoring_cute import (
            allocate_scoring_workspace,
            prepare_scoring,
        )

        scoring_workspace = allocate_scoring_workspace(
            capacity, score_width, score_group_rows, region_scores.device
        )
        # Metadata is computed in the forward graph, hence refreshed on replay.
        # Prefill/tiled/CLEAR_FUTURE paths retain the original implementation.
        if (
            # TODO(yhyang201): frozen SGLANG_OPT_STEP5_INDEXER_NORM_PACKED; prune dead paths.
            step5_options.INDEXER_NORM_PACKED
            and not prefill
            and use_cute
            and rows == capacity
            and req_to_token is not None
            and row_to_req is not None
            and positions is not None
        ):
            prepare_scoring(scoring_workspace, req_to_token, row_to_req, positions)
            scoring_metadata_ready = True
    return Step5SelectionWorkspace(
        packed=torch.empty((rows, 513), dtype=torch.int64, device=device),
        counts=torch.empty(rows, dtype=torch.int32, device=device),
        lengths=torch.empty(rows, dtype=torch.int32, device=device),
        region_scores=region_scores,
        use_cute=use_cute,
        scoring_workspace=scoring_workspace,
        scoring_metadata_ready=scoring_metadata_ready,
        ids=(
            None
            if use_cute and not use_deepselect
            else torch.empty((capacity, 513), dtype=torch.int32, device=device)
        ),
        use_deepselect=use_deepselect,
        prefill_scoring=prefill_scoring,
        topk_lengths=topk_lengths,
        topk_lengths_ready=topk_lengths_ready,
        topk_ids=(
            torch.empty(
                (capacity, STEP5_HISTORY_REGIONS), dtype=torch.int32, device=device
            )
            if use_deepselect
            else None
        ),
        tiles=tuple(
            (start, min(start + capacity, rows))
            for start in range(0, rows, capacity or 1)
        ),
    )


def select_step5_regions(
    *,
    index_q: torch.Tensor,
    index_weights: torch.Tensor,
    compressed_index_keys: torch.Tensor,
    req_to_token: torch.Tensor,
    row_to_req: torch.Tensor,
    positions: torch.Tensor,
    score_width: int,
    score_group_rows: int = 1,
    scoring_weight_scale: float | None = None,
    packed_qw_ready: bool = False,
    workspace: Step5SelectionWorkspace | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return packed region IDs, counts and lengths without owning request state.

    CSA publication precedes this call. Every row sees the same completed
    compressed_index_keys and causal position as the untiled path. Slicing only query rows
    keeps request IDs, logical regions and subsequent prefill groups unchanged.
    Scratch is reused on the caller's stream; Graph loops depend on shapes only.
    """
    if workspace is None:
        # Standalone kernel callers may allocate their own one-call workspace.
        workspace = allocate_selection_workspace(
            rows=index_q.shape[0],
            score_width=score_width,
            score_group_rows=score_group_rows,
            device=index_q.device,
            row_to_req=row_to_req,
            positions=positions,
        )
    packed, counts, lengths = workspace.packed, workspace.counts, workspace.lengths
    region_scores, ids = workspace.region_scores, workspace.ids
    plan = _prefill_scoring_plan(workspace, index_q)
    if packed_qw_ready and (
        not workspace.scoring_metadata_ready
        or not workspace.use_cute
        or len(workspace.tiles) != 1
        or plan is not None
    ):
        raise ValueError("Packed Q/W requires prepared single-tile decode scoring")
    for tile_index, (start, end) in enumerate(workspace.tiles):
        tile_scores = region_scores[: end - start]
        if packed_qw_ready:
            from sglang.kernels.ops.attention.step5.scoring_cute import (
                launch_prepared_scoring,
            )

            launch_prepared_scoring(
                compressed_index_keys,
                tile_scores,
                workspace.scoring_workspace,
                end - start,
            )
        elif plan is not None:
            step5_prefill_scores(
                index_q[start:end],
                index_weights[start:end],
                compressed_index_keys=compressed_index_keys,
                tile=plan.tiles[tile_index],
                plan=plan,
                out=region_scores,
                weight_scale=scoring_weight_scale,
            )
        else:
            step5_paged_scores(
                index_q[start:end],
                index_weights[start:end],
                compressed_index_keys=compressed_index_keys,
                req_to_token=req_to_token,
                row_to_req=row_to_req[start:end],
                positions=positions[start:end],
                out=tile_scores,
                clear_future=not workspace.use_cute,
                group_rows=score_group_rows,
                workspace=workspace.scoring_workspace,
                weight_scale=scoring_weight_scale,
            )
        if workspace.use_deepselect:
            step5_select_regions_deepselect(
                tile_scores,
                positions[start:end],
                row_to_req[start:end],
                packed[start:end],
                counts[start:end],
                req_to_token=req_to_token,
                causal_lengths=lengths[start:end],
                history_lengths=workspace.topk_lengths[start:end],
                topk_ids=workspace.topk_ids[: end - start],
                region_ids=ids[: end - start],
                history_ready=workspace.topk_lengths_ready,
            )
        elif workspace.use_cute:
            from sglang.kernels.ops.attention.step5.selection_cute import (
                step5_select_regions_cute,
            )

            step5_select_regions_cute(
                tile_scores,
                positions[start:end],
                row_to_req[start:end],
                packed[start:end],
                counts[start:end],
                req_to_token=req_to_token,
                causal_lengths=lengths[start:end],
            )
        else:
            tile_ids = ids[: end - start]
            step5_select_regions(
                tile_scores,
                positions=positions[start:end],
                row_to_req=row_to_req[start:end],
                region_ids=tile_ids,
                use_cute=workspace.use_cute,
                counts=counts[start:end],
            )
            step5_pack_decode_regions(
                tile_ids,
                counts[start:end],
                req_to_token=req_to_token,
                row_to_req=row_to_req[start:end],
                positions=positions[start:end],
                packed_regions=packed[start:end],
                causal_lengths=lengths[start:end],
            )
    return packed, counts, lengths


def _prefill_scoring_plan(workspace, index_q):
    state = workspace.prefill_scoring
    if state is None:
        return None
    if "plan" not in state:
        heads = int(index_q.shape[1])
        plan = None
        if can_use_step5_scoring_prefill(index_q.device, heads):
            prefixes, lengths, request_slots = state["spans"]
            plan = build_step5_prefill_scoring_plan(
                tiles=workspace.tiles,
                prefixes=list(prefixes),
                lengths=list(lengths),
                request_slots=request_slots,
                req_to_token=state["req_to_token"],
                capacity=workspace.region_scores.shape[0],
                heads=heads,
                device=index_q.device,
            )
        state["plan"] = plan
    return state["plan"]
