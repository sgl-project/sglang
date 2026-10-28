# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch

from sglang.kernels.ops.attention.step5 import options as step5_options

STEP5_KV_PAGE_TOKENS = 64
STEP5_REGION_TOKENS = 8
STEP5_PREFILL_SCORING_HEADS = 4
STEP5_PREFILL_SCORING_BLOCK_Q = 32


def can_use_step5_scoring_prefill(device: torch.device, heads: int) -> bool:
    return (
        # TODO(yhyang201): frozen SGLANG_OPT_STEP5_SCORING_PREFILL_SM100; prune dead paths.
        step5_options.SCORING_PREFILL_SM100
        and heads == STEP5_PREFILL_SCORING_HEADS
        and device.type == "cuda"
        and torch.version.hip is None
        and torch.cuda.get_device_capability(device)[0] == 10
    )


@dataclass
class Step5PrefillScoringTile:
    rows: int
    rows_padded: int
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    block_table: torch.Tensor


@dataclass
class Step5PrefillScoringPlan:
    heads: int
    tiles: list
    q: torch.Tensor
    weights: torch.Tensor
    q_runtime: torch.Tensor
    kernel_weights: torch.Tensor


def build_step5_prefill_scoring_plan(
    *,
    tiles,
    prefixes,
    lengths,
    request_slots,
    req_to_token,
    capacity,
    heads,
    device,
):
    align = STEP5_PREFILL_SCORING_BLOCK_Q
    starts = [0]
    for n in lengths:
        starts.append(starts[-1] + n)
    specs = []
    for tile_start, tile_end in tiles:
        cu_q, cu_k, slots, pages = [0], [0], [], []
        for i, (prefix, length) in enumerate(zip(prefixes, lengths)):
            lo = max(tile_start, starts[i])
            hi = min(tile_end, starts[i] + length)
            if lo >= hi:
                continue
            regions = (
                prefix + hi - starts[i] + STEP5_REGION_TOKENS - 1
            ) // STEP5_REGION_TOKENS
            cu_q.append(cu_q[-1] + hi - lo)
            cu_k.append(cu_k[-1] + regions)
            slots.append(i)
            pages.append(
                (regions * STEP5_REGION_TOKENS + STEP5_KV_PAGE_TOKENS - 1)
                // STEP5_KV_PAGE_TOKENS
            )
        specs.append((cu_q, cu_k, slots, pages) if slots else None)
    flat = [v for spec in specs if spec is not None for part in spec[:3] for v in part]
    staged = (
        torch.tensor(flat, dtype=torch.int32, pin_memory=True).to(
            device, non_blocking=True
        )
        if flat
        else None
    )
    plans = []
    offset = 0
    for spec in specs:
        if spec is None:
            plans.append(None)
            continue
        cu_q, cu_k, slots, pages = spec
        cu_q_t = staged[offset : offset + len(cu_q)]
        offset += len(cu_q)
        cu_k_t = staged[offset : offset + len(cu_k)]
        offset += len(cu_k)
        slot_idx = staged[offset : offset + len(slots)]
        offset += len(slots)
        slot = request_slots[slot_idx.long()].long()
        cols = max(pages) * STEP5_KV_PAGE_TOKENS
        table = (
            req_to_token[slot][:, 0:cols:STEP5_KV_PAGE_TOKENS] // STEP5_KV_PAGE_TOKENS
        ).to(torch.int32)
        rows = cu_q[-1]
        plans.append(
            Step5PrefillScoringTile(
                rows=rows,
                rows_padded=(rows + align - 1) // align * align,
                cu_seqlens_q=cu_q_t,
                cu_seqlens_k=cu_k_t,
                block_table=table.contiguous(),
            )
        )
    cap = (capacity + align - 1) // align * align
    return Step5PrefillScoringPlan(
        heads=heads,
        tiles=plans,
        q=torch.empty((cap, 1, heads, 256), dtype=torch.bfloat16, device=device),
        weights=torch.empty((cap, 1, heads), dtype=torch.float32, device=device),
        q_runtime=torch.empty(
            (cap * heads, 256), dtype=torch.float8_e4m3fn, device=device
        ),
        kernel_weights=torch.empty((cap, heads), dtype=torch.float32, device=device),
    )


def step5_prefill_scores(
    index_q: torch.Tensor,
    index_weights: torch.Tensor,
    *,
    compressed_index_keys: torch.Tensor,
    tile,
    plan: Step5PrefillScoringPlan,
    out: torch.Tensor,
    weight_scale: float | None,
) -> None:
    if tile is None:
        return
    from sglang.kernels.ops.attention.step5.cute.scoring_prefill_sm100 import (
        launch_prefill_paged_scoring,
    )

    heads = plan.heads
    rows, padded = tile.rows, tile.rows_padded
    if weight_scale is None and index_weights.dtype == torch.bfloat16:
        weight_scale = 0.5
    q = plan.q[:padded]
    w = plan.weights[:padded]
    q[:rows].copy_(index_q[:rows].view(rows, 1, heads, index_q.shape[-1]))
    w[:rows].copy_(index_weights[:rows].view(rows, 1, heads))
    if weight_scale is not None:
        w[:rows].mul_(weight_scale)
    launch_prefill_paged_scoring(
        q,
        w,
        compressed_index_keys,
        tile.block_table,
        tile.cu_seqlens_q,
        tile.cu_seqlens_k,
        int(compressed_index_keys.shape[1]),
        q_runtime=plan.q_runtime[: padded * heads],
        kernel_weights=plan.kernel_weights[:padded],
        out=out[:padded],
    )
