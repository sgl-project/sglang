"""Ascend KeyPool write-plan data and procedural buffer preparation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch


@dataclass(frozen=True)
class AscendKpoolWritePlan:
    """Stable write-plan buffers for one forward or graph capture bucket.

    The last operator request consumes graph padding. Real requests retain their
    computed TH rows even when invalid or unaccepted. Persistent tails are BF16;
    only the per-layer CANN state is FP32.
    """

    num_tokens: int
    cmp_ratio: int
    indexer_head_dim: int
    index_page_size: int
    tail_capacity: int
    mode: Literal["normal", "mtp"]
    max_query_len: int

    start_pos: torch.Tensor
    commit_lens: torch.Tensor
    req_pool_indices: torch.Tensor
    valid_requests: torch.Tensor
    query_start_loc: torch.Tensor

    state_start_pos: torch.Tensor
    state_block_table: torch.Tensor
    state_cache: torch.Tensor
    pooled_slot_mapping: torch.Tensor
    tail_write_indices: torch.Tensor
    tail_state_indices: torch.Tensor
    tail_token_indices: torch.Tensor
    _history_indices: torch.Tensor
    _history_valid: torch.Tensor


def _buffer_specs(batch, tokens, ratio, width, write_width, max_query_len):
    requests = batch + 1
    return {
        "query_start_loc": ((batch + 2,), torch.int32),
        "start_pos": ((batch,), torch.int32),
        "commit_lens": ((batch,), torch.int32),
        "req_pool_indices": ((batch,), torch.int32),
        "valid_requests": ((batch,), torch.bool),
        "state_start_pos": ((requests,), torch.int32),
        "state_block_table": (
            (requests, 1 + (max_query_len + ratio - 1) // ratio),
            torch.int32,
        ),
        "state_cache": ((1 + 2 * requests, ratio, 2 * width), torch.float32),
        "pooled_slot_mapping": (
            (min(tokens, tokens // ratio + requests),),
            torch.int64,
        ),
        "tail_write_indices": ((requests, write_width), torch.int64),
        "tail_state_indices": ((requests, write_width), torch.int64),
        "tail_token_indices": ((requests, write_width), torch.int64),
        "_history_indices": ((batch, ratio), torch.int64),
        "_history_valid": ((batch, ratio), torch.bool),
    }


def init_kpool_write_plan(
    *,
    batch_size,
    num_tokens,
    cmp_ratio,
    indexer_head_dim,
    index_page_size,
    tail_capacity,
    mode,
    max_query_len,
    device,
) -> AscendKpoolWritePlan:
    if mode not in ("normal", "mtp"):
        raise ValueError("KeyPool mode must be normal or mtp")
    if cmp_ratio <= 1 or tail_capacity < cmp_ratio or index_page_size <= 0:
        raise ValueError("invalid KeyPool cache geometry")
    write_width = max_query_len if mode == "mtp" else cmp_ratio
    if mode == "mtp" and tail_capacity < cmp_ratio + write_width:
        raise ValueError("KeyPool rollback ring must cover r+N rows")
    buffers = {
        name: torch.zeros(shape, dtype=dtype, device=device)
        for name, (shape, dtype) in _buffer_specs(
            batch_size,
            num_tokens,
            cmp_ratio,
            indexer_head_dim,
            write_width,
            max_query_len,
        ).items()
    }
    plan = AscendKpoolWritePlan(
        num_tokens=num_tokens,
        cmp_ratio=cmp_ratio,
        indexer_head_dim=indexer_head_dim,
        index_page_size=index_page_size,
        tail_capacity=tail_capacity,
        mode=mode,
        max_query_len=max_query_len,
        **buffers,
    )
    request = torch.arange(batch_size + 1, dtype=torch.int32, device=device)
    plan.state_block_table.copy_(
        (batch_size + 2 + request)[:, None].expand_as(plan.state_block_table)
    )
    plan.state_block_table[:, 0].copy_(request + 1)
    return plan


def update_kpool_write_plan(
    plan: AscendKpoolWritePlan,
    *,
    start_pos,
    query_lens,
    commit_lens,
    req_pool_indices,
    valid_requests,
    index_page_table,
):
    """Refresh all request inputs and final write addresses in place."""
    for name, value in (
        ("start_pos", start_pos),
        ("commit_lens", commit_lens),
        ("req_pool_indices", req_pool_indices),
        ("valid_requests", valid_requests),
    ):
        getattr(plan, name).copy_(value)
    plan.query_start_loc[0].zero_()
    plan.query_start_loc[1:-1].copy_(query_lens.cumsum(0))
    plan.query_start_loc[-1] = plan.num_tokens
    plan.state_start_pos[:-1].copy_(start_pos % plan.cmp_ratio)
    plan.state_start_pos[-1].zero_()
    _update_write_mappings(plan, index_page_table)


def _update_write_mappings(plan: AscendKpoolWritePlan, index_page_table):
    ratio, page_size = plan.cmp_ratio, plan.index_page_size
    batch = plan.req_pool_indices.numel()
    p = plan.start_pos.long()
    q = (plan.query_start_loc[1 : batch + 1] - plan.query_start_loc[:batch]).long()
    active = plan.valid_requests
    owner = torch.where(active, plan.req_pool_indices.long(), 0)
    slots = plan.pooled_slot_mapping
    plan.tail_write_indices.fill_(-1)
    plan.tail_state_indices.fill_(-1)
    plan.tail_token_indices.fill_(-1)
    # Count every computed group, including invalid requests; acceptance
    # only masks destinations and never shifts the next request's source.
    count = (p + q) // ratio - p // ratio
    offsets = count.cumsum(0) - count
    j = torch.arange((plan.max_query_len + ratio - 1) // ratio, device=p.device)
    source = offsets[:, None] + j
    computed = (j < count[:, None]) & (source < slots.numel())
    accepted = torch.minimum(plan.commit_lens.long().clamp_min(0), q)
    committed = (p + accepted) // ratio - p // ratio
    group = p[:, None] // ratio + j
    column = group // page_size
    valid = (
        active[:, None]
        & computed
        & (j < committed[:, None])
        & (column >= 0)
        & (column < index_page_table.shape[1])
    )
    if index_page_table.shape[1]:
        page = index_page_table.long().gather(
            1, column.clamp(0, index_page_table.shape[1] - 1)
        )
        destination = page * page_size + group % page_size
        valid = valid & (page >= 0)
    else:
        destination = torch.full_like(group, -1)
    mapping = slots.new_full((slots.numel() + 1,), -1)
    mapping.scatter_(
        0,
        torch.where(computed, source, slots.numel()).flatten(),
        torch.where(valid, destination, -1).flatten(),
    )
    slots.copy_(mapping[:-1])

    col = torch.arange(plan.tail_write_indices.shape[1], device=p.device)
    if plan.mode == "mtp":
        position = p[:, None] + col
        valid = active[:, None] & (col < q[:, None])
        token = plan.query_start_loc[:batch].long()[:, None] + col
        plan.tail_token_indices[:batch].copy_(torch.where(valid, token, -1))
    else:
        end = p + q
        remain = end % ratio
        position = end[:, None] - remain[:, None] + col
        valid = active[:, None] & (col < remain[:, None])
        request = torch.arange(batch, device=p.device)
        block = 1 + request + torch.where(p % ratio + q >= ratio, batch + 1, 0)
        state_row = block[:, None] * ratio + col
        plan.tail_state_indices[:batch].copy_(torch.where(valid, state_row, -1))
    destination = owner[:, None] * plan.tail_capacity + position % plan.tail_capacity
    plan.tail_write_indices[:batch].copy_(torch.where(valid, destination, -1))

    col = torch.arange(ratio, device=p.device)
    history = p[:, None] - p[:, None] % ratio + col
    plan._history_indices.copy_(
        owner[:, None] * plan.tail_capacity + history % plan.tail_capacity
    )
    plan._history_valid.copy_(active[:, None] & (col < p[:, None] % ratio))


def pack_compressed_tail(
    plan: AscendKpoolWritePlan, tail_k: torch.Tensor, tail_gate: torch.Tensor
) -> torch.Tensor:
    """Gather BF16 rings into FP32 state; write mappings are already complete."""
    if tail_k.shape != tail_gate.shape or tail_k.shape[1:] != (
        plan.tail_capacity,
        plan.indexer_head_dim,
    ):
        raise ValueError("persistent tail shape differs from workspace geometry")
    if tail_k.dtype != torch.bfloat16 or tail_gate.dtype != torch.bfloat16:
        raise ValueError("persistent KeyPool tails must be BF16")
    key = tail_k.view(-1, plan.indexer_head_dim)[plan._history_indices].float()
    gate = tail_gate.view(-1, plan.indexer_head_dim)[plan._history_indices].float()
    plan.state_cache.zero_()
    plan.state_cache[
        1 : plan.req_pool_indices.numel() + 1, :, : plan.indexer_head_dim
    ].copy_(torch.where(plan._history_valid[..., None], key, 0))
    plan.state_cache[
        1 : plan.req_pool_indices.numel() + 1, :, plan.indexer_head_dim :
    ].copy_(torch.where(plan._history_valid[..., None], gate, 0))
    return plan.state_cache
