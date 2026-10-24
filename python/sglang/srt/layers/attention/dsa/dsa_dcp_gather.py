"""Gather the DCP-striped latent KV of an extend batch into one position-ordered buffer per layer."""

from __future__ import annotations

import msgspec
import torch


class DcpGatherPlan(msgspec.Struct, frozen=True):
    # Physical rows this rank contributes, in (request, position) order, padded to n_max.
    send_rows: torch.Tensor
    # Index into the [dcp_size * n_max] all-gather result for every (request, position).
    recv_rows: torch.Tensor
    # [bs, max_len] int32: row of position p of request b in the gathered buffer.
    page_table: torch.Tensor
    num_tokens: int
    num_rows: int
    # Widened slot -> gathered row for every slot of this batch (fused top-k emits slots).
    slot_to_row: torch.Tensor


def build_dcp_gather_plan(
    *,
    page_table: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    dcp_size: int,
    dcp_rank: int,
    physical_page_size: int,
    slot_to_row: torch.Tensor,
) -> DcpGatherPlan:
    device = page_table.device
    bs, max_len = page_table.shape
    lens = seq_lens_cpu.to(torch.int64)
    total = int(lens.sum())
    base = (torch.cumsum(lens, 0) - lens).to(device)
    req_idx = torch.repeat_interleave(torch.arange(bs), lens).to(device)
    pos = torch.arange(total, device=device) - base[req_idx]
    slots = page_table[req_idx, pos].to(torch.int64)
    owner = slots % dcp_size
    local_row = slots // dcp_size
    # Owners alternate within every widened page, so no rank holds more than total / dcp + bs rows.
    n_max = total // dcp_size + bs
    order = torch.argsort(owner, stable=True)
    counts = torch.bincount(owner, minlength=dcp_size)
    starts = torch.cumsum(counts, 0) - counts
    rank_index = torch.empty_like(owner)
    rank_index[order] = torch.arange(total, device=device) - starts[owner[order]]
    send_rows = torch.zeros(n_max + 1, dtype=torch.int64, device=device)
    send_rows.scatter_(0, torch.where(owner == dcp_rank, rank_index, n_max), local_row)
    recv_rows = owner * n_max + rank_index
    gathered_table = (
        base[:, None] + torch.arange(max_len, device=device)[None, :]
    ).clamp_(max=total - 1)
    num_rows = -(-total // physical_page_size) * physical_page_size
    slot_to_row[slots] = torch.arange(total, device=device, dtype=slot_to_row.dtype)
    return DcpGatherPlan(
        send_rows=send_rows[:n_max],
        recv_rows=recv_rows,
        page_table=gathered_table.to(torch.int32),
        num_tokens=total,
        num_rows=num_rows,
        slot_to_row=slot_to_row,
    )


def remap_slots_to_gathered_rows(
    page_table_1: torch.Tensor, slot_to_row: torch.Tensor
) -> torch.Tensor:
    """Fused top-k emits widened KV slots; translate them into gathered rows, keeping -1 padding."""
    rows = slot_to_row[page_table_1.clamp(min=0).to(torch.int64)]
    return torch.where(page_table_1 >= 0, rows.to(page_table_1.dtype), page_table_1)


def gather_dcp_prefix_kv(
    k_cache: torch.Tensor, plan: DcpGatherPlan, group, kv_cache_dim: int
) -> torch.Tensor:
    """All-gather every rank's owned rows and reorder them by position; returns [num_rows, kv_cache_dim]."""
    rows = k_cache.view(-1, kv_cache_dim)
    dtype = rows.dtype
    rows = rows.view(torch.uint8)
    send = rows.index_select(0, plan.send_rows)
    recv = group.all_gather(send, dim=0)
    out = torch.empty(
        (plan.num_rows, rows.shape[1]), dtype=torch.uint8, device=rows.device
    )
    torch.index_select(recv, 0, plan.recv_rows, out=out[: plan.num_tokens])
    return out.view(dtype)
