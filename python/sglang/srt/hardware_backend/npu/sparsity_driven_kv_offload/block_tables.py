"""CPU metadata for compact sparse-KV prefill gathers."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def prepare_prefill_gather_rows(
    req_ids_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
    request_rows: int,
    context_len: int,
) -> tuple[torch.Tensor, int]:
    """Resolve compact gather shape without synchronizing a device stream.

    NPU Boolean indexing needs a device-to-host shape read. During overlap
    scheduling that read can wait on collectives whose next inputs have not
    been submitted yet. ForwardBatch already carries these values on CPU.
    """
    if req_ids_cpu is None or seq_lens_cpu is None:
        raise RuntimeError("Prefill KV gather requires CPU request and length metadata")
    if req_ids_cpu.device.type != "cpu" or seq_lens_cpu.device.type != "cpu":
        raise RuntimeError("Prefill KV gather metadata must be on CPU")
    if req_ids_cpu.dim() != 1 or req_ids_cpu.shape != seq_lens_cpu.shape:
        raise RuntimeError(
            "Prefill request IDs and lengths must be matching 1-D tensors"
        )
    keep = torch.nonzero((seq_lens_cpu > 0) & (req_ids_cpu >= 0)).flatten()
    requests = req_ids_cpu[keep]
    lengths = seq_lens_cpu[keep]
    if bool((requests >= request_rows).any()):
        raise RuntimeError(
            f"Prefill request ID outside sparse pool size {request_rows}"
        )
    if bool((lengths > context_len).any()):
        raise RuntimeError(
            f"Prefill sequence length exceeds sparse context {context_len}"
        )
    return keep, int(lengths.sum())


@dataclass
class FusedPrefillPlan:
    """Forward-local physical mapping and borrowed same-stream workspace."""

    stream: object
    total_tokens: int
    source: torch.Tensor
    destination: torch.Tensor
    valid: torch.Tensor
    current_mapping: object = None


def make_fused_prefill_plan(batch, manager, stream):
    keep, total = prepare_prefill_gather_rows(
        batch.req_pool_indices_cpu,
        batch.seq_lens_cpu,
        manager.size,
        manager.max_context_len,
    )
    if (
        batch.req_pool_indices.numel() != batch.seq_lens.numel()
        or batch.req_pool_indices.numel() != batch.req_pool_indices_cpu.numel()
    ):
        raise RuntimeError("Prefill CPU and device metadata must have matching sizes")
    device = manager.device
    req_pool_indices = batch.req_pool_indices.to(device=device, dtype=torch.long)
    lengths = batch.seq_lens.to(device=device, dtype=torch.long)
    if keep.numel() != req_pool_indices.numel():
        keep = keep.to(device=device, dtype=torch.long, non_blocking=True)
        req_pool_indices = req_pool_indices.index_select(0, keep)
        lengths = lengths.index_select(0, keep)
    starts = torch.cumsum(lengths, 0) - lengths
    token_to_slots = torch.repeat_interleave(
        req_pool_indices, lengths, output_size=total
    )
    starts = torch.repeat_interleave(starts, lengths, output_size=total)
    destination = torch.arange(total, device=device, dtype=torch.long)
    token_positions = destination - starts
    source = manager.host_source_indices(token_to_slots, token_positions).contiguous()
    return FusedPrefillPlan(
        stream,
        total,
        source,
        destination,
        torch.ones(total, device=device, dtype=torch.bool),
    )


def fused_prefill_current_mapping(plan, batch, current_valid, host_token_capacity):
    """Return fixed-shape D2D mapping or None for ambiguous CPU layouts."""
    req = batch.req_pool_indices_cpu
    lengths = batch.seq_lens_cpu
    ext = batch.extend_seq_lens_cpu
    prefix = batch.extend_prefix_lens_cpu
    loc = batch.out_cache_loc
    n = req.numel()
    if (
        ext is None
        or prefix is None
        or loc is None
        or len(ext) != n
        or len(prefix) != n
        or lengths.numel() != n
    ):
        return None
    counts = [int(x) for x in ext]
    prefixes = [int(x) for x in prefix]
    lens = [int(x) for x in lengths]
    requests = [int(x) for x in req]
    active = [r for r, length in zip(requests, lens) if r >= 0 and length > 0]
    if len(set(active)) != len(active) or any(
        e < 0 or p < 0 or e + p != length
        for e, p, length in zip(counts, prefixes, lens)
    ):
        return None
    rows = int(loc.numel())
    if rows != current_valid.numel() or not rows or not plan.total_tokens:
        return None
    if sum(counts) == rows:
        slots = counts
    elif n and rows % n == 0 and all(e <= rows // n for e in counts):
        slots = [rows // n] * n
    else:
        return None
    dest, eligible = [], []
    offset = 0
    for req_id, length, pre, ext_len, width in zip(
        requests, lens, prefixes, counts, slots
    ):
        for local in range(width):
            is_valid = req_id > 0 and length > 0 and local < ext_len
            dest.append(offset + pre + local if is_valid else 0)
            eligible.append(is_valid)
        if req_id >= 0 and length > 0:
            offset += length
    device = plan.source.device
    dst = torch.tensor(dest, device=device, dtype=torch.long)
    eligible = torch.tensor(eligible, device=device, dtype=torch.bool)
    physical = loc.to(device=device, dtype=torch.long).reshape(-1)
    valid = (
        eligible & current_valid & (physical >= 0) & (physical < host_token_capacity)
    )
    safe_physical = physical.clamp(0, host_token_capacity - 1)
    counts_by_physical = torch.zeros(
        host_token_capacity, device=device, dtype=torch.int32
    )
    counts_by_physical.scatter_add_(0, safe_physical, valid.to(torch.int32))
    valid = (
        valid
        & (counts_by_physical[safe_physical] == 1)
        & (plan.source[dst] == physical)
    )
    occupied = torch.zeros(plan.total_tokens, device=device, dtype=torch.int32)
    occupied.scatter_add_(0, dst, valid.to(torch.int32))
    return (
        torch.arange(rows, device=device, dtype=torch.long),
        dst,
        valid.contiguous(),
        (occupied == 0).contiguous(),
    )
