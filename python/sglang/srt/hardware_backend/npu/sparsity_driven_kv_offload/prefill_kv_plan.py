"""Sparse-KV prefill mappings with CPU shape validation."""

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


def make_fused_prefill_plan(batch, manager, stream, *, prepared=None):
    _, total = (
        prepare_prefill_gather_rows(
            batch.req_pool_indices_cpu,
            batch.seq_lens_cpu,
            manager.size,
            manager.max_context_len,
        )
        if prepared is None
        else prepared
    )
    if (
        batch.req_pool_indices.numel() != batch.seq_lens.numel()
        or batch.req_pool_indices.numel() != batch.req_pool_indices_cpu.numel()
    ):
        raise RuntimeError("Prefill CPU and device metadata must have matching sizes")
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.row_metadata import (
        prefill_plan,
    )

    source, destination, valid = prefill_plan(batch, manager, total)
    return FusedPrefillPlan(stream, total, source, destination, valid)


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
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.row_metadata import (
        prefill_current_mapping,
    )

    return prefill_current_mapping(
        plan,
        batch,
        current_valid,
        host_token_capacity,
        slots,
        requests,
        lens,
        prefixes,
        counts,
    )
