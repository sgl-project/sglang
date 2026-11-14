"""Sparse-KV prefill mappings."""

from __future__ import annotations

from dataclasses import dataclass

import torch


def prepare_prefill_gather_rows(
    req_ids_cpu: torch.Tensor,
    seq_lens_cpu: torch.Tensor,
) -> tuple[torch.Tensor, int]:
    """Resolve compact gather shape without synchronizing a device stream.

    NPU Boolean indexing needs a device-to-host shape read. During overlap
    scheduling that read can wait on collectives whose next inputs have not
    been submitted yet. ForwardBatch already carries these values on CPU.
    """
    keep = torch.nonzero((seq_lens_cpu > 0) & (req_ids_cpu >= 0)).flatten()
    lengths = seq_lens_cpu[keep]
    return keep, int(lengths.sum())


@dataclass
class FusedPrefillPlan:
    """Forward-local physical mapping and borrowed same-stream workspace."""

    total_tokens: int
    source: torch.Tensor
    destination: torch.Tensor
    valid: torch.Tensor
    current_mapping: object = None


def make_fused_prefill_plan(batch, manager, *, prepared=None, current_valid=None):
    _, total = (
        prepare_prefill_gather_rows(
            batch.req_pool_indices_cpu,
            batch.seq_lens_cpu,
        )
        if prepared is None
        else prepared
    )
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.row_metadata import (
        prefill_plan,
    )

    current_metadata = None
    max_current_width = 0
    if current_valid is not None:
        current_metadata, max_current_width = _current_metadata(batch, current_valid)
    source, destination, valid, current_mapping = prefill_plan(
        batch,
        manager,
        total,
        current_valid=(
            current_valid.reshape(-1) if current_metadata is not None else None
        ),
        current_metadata=current_metadata,
        max_current_width=max_current_width,
    )
    return FusedPrefillPlan(total, source, destination, valid, current_mapping)


def _current_metadata(batch, current_valid):
    """Build native fused-plan metadata for compact or padded current rows."""
    req = batch.req_pool_indices_cpu
    ext = batch.extend_seq_lens_cpu
    prefix = batch.extend_prefix_lens_cpu
    n = req.numel()
    counts = [int(x) for x in ext]
    prefixes = [int(x) for x in prefix]
    rows = int(current_valid.numel())
    if not rows:
        return None, 0
    if sum(counts) == rows:
        slots = counts
    else:
        slots = [rows // n] * n
    starts, offset = [], 0
    for width in slots:
        starts.append(offset)
        offset += width
    metadata = torch.tensor(
        starts + prefixes + counts + slots,
        device=batch.req_pool_indices.device,
        dtype=torch.long,
    )
    return metadata, max(slots, default=0)
