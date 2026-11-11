"""Adapt sparse-KV forward metadata to the companion NPU tensor operators."""

from __future__ import annotations

import importlib
from functools import lru_cache

import torch


@lru_cache(maxsize=1)
def _kernel_module():
    return importlib.import_module(
        "sgl_kernel_npu.sparsity_driven_kv_offload.row_metadata"
    )


def mtp_rows(
    batch,
    query_width,
    request_rows,
    context_len,
    *,
    publication_rows=None,
    host_capacity=0,
):
    """Pass tensor inputs while retaining framework forward-mode policy."""
    counts = (
        batch.spec_info.num_accept_tokens
        if batch.forward_mode.is_draft_extend_v2()
        else None
    )
    return _kernel_module().mtp_rows(
        batch.req_pool_indices,
        batch.seq_lens,
        batch.positions,
        batch.out_cache_loc,
        counts,
        query_width,
        request_rows,
        context_len,
        publication_rows=publication_rows,
        host_capacity=host_capacity,
    )


def mtp_publication_rows(batch, num_tokens, request_rows, context_len, host_capacity):
    """Fuse MTP masks, TP padding, physical bounds and publication indices."""
    from sglang.srt.hardware_backend.npu.sparsity_driven_kv_offload.mtp import (
        mtp_row_geometry,
    )

    _, width, _ = mtp_row_geometry(batch, num_tokens)
    if batch.out_cache_loc is None or batch.out_cache_loc.numel() != num_tokens:
        raise ValueError("Sparse MTP publication requires one physical row per token")
    return mtp_rows(
        batch,
        width,
        request_rows,
        context_len,
        publication_rows=num_tokens,
        host_capacity=host_capacity,
    )


def prefill_plan(batch, manager, total):
    """Keep CPU-derived output sizing outside the tensor-only device operator."""
    return _kernel_module().prefill_plan(
        batch.req_pool_indices,
        batch.seq_lens,
        manager.req_to_token_pool.req_to_token,
        total,
        int(batch.seq_lens_cpu.max()) if total else 0,
        manager.actual_request_rows,
        manager.logical_max_context_len,
    )


def prefill_current_mapping(
    plan, batch, current_valid, capacity, slots, requests, lengths, prefixes, extends
):
    """Build a D2D map while rejecting duplicate physical current rows."""
    locations = batch.out_cache_loc.reshape(-1).contiguous()
    current_valid = current_valid.reshape(-1).contiguous()
    ends, starts, live_ends = [], [], []
    row_end = full_start = 0
    for width, req, length, ext in zip(slots, requests, lengths, extends):
        starts.append(full_start)
        live_ends.append(row_end + ext if req > 0 and length > 0 else row_end)
        row_end += width
        ends.append(row_end)
        if req >= 0 and length > 0:
            full_start += length
    device = plan.source.device
    metadata = torch.tensor(
        ends + starts + prefixes + live_ends, device=device, dtype=torch.long
    )
    return _kernel_module().prefill_current_mapping(
        plan.source,
        locations,
        current_valid,
        metadata,
        capacity,
        plan.total_tokens,
        len(slots),
    )


def copy_rows_metadata(sources, destinations, source_rows, destination_rows, valid):
    """Build descriptors through the matching companion kernel package."""
    return _kernel_module().copy_rows_metadata(
        sources, destinations, source_rows, destination_rows, valid
    )
