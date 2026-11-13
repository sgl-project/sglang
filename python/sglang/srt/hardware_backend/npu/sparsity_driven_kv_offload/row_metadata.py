"""Adapt sparse-KV forward metadata to the companion NPU tensor operators."""

from __future__ import annotations

from sgl_kernel_npu.sparsity_driven_kv_offload import row_metadata as kernels


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
    return kernels.mtp_rows(
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


def prefill_plan(
    batch,
    manager,
    total,
    *,
    current_valid=None,
    current_metadata=None,
    max_current_width=0,
):
    """Keep CPU-derived output sizing outside the tensor-only device operator."""
    return kernels.prefill_plan(
        batch.req_pool_indices,
        batch.seq_lens,
        manager.req_to_token_pool.req_to_token,
        total,
        int(batch.seq_lens_cpu.max()) if total else 0,
        manager.actual_request_rows,
        manager.logical_max_context_len,
        current_metadata=current_metadata,
        current_valid=current_valid,
        max_current_width=max_current_width,
    )


def copy_rows_metadata(sources, destinations, source_rows, destination_rows, valid):
    """Build descriptors through the matching companion kernel package."""
    return kernels.copy_rows_metadata(
        sources, destinations, source_rows, destination_rows, valid
    )
