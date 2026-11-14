"""Adapt sparse-KV forward metadata to the companion NPU tensor operators."""

from __future__ import annotations

from sgl_kernel_npu.sparsity_driven_kv_offload import row_metadata as kernels


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
