"""Shape graph-stable Copy-SFA output storage for the active query rows."""

from __future__ import annotations

import torch


def copy_sfa_host_row_stride(
    logical_context_len: int, enabled: bool, page_size: int = 128
) -> int:
    """Return the host-row stride without changing an opted-out manager."""
    if logical_context_len <= 0 or page_size <= 0:
        raise ValueError("context length and page size must be positive")
    if not enabled:
        return logical_context_len
    return ((logical_context_len + page_size - 1) // page_size) * page_size


def view_copy_sfa_output(
    output_buffer: torch.Tensor,
    batch_size: int,
    query_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Return a contiguous active output view without allocating storage."""
    required_elements = batch_size * query_heads * head_dim
    if batch_size < 0 or query_heads < 0 or head_dim < 0:
        raise ValueError("Copy-SFA output dimensions must be non-negative")
    if required_elements > output_buffer.numel():
        raise ValueError(
            "Copy-SFA output view exceeds the preallocated buffer: "
            f"requires {required_elements}, has {output_buffer.numel()} elements"
        )
    return output_buffer.view(-1)[:required_elements].view(
        batch_size, query_heads, head_dim
    )
