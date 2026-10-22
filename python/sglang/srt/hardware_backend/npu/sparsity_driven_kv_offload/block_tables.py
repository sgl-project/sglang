"""Utilities for constructing graph-stable sparse-KV block tables."""

from __future__ import annotations

import torch


def fill_request_block_table(
    req_ids: torch.Tensor,
    valid_req: torch.Tensor,
    page_offsets: torch.Tensor,
    pages_per_request: int,
    output: torch.Tensor,
) -> None:
    """Fill row-major physical page IDs for request rows, in place."""
    if req_ids.dim() != 1 or valid_req.shape != req_ids.shape:
        raise ValueError("req_ids and valid_req must be matching 1-D tensors")
    if output.dim() != 2 or output.shape != (req_ids.numel(), page_offsets.numel()):
        raise ValueError("output must have shape [batch, pages_per_request]")
    if output.dtype != torch.int32 or req_ids.dtype != torch.int32:
        raise ValueError("request IDs and output must use int32")
    safe_req_ids = torch.where(valid_req, req_ids, torch.zeros_like(req_ids))
    torch.mul(
        safe_req_ids.view(-1, 1).expand_as(output),
        pages_per_request,
        out=output,
    )
    torch.add(output, page_offsets.view(1, -1), out=output)
