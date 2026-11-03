"""Utilities for constructing graph-stable sparse-KV block tables."""

from __future__ import annotations

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
