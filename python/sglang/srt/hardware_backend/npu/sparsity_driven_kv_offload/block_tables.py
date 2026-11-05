"""CPU metadata for compact sparse-KV prefill gathers."""

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
