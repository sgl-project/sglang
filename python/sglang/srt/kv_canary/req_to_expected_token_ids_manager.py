from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from sglang.kernels.ops.kv_canary.scatter_req_token_ids import (
    launch_scatter_req_token_ids_kernel,
)
from sglang.srt.utils.common import flatten_arrays_to_int64_tensor

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch


def compute_req_all_ids_info(
    reqs: list[Req],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Snapshot per-req (origin_input_ids + output_ids) as pinned CPU int64 tensors.

    Returns:
        ``(req_all_ids_flat, req_all_ids_lens)`` — both pinned CPU int64. ``flat`` is the
        flattened ``cat(r.origin_input_ids, r.output_ids) for r in reqs``; ``lens`` is
        per-req ``len(origin_input_ids) + len(output_ids)``.
    """
    parts = [arr for req in reqs for arr in (req.origin_input_ids, req.output_ids)]
    req_all_ids_flat = flatten_arrays_to_int64_tensor(
        parts, device=torch.device("cpu"), pin=True
    )
    req_all_ids_lens = torch.tensor(
        [len(req.origin_input_ids) + len(req.output_ids) for req in reqs],
        dtype=torch.int64,
        pin_memory=True,
    )
    return req_all_ids_flat, req_all_ids_lens


def populate_req_to_expected_token_ids(
    *,
    forward_batch: ForwardBatch,
    req_to_verify_expected_tokens: Optional[torch.Tensor],
) -> None:
    """Scatter the forward batch's per-req token-id snapshot into the device-side pool."""
    req_all_ids_flat_cpu = forward_batch.req_all_ids_flat
    req_all_ids_lens_cpu = forward_batch.req_all_ids_lens
    if req_all_ids_flat_cpu is None or req_all_ids_lens_cpu is None:
        return
    if req_to_verify_expected_tokens is None:
        return

    bs = int(forward_batch.req_pool_indices.shape[0])
    if bs == 0:
        return
    if int(req_all_ids_lens_cpu.shape[0]) != bs:
        raise RuntimeError(
            f"kv-canary: req_all_ids_lens length {int(req_all_ids_lens_cpu.shape[0])} != "
            f"batch_size {bs}; ForwardBatch snapshot diverged"
        )

    offsets_cpu = torch.zeros(bs + 1, dtype=torch.int64, pin_memory=True)
    offsets_cpu[1:] = torch.cumsum(req_all_ids_lens_cpu, dim=0)
    total_tokens = int(offsets_cpu[bs].item())
    if total_tokens != int(req_all_ids_flat_cpu.shape[0]):
        raise RuntimeError(
            f"kv-canary: cumsum(req_all_ids_lens)={total_tokens} != "
            f"req_all_ids_flat.numel()={int(req_all_ids_flat_cpu.shape[0])}; snapshot inconsistent"
        )
    if total_tokens == 0:
        return

    device = req_to_verify_expected_tokens.device
    req_all_ids_flat_dev = req_all_ids_flat_cpu.to(device, non_blocking=True)
    offsets_dev = offsets_cpu.to(device, non_blocking=True)
    req_pool_indices_dev = forward_batch.req_pool_indices.to(
        device=device, dtype=torch.int64
    )

    launch_scatter_req_token_ids_kernel(
        flat_in=req_all_ids_flat_dev,
        offsets=offsets_dev,
        req_pool_indices=req_pool_indices_dev,
        pool_out=req_to_verify_expected_tokens,
    )
