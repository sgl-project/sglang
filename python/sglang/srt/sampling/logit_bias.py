"""The logit_bias sampling plugin: a dense [bs, vocab] additive bias, built at
each launch from the step's requests."""

from __future__ import annotations

from typing import TYPE_CHECKING, List, Optional

import torch

from sglang.srt.utils.device import async_h2d

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_batch_info import SamplingBatchInfo


def build_logit_bias(
    reqs: List[Req], *, vocab_size: int, device: str
) -> Optional[torch.Tensor]:
    """The step's bias, one row per request; None when no request carries one."""
    if all(r.sampling_params.logit_bias is None for r in reqs):
        return None
    rows, cols, vals = [], [], []
    for i, r in enumerate(reqs):
        if r.sampling_params.logit_bias is None:
            continue
        # Dedup on int(key) first ("1" and "01" collide): duplicate indices make
        # the index_put below nondeterministic on CUDA.
        row_bias = {
            int(key): value for key, value in r.sampling_params.logit_bias.items()
        }
        rows.extend([i] * len(row_bias))
        cols.extend(row_bias.keys())
        vals.extend(row_bias.values())
    logit_bias = torch.zeros(len(reqs), vocab_size, device=device)
    if rows:
        index = async_h2d([rows, cols], dtype=torch.int64, device=device)
        values = async_h2d(vals, dtype=logit_bias.dtype, device=device)
        logit_bias[index[0], index[1]] = values
    return logit_bias


def apply_logit_bias(
    logits: torch.Tensor,
    sampling_info: SamplingBatchInfo,
    *,
    rows_per_request: int = 1,
) -> None:
    """Add each request's bias to its `rows_per_request` consecutive logits rows."""
    if sampling_info.logit_bias is None:
        return
    # In the bias's own dtype, so the sum rounds once to the logits dtype.
    logits.unflatten(0, (-1, rows_per_request)).add_(
        sampling_info.logit_bias[:, None, :]
    )
