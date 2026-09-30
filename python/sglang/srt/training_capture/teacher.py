"""Compact raw teacher scores, captured before any serving-side mutation."""

from __future__ import annotations

import msgspec
import torch
from sglang.srt.training_capture.protocol import ContractError


class TeacherRows(msgspec.Struct, frozen=True):
    token_ids: torch.Tensor
    logits: torch.Tensor
    logsumexp: torch.Tensor


@torch.no_grad()
def capture_teacher(
    raw_logits: torch.Tensor, vocab_size: int, row_indices: torch.Tensor | None = None
) -> TeacherRows:
    """All returned tensors own storage independent of the logits/graph buffer."""
    if raw_logits.ndim != 2 or not raw_logits.is_floating_point():
        raise ContractError(
            "teacher logits must be a floating [rows, vocabulary] tensor"
        )
    if not 128 <= vocab_size <= raw_logits.shape[1]:
        raise ContractError("teacher capture requires the complete unpadded vocabulary")
    scores = raw_logits[:, :vocab_size]
    if row_indices is not None:
        scores = scores.index_select(0, row_indices)
    values, ids = torch.topk(scores, k=128, dim=-1, sorted=True)
    return TeacherRows(
        token_ids=ids.to(torch.int32),
        logits=values.float(),
        logsumexp=torch.logsumexp(scores.float(), dim=-1),
    )
