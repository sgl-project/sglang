"""Input contracts shared by the masked and contiguous row movers."""

from __future__ import annotations

import torch


def check_source_rows(
    hidden_states: torch.Tensor, topk_ids: torch.Tensor, top_k: int
) -> None:
    if hidden_states.dtype != torch.bfloat16:
        raise ValueError("the row movers take BF16 [num_tokens, hidden] input")
    if hidden_states.ndim != 2 or not hidden_states.is_contiguous():
        raise ValueError("hidden_states must be contiguous [num_tokens, hidden]")
    if not topk_ids.is_contiguous():
        raise ValueError("topk_ids must be contiguous")
    if topk_ids.numel() != hidden_states.size(0) * top_k:
        raise ValueError(
            f"topk_ids carries {topk_ids.numel()} pairs for "
            f"{hidden_states.size(0)} tokens x top_k={top_k}"
        )
