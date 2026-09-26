"""Strictly-gated TP-sharded greedy argmax helpers.

The fast path intentionally handles only the case where sampling is exactly
``argmax(logits)``. Any operation which can inspect or mutate the full vocab
logits must keep using the regular all-gather path.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist

from sglang.srt.environ import envs


def can_use_tp_sharded_greedy(forward_batch) -> bool:
    """Return a rank-consistent, batch-only eligibility decision.

    LM-head layout and TP topology checks are applied later by
    :class:`LogitsProcessor`, where the sharded weight is available.
    """

    if not envs.SGLANG_ENABLE_TP_SHARDED_GREEDY.get():
        return False

    sampling_info = getattr(forward_batch, "sampling_info", None)
    if sampling_info is None or not sampling_info.is_all_greedy:
        return False
    if getattr(forward_batch, "return_logprob", False):
        return False
    if any(getattr(forward_batch, "top_logprobs_nums", None) or []):
        return False
    if any(x for x in (getattr(forward_batch, "token_ids_logprobs", None) or [])):
        return False
    if getattr(forward_batch, "is_prefill_only", False):
        return False
    if getattr(forward_batch, "spec_info", None) is not None:
        return False

    forward_mode = forward_batch.forward_mode
    if forward_mode.is_target_verify() or forward_mode.is_draft_extend_v2():
        return False

    if sampling_info.has_custom_logit_processor:
        return False
    if sampling_info.grammars or sampling_info.grammar_mask is not None:
        return False
    if sampling_info.logit_bias is not None:
        return False
    if sampling_info.acc_additive_penalties is not None:
        return False
    if sampling_info.acc_scaling_penalties is not None:
        return False
    if (
        sampling_info.penalizer_orchestrator is not None
        and sampling_info.penalizer_orchestrator.is_required
    ):
        return False
    if any(sampling_info.return_sampling_masks or []):
        return False

    # The current draft deliberately leaves CUDA-graph buffers on the proven
    # full-logits path. A follow-up can capture the tiny candidate collectives.
    if getattr(forward_batch, "next_token_logits_buffer", None) is not None:
        return False
    return True


def select_global_argmax_candidates(
    candidate_values: torch.Tensor, candidate_token_ids: torch.Tensor
) -> torch.Tensor:
    """Select a global argmax, breaking exact ties by the lowest token id.

    Both inputs have shape ``[tp_size, batch]``. NaNs must already have been
    sanitized using the same policy as the regular sampler.
    """

    if (
        candidate_values.ndim != 2
        or candidate_token_ids.shape != candidate_values.shape
    ):
        raise ValueError("candidate values and token ids must have shape [tp, batch]")
    max_id = torch.iinfo(candidate_token_ids.dtype).max
    is_nan = torch.isnan(candidate_values)
    nan_ids = torch.where(is_nan, candidate_token_ids, max_id).min(dim=0).values

    # torch.argmax treats NaN as the winning value and returns its first index.
    # Reproduce that behavior across shards when sanitization is disabled.
    max_values = candidate_values.max(dim=0).values
    tied_ids = torch.where(
        candidate_values == max_values.unsqueeze(0),
        candidate_token_ids,
        max_id,
    )
    finite_ids = tied_ids.min(dim=0).values
    return torch.where(is_nan.any(dim=0), nan_ids, finite_ids)


def tp_sharded_greedy_argmax(
    local_logits: torch.Tensor,
    *,
    vocab_start: int,
    vocab_end: int,
    process_group,
    world_size: int,
) -> Optional[torch.Tensor]:
    """All-gather one ``(value, global_token_id)`` candidate per row/rank."""

    valid_width = vocab_end - vocab_start
    if (
        local_logits.ndim != 2
        or valid_width <= 0
        or valid_width > local_logits.shape[1]
        or world_size <= 1
    ):
        return None

    local_values, local_indices = local_logits[:, :valid_width].max(dim=-1)
    local_token_ids = local_indices.to(torch.int32).add_(vocab_start)

    # One collective for the pair. The token id occupies the second float32
    # lane as raw int32 bits, avoiding precision loss and a second NCCL launch.
    packed = torch.empty(
        (local_values.numel(), 2), dtype=torch.float32, device=local_values.device
    )
    packed[:, 0] = local_values
    packed.view(torch.int32)[:, 1] = local_token_ids
    gathered = torch.empty(
        (world_size * local_values.numel(), 2),
        dtype=packed.dtype,
        device=packed.device,
    )
    dist.all_gather_into_tensor(gathered, packed, group=process_group)
    batch = local_values.numel()
    return select_global_argmax_candidates(
        gathered[:, 0].view(world_size, batch),
        gathered.view(torch.int32)[:, 1].view(world_size, batch),
    ).to(torch.int64)
