"""Target-model sampling supports for chain EAGLE/NEXTN verification.

Capture the verifier's already-filtered distribution without changing sampling.
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist

from sglang.srt.layers.logits_processor import SamplingMaskOutput, SamplingMaskStatus


def spec_sampling_mask_unsupported_reason(
    *,
    spec_algorithm,
    eagle_topk: int,
    use_rejection_sampling: bool,
    accept_thresholds: Tuple[float, float],
    min_p: float,
    cuda: bool,
    simulate_acceptance: bool,
) -> Optional[str]:
    if not cuda:
        return "return_sampling_mask with speculative decoding requires CUDA."
    if simulate_acceptance:
        return "return_sampling_mask does not support simulated acceptance."
    if not spec_algorithm.is_eagle() or spec_algorithm.is_frozen_kv_mtp():
        return "return_sampling_mask with speculative decoding requires EAGLE/NEXTN."
    if eagle_topk != 1 or use_rejection_sampling:
        return (
            "return_sampling_mask with speculative decoding requires "
            "--speculative-eagle-topk 1 without rejection sampling."
        )
    if accept_thresholds != (1.0, 1.0):
        # Lower thresholds accept drafts the target distribution would not sample.
        return (
            "return_sampling_mask with speculative decoding requires the default "
            "speculative accept thresholds."
        )
    if min_p > 0:
        return "return_sampling_mask with speculative decoding does not support min_p."
    return None


def verify_sampling_mask_output(
    *,
    mask_req_rows: torch.Tensor,
    mask_probs: Optional[torch.Tensor],
    predict: torch.Tensor,
    accept_index: torch.Tensor,
    draft_token_num: int,
    max_tokens: int,
    support_capture_indices: Optional[torch.Tensor],
    sync_groups: Tuple,
) -> SamplingMaskOutput:
    """One support row per verify position of each opted-in request, request-major.

    ``mask_probs`` is ``None`` for greedy verify, whose support is the emitted token.
    Positions past the accept length are padding: status OK, length 0.
    """
    accept_rows = accept_index.index_select(0, mask_req_rows).long()
    valid_2d = accept_rows >= 0
    valid = valid_2d.view(-1)
    accept_rows = accept_rows.clamp(min=0)
    tokens = predict.index_select(0, accept_rows.view(-1)).long().view(-1, 1)
    num_rows = tokens.shape[0]
    statuses = torch.full(
        (num_rows,), SamplingMaskStatus.OK, dtype=torch.int32, device=tokens.device
    )
    support_rows = None
    support_logprobs = None
    if support_capture_indices is not None:
        stride = accept_index.shape[1]
        support_rows = (
            support_capture_indices.view(-1, 1) * stride
            + torch.arange(stride, device=tokens.device)
        ).view(-1)
    if mask_probs is None:
        token_ids = tokens.int()
        lengths = valid.int()
        selected_logprobs = torch.zeros(num_rows, device=tokens.device)
        if support_rows is not None:
            support_logprobs = torch.zeros(
                (support_rows.numel(), 1), device=tokens.device
            )
    else:
        # Row of mask_probs: request slot * draft_token_num + position within the request.
        slot_first_row = (
            torch.arange(mask_req_rows.numel(), device=tokens.device).view(-1, 1)
            - mask_req_rows.view(-1, 1)
        ) * draft_token_num
        local_rows = torch.where(valid_2d, accept_rows + slot_first_row, 0)
        rows = mask_probs.index_select(0, local_rows.view(-1))
        selected = rows.gather(1, tokens).squeeze(1).float()
        support_mass = rows.sum(dim=-1, dtype=torch.float32)
        selected_logprobs = torch.log(selected / support_mass)
        realized_lengths = (rows > 0).sum(dim=-1, dtype=torch.int32)
        statuses[realized_lengths > max_tokens] = SamplingMaskStatus.OVERFLOW
        statuses[~((selected > 0) & torch.isfinite(selected_logprobs))] = (
            SamplingMaskStatus.INVALID
        )
        statuses[~valid] = SamplingMaskStatus.OK
        packed_size = min(max_tokens, rows.shape[-1])
        packed_probs, token_ids = rows.topk(packed_size, dim=-1)
        token_ids = token_ids.int()
        if support_rows is not None:
            support_logprobs = torch.log(
                packed_probs.index_select(0, support_rows).float()
                / support_mass.index_select(0, support_rows).unsqueeze(-1)
            )
        lengths = torch.where(valid, realized_lengths.clamp(max=packed_size), 0).int()
    # All replicas must make the same request-abort decision.
    for group in sync_groups:
        if (
            dist.is_initialized()
            and group is not None
            and dist.get_world_size(group) > 1
        ):
            dist.all_reduce(statuses, op=dist.ReduceOp.MAX, group=group)
    return SamplingMaskOutput(
        token_ids=token_ids,
        lengths=lengths,
        selected_logprobs=selected_logprobs,
        statuses=statuses,
        support_logprobs=support_logprobs,
    )
