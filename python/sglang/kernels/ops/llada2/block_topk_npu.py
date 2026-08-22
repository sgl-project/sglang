"""LLaDA2 block routing for Ascend NPU."""

import torch
import torch.nn.functional as F
import torch_npu


def _lower_id_ties(
    scores: torch.Tensor,
    threshold: torch.Tensor,
    k: int,
) -> torch.Tensor:
    greater = scores > threshold
    equal = scores == threshold
    num_ties = k - greater.sum(dim=-1, keepdim=True)
    tie_rank = equal.to(torch.int32).cumsum(dim=-1)
    selected = greater | (equal & (tie_rank <= num_ties))
    expert_ids = torch.arange(scores.shape[-1], device=scores.device)
    selected_ids = torch.where(selected, expert_ids, scores.shape[-1])
    return selected_ids.topk(k, dim=-1, largest=False).values


def _ordered_lower_id_ties(
    scores: torch.Tensor,
    ordered_topk_scores: torch.Tensor,
) -> torch.Tensor:
    expert_ids = torch.arange(scores.shape[-1], device=scores.device)
    selected_ids = []
    for rank in range(ordered_topk_scores.shape[-1]):
        score = ordered_topk_scores[:, rank : rank + 1]
        same_score = scores == score
        occurrence = (ordered_topk_scores[:, :rank] == score).sum(
            dim=-1, keepdim=True
        ) + 1
        tie_rank = same_score.to(torch.int32).cumsum(dim=-1)
        candidates = torch.where(
            tie_rank == occurrence,
            expert_ids,
            scores.shape[-1],
        )
        selected_ids.append(candidates.min(dim=-1).values)
    return torch.stack(selected_ids, dim=-1)


def block_topk_npu(
    router_logits: torch.Tensor,
    correction_bias: torch.Tensor,
    block_size: int,
    expert_capacity: int,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select token experts from a block-level expert capacity."""
    num_tokens, num_experts = router_logits.shape
    device = router_logits.device
    if num_tokens == 0:
        return (
            torch.empty((0, top_k), dtype=torch.float32, device=device),
            torch.empty((0, top_k), dtype=torch.int32, device=device),
        )

    base_scores = torch.sigmoid(router_logits.float())
    routing_scores = base_scores + correction_bias.float()

    num_blocks = (num_tokens + block_size - 1) // block_size
    padded_num_tokens = num_blocks * block_size
    pad_tokens = padded_num_tokens - num_tokens
    if pad_tokens:
        base_scores = F.pad(base_scores, (0, 0, 0, pad_tokens), value=0.0)
        routing_scores = F.pad(
            routing_scores,
            (0, 0, 0, pad_tokens),
            value=float("-inf"),
        )

    routing_scores_blocked = routing_scores.view(num_blocks, block_size, num_experts)
    block_expert_scores = routing_scores_blocked.max(dim=1).values

    capacity_scores = block_expert_scores.topk(expert_capacity, dim=-1).values
    capacity_threshold = capacity_scores.min(dim=-1, keepdim=True).values
    capacity_ids = _lower_id_ties(
        block_expert_scores,
        capacity_threshold,
        expert_capacity,
    )

    capacity_ids_per_token = capacity_ids.unsqueeze(1).expand(-1, block_size, -1)
    capacity_scores = routing_scores_blocked.gather(2, capacity_ids_per_token)
    flat_capacity_scores = capacity_scores.view(padded_num_tokens, expert_capacity)
    _, provisional_local_ids, _ = torch_npu.npu_moe_gating_top_k(
        flat_capacity_scores,
        top_k,
        norm_type=0,
    )
    ordered_topk_scores = flat_capacity_scores.gather(
        1, provisional_local_ids.to(torch.int64)
    )
    local_ids = _ordered_lower_id_ties(
        flat_capacity_scores,
        ordered_topk_scores,
    ).view(num_blocks, block_size, top_k)
    ids = capacity_ids_per_token.gather(2, local_ids).view(padded_num_tokens, top_k)

    selected_base_scores = base_scores.gather(1, ids)
    if top_k > 1:
        weight_sum = selected_base_scores.sum(dim=-1, keepdim=True)
        weights = torch.where(
            weight_sum > 1e-30,
            selected_base_scores / weight_sum.clamp_min(1e-30),
            torch.full_like(selected_base_scores, 1.0 / top_k),
        )
    else:
        weights = selected_base_scores

    return (
        weights[:num_tokens],
        ids[:num_tokens].to(torch.int32),
    )
