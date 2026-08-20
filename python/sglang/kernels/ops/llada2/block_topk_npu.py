"""LLaDA2 block routing for Ascend NPU."""

import torch
import torch.nn.functional as F


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

    # A stable descending sort returns the lowest expert id first among equal
    # scores, which is exactly the required tie-break, and it does so without
    # perturbing close but unequal scores.
    capacity_order = block_expert_scores.argsort(dim=-1, descending=True, stable=True)
    capacity_ids = capacity_order[:, :expert_capacity].sort(dim=-1).values

    capacity_ids_per_token = capacity_ids.unsqueeze(1).expand(-1, block_size, -1)
    capacity_scores = routing_scores_blocked.gather(2, capacity_ids_per_token)
    flat_capacity_scores = capacity_scores.view(padded_num_tokens, expert_capacity)
    local_ids = flat_capacity_scores.argsort(dim=-1, descending=True, stable=True)[
        :, :top_k
    ].view(num_blocks, block_size, top_k)
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
