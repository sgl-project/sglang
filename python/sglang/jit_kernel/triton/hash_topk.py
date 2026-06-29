"""HIP fallback for ``hash_topk``: ``csrc/deepseek_v4/hash_topk.cuh`` uses
CUDA-only primitives, so on ROCm we dispatch to this Triton implementation.
"""

from __future__ import annotations

from typing import Tuple

import torch
import triton
import triton.language as tl


@triton.jit
def _hash_topk_triton_kernel(
    router_logits_ptr,
    input_ids_ptr,
    tid2eid_ptr,
    topk_weights_ptr,
    topk_ids_ptr,
    num_routed_experts: tl.constexpr,
    topk_routed: tl.constexpr,
    topk_fused: tl.constexpr,
    routed_scaling_factor,
    BLOCK_K: tl.constexpr,
):
    token_pos = tl.program_id(0)
    token_id = tl.load(input_ids_ptr + token_pos).to(tl.int64)

    k_off = tl.arange(0, BLOCK_K)
    routed_mask = k_off < topk_routed
    fused_mask = k_off < topk_fused
    is_shared = k_off >= topk_routed

    expert_id = tl.load(
        tid2eid_ptr + token_id * topk_routed + k_off,
        mask=routed_mask,
        other=0,
    ).to(tl.int32)
    logit = tl.load(
        router_logits_ptr + token_pos * num_routed_experts + expert_id,
        mask=routed_mask,
        other=0.0,
    ).to(tl.float32)

    softplus = tl.maximum(logit, 0.0) + tl.log(1.0 + tl.exp(-tl.abs(logit)))
    weight = tl.sqrt(softplus)
    weight = tl.where(routed_mask, weight, 0.0)
    routed_sum = tl.sum(weight, axis=0)

    shared_weight = 1.0 / routed_scaling_factor
    final_weight = tl.where(is_shared, shared_weight, weight / routed_sum)
    shared_id = num_routed_experts + (k_off - topk_routed)
    final_id = tl.where(is_shared, shared_id, expert_id).to(tl.int32)

    out_off = token_pos * topk_fused + k_off
    tl.store(topk_weights_ptr + out_off, final_weight, mask=fused_mask)
    tl.store(topk_ids_ptr + out_off, final_id, mask=fused_mask)


def hash_topk_triton(
    router_logits: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    num_fused_shared_experts: int,
    routed_scaling_factor: float,
    scoring_func: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    assert scoring_func == "sqrtsoftplus"

    num_tokens = router_logits.size(0)
    num_routed_experts = router_logits.size(1)
    topk_routed = tid2eid.size(1)
    topk_fused = topk_routed + num_fused_shared_experts

    topk_weights = torch.empty(
        (num_tokens, topk_fused), dtype=torch.float32, device=router_logits.device
    )
    topk_ids = torch.empty(
        (num_tokens, topk_fused), dtype=torch.int32, device=router_logits.device
    )
    if num_tokens == 0:
        return topk_weights, topk_ids

    block_k = max(triton.next_power_of_2(topk_fused), 1)
    _hash_topk_triton_kernel[(num_tokens,)](
        router_logits,
        input_ids,
        tid2eid,
        topk_weights,
        topk_ids,
        num_routed_experts=num_routed_experts,
        topk_routed=topk_routed,
        topk_fused=topk_fused,
        routed_scaling_factor=float(routed_scaling_factor),
        BLOCK_K=block_k,
        num_warps=1,
    )
    return topk_weights, topk_ids
