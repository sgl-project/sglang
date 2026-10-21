from __future__ import annotations

from typing import Optional

import torch


def is_mega_gate_available() -> bool:
    try:
        from deep_gemm import bf16_mega_gate as impl
    except ImportError:
        return False
    return callable(impl)


def bf16_mega_gate(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    top_k: int,
    *,
    routed_scaling_factor: float,
    ep_rank: int,
    bias: Optional[torch.Tensor] = None,
    image_bias: Optional[torch.Tensor] = None,
    image_token_mask: Optional[torch.Tensor] = None,
    valid_token_mask: Optional[torch.Tensor] = None,
    hash_topk_ids: Optional[torch.Tensor] = None,
    out: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    import deep_gemm

    if valid_token_mask is not None:
        hidden_states = torch.where(valid_token_mask[:, None], hidden_states, 0)

    if out is None:
        ids = torch.empty(
            (hidden_states.shape[0], top_k),
            dtype=torch.int64,
            device=hidden_states.device,
        )
        weights = torch.empty_like(ids, dtype=torch.float32)
    else:
        ids, weights = out
    deep_gemm.bf16_mega_gate(
        hidden_states,
        weight,
        top_k,
        use_shared_as_routed=False,
        num_shared_experts=0,
        routed_scaling_factor=routed_scaling_factor,
        ep_rank=ep_rank,
        scoring_func="sqrtsoftplus",
        mask=valid_token_mask,
        bias=bias,
        image_bias=image_bias,
        image_token_mask=image_token_mask,
        fix_routing_mask=(
            torch.ones_like(ids[:, 0], dtype=torch.bool)
            if hash_topk_ids is not None
            else None
        ),
        unmapped_topk_idx=hash_topk_ids,
        out=(ids, weights),
    )
    return weights, ids
