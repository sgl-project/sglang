"""Helion post-expert unpermute, RMSNorm, and weighted-sum fusion for Muse Spark v1.2."""

from __future__ import annotations

import helion
import helion.language as hl
import torch

POST_EXPERT_FINALIZE_CONFIGS: dict[str, helion.Config] = {
    "short": helion.Config(
        atomic_indexing=[],
        block_sizes=[],
        indexing=[
            "pointer",
            "tensor_descriptor",
            "tensor_descriptor",
            "pointer",
            "pointer",
        ],
        load_eviction_policies=["", "first", "", ""],
        num_stages=7,
        num_warps=16,
        pid_type="flat",
        range_flattens=[None, None],
        range_multi_buffers=[None, None],
        range_num_stages=[],
        range_unroll_factors=[0, 0],
        static_ranges=[True],
    ),
    "long": helion.Config(
        block_sizes=[],
        num_stages=7,
        num_warps=4,
        pid_type="flat",
        range_flattens=[None, None],
        range_multi_buffers=[None, None],
        range_unroll_factors=[0, 0],
        static_ranges=[True],
    ),
}


def _post_expert_specialization_key(
    gemm2_output: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    output: torch.Tensor,
    *_: object,
) -> int:
    num_tokens = int(output.shape[0])
    if num_tokens <= 0:
        return 0
    return 1 if num_tokens == 1 else 2


@helion.kernel(
    key=_post_expert_specialization_key,
    static_shapes=False,
)
def _post_expert_finalize_helion_kernel(
    gemm2_output: torch.Tensor,
    expert_weights: torch.Tensor,
    expanded_idx_to_permuted_idx: torch.Tensor,
    rms_norm_weight: torch.Tensor,
    output: torch.Tensor,
    topk: hl.constexpr,
    hidden_dim: hl.constexpr,
    eps: hl.constexpr,
    weight_first: hl.constexpr,
) -> None:
    """Write the in-place TRT-LLM MoE finalize output."""
    num_tokens, _ = output.size()
    hl.specialize(gemm2_output.stride())
    hl.specialize(expert_weights.stride())
    hl.specialize(expanded_idx_to_permuted_idx.stride())
    hl.specialize(rms_norm_weight.stride())
    hl.specialize(output.stride())
    index_map = expanded_idx_to_permuted_idx.view(num_tokens, topk)
    weights = expert_weights.view(num_tokens, topk)

    for token_idx in hl.grid(num_tokens):
        # Match the CuTeDSL decode path's programmatic dependent launch. The
        # caller enables PDL only for grids small enough to avoid producer
        # starvation; on larger prefill grids these instructions are inert.
        hl.inline_triton("tl.extra.cuda.gdc_wait()", args=(), output_like=None)
        accumulated = torch.zeros(
            (hidden_dim,), dtype=torch.bfloat16, device=output.device
        )
        for expert_slot in range(topk):
            permuted_idx = index_map[token_idx, expert_slot]
            if permuted_idx >= 0:
                values = gemm2_output[permuted_idx, :hidden_dim].to(torch.float32)
                norm_weight = rms_norm_weight[:].to(torch.float32)
                if weight_first:
                    values = values * norm_weight
                    variance = torch.mean(values * values, dim=-1)
                    normalized = values * torch.rsqrt(variance + eps)
                else:
                    variance = torch.mean(values * values, dim=-1)
                    normalized = values * torch.rsqrt(variance + eps) * norm_weight
                route_weight = weights[token_idx, expert_slot].to(torch.float32)
                accumulated = (
                    accumulated.to(torch.float32) + normalized * route_weight
                ).to(torch.bfloat16)
        output[token_idx, :] = accumulated
        hl.inline_triton(
            "tl.extra.cuda.gdc_launch_dependents()", args=(), output_like=None
        )


__all__ = [
    "POST_EXPERT_FINALIZE_CONFIGS",
    "_post_expert_finalize_helion_kernel",
]
