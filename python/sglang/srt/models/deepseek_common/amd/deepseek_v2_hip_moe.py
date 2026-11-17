"""ROCm glue of DeepseekV2MoE: V4.1's split-K decode router and the fused all-reduce + mHC
post of the DeepSeek-V4 decode batches."""

from __future__ import annotations

from typing import Optional, Tuple

import torch

from sglang.srt.models.deepseek_common.utils import _use_aiter

if _use_aiter:
    from sglang.kernels.ops.gemm.router_gemv_hip import (
        ROCM_GLM_ROUTER_MAX_TOKENS,
        rocm_gemv_split_k_max_tokens,
    )
    from sglang.kernels.ops.moe.rocm_router_gate import rocm_router_max_tokens
    from sglang.srt.layers.rocm_linear_utils import rocm_dsv3_router_split_k


def router_max_tokens(gate, config, is_hash_moe: bool) -> int:
    """Rows up to which the split-K router serves gate (a MoEGate), -1 when it never does:
    V4.1 and single-group sigmoid routing (GLM-5.3-Flash), whose Triton router sums the
    partials; DSv4 keeps aiter's router GEMM and gate."""
    if (
        _use_aiter
        and not gate.is_deepseek_v4
        and not is_hash_moe
        and getattr(config, "scoring_func", None) == "sigmoid"
        and getattr(config, "topk_method", None) == "noaux_tc"
        and (getattr(config, "n_group", None) or 1) == 1
        and (getattr(config, "topk_group", None) or 1) == 1
    ):
        if (
            rocm_gemv_split_k_max_tokens(
                n=config.n_routed_experts,
                k=config.hidden_size,
                weight_dtype=gate.weight.dtype,
            )
            < 0
        ):
            return -1
        return ROCM_GLM_ROUTER_MAX_TOKENS
    if not (
        _use_aiter
        and gate.is_deepseek_v4
        and getattr(config, "model_type", None) == "deepseek_v41"
        and not is_hash_moe
    ):
        return -1
    return rocm_router_max_tokens(
        num_experts=config.n_routed_experts,
        hidden_size=config.hidden_size,
        topk=config.num_experts_per_tok,
        weight_dtype=gate.weight.dtype,
    )


def batch_has_images(forward_batch) -> bool:
    """Whether a batch can carry image tokens: only extend batches with image inputs do, so
    every other batch takes the fused top-k instead of the vision one."""
    if forward_batch is None:
        return True
    return (
        forward_batch.forward_mode.is_extend() and forward_batch.contains_image_inputs()
    )


def forward_gate(
    moe, hidden_states: torch.Tensor, gemm_output_zero_allocator, *, fused_gate: bool
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """The router logits of moe (a DeepseekV2MoE) and, on the split-K decode router, the
    partials moe.topk sums into them; fused_gate=False when anything else reads them."""
    if fused_gate and _use_aiter and not moe.is_hash:
        logits_and_partials = rocm_dsv3_router_split_k(moe.gate, hidden_states)
        if logits_and_partials is not None:
            return logits_and_partials
    return moe.gate(hidden_states, gemm_output_zero_allocator), None
