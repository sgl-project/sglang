"""Qwen routing with a fixed FP32 normalization and BF16 weight boundary, and the Qwen3.5 MoE shared-expert gate."""

import torch

from sglang.srt.layers.moe.topk import (
    StandardTopKOutput,
    TopK,
    capture_routed_experts_if_allowed,
)
from sglang.srt.true_on_policy.qwen_route import route_qwen_logits


class TopQwenRouter(TopK):
    def forward(
        self,
        hidden_states: torch.Tensor,
        router_logits: torch.Tensor,
        *,
        num_token_non_padded=None,
        expert_location_dispatch_info=None,
    ):
        """The fused TOP routing: FP32 softmax, top-k by probability then expert id, FP32 renormalize, one cast."""
        if expert_location_dispatch_info is not None:
            raise NotImplementedError(
                "[top] expert-location dispatch (EPLB) is not in the declared program"
            )
        # Callers omit the padded-row bound; TopK reads it from the forward context. The placement
        # info it also resolves is not read: this router dispatches by logical expert id.
        num_token_non_padded, _ = self._resolve_routing_inputs(
            num_token_non_padded, None
        )
        routing_weights, selected_experts = route_qwen_logits(
            router_logits, self.topk_config.top_k, hidden_states.dtype
        )
        if num_token_non_padded is not None:
            from sglang.srt.layers.moe.topk import _mask_topk_ids_padded_region

            _mask_topk_ids_padded_region(selected_experts, num_token_non_padded)
        # Slot order with pads already -1: the trainer's rollout routing replay reads these rows back.
        capture_routed_experts_if_allowed(
            self.topk_config, self.layer_id, selected_experts
        )
        return StandardTopKOutput(
            topk_weights=routing_weights,
            topk_ids=selected_experts,
            router_logits=router_logits,
        )


def top_qwen_router(config, quant_config):
    """TopQwenRouter for Qwen2MoeSparseMoeBlock under TOP after admission, else None."""
    from sglang.srt.true_on_policy.config import is_true_on_policy_enabled

    if not is_true_on_policy_enabled():
        return None
    from sglang.srt.true_on_policy.moe_ep import validate_qwen_moe

    validate_qwen_moe(config, quant_config, block="Qwen2MoeSparseMoeBlock")
    return TopQwenRouter


# Hidden sizes the shared gate is qualified on: Qwen3.5/3.6-35B-A3B (2048), Qwen3.8-Flash-Next (2560).
# Admission reads this (moe_ep._qualified_experts).
SHARED_GATE_HIDDEN_SIZES = (2048, 2560)

# Narrow tiles for the one-column gate: the default 128x128x64 tile wastes 127 of 128 columns and serializes the
# 2,048-deep K loop on one program. Tuned on H200 under a CUDA graph at 32-2,048 rows; re-tune for a new hidden
# size. Still batch invariant (fixed config).
_SHARED_GATE_CONFIG = dict(
    BLOCK_SIZE_M=32,
    BLOCK_SIZE_N=16,
    BLOCK_SIZE_K=128,
    GROUP_SIZE_M=8,
    num_stages=3,
    num_warps=4,
)


def shared_gate_linear(x, weight):
    """Compute the Qwen scalar shared-expert gate on the batch-invariant persistent matmul, narrow tiles.

    The generic BI dispatcher requires DeepGEMM-compatible widths, so this
    model-owned operation is explicit on both serving and trainer paths.
    """
    from sglang.srt.batch_invariant_ops.batch_invariant_ops import (
        _matmul_persistent_triton,
    )

    if (
        x.ndim != 2
        or x.shape[1] not in SHARED_GATE_HIDDEN_SIZES
        or weight.shape != (1, x.shape[1])
        or x.dtype != torch.bfloat16
        or weight.dtype != torch.bfloat16
        or not x.is_cuda
        or weight.device != x.device
    ):
        raise ValueError(
            f"TOP Qwen shared gate requires CUDA BF16 [M,H] and [1,H], H in {SHARED_GATE_HIDDEN_SIZES}"
        )
    if x.shape[0] == 0:
        return x.new_empty((0, 1))
    return _matmul_persistent_triton(
        x.contiguous(), weight.t(), out_dtype=torch.bfloat16, config=_SHARED_GATE_CONFIG
    )


def gated_shared_output(hidden_states, gate_weight, shared_output):
    """Qwen3.5 shared expert scaled by its sigmoid gate, on the TOP gate GEMM."""
    return shared_gate_linear(hidden_states, gate_weight).sigmoid() * shared_output
