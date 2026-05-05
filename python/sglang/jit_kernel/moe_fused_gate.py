from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_moe_fused_gate_module() -> Module:
    return load_jit(
        "moe_fused_gate",
        cuda_files=["moe/moe_fused_gate.cuh"],
        cuda_wrappers=[
            ("moe_fused_gate", "moe_fused_gate_detail::MoeFusedGateKernel::run")
        ],
    )


def moe_fused_gate(
    input: torch.Tensor,
    bias: torch.Tensor,
    num_expert_group: int,
    topk_group: int,
    topk: int,
    num_fused_shared_experts: int = 0,
    routed_scaling_factor: float = 1.0,
    apply_routed_scaling_factor_on_output: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused MoE gate: sigmoid + bias + group-filter + topk + renorm.

    Args:
        input: Gating logits of shape ``[num_tokens, num_experts]``.
               Supported dtypes: ``float32``, ``float16``, ``bfloat16``.
        bias:  Correction bias of shape ``[num_experts]``, same dtype as ``input``.
        num_expert_group: Number of expert groups.
               ``num_experts`` must be divisible by this value and
               ``num_experts / num_expert_group`` must be <= 32.
        topk_group: Number of expert groups to keep per token.
        topk: Total number of experts selected per token (including any
              fused shared experts).
        num_fused_shared_experts: Number of shared-expert slots to append.
               When > 0, the last ``num_fused_shared_experts`` columns of the
               output indices are set to ``num_experts``, ``num_experts+1``, …
               and their weights are ``sum(routed_weights) / routed_scaling_factor``.
        routed_scaling_factor: Scaling factor applied to weights (or used to
               compute shared-expert weight). Default is ``1.0``.
        apply_routed_scaling_factor_on_output: If ``True``, multiply all
               selected weights by ``routed_scaling_factor`` after normalisation.

    Returns:
        A tuple ``(topk_weights, topk_ids)`` where:

        * ``topk_weights`` – ``float32`` tensor of shape ``[num_tokens, topk]``
          with the normalised expert weights.
        * ``topk_ids`` – ``int32`` tensor of shape ``[num_tokens, topk]``
          with the selected expert indices.

    Raises:
        RuntimeError: If ``num_experts`` is not a power of two, or
                      ``num_experts / num_expert_group > 32``.
    """
    num_tokens, num_experts = input.shape

    topk_weights = torch.empty(
        (num_tokens, topk), dtype=torch.float32, device=input.device
    )
    topk_ids = torch.empty((num_tokens, topk), dtype=torch.int32, device=input.device)

    module = _jit_moe_fused_gate_module()
    module.moe_fused_gate(
        input,
        bias,
        topk_weights,
        topk_ids,
        num_expert_group,
        topk_group,
        topk,
        num_fused_shared_experts,
        routed_scaling_factor,
        apply_routed_scaling_factor_on_output,
    )

    return topk_weights, topk_ids
