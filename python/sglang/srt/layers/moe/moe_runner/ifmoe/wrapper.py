"""IFMoe fused function for sglang — PyTorch binding."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

from .quant_info import IFMoeQuantInfo

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        StandardCombineInput,
        StandardDispatchOutput,
    )

logger = logging.getLogger(__name__)

HIDDEN_SIZE = 7168
NUM_EXPERTS_GLOBAL = 256


# Module-level caches to avoid per-call allocation.
_dummy_scale_cache = {}
_zero_bias_cache = {}


def _get_dummy_scale(device):
    key = device.index if hasattr(device, "index") else device
    t = _dummy_scale_cache.get(key)
    if t is None:
        t = torch.empty(0, dtype=torch.float32, device=device)
        _dummy_scale_cache[key] = t
    return t


def _get_zero_bias(device):
    key = device.index if hasattr(device, "index") else device
    t = _zero_bias_cache.get(key)
    if t is None:
        t = torch.zeros(NUM_EXPERTS_GLOBAL, dtype=torch.bfloat16, device=device)
        _zero_bias_cache[key] = t
    return t


def fused_experts_ifmoe(
    dispatch_output: "StandardDispatchOutput",
    quant_info: IFMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> "StandardCombineInput":
    """Bridge sglang's MoE dispatch to the IFMoe CUDA kernel."""
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    hidden_states = dispatch_output.hidden_states
    topk_output = dispatch_output.topk_output
    T = hidden_states.shape[0]

    if T == 0:
        return StandardCombineInput(
            hidden_states=torch.empty(
                0, HIDDEN_SIZE, dtype=hidden_states.dtype, device=hidden_states.device
            ),
        )

    router_logits = topk_output.router_logits

    dummy_scale = _get_dummy_scale(hidden_states.device)
    routing_bias = quant_info.routing_bias
    if routing_bias is None:
        routing_bias = _get_zero_bias(hidden_states.device)

    # The kernel quantizes BF16 hidden states internally. Keep top-k tensors in the
    # dtype/layout expected by the C++ binding without forcing copies on the fast path.
    topk_ids = topk_output.topk_ids
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    if not topk_ids.is_contiguous():
        topk_ids = topk_ids.contiguous()

    topk_weights = topk_output.topk_weights
    if topk_weights.dtype != torch.float32:
        topk_weights = topk_weights.to(torch.float32)
    if not topk_weights.is_contiguous():
        topk_weights = topk_weights.contiguous()

    from .binding_torch import kernel

    # routed_scaling_factor must be applied inside the kernel for sglang versions where
    # the TopK layer does NOT apply it externally (sglang 0.1.dev8373 with is_ifmoe()).
    # The kernel's ext_routing_remap multiplies ext_topk_weights by rsf inside.
    rsf = float(getattr(runner_config, "routed_scaling_factor", None) or 1.0)
    output = kernel(
        router_logits if router_logits.is_contiguous() else router_logits.contiguous(),
        routing_bias if routing_bias.is_contiguous() else routing_bias.contiguous(),
        hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous(),
        dummy_scale,
        quant_info.w13_weight,
        quant_info.w13_weight_scale,
        quant_info.w2_weight,
        quant_info.w2_weight_scale,
        int(quant_info.local_expert_offset),
        rsf,
        topk_ids,
        topk_weights,
    )

    if output.dtype != hidden_states.dtype:
        output = output.to(hidden_states.dtype)
    return StandardCombineInput(hidden_states=output)
