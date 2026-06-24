"""Fail-closed dispatcher for the registered Artemis fused-MoE shapes."""

from __future__ import annotations

import os
import sys
from typing import TYPE_CHECKING

import torch
from sglang.srt.environ import envs

from .registry import FusedMoeKernelSpec, find_kernel_spec

if TYPE_CHECKING:
    from sglang.srt.layers.moe.moe_runner.aiter import (
        AiterMoeQuantInfo,
        AiterRunnerInput,
    )
    from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig

_MARKER = "[artemis-dsr1-fp8-fused-moe]"
_SEEN_DISPATCH: set[str] = set()


def is_artemis_fused_moe_enabled() -> bool:
    """Return whether the opt-in Artemis kernel dispatcher is enabled."""
    return envs.ARTEMIS_KERNELS.get()


def _device_arch(tensor: torch.Tensor) -> str:
    index = tensor.device.index
    if index is None:
        index = torch.cuda.current_device()
    return str(torch.cuda.get_device_properties(index).gcnArchName).split(":", 1)[0]


def _shape_spec(
    runner_input: AiterRunnerInput,
    runner_config: MoeRunnerConfig,
) -> FusedMoeKernelSpec | None:
    hidden = runner_input.hidden_states
    topk_ids = runner_input.topk_ids
    topk_weights = runner_input.topk_weights
    if hidden.ndim != 2 or topk_ids.ndim != 2 or topk_weights.ndim != 2:
        return None
    return find_kernel_spec(
        tokens=hidden.shape[0],
        hidden_size=runner_config.hidden_size,
        intermediate_size_per_partition=runner_config.intermediate_size_per_partition,
        experts=runner_config.num_local_experts,
        top_k=runner_config.top_k,
    )


def _contract_mismatch(
    spec: FusedMoeKernelSpec,
    runner_input: AiterRunnerInput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> str | None:
    hidden = runner_input.hidden_states
    topk_ids = runner_input.topk_ids
    topk_weights = runner_input.topk_weights
    w1 = quant_info.w13_weight
    w2 = quant_info.w2_weight
    w1_scale = quant_info.w13_scale
    w2_scale = quant_info.w2_scale

    if hidden.shape != spec.hidden_shape or hidden.dtype is not torch.bfloat16:
        return "hidden"
    if w1.shape != spec.w13_shape or w1.dtype is not torch.float8_e4m3fn:
        return "w1"
    if w2.shape != spec.w2_shape or w2.dtype is not torch.float8_e4m3fn:
        return "w2"
    if w1_scale is None or w1_scale.dtype is not torch.float32:
        return "w1_scale_dtype"
    if w2_scale is None or w2_scale.dtype is not torch.float32:
        return "w2_scale_dtype"
    if w1_scale.shape != spec.w13_scale_shape or not w1_scale.is_contiguous():
        return "w1_scale_layout"
    if w2_scale.shape != spec.w2_scale_shape or not w2_scale.is_contiguous():
        return "w2_scale_layout"
    if topk_ids.shape != spec.topk_shape or topk_ids.dtype is not torch.int32:
        return "topk_ids"
    if topk_weights.shape != spec.topk_shape or topk_weights.dtype is not torch.float32:
        return "topk_weights"
    if getattr(quant_info.quant_type, "value", None) != "per_128x128":
        return "quant_type"
    if quant_info.expert_mask is not None:
        return "expert_mask"
    if quant_info.a13_scale is not None or quant_info.a2_scale is not None:
        return "prequantized_activation"
    if quant_info.b13 is not None or quant_info.b2 is not None:
        return "bias"
    if quant_info.doweight_stage1 or quant_info.hidden_pad or quant_info.intermediate_pad:
        return "quant_flags"
    if (
        runner_config.num_experts != spec.experts
        or runner_config.num_local_experts != spec.experts
        or runner_config.num_fused_shared_experts != 1
        or runner_config.params_dtype is not torch.bfloat16
        or runner_config.activation != "silu"
        or not runner_config.is_gated
    ):
        return "runner_config"
    if runner_config.apply_router_weight_on_input or runner_config.no_combine:
        return "runner_flags"
    tensors = (hidden, w1, w2, w1_scale, w2_scale, topk_ids, topk_weights)
    if any(tensor.device.type != "cuda" or not tensor.is_contiguous() for tensor in tensors):
        return "layout"
    if len({tensor.device for tensor in tensors}) != 1:
        return "device"
    if _device_arch(hidden) != spec.architecture:
        return "arch"
    return None


def _allocate_preparation(
    hidden: torch.Tensor,
    block_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.empty_like(hidden, dtype=torch.float8_e4m3fn),
        torch.empty(
            (hidden.shape[0], hidden.shape[1] // block_size),
            dtype=torch.float32,
            device=hidden.device,
        ),
        torch.empty_like(hidden),
    )


def launch_fused_moe(
    spec: FusedMoeKernelSpec,
    output: torch.Tensor,
    hidden: torch.Tensor,
    quantized: torch.Tensor,
    scales: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
    w1_scale: torch.Tensor,
    w2_scale: torch.Tensor,
) -> torch.Tensor:
    """Compose the preparation, W1, and W2 stages selected by the registry."""
    from sglang.srt.artemis_kernels.compat import ensure_triton_compat

    from .kernels.moe_workspaces import get_route_workspaces

    ensure_triton_compat()
    if spec.tokens != 32:  # The registry is closed; this protects future edits.
        raise ValueError(f"unregistered fused-MoE specialization: {spec.name}")
    from .kernels.moe_prep_gluon_m32 import moe_prep_gluon_m32 as prep
    from .kernels.moe_w1_gluon_m32 import moe_w1_gluon_m32 as w1_stage
    from .kernels.moe_w2_gluon_m32 import moe_w2_gluon_m32 as w2_stage

    route_tokens, route_weights, route_counts, route_slots, contribution = get_route_workspaces(
        hidden.device, spec.tokens
    )
    prep(
        hidden,
        quantized,
        scales,
        output,
        topk_ids,
        topk_weights,
        route_tokens,
        route_weights,
        route_counts,
        route_slots,
    )
    intermediate, intermediate_scale = w1_stage(
        quantized,
        w1,
        scales,
        w1_scale,
        route_tokens,
        route_counts,
    )
    return w2_stage(
        output,
        contribution,
        w2,
        w2_scale,
        intermediate,
        intermediate_scale,
        route_tokens,
        route_weights,
        route_counts,
        route_slots,
    )


def _emit_dispatch(spec: FusedMoeKernelSpec) -> None:
    if spec.name in _SEEN_DISPATCH:
        return
    _SEEN_DISPATCH.add(spec.name)
    print(
        f"{_MARKER} event=dispatch pid={os.getpid()} kernel={spec.name} "
        f"shape={spec.tokens}x{spec.hidden_size} experts={spec.experts} topk={spec.top_k}",
        file=sys.stderr,
        flush=True,
    )


def dispatch_fused_moe(
    runner_input: AiterRunnerInput,
    quant_info: AiterMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> torch.Tensor | None:
    """Dispatch a registered specialization, or return ``None`` for stock."""
    if not is_artemis_fused_moe_enabled():
        return None
    spec = _shape_spec(runner_input, runner_config)
    if spec is None or _contract_mismatch(spec, runner_input, quant_info, runner_config):
        return None

    hidden = runner_input.hidden_states
    topk_ids = runner_input.topk_ids
    topk_weights = runner_input.topk_weights
    quantized, scales, output = _allocate_preparation(hidden, spec.block_size)
    result = launch_fused_moe(
        spec,
        output,
        hidden,
        quantized,
        scales,
        quant_info.w13_weight,
        quant_info.w2_weight,
        topk_ids,
        topk_weights,
        quant_info.w13_scale,
        quant_info.w2_scale,
    )
    _emit_dispatch(spec)
    return result


__all__ = [
    "dispatch_fused_moe",
    "is_artemis_fused_moe_enabled",
    "launch_fused_moe",
]
