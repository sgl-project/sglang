"""Canonical gfx950 MXFP4 decode execution for SharedEP."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

import torch

from sglang.srt.layers.moe.moe_runner.base import MoeRunnerConfig
from sglang.srt.layers.moe.moe_runner.shared_ep import (
    SharedEpQuantCapability,
    SharedEpQuantInfo,
    SharedEpQuantization,
)
from sglang.srt.layers.moe.shared_ep.kernels import (
    prepare_routes,
    reduce_owner_output,
)
from sglang.srt.layers.moe.shared_ep.layout import SharedEpInputViews
from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

if TYPE_CHECKING:
    from sglang.srt.layers.moe.shared_ep.backend import SharedEpDispatchOutput


def publish_bf16_owner_input(
    target: SharedEpInputViews,
    *,
    source: torch.Tensor,
    source_ids: torch.Tensor,
    source_weights: torch.Tensor,
) -> None:
    """Publish BF16 activations and canonical Top-K metadata into owner storage."""

    if source.ndim != 2 or source.dtype != torch.bfloat16:
        raise TypeError("SharedEP MXFP4 owner activations must be 2D BF16")
    if not source.is_contiguous():
        raise ValueError("SharedEP MXFP4 owner activations must be contiguous")
    if target.activations.ndim != 2 or target.activations.dtype != torch.bfloat16:
        raise TypeError("SharedEP MXFP4 target must expose directly addressable BF16")
    if target.scales is not None:
        raise ValueError("SharedEP MXFP4 BF16 owner storage must not expose FP8 scales")
    if source_ids.ndim != 2 or source_weights.ndim != 2:
        raise ValueError("SharedEP MXFP4 routes must be two-dimensional")
    if source_ids.shape != source_weights.shape:
        raise ValueError("SharedEP MXFP4 route id and weight shapes must match")
    if source_ids.dtype not in (torch.int32, torch.int64):
        raise TypeError("SharedEP MXFP4 route ids must use int32 or int64")
    if source_weights.dtype != torch.float32:
        raise TypeError("SharedEP MXFP4 route weights must use float32")

    num_tokens, hidden_size = source.shape
    if not 0 <= num_tokens <= target.activations.shape[0]:
        raise ValueError(
            f"SharedEP MXFP4 input has {num_tokens} tokens, "
            f"capacity is {target.activations.shape[0]}"
        )
    if hidden_size != target.activations.shape[1]:
        raise ValueError("SharedEP MXFP4 activation hidden size does not match storage")
    if source_ids.shape != (num_tokens, target.topk_ids.shape[1]):
        raise ValueError("SharedEP MXFP4 route shape does not match owner storage")
    tensors = (
        source,
        source_ids,
        source_weights,
        target.activations,
        target.topk_ids,
        target.topk_weights,
    )
    if any(tensor.device != source.device for tensor in tensors[1:]):
        raise ValueError("SharedEP MXFP4 owner tensors must use one device")

    target.topk_ids.fill_(-1)
    target.topk_weights.zero_()
    if num_tokens:
        target.activations[:num_tokens].copy_(source)
        target.topk_ids[:num_tokens].copy_(source_ids)
        target.topk_weights[:num_tokens].copy_(source_weights)


def _valid_fp4_dtype(dtype: torch.dtype) -> bool:
    fp4_dtype = getattr(torch, "float4_e2m1fn_x2", None)
    return dtype == torch.uint8 or (fp4_dtype is not None and dtype == fp4_dtype)


def _valid_e8m0_dtype(dtype: torch.dtype) -> bool:
    e8m0_dtype = getattr(torch, "float8_e8m0fnu", None)
    return dtype == torch.uint8 or (e8m0_dtype is not None and dtype == e8m0_dtype)


def validate_shared_ep_mxfp4_weights(
    dispatch_output: SharedEpDispatchOutput,
    quant_info: SharedEpQuantInfo,
    runner_config: MoeRunnerConfig,
) -> None:
    """Validate SGLang's profile and canonical tensor metadata before AITER."""

    profile = dispatch_output.profile
    if profile.quantization is not SharedEpQuantization.MXFP4:
        raise TypeError("SharedEP MXFP4 execution requires an MXFP4 profile")
    if not isinstance(quant_info, SharedEpQuantInfo):
        raise TypeError("SharedEP MXFP4 requires runner-neutral quant metadata")
    quant_info.require_decode_capability(SharedEpQuantCapability.CANONICAL_MXFP4)
    if tuple(quant_info.block_shape) != (1, 32):
        raise ValueError(
            f"SharedEP MXFP4 requires canonical (1, 32) groups, "
            f"got {quant_info.block_shape}"
        )
    if quant_info.w13_scale is None or quant_info.w2_scale is None:
        raise ValueError("SharedEP MXFP4 requires W13 and W2 E8M0 scales")

    expected_shapes = (
        (
            profile.num_local_experts,
            profile.intermediate_size * 2,
            profile.hidden_size // 2,
        ),
        (
            profile.num_local_experts,
            profile.hidden_size,
            profile.intermediate_size // 2,
        ),
        (
            profile.num_local_experts,
            profile.intermediate_size * 2,
            profile.hidden_size // 32,
        ),
        (
            profile.num_local_experts,
            profile.hidden_size,
            profile.intermediate_size // 32,
        ),
    )
    tensors = (
        quant_info.w13_weight,
        quant_info.w2_weight,
        quant_info.w13_scale,
        quant_info.w2_scale,
    )
    observed_shapes = tuple(tuple(tensor.shape) for tensor in tensors)
    if observed_shapes != expected_shapes:
        raise ValueError(
            "SharedEP canonical MXFP4 tensor shapes do not match the profile: "
            f"observed={observed_shapes}, expected={expected_shapes}"
        )
    for name, weight in (
        ("w13_weight", quant_info.w13_weight),
        ("w2_weight", quant_info.w2_weight),
    ):
        if not _valid_fp4_dtype(weight.dtype):
            raise TypeError(f"{name} must contain packed OCP E2M1 values")
        if not weight.is_contiguous():
            raise ValueError(f"{name} must use canonical contiguous storage")
    for name, scale in (
        ("w13_scale", quant_info.w13_scale),
        ("w2_scale", quant_info.w2_scale),
    ):
        if not _valid_e8m0_dtype(scale.dtype):
            raise TypeError(f"{name} must contain canonical E8M0 bytes")
        if not scale.is_contiguous():
            raise ValueError(f"{name} must use canonical contiguous storage")

    if runner_config.activation != "silu" or not runner_config.is_gated:
        raise ValueError("SharedEP MXFP4 supports gated SiLU experts only")
    if runner_config.gemm1_alpha is not None:
        raise ValueError("SharedEP MXFP4 does not support gemm1_alpha")
    if runner_config.gemm1_clamp_limit is not None:
        raise ValueError("SharedEP MXFP4 does not support gemm1_clamp_limit")


def _load_aiter_mxfp4_ops() -> tuple[Callable, Callable]:
    from aiter.ops.triton.moe.shared_ep_mxfp4 import (
        shared_ep_mxfp4_w2,
        shared_ep_mxfp4_w13,
    )

    return shared_ep_mxfp4_w13, shared_ep_mxfp4_w2


def run_shared_ep_mxfp4(
    dispatch_output: SharedEpDispatchOutput,
    quant_info: SharedEpQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    """Run canonical W13/W2 and write W2 into global owner route slots."""

    validate_shared_ep_mxfp4_weights(dispatch_output, quant_info, runner_config)
    profile = dispatch_output.profile
    state = dispatch_output.state
    routes = prepare_routes(
        profile,
        state.global_input.topk_ids,
        state.global_input.topk_weights,
        state.input_epoch.allocation.local_storage,
        state.input_epoch.epoch,
        local_expert_start=dispatch_output.local_expert_start,
    )

    owner_activations = dispatch_output.hidden_states
    owner_rows = owner_activations.numel() // profile.hidden_size
    route_capacity = owner_rows * profile.top_k
    route_weights = routes.local_weights.view(owner_rows, profile.top_k)
    intermediate = torch.empty(
        (route_capacity, profile.intermediate_size),
        dtype=torch.bfloat16,
        device=owner_activations.device,
    )
    global_output = state.global_output.view(route_capacity, profile.hidden_size)
    if not global_output.is_contiguous():
        raise RuntimeError(
            "SharedEP MXFP4 direct output requires a dense rank-major HIP VMM "
            "mapping with no rank or route padding"
        )

    shared_ep_mxfp4_w13, shared_ep_mxfp4_w2 = _load_aiter_mxfp4_ops()
    intermediate = shared_ep_mxfp4_w13(
        owner_activations,
        quant_info.w13_weight,
        quant_info.w13_scale,
        routes.sorted_token_ids,
        routes.expert_ids,
        routes.num_tokens_post_padded,
        top_k=profile.top_k,
        route_block_size=profile.block_size_m,
        out=intermediate,
        weight_layout=quant_info.weight_layout.value,
        scale_layout=quant_info.scale_layout.value,
        config=profile.w13_kernel_config(dispatch_output.num_tokens),
        swiglu_limit=runner_config.swiglu_limit,
        check_route_values=False,
    )
    direct_output = shared_ep_mxfp4_w2(
        intermediate,
        quant_info.w2_weight,
        quant_info.w2_scale,
        route_weights,
        routes.sorted_token_ids,
        routes.expert_ids,
        routes.num_tokens_post_padded,
        top_k=profile.top_k,
        route_block_size=profile.block_size_m,
        out=global_output,
        weight_layout=quant_info.weight_layout.value,
        scale_layout=quant_info.scale_layout.value,
        config=profile.w2_kernel_config(dispatch_output.num_tokens),
        check_route_values=False,
    )
    if direct_output.data_ptr() != global_output.data_ptr():
        raise RuntimeError("AITER MXFP4 W2 did not preserve the direct output buffer")

    state.output_epoch.publish()
    state.output_epoch.wait_all()
    output = reduce_owner_output(
        state.local_output,
        num_tokens=dispatch_output.num_tokens,
    )
    return StandardCombineInput(hidden_states=output)
