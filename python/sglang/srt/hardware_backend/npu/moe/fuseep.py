"""Ascend FuseEP fused dispatch+GEMM+combine forward path.

Follows the mega_moe shape: a free-function bypass invoked from
``FusedMoE.forward`` when ``--moe-a2a-backend ascend_fuseep`` is set, plus a
weight-postprocess helper that NPU quant_methods call from their
``process_weights_after_loading`` when the same backend is selected.
"""

from __future__ import annotations

import importlib.metadata
import inspect
import json
import logging
import math
from functools import lru_cache
from typing import TYPE_CHECKING

import torch

from sglang.srt.environ import envs
from sglang.srt.hardware_backend.npu.utils import npu_format_cast
from sglang.srt.layers.moe.token_dispatcher.deepep import DeepEPBuffer
from sglang.srt.layers.moe.utils import DeepEPMode
from sglang.srt.runtime_context import get_exec, get_parallel

if TYPE_CHECKING:
    from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
    from sglang.srt.layers.moe.topk import TopKOutput


logger = logging.getLogger(__name__)

_PARAMS_BYTES = 2  # bf16 — Ascend's Dispatch & Combine does not support fp16


def _encode_shared_scale(scale: torch.Tensor) -> torch.Tensor:
    """Per-channel float scales -> the MegaMoE scale-bit uint64 encoding."""
    return (
        scale.reshape(-1)
        .contiguous()
        .view(torch.int32)
        .to(torch.int64)
        .bitwise_and(0xFFFFFFFF)
        .view(torch.uint64)
    )


def _requantize_shared_w4(weight: torch.Tensor, scale: torch.Tensor):
    if weight.dtype != torch.int8 or weight.ndim != 2 or weight.shape[1] % 8:
        raise ValueError(
            "shared W4 requires INT8 [inputs, channels] with channels divisible by 8"
        )
    scale = scale.detach().cpu().to(torch.float32).reshape(-1)
    if (
        scale.numel() != weight.shape[1]
        or not torch.isfinite(scale).all()
        or (scale < 0).any()
    ):
        raise ValueError("shared W4 requires finite nonnegative per-channel scales")
    values = weight.detach().cpu().to(torch.float32)
    maximum = values.abs().amax(dim=0).clamp_min(1.0)
    quantized = (values * (7.0 / maximum)).round().clamp(-7, 7).to(torch.int8)
    low = quantized[:, 0::2].bitwise_and(15).to(torch.uint8)
    high = quantized[:, 1::2].bitwise_and(15).to(torch.uint8).bitwise_left_shift(4)
    packed = low.bitwise_or(high).view(torch.int8).contiguous()
    scale4 = (scale * maximum / 7.0).contiguous()
    bias = (8.0 * quantized.to(torch.float32).sum(dim=0) * scale4).contiguous()
    return packed, scale4, bias


def _route_shared_w4(
    topk_ids, topk_weights, local_experts, ep_size, ep_rank, shared_scale
):
    if local_experts <= 0 or ep_size <= 1 or not 0 <= ep_rank < ep_size:
        raise ValueError("shared W4 routing requires valid EP geometry")
    if (
        topk_ids.ndim != 2
        or topk_ids.shape != topk_weights.shape
        or topk_ids.shape[1] >= 16
    ):
        raise ValueError(
            "shared W4 routing requires matching top-k tensors with room for one expert"
        )
    if (local_experts + 1) * ep_size > 1024 or not 0.0 < shared_scale <= 1.0:
        raise ValueError(
            "shared W4 routing exceeds MegaMoE limits or has invalid scale"
        )
    remapped = torch.where(
        topk_ids >= 0,
        topk_ids + torch.div(topk_ids, local_experts, rounding_mode="floor"),
        topk_ids,
    )
    shared_ids = topk_ids.new_full(
        (topk_ids.shape[0], 1), ep_rank * (local_experts + 1) + local_experts
    )
    shared_weights = topk_weights.new_full((topk_weights.shape[0], 1), shared_scale)
    shared_weights = torch.where(
        (topk_ids >= 0).any(dim=1, keepdim=True), shared_weights, 0.0
    )
    return torch.cat((remapped, shared_ids), dim=1), torch.cat(
        (topk_weights, shared_weights), dim=1
    )


def _check_shared_geometry(hidden: int, intermediate: int) -> None:
    """The native operator's shared-geometry envelope (same as routed)."""
    if (
        not 1024 <= intermediate <= 3072
        or intermediate % 512
        or not 1024 <= hidden <= 8192
        or hidden % 512
    ):
        raise ValueError(
            f"shared geometry outside MegaMoE limits "
            f"(hidden={hidden}, intermediate={intermediate})"
        )


def prepare_fuseep_shared_w4(moe_module):
    shared = moe_module.shared_experts
    experts = moe_module.experts
    if shared is None or getattr(moe_module, "num_fused_shared_experts", 0):
        raise ValueError("shared W4 requires separate shared experts")
    if not getattr(experts, "_mega_moe_weights_processed", False):
        raise ValueError("shared W4 requires processed MegaMoE weights")
    if getattr(experts, "_fuseep_shared_w4", None) is not None:
        return
    local_experts = experts.num_local_experts
    ep_size = experts.moe_ep_size
    if experts.moe_tp_size != 1 or experts.num_experts != local_experts * ep_size:
        raise ValueError("shared W4 requires uniform EP experts and MoE TP=1")
    _check_shared_geometry(experts.hidden_size, experts.intermediate_size_per_partition)
    if local_experts >= 128 or (local_experts + 1) * ep_size > 1024:
        raise ValueError("shared W4 exceeds MegaMoE expert limit")
    if experts.moe_runner_config.top_k >= 16:
        raise ValueError("shared W4 requires a free routing slot")
    output_scale = (
        1.0
        if moe_module.routed_scaling_factor_fused_in_topk
        else 1.0 / float(moe_module.routed_scaling_factor)
    )
    if not 0.0 < output_scale <= 1.0:
        raise ValueError("shared W4 requires a supported routed output scale")
    prepared = {}
    for prefix, proj, inputs, channels in (
        (
            "w13",
            shared.gate_up_proj,
            experts.hidden_size,
            2 * experts.intermediate_size_per_partition,
        ),
        (
            "w2",
            shared.down_proj,
            experts.intermediate_size_per_partition,
            experts.hidden_size,
        ),
    ):
        weight = proj.weight.data
        if weight.dtype != torch.int8 or tuple(weight.shape) not in (
            (inputs, channels),
            (channels, inputs),
        ):
            raise ValueError(
                f"shared W4 {prefix} has incompatible weight shape or dtype"
            )
        if tuple(weight.shape) != (inputs, channels):
            weight = weight.transpose(0, 1)
        offset = getattr(proj, "weight_offset", None)
        if offset is not None and torch.count_nonzero(offset).item():
            raise ValueError(f"shared W4 {prefix} has nonzero offset")
        linear_bias = getattr(proj, "bias", None)
        if linear_bias is not None and torch.count_nonzero(linear_bias).item():
            raise ValueError(f"shared W4 {prefix} has unsupported linear bias")
        packed, scale, bias = _requantize_shared_w4(weight, proj.weight_scale)
        device = weight.device
        for suffix in ("weight", "weight_scale", "scale_bias"):
            value = getattr(experts, f"{prefix}_{suffix}")
            if not isinstance(value, list) or len(value) != local_experts:
                raise ValueError(
                    f"shared W4 requires one {prefix}_{suffix} per routed expert"
                )
        prepared[f"{prefix}_weight"] = npu_format_cast(packed.to(device)).view(
            torch.int32
        )
        prepared[f"{prefix}_weight_scale"] = _encode_shared_scale(scale.to(device))
        prepared[f"{prefix}_scale_bias"] = bias.to(device)
    for name, value in prepared.items():
        setattr(experts, name, [*getattr(experts, name), value])
    experts._fuseep_shared_w4 = dict(local_experts=local_experts, scale=output_scale)


def fuseep_dispatch_expert_count(layer: FusedMoE) -> int:
    return layer.num_experts + (
        layer.moe_ep_size if getattr(layer, "_fuseep_shared_w4", None) else 0
    )


def _get_fuseep_buffer(layer: FusedMoE):
    DeepEPBuffer.set_dispatch_mode_as_low_latency()
    return DeepEPBuffer.get_deepep_buffer(
        get_parallel().moe_ep_group.device_group,
        layer.hidden_size,
        _PARAMS_BYTES,
        DeepEPMode.LOW_LATENCY,
        envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get(),
        fuseep_dispatch_expert_count(layer),
    )


def forward_fuseep(
    layer: FusedMoE,
    hidden_states: torch.Tensor,
    topk_output: TopKOutput,
) -> torch.Tensor:
    mode = get_exec().moe.fuseep_mode
    if mode == 3:
        capacity = _mega_moe_capacity()
        _log_mega_moe_geometry(
            hidden_states.shape[0],
            hidden_states.shape[1],
            capacity,
            layer.moe_ep_size,
            layer.moe_tp_size,
        )
        if hidden_states.dtype != torch.bfloat16:
            raise ValueError("FuseEP mode 3 requires BF16 hidden states")
        if hidden_states.shape[0] > capacity:
            raise ValueError(
                f"FuseEP mode 3 input has {hidden_states.shape[0]} tokens, exceeding "
                f"SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK={capacity}"
            )
        if not getattr(layer, "_mega_moe_weights_processed", False):
            raise RuntimeError("FuseEP mode 3 expert weights have not been prepared")
        topk_ids = topk_output.topk_ids.to(torch.int32)
        topk_weights = topk_output.topk_weights.to(torch.float32)
        shared = getattr(layer, "_fuseep_shared_w4", None)
        if shared is not None:
            topk_ids, topk_weights = _route_shared_w4(
                topk_ids,
                topk_weights,
                shared["local_experts"],
                layer.moe_ep_size,
                layer.moe_ep_rank,
                shared["scale"],
            )
        buffer = _get_fuseep_buffer(layer)
        hidden_states, _ = buffer.fused_deep_moe(
            x=hidden_states,
            topk_idx=topk_ids,
            topk_weights=topk_weights,
            gmm1_permuted_weight=layer.w13_weight,
            gmm1_permuted_weight_scale=getattr(layer, "w13_weight_scale", None),
            gmm2_weight=layer.w2_weight,
            gmm2_weight_scale=getattr(layer, "w2_weight_scale", None),
            num_max_dispatch_tokens_per_rank=capacity,
            num_experts=fuseep_dispatch_expert_count(layer),
            fuse_mode=3,
            quant_mode=layer._mega_moe_quant_mode,
            **layer._mega_moe_activation_params,
            l1_bias=getattr(layer, "w13_scale_bias", None),
            l2_bias=getattr(layer, "w2_scale_bias", None),
        )
        return hidden_states

    buf = _get_fuseep_buffer(layer)
    hidden_states, _ = buf.fused_deep_moe(
        hidden_states,
        topk_idx=topk_output.topk_ids,
        topk_weights=topk_output.topk_weights,
        gmm1_permuted_weight=layer.w13_weight,
        gmm1_permuted_weight_scale=layer.w13_weight_scale,
        gmm2_weight=layer.w2_weight,
        gmm2_weight_scale=layer.w2_weight_scale,
        num_max_dispatch_tokens_per_rank=(
            envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
        ),
        num_experts=layer.num_experts,
        fuse_mode=mode,
    )
    return hidden_states


@lru_cache(maxsize=128)
def _log_mega_moe_geometry(rows: int, hidden: int, capacity: int, ep: int, tp: int):
    logger.info(
        "A3 MegaMoE input_shape=(%d, %d) dispatch_capacity=%d ep_size=%d moe_tp_size=%d",
        rows,
        hidden,
        capacity,
        ep,
        tp,
    )


def _mega_moe_capacity() -> int:
    capacity = envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get()
    if not 1 <= capacity <= 4096:
        raise ValueError(
            "A3 FuseEP mode 3 requires "
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK in [1, 4096]"
        )
    return capacity


def _mega_moe_activation(config):
    activation = config.activation
    if not getattr(config, "is_gated", True):
        raise ValueError("MegaMoE requires a gated activation")
    if getattr(config, "swiglu_limit", None) not in (None, 0):
        raise ValueError("MegaMoE does not expose clipped SwiGLU through DeepEP")
    if getattr(config, "gemm1_beta", None) is not None:
        raise ValueError("MegaMoE does not expose gemm1_beta through DeepEP")
    if getattr(config, "apply_router_weight_on_input", False) or getattr(
        config, "no_combine", False
    ):
        raise ValueError(
            "MegaMoE requires routing weights on the combined expert output"
        )
    alpha = getattr(config, "gemm1_alpha", None)
    clamp = getattr(config, "gemm1_clamp_limit", None)
    if activation in ("silu", "swiglu"):
        if alpha not in (None, 1.0) or clamp is not None:
            raise ValueError(
                "Ordinary SwiGLU cannot use SiTU/OAI activation parameters"
            )
        return dict(activation="swiglu")
    if activation in ("situ", "situglu"):
        beta = 4.0 if alpha is None else float(alpha)
        linear_beta = None if clamp is None else float(clamp)
        if (
            not math.isfinite(beta)
            or beta <= 0
            or (
                linear_beta is not None
                and (not math.isfinite(linear_beta) or linear_beta < 0)
            )
        ):
            raise ValueError(
                "SiTU requires finite positive beta and nonnegative linear_beta"
            )
        # None explicitly preserves the model's unclamped SiTU variant.
        return dict(activation="situ", beta=beta, linear_beta=linear_beta)
    raise ValueError(
        f"MegaMoE does not support activation {activation!r}; use SwiGLU or SiTU"
    )


def _check_mega_moe_operator(operator):
    # Validate the A3 invocation, independent of DeepEP's optional MXFP kwargs.
    arguments = dict.fromkeys(
        (
            "x",
            "topk_ids",
            "topk_weights",
            "l1_weights",
            "l2_weights",
            "sym_buffer",
            "l1_weights_sf",
            "l2_weights_sf",
            "l1_bias",
            "l2_bias",
            "x_active_mask",
        )
    )
    arguments.update(activation="swiglu", activation_params=None)
    try:
        inspect.signature(operator).bind(**arguments)
    except TypeError as exc:
        raise RuntimeError(
            "DeepEP's MegaMoE loader returned an incompatible A3 operator; "
            "upgrade to matching CANN toolkit/operator packages exposing "
            f"activation_params and use a compatible DeepEP build: {exc}"
        ) from exc
    return sorted(arguments)


@lru_cache(maxsize=1)
def _check_mega_moe_support():
    try:
        import cann_ops_transformer
        import deep_ep
        from cann_ops_transformer.ops import get_symm_buffer_for_mega_moe, mega_moe
        from deep_ep import Buffer
        from deep_ep.buffer import FuseMode
        from deep_ep.strategies import fused_strategy

        buffer_signature = inspect.signature(Buffer.fused_deep_moe)
        for name, module, package in (
            ("deep_ep", deep_ep, "deep-ep"),
            ("cann_ops_transformer", cann_ops_transformer, "cann-ops-transformer"),
        ):
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = getattr(module, "__version__", None)
            logger.warning(
                "MegaMoE package %s: path=%s version=%s", name, module.__file__, version
            )
        required = {
            "fuse_mode",
            "quant_mode",
            "activation",
            "beta",
            "linear_beta",
            "l1_bias",
            "l2_bias",
        }
        if int(FuseMode.MEGA_MOE) != 3 or not required.issubset(
            buffer_signature.parameters
        ):
            raise RuntimeError("DeepEP does not expose the current MegaMoE API")
        if not callable(mega_moe) or not callable(get_symm_buffer_for_mega_moe):
            raise RuntimeError("ops-transformer does not expose MegaMoE")
        operator_signature = inspect.signature(mega_moe)
        # DeepEP owns native-operator compatibility; do not patch its strategies here.
        get_symm_buffer_for_mega_moe, compatible_operator = (
            fused_strategy.MegaMoeStrategy._load_mega_moe_ops()
        )
        keywords = _check_mega_moe_operator(compatible_operator)
        diagnostic = {
            "deep_ep": {
                "path": deep_ep.__file__,
                "version": getattr(deep_ep, "__version__", None),
            },
            "cann_ops_transformer": {
                "path": cann_ops_transformer.__file__,
                "version": getattr(cann_ops_transformer, "__version__", None),
            },
            "fused_deep_moe_signature": str(buffer_signature),
            "mega_moe_signature": str(operator_signature),
            "strategy_path": fused_strategy.__file__,
            "operator_keywords": keywords,
            "legacy_swiglu_adapter": compatible_operator is not mega_moe,
        }
        logger.warning("MegaMoE dependencies: %s", json.dumps(diagnostic))
    except (
        ImportError,
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
        OSError,
        SyntaxError,
    ) as exc:
        raise RuntimeError(
            "FuseEP mode 3 requires matching CANN, ops-transformer and "
            "DeepEP packages exposing FuseMode.MEGA_MOE, activation and bias "
            f"arguments; dependency check failed: {exc}"
        ) from exc
    soc = torch.npu.get_device_name()
    if not soc.startswith("Ascend910_93"):
        raise ValueError(f"FuseEP mode 3 currently supports Ascend A3, got {soc}")
    return get_symm_buffer_for_mega_moe, compatible_operator


def _process_mega_moe_weights(layer: torch.nn.Module) -> None:
    from sglang.srt.hardware_backend.npu.quantization.moe_methods import (
        NPUUnquantMoEMethod,
        NPUW4A8Int8MoEMethod,
        NPUW8A8Int8MoEMethod,
    )

    if getattr(layer, "_mega_moe_weights_processed", False):
        return
    kernels = (layer.w13_kernel, layer.w2_kernel)
    if type(kernels[0]) is not type(kernels[1]) or type(kernels[0]) not in (
        NPUW4A8Int8MoEMethod,
        NPUW8A8Int8MoEMethod,
        NPUUnquantMoEMethod,
    ):
        raise ValueError(
            "A3 MegaMoE requires homogeneous BF16, W4A8 INT, or W8A8 INT experts; FP8/MXFP/FP4 layouts need a separate platform adapter"
        )
    unquantized = type(kernels[0]) is NPUUnquantMoEMethod
    w4a8 = isinstance(kernels[0], NPUW4A8Int8MoEMethod)
    if w4a8 and any(
        not kernel.is_per_channel_weight or kernel.activation_use_clip
        for kernel in kernels
    ):
        raise ValueError(
            "FuseEP mode 3 requires per-channel scales without activation clipping"
        )
    config = layer.moe_runner_config
    activation_params = _mega_moe_activation(config)
    if layer.num_fused_shared_experts:
        raise ValueError("FuseEP mode 3 requires separate shared experts")
    ep = layer.moe_ep_size
    if layer.moe_tp_size != 1 or ep not in (2, 4, 8, 16, 32, 64):
        raise ValueError("A3 FuseEP mode 3 requires MoE TP=1 and EP=2/4/8/16/32/64")
    experts = layer.num_local_experts
    hidden = layer.hidden_size
    intermediate = layer.intermediate_size_per_partition
    if (
        not ep <= layer.num_experts <= 1024
        or layer.num_experts != experts * ep
        or experts > 128
        or not 1024 <= hidden <= 8192
        or hidden % 512
        or not 1024 <= intermediate <= 3584
        or intermediate % 512
        or not 1 <= config.top_k <= 16
    ):
        raise ValueError("Expert geometry is outside A3 MegaMoE limits")

    # Validate both projections before replacing registered parameters with lists.
    source = {}
    for prefix, channels, inputs in (
        ("w13", 2 * intermediate, hidden),
        ("w2", hidden, intermediate),
    ):
        weight = getattr(layer, f"{prefix}_weight")
        scale = getattr(layer, f"{prefix}_weight_scale", None)
        shape = (experts, channels // (2 if w4a8 else 1), inputs)
        dtype = torch.bfloat16 if unquantized else torch.int8
        if weight.dtype != dtype or tuple(weight.shape) != shape:
            raise ValueError(
                f"Invalid {prefix} checkpoint weight; expected {dtype} storage {shape}"
            )
        if unquantized:
            if scale is not None or any(
                getattr(layer, f"{prefix}_{suffix}", None) is not None
                for suffix in ("bias", "weight_bias", "scale_bias")
            ):
                raise ValueError(
                    "BF16 MegaMoE requires weights without quantization scales or expert bias"
                )
            source[prefix] = weight.data, None, None
            continue
        if (
            scale is None
            or scale.dtype != torch.float32
            or tuple(scale.shape) != (experts, channels, 1)
            or scale.device != weight.device
            or not torch.isfinite(scale).all().item()
            or (scale < 0).any().item()
            or hasattr(layer, f"{prefix}_weight_scale_second")
        ):
            raise ValueError(
                f"FuseEP mode 3 requires finite per-channel {prefix} scales"
            )
        offset = getattr(layer, f"{prefix}_weight_offset", None)
        if offset is not None and torch.count_nonzero(offset).item():
            raise ValueError(
                f"FuseEP mode 3 does not support nonzero {prefix} weight offsets"
            )
        bias = getattr(layer, f"{prefix}_scale_bias", None)
        if w4a8:
            if (
                bias is None
                or bias.ndim != 3
                or tuple(bias.shape[:2]) != (experts, channels)
                or bias.shape[2] < 1
                or bias.dtype != torch.float32
                or bias.device != weight.device
                or not torch.isfinite(bias).all().item()
            ):
                raise ValueError(
                    f"FuseEP mode 3 requires loaded {prefix}_scale_bias compensation"
                )
        elif any(
            getattr(layer, f"{prefix}_{suffix}", None) is not None
            for suffix in ("bias", "weight_bias", "scale_bias")
        ):
            raise ValueError("FuseEP mode 3 W8A8 does not support expert bias")
        source[prefix] = weight.data, scale.data, bias

    _mega_moe_capacity()
    _check_mega_moe_support()
    converted = {}
    for prefix, (weight, scale, bias) in source.items():
        weights = [
            npu_format_cast(expert.transpose(-2, -1).contiguous())
            for expert in weight.unbind(0)
        ]
        converted[f"{prefix}_weight"] = (
            [expert.view(torch.int32) for expert in weights] if w4a8 else weights
        )
        if not unquantized:
            converted[f"{prefix}_weight_scale"] = [
                expert.reshape(-1)
                .contiguous()
                .view(torch.int32)
                .to(torch.int64)
                .bitwise_and(0xFFFFFFFF)
                .view(torch.uint64)
                for expert in scale.unbind(0)
            ]
        if w4a8:
            # MoE TP=1 owns all loaded W2 compensation segments.
            converted[f"{prefix}_scale_bias"] = [
                expert.sum(dim=-1) for expert in bias.data.unbind(0)
            ]
    for name, value in converted.items():
        delattr(layer, name)
        setattr(layer, name, value)
    layer._mega_moe_activation_params = activation_params
    layer._mega_moe_quant_mode = 0 if unquantized else 1
    layer._mega_moe_weights_processed = True


def _permute_w13_weight_scale(w: torch.Tensor, tile_n: int) -> torch.Tensor:
    if tile_n % 2 != 0:
        raise ValueError(f"tile_n must be even, got {tile_n}")

    *dims, n = w.shape
    if n % tile_n != 0:
        raise ValueError(f"Last dimension {n} must be divisible by tile_n {tile_n}")

    w_reshaped = w.reshape(*dims, 2, n // tile_n, tile_n // 2)
    perm_order = list(range(len(dims))) + [-2, -3, -1]
    return w_reshaped.permute(perm_order).reshape(*dims, n)


def _reshape_w13_weight(
    weight: torch.Tensor, dim: int, chunk_size: int = 64
) -> torch.Tensor:
    # Achieving greater computing power through reshape on Ascend.
    original_shape = weight.shape
    if dim < 0:
        dim += len(original_shape)

    if original_shape[dim] % (2 * chunk_size) != 0:
        raise ValueError(
            f"Dimension {dim} size {original_shape[dim]} must be divisible by "
            f"{2 * chunk_size}"
        )

    new_shape = (
        *original_shape[:dim],
        2,
        original_shape[dim] // (2 * chunk_size),
        chunk_size,
        *original_shape[dim + 1 :],
    )

    weight = weight.view(new_shape)
    weight = weight.transpose(dim, dim + 1).contiguous()
    return weight.view(*original_shape[:dim], -1, *original_shape[dim + 1 :])


def _release_weight_cache(weight: torch.Tensor) -> torch.Tensor:
    # .contiguous() introduces additional memory overhead; release with resize_(0)
    origin_weight = weight.data.transpose(1, 2)
    new_weight = origin_weight.contiguous()
    origin_weight.untyped_storage().resize_(0)
    return new_weight


def _scale_from_float_to_int64(scale: torch.Tensor) -> torch.nn.Parameter:
    import numpy as np

    converted = torch.from_numpy(
        np.frombuffer(
            scale.cpu().to(torch.float32).numpy().tobytes(), dtype=np.int32
        ).astype(np.int64)
    ).to(scale.device)
    return torch.nn.Parameter(converted, requires_grad=False)


def process_fuseep_weights(layer: torch.nn.Module, weight_prefix: str) -> None:
    """Prepare a FuseEP projection; mode 3 converts both projections together."""
    mode = get_exec().moe.fuseep_mode
    if mode == 3:
        _process_mega_moe_weights(layer)
        return
    if mode == 1:
        # -- The fused MoE optimization mode "1": dispatch_gmm_combine_decode --
        if weight_prefix == "w13":
            cpu_w13 = layer.w13_weight.data.transpose(1, 2).cpu()
            layer.w13_weight.data = _reshape_w13_weight(cpu_w13, -1).npu()
            w13_scale = layer.w13_weight_scale.data.squeeze(-1).contiguous()
            w13_scale = _permute_w13_weight_scale(w13_scale, 128)
            layer.w13_weight_scale = torch.nn.Parameter(
                w13_scale.to(torch.float32), requires_grad=False
            )
            layer.w13_weight.data = npu_format_cast(layer.w13_weight.data)
        else:  # weight_prefix == "w2"
            layer.w2_weight.data = npu_format_cast(layer.w2_weight.data)
            w2_scale = layer.w2_weight_scale.data.squeeze(-1).contiguous()
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_scale.to(torch.float32), requires_grad=False
            )
    elif mode == 2:
        # -- The fused MoE optimization mode "2": dispatch_ffn_combine --
        if weight_prefix == "w13":
            w13_weight = _release_weight_cache(layer.w13_weight)
            layer.w13_weight.data = npu_format_cast(w13_weight)
            layer.w13_weight_scale.data = layer.w13_weight_scale.data.view(
                layer.w13_weight_scale.data.shape[0], -1
            )
            layer.w13_weight_scale = _scale_from_float_to_int64(
                layer.w13_weight_scale.data
            )
        else:  # weight_prefix == "w2"
            w2_weight = _release_weight_cache(layer.w2_weight)
            layer.w2_weight.data = npu_format_cast(w2_weight)
            w2_scale = layer.w2_weight_scale.data.squeeze(-1).contiguous()
            layer.w2_weight_scale = torch.nn.Parameter(
                w2_scale.to(torch.float32), requires_grad=False
            )
            layer.w2_weight_scale = _scale_from_float_to_int64(
                layer.w2_weight_scale.data
            )

    # -- offsets (exist or not, same logic for both prefixes) ---------------
    offset_attr = f"{weight_prefix}_weight_offset"
    if hasattr(layer, offset_attr):
        setattr(
            layer,
            offset_attr,
            torch.nn.Parameter(
                getattr(layer, offset_attr).data.squeeze(-1).contiguous(),
                requires_grad=False,
            ),
        )
