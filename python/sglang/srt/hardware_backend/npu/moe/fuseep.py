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


_PARAMS_BYTES = 2  # bf16 — Ascend's Dispatch & Combine does not support fp16


def _get_fuseep_buffer(layer: FusedMoE):
    DeepEPBuffer.set_dispatch_mode_as_low_latency()
    return DeepEPBuffer.get_deepep_buffer(
        get_parallel().moe_ep_group.device_group,
        layer.hidden_size,
        _PARAMS_BYTES,
        DeepEPMode.LOW_LATENCY,
        envs.SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK.get(),
        layer.num_experts,
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
        buffer = _get_fuseep_buffer(layer)
        hidden_states, _ = buffer.fused_deep_moe(
            x=hidden_states,
            topk_idx=topk_output.topk_ids.to(torch.int32),
            topk_weights=topk_output.topk_weights.to(torch.float32),
            gmm1_permuted_weight=layer.w13_weight,
            gmm1_permuted_weight_scale=layer.w13_weight_scale,
            gmm2_weight=layer.w2_weight,
            gmm2_weight_scale=layer.w2_weight_scale,
            num_max_dispatch_tokens_per_rank=capacity,
            num_experts=layer.num_experts,
            fuse_mode=3,
            quant_mode=1,
            activation="swiglu",
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
    logging.getLogger(__name__).info(
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


def _check_mega_moe_operator(operator):
    # Validate the INT invocation, independent of DeepEP's optional MXFP kwargs.
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
            "DeepEP's MegaMoE loader returned an incompatible INT operator; "
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

        arguments = inspect.signature(Buffer.fused_deep_moe).parameters
        for name, module, package in (
            ("deep_ep", deep_ep, "deep-ep"),
            ("cann_ops_transformer", cann_ops_transformer, "cann-ops-transformer"),
        ):
            try:
                version = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                version = getattr(module, "__version__", None)
            logging.getLogger(__name__).warning(
                "MegaMoE package %s: path=%s version=%s", name, module.__file__, version
            )
        logging.getLogger(__name__).warning(
            "MegaMoE signatures: fused_deep_moe%s mega_moe%s",
            inspect.signature(Buffer.fused_deep_moe),
            inspect.signature(mega_moe),
        )
        required = {"fuse_mode", "activation", "l1_bias", "l2_bias"}
        if int(FuseMode.MEGA_MOE) != 3 or not required.issubset(arguments):
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
            "fused_deep_moe_signature": str(inspect.signature(Buffer.fused_deep_moe)),
            "mega_moe_signature": str(operator_signature),
            "strategy_path": fused_strategy.__file__,
            "int_operator_keywords": sorted(keywords),
            "legacy_swiglu_adapter": compatible_operator is not mega_moe,
        }
        logging.getLogger(__name__).warning(
            "MegaMoE dependencies: %s", json.dumps(diagnostic)
        )
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
        NPUW4A8Int8MoEMethod,
        NPUW8A8Int8MoEMethod,
    )

    if getattr(layer, "_mega_moe_weights_processed", False):
        return
    kernels = (layer.w13_kernel, layer.w2_kernel)
    if type(kernels[0]) is not type(kernels[1]) or type(kernels[0]) not in (
        NPUW4A8Int8MoEMethod,
        NPUW8A8Int8MoEMethod,
    ):
        raise ValueError("FuseEP mode 3 requires homogeneous ModelSlim W4A8/W8A8 INT")
    w4a8 = isinstance(kernels[0], NPUW4A8Int8MoEMethod)
    if w4a8 and any(
        not kernel.is_per_channel_weight or kernel.activation_use_clip
        for kernel in kernels
    ):
        raise ValueError(
            "FuseEP mode 3 requires per-channel scales without activation clipping"
        )
    config = layer.moe_runner_config
    if (
        config.activation != "silu"
        or config.swiglu_limit is not None
        or config.gemm1_alpha is not None
        or config.gemm1_clamp_limit is not None
    ):
        raise ValueError("FuseEP mode 3 currently requires ordinary SwiGLU")
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
        or not 1024 <= intermediate <= 3072
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
        scale = getattr(layer, f"{prefix}_weight_scale")
        shape = (experts, channels // (2 if w4a8 else 1), inputs)
        if weight.dtype != torch.int8 or tuple(weight.shape) != shape:
            raise ValueError(
                f"Invalid {prefix} checkpoint weight; expected INT8 storage {shape}"
            )
        if (
            scale.dtype != torch.float32
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
    if get_exec().moe.fuseep_mode == 3:
        _process_mega_moe_weights(layer)
        return
    if get_exec().moe.fuseep_mode == 1:
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
    elif get_exec().moe.fuseep_mode == 2:
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
