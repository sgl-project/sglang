# SPDX-License-Identifier: Apache-2.0
"""Pi0.5's fused, per-tensor ModelOpt FP8 checkpoint contract."""

from __future__ import annotations

import re

import torch
from torch import nn

_COMPONENT_PREFIXES = {
    "paligemma": "paligemma_with_expert.paligemma.model.language_model.layers.",
    "action_expert": "paligemma_with_expert.gemma_expert.model.layers.",
}
_PROJECTION = re.compile(
    r"\d+\.(?:self_attn\.(?:qkv_proj|o_proj)|mlp\.(?:gate_up_proj|down_proj))$"
)


def projection_names(model: nn.Module, components: list[str]) -> list[str]:
    """Select only transformer projections, including already fused QKV/gate-up."""
    if not components or len(set(components)) != len(components):
        raise ValueError("FP8 components must be nonempty and unique")
    if set(components) - _COMPONENT_PREFIXES.keys():
        raise ValueError(f"Unsupported Pi0.5 FP8 components: {components}")
    return [
        name
        for name, _ in model.named_modules()
        if any(
            name.startswith(_COMPONENT_PREFIXES[component])
            and _PROJECTION.fullmatch(name[len(_COMPONENT_PREFIXES[component]) :])
            for component in components
        )
    ]


def validate_quantization_config(config: dict) -> list[str]:
    if (
        config.get("quant_method") != "modelopt"
        or config.get("quant_algo") != "FP8"
        or config.get("pi05_fused_projections") is not True
    ):
        raise ValueError("Pi0.5 requires a fused-projection ModelOpt FP8 checkpoint")
    components = config.get("components")
    if not isinstance(components, list) or not components:
        raise ValueError("Pi0.5 FP8 checkpoint must declare components")
    if (
        len(set(components)) != len(components)
        or set(components) - _COMPONENT_PREFIXES.keys()
    ):
        raise ValueError(f"Unsupported Pi0.5 FP8 components: {components}")
    return components


def replace_projections(
    model: nn.Module, components: list[str], *, quantized: bool
) -> list[str]:
    """Use one scale for each fused execution unit, never independent shard scales.

    The calibration version uses ordinary nn.Linear so ModelOpt can instrument
    it. The inference version uses a single-partition ReplicatedLinear; the
    original fused forward already splits the result into Q/K/V or gate/up.
    Call after the core's BF16/FP32 precision policy, before loading weights.
    """
    from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear
    from sglang.multimodal_gen.runtime.layers.quantization.modelopt_fp8 import (
        ModelOptFp8Config,
    )

    names = projection_names(model, components)
    for component in components:
        if not any(name.startswith(_COMPONENT_PREFIXES[component]) for name in names):
            raise ValueError(f"No Pi0.5 projections found for {component}")
    for name in names:
        module = model.get_submodule(name)
        output_size, input_size = module.weight.shape
        if quantized:
            replacement = ReplicatedLinear(
                input_size,
                output_size,
                bias=module.bias is not None,
                params_dtype=torch.bfloat16,
                quant_config=ModelOptFp8Config(),
                prefix=name,
            ).to(device=module.weight.device)
        else:
            replacement = nn.Linear(
                input_size,
                output_size,
                bias=module.bias is not None,
                device=module.weight.device,
                dtype=module.weight.dtype,
            )
            with torch.no_grad():
                replacement.weight.copy_(module.weight)
                if module.bias is not None:
                    replacement.bias.copy_(module.bias)
        parent_name, _, child_name = name.rpartition(".")
        setattr(model.get_submodule(parent_name), child_name, replacement)
    return names


def finalize_fp8_weights(model: nn.Module, names: list[str]) -> None:
    for name in names:
        layer = model.get_submodule(name)
        for scale_name in ("weight_scale", "input_scale"):
            scale = getattr(layer, scale_name)
            if scale.numel() != 1 or not bool(
                (torch.isfinite(scale) & (scale > 0)).all()
            ):
                raise ValueError(f"Invalid per-tensor FP8 {scale_name} for {name}")
        layer.quant_method.process_weights_after_loading(layer)
