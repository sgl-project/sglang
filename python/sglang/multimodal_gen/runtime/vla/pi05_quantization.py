# SPDX-License-Identifier: Apache-2.0
"""Pi0.5's fused, per-tensor ModelOpt FP8 checkpoint contract."""

from __future__ import annotations

import re

import torch
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import ReplicatedLinear

_COMPONENT_PREFIXES = {
    "paligemma": "paligemma_with_expert.paligemma.model.language_model.layers.",
    "action_expert": "paligemma_with_expert.gemma_expert.model.layers.",
    "vision": "paligemma_with_expert.paligemma.model.vision_tower.",
    "projector": "paligemma_with_expert.paligemma.model.multi_modal_projector.",
    "action_heads": "action_",
}
_PROJECTION = re.compile(
    r"\d+\.(?:self_attn\.(?:qkv_proj|o_proj)|mlp\.(?:gate_up_proj|down_proj))$"
)

_VISION_PROJECTION = re.compile(
    r"(?:vision_model\.)?encoder\.layers\.\d+\.(?:self_attn\.(?:qkv_proj|proj)|mlp\.fc[12])$"
)
DEFAULT_COMPONENTS = list(_COMPONENT_PREFIXES)


def _selected(name: str, component: str) -> bool:
    prefix = _COMPONENT_PREFIXES[component]
    if not name.startswith(prefix):
        return False
    suffix = name[len(prefix) :]
    if component == "vision":
        return bool(_VISION_PROJECTION.fullmatch(suffix))
    if component == "projector":
        return suffix == "linear"
    if component == "action_heads":
        return suffix in ("in_proj", "out_proj")
    return bool(_PROJECTION.fullmatch(suffix))


def _tuple_output(module, inputs, output):
    # Shared SigLIP linears return (output, bias), unlike nn.Linear.
    return output, None


class Pi05Fp8Linear(ReplicatedLinear):
    """Preserve each caller's tensor/tuple interface and high-precision output."""

    def __init__(self, *args, tensor_output: bool, output_dtype: torch.dtype, **kwargs):
        super().__init__(*args, **kwargs)
        self.tensor_output = tensor_output
        self.output_dtype = output_dtype
        self.in_features = self.input_size
        self.out_features = self.output_size

    def forward(self, x):
        output, bias = super().forward(x.to(torch.bfloat16))
        output = output.to(self.output_dtype)
        return output if self.tensor_output else (output, bias)


def projection_names(model: nn.Module, components: list[str]) -> list[str]:
    """Select executed Linears; exclude Conv2d, time MLP and AdaRMS dense."""
    if not components or len(set(components)) != len(components):
        raise ValueError("FP8 components must be nonempty and unique")
    if set(components) - _COMPONENT_PREFIXES.keys():
        raise ValueError(f"Unsupported Pi0.5 FP8 components: {components}")
    return [
        name
        for name, _ in model.named_modules()
        if any(_selected(name, component) for component in components)
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
            replacement = Pi05Fp8Linear(
                input_size,
                output_size,
                bias=module.bias is not None,
                params_dtype=torch.bfloat16,
                quant_config=ModelOptFp8Config(),
                prefix=name,
                tensor_output=isinstance(module, nn.Linear),
                output_dtype=module.weight.dtype,
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
            if not isinstance(module, nn.Linear):
                replacement.register_forward_hook(_tuple_output)
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
