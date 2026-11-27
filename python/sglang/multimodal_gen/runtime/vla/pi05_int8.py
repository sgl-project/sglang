# SPDX-License-Identifier: Apache-2.0
"""Online W8A8 INT8 projections for Pi0.5 on sm80+ (Ampere has no FP8 tensor cores).

Weights are quantized symmetrically per output channel after the BF16 checkpoint is loaded; activations are
quantized per token at runtime. Both scales are applied in the INT8 GEMM epilogue, so no calibration data or
quantized checkpoint is needed. The projection set is the one the ModelOpt FP8 path uses.
"""

from __future__ import annotations

import torch
from torch import nn

from sglang.multimodal_gen.runtime.vla.pi05_quantization import get_projection_names

INT8_COMPONENTS = ("paligemma", "action_expert", "vision", "projector")


def int8_supported(device: torch.device) -> bool:
    return (
        device.type == "cuda"
        and torch.version.cuda is not None
        and torch.cuda.get_device_capability(device) >= (8, 0)
    )


def validate_int8_components(components: list[str]) -> list[str]:
    if not isinstance(components, list) or not all(
        isinstance(c, str) for c in components
    ):
        raise ValueError("int8_components must be a list of strings")
    if len(set(components)) != len(components):
        raise ValueError("int8_components must be unique")
    unknown = set(components) - set(INT8_COMPONENTS)
    if unknown:
        raise ValueError(
            f"Unsupported Pi0.5 INT8 components {sorted(unknown)}; choose from {INT8_COMPONENTS}"
        )
    return components


def _quantize_weight_per_channel(
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    w = weight.detach().float()
    scale = w.abs().amax(dim=1).clamp(min=1e-8) / 127.0
    q = torch.round(w / scale[:, None]).clamp_(-128, 127).to(torch.int8)
    return q, scale.contiguous()


class Pi05Int8Linear(nn.Module):
    """Drop-in for a Pi0.5 projection: keeps the caller's tensor or (output, bias) interface."""

    def __init__(
        self,
        *,
        weight: torch.Tensor,
        bias: torch.Tensor | None,
        tensor_output: bool,
        output_dtype: torch.dtype,
    ):
        super().__init__()
        q, scale = _quantize_weight_per_channel(weight)
        self.out_features, self.in_features = q.shape
        self.register_buffer("weight", q, persistent=False)
        self.register_buffer("weight_scale", scale, persistent=False)
        self.register_buffer(
            "bias",
            None if bias is None else bias.detach().to(torch.bfloat16).contiguous(),
            persistent=False,
        )
        self.tensor_output = tensor_output
        self.output_dtype = output_dtype

    @classmethod
    def from_module(cls, module: nn.Module) -> Pi05Int8Linear:
        if not isinstance(module, nn.Linear) and module.skip_bias_add:
            raise ValueError("Pi0.5 INT8 does not support skip_bias_add projections")
        return cls(
            weight=module.weight,
            bias=module.bias,
            tensor_output=isinstance(module, nn.Linear),
            output_dtype=module.weight.dtype,
        )

    def forward(self, x: torch.Tensor):
        from sgl_kernel import int8_scaled_mm

        from sglang.kernels.ops.quantization.int8_kernel import per_token_quant_int8

        leading = x.shape[:-1]
        x_q, x_scale = per_token_quant_int8(
            x.reshape(-1, self.in_features).to(torch.bfloat16)
        )
        # int8_scaled_mm takes the weight as a [K, N] column-major operand.
        out = int8_scaled_mm(
            x_q, self.weight.t(), x_scale, self.weight_scale, torch.bfloat16, self.bias
        )
        out = out.reshape(*leading, self.out_features).to(self.output_dtype)
        return out if self.tensor_output else (out, None)


def quantize_projections_int8(model: nn.Module, components: list[str]) -> list[str]:
    """Replace loaded BF16 projections in place; call after weights are loaded."""
    names = get_projection_names(model, validate_int8_components(components))
    for name in names:
        module = model.get_submodule(name)
        parent_name, _, child_name = name.rpartition(".")
        setattr(
            model.get_submodule(parent_name),
            child_name,
            Pi05Int8Linear.from_module(module),
        )
    return names
