"""A ModelOpt FP8 linear stand-in for the norm+quant activation-gate tests."""

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers.quantization.modelopt_quant import (
    ModelOptFp8LinearMethod,
)


def fp8_linear(input_scale: float) -> nn.Module:
    linear = nn.Module()
    linear.quant_method = object.__new__(ModelOptFp8LinearMethod)
    linear.register_parameter(
        "input_scale",
        nn.Parameter(
            torch.tensor(input_scale, dtype=torch.float32, device="cuda"),
            requires_grad=False,
        ),
    )
    return linear
