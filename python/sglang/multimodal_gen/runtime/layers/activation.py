# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0
# Adapted from vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/model_executor/layers/activation.py
"""Custom activation functions."""
import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# TODO (will): remove this dependency
from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp


@CustomOp.register("silu_and_mul")
class SiluAndMul(CustomOp):
    """An activation function for SwiGLU.

    The function computes x -> silu(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return: (num_tokens, d) or (batch_size, seq_len, d)
    """

    def __init__(self) -> None:
        super().__init__()

    def forward_cuda(self, *args, **kwargs) -> Any:
        return self.forward_native(*args, **kwargs)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.silu(x[..., :d]) * x[..., d:]


@CustomOp.register("gelu_and_mul")
class GeluAndMul(CustomOp):
    """An activation function for GeGLU.

    The function computes x -> GELU(x[:d]) * x[d:] where d = x.shape[-1] // 2.

    Shapes:
        x: (batch_size, seq_len, 2 * d) or (num_tokens, 2 * d)
        return: (batch_size, seq_len, d) or (num_tokens, d)
    """

    def __init__(self, approximate: str = "none"):
        super().__init__()
        self.approximate = approximate
        if approximate not in ("none", "tanh"):
            raise ValueError(f"Unknown approximate mode: {approximate}")

    def forward_cuda(self, *args, **kwargs) -> Any:
        return self.forward_native(*args, **kwargs)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        d = x.shape[-1] // 2
        return F.gelu(x[..., :d], approximate=self.approximate) * x[..., d:]

    def extra_repr(self) -> str:
        return f"approximate={repr(self.approximate)}"


@CustomOp.register("gelu_new")
class NewGELU(CustomOp):

    def __init__(self):
        super().__init__()

    def forward_cuda(self, *args, **kwargs) -> Any:
        return self.forward_native(*args, **kwargs)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        c = math.sqrt(2.0 / math.pi)
        return 0.5 * x * (1.0 + torch.tanh(c * (x + 0.044715 * torch.pow(x, 3.0))))


@CustomOp.register("quick_gelu")
class QuickGELU(CustomOp):
    # https://github.com/huggingface/transformers/blob/main/src/transformers/activations.py#L90
    def __init__(self):
        super().__init__()

    def forward_cuda(self, *args, **kwargs) -> Any:
        return self.forward_native(*args, **kwargs)

    def forward_native(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-native implementation equivalent to forward()."""
        return x * torch.sigmoid(1.702 * x)


_ACTIVATION_REGISTRY = {
    "gelu": nn.GELU,
    "gelu_new": NewGELU,
    "gelu_pytorch_tanh": lambda: nn.GELU(approximate="tanh"),
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "quick_gelu": QuickGELU,
}


def get_act_fn(act_fn_name: str) -> nn.Module:
    """Get an activation function by name."""
    act_fn_name = act_fn_name.lower()
    if act_fn_name not in _ACTIVATION_REGISTRY:
        raise ValueError(f"Activation function {act_fn_name!r} is not supported.")

    return _ACTIVATION_REGISTRY[act_fn_name]()


_ACTIVATION_AND_MUL_REGISTRY = {
    "gelu": GeluAndMul,
    "silu": SiluAndMul,
}


def get_act_and_mul_fn(act_fn_name: str) -> nn.Module:
    """Get an activation-and-mul (i.e. SiluAndMul) function by name."""
    act_fn_name = act_fn_name.lower()
    if act_fn_name not in _ACTIVATION_AND_MUL_REGISTRY:
        raise ValueError(f"Activation function {act_fn_name!r} is not supported.")

    return _ACTIVATION_AND_MUL_REGISTRY[act_fn_name]()
