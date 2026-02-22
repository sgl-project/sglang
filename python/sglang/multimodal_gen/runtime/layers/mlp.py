# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

from typing import Optional

import torch
import torch.nn as nn
from diffusers.models.activations import (
    GEGLU,
    GELU,
    ApproximateGELU,
    LinearActivation,
    SwiGLU,
)

from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)
from sglang.multimodal_gen.runtime.layers.quantization import QuantizationConfig


class MLP(nn.Module):
    """
    MLP for DiT blocks, NO gated linear units
    """

    def __init__(
        self,
        input_dim: int,
        mlp_hidden_dim: int,
        output_dim: int | None = None,
        bias: bool = True,
        act_type: str = "gelu_pytorch_tanh",
        dtype: torch.dtype | None = None,
        prefix: str = "",
        quant_config: QuantizationConfig = None,
    ):
        super().__init__()
        self.fc_in = ColumnParallelLinear(
            input_dim,
            mlp_hidden_dim,
            bias=True,
            gather_output=False,
            quant_config=quant_config,
        )

        self.act = get_act_fn(act_type)
        if output_dim is None:
            output_dim = input_dim
        self.fc_out = RowParallelLinear(
            mlp_hidden_dim,
            output_dim,
            bias=True,
            input_is_parallel=True,
            quant_config=quant_config,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc_in(x)
        x = self.act(x)
        x, _ = self.fc_out(x)
        return x


class FeedForward(nn.Module):
    r"""
    A feed-forward layer.

    Parameters:
        dim (`int`): The number of channels in the input.
        dim_out (`int`, *optional*): The number of channels in the output. If not given, defaults to `dim`.
        mult (`int`, *optional*, defaults to 4): The multiplier to use for the hidden dimension.
        activation_fn (`str`, *optional*, defaults to `"geglu"`): Activation function to be used in feed-forward.
        bias (`bool`, defaults to True): Whether to use a bias in the linear layer.
    """

    def __init__(
        self,
        dim: int,
        dim_out: Optional[int] = None,
        mult: int = 4,
        activation_fn: str = "geglu",
        inner_dim=None,
        bias: bool = True,
    ):
        super().__init__()
        if inner_dim is None:
            inner_dim = int(dim * mult)
        dim_out = dim_out if dim_out is not None else dim

        if activation_fn == "gelu":
            act_fn = GELU(dim, inner_dim, bias=bias)
        if activation_fn == "gelu-approximate":
            act_fn = GELU(dim, inner_dim, approximate="tanh", bias=bias)
        elif activation_fn == "geglu":
            act_fn = GEGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "geglu-approximate":
            act_fn = ApproximateGELU(dim, inner_dim, bias=bias)
        elif activation_fn == "swiglu":
            act_fn = SwiGLU(dim, inner_dim, bias=bias)
        elif activation_fn == "linear-silu":
            act_fn = LinearActivation(dim, inner_dim, bias=bias, activation="silu")

        self.net = nn.ModuleList([])
        # project in
        self.net.append(act_fn)
        # dummy dropout layer to match with checkpoints compatible with diffusers
        self.net.append(nn.Dropout(0.0))
        # project out
        self.net.append(nn.Linear(inner_dim, dim_out, bias=bias))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        for module in self.net:
            hidden_states = module(hidden_states)
        return hidden_states
