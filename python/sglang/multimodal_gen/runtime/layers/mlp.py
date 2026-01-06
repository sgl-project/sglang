# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo

# SPDX-License-Identifier: Apache-2.0

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.layers.activation import get_act_fn
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    RowParallelLinear,
)


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
    ):
        super().__init__()
        self.act_type = act_type.lower()
        self.fc_in = ColumnParallelLinear(
            input_dim,
            mlp_hidden_dim,
            bias=True,
            gather_output=False,
        )

        self.act = get_act_fn(act_type)
        self.act_fp32 = self.act_type in {
            "gelu",
            "gelu_new",
            "gelu_pytorch_tanh",
            "quick_gelu",
            "silu",
        }
        if output_dim is None:
            output_dim = input_dim
        self.fc_out = RowParallelLinear(
            mlp_hidden_dim,
            output_dim,
            bias=True,
            input_is_parallel=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, _ = self.fc_in(x)
        if self.act_fp32 and x.dtype in (torch.float16, torch.bfloat16):
            x = self.act(x.float()).to(dtype=x.dtype)
        else:
            x = self.act(x)
        x, _ = self.fc_out(x)
        return x
