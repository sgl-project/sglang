# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project

from mindspore import Tensor, mint, nn


class SwiGLU(nn.Cell):
    """An activation function for SwiGLU

    Shapes:
        x: (num_tokens, 2 * d) or (batch_size, seq_len, 2 * d)
        return (num_tokens, d) or (batch_size, seq_len, d)
    """

    def __init__(self) -> None:
        super().__init__()
        self.silu = nn.SiLU()

    def construct(self, x: Tensor) -> Tensor:
        hidden_size = x.shape[-1] // 2
        size = [hidden_size, hidden_size]
        gate, up = mint.split(x, size, dim=-1)
        gate = self.silu(gate)
        hidden = mint.mul(up, gate)
        return hidden
