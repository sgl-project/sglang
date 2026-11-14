# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Prepared first dense MLP, sharing projection and SiTU decode arithmetic."""

from functools import cache

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.runtime import make_fake_stream

from sglang.kernels.jit.kimi_k3.host import (
    architecture,
    dynamic_fake,
    dynamic_view,
    stream,
)
from sglang.kernels.ops.moe.kimi_k3_specialized.dense import DenseSitu
from sglang.srt.kernel_plugins.nvidia.kimi_k3.cutedsl.gemm.projection import (
    PreparedProjection,
)


@cache
def _compile_activation(hidden, padded, arch):
    args = (dynamic_fake(2), dynamic_fake(2), *map(cutlass.Float32, (4, 25)))
    kernel = DenseSitu(hidden, padded)
    return cute.compile(kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}")


class PreparedDenseMLP:
    """Dense MLP; TP16 pads down K 2112 -> 2176 and the activation zero-fills it."""

    def __init__(self, mlp, batches, reduce=None):
        gate_up, down = mlp.gate_up_proj.weight, mlp.down_proj.weight
        self.device, self.reduce = gate_up.device, reduce
        self.hidden = hidden = gate_up.shape[0] // 2
        self.padded = padded = (hidden + 127) // 128 * 128
        linear = -1 if mlp.act_fn.linear_beta is None else mlp.act_fn.linear_beta
        self.scalars = tuple(map(cutlass.Float32, (mlp.act_fn.beta, linear)))
        if padded != hidden:
            down = torch.nn.functional.pad(down.detach(), (0, padded - hidden))
        self.gates = gate_up.new_empty((64, 2 * hidden))
        self.hidden_rows = gate_up.new_empty((64, padded))
        tile = {b: next(n for n in (8, 16, 32, 64) if b <= n) for b in batches}
        up_tiles = {8: (64, 8, 4, 8), 16: (64, 16, 2, 8), 32: (64, 32, 2, 8)}
        up_tiles[64] = (64, 64, 2, 5)
        down_tiles = {8: (128, 8, 1, 6), 16: (128, 16, 1, 6), 32: (128, 32, 1, 5)}
        down_tiles[64] = (64, 64, 1, 5)
        policy = lambda tiles: {b: tiles[n] for b, n in tile.items()}
        self.gate = PreparedProjection(gate_up, policy(up_tiles))
        self.down = PreparedProjection(down, policy(down_tiles))
        self.gate.prepare(), self.down.prepare()
        self.activation = _compile_activation(hidden, padded, architecture(self.device))

    def run(self, x, out):
        gate, hidden = self.gates[: x.shape[0]], self.hidden_rows[: x.shape[0]]
        self.gate.run(x, gate)
        views = (dynamic_view(gate), dynamic_view(hidden))
        self.activation(*views, *self.scalars, stream(self.device))
        self.down.run(hidden, out)
        if self.reduce is not None:
            reduced = self.reduce(out)
            if reduced.data_ptr() != out.data_ptr():
                out.copy_(reduced)
        return out
