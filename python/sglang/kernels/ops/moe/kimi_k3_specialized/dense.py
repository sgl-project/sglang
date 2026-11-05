# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Prepared first dense MLP, sharing projection and SiTU decode arithmetic."""

import cutlass
import cutlass.cute as cute

from sglang.kernels.jit.kimi_k3.primitives import launch
from sglang.kernels.ops.communication.kimi_k3_specialized.shared import situ


class DenseSitu:
    def __init__(self, hidden, padded):
        self.hidden, self.padded = hidden, padded

    @cute.jit
    def __call__(self, *args):
        grid = (cute.ceil_div(self.padded, 256), args[0].shape[0], 1)
        launch(self.kernel, args, grid, 256)

    @cute.kernel
    def kernel(self, args):
        x, out, beta, linear = args[0:4]
        tile, row, _ = cute.arch.block_idx()
        tid, _, _ = cute.arch.thread_idx()
        col = tile * 256 + tid
        cute.arch.griddepcontrol_wait()
        if col < self.padded:
            value = cutlass.Float32(0)
            if col < self.hidden:
                gate = cutlass.Float32(x[row, col])
                up = cutlass.Float32(x[row, self.hidden + col])
                value = situ(gate, up, beta, linear, native=True)
            out[row, col] = cutlass.BFloat16(value)
        cute.arch.griddepcontrol_launch_dependents()
