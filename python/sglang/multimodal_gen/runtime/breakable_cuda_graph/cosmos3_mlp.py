# SPDX-License-Identifier: Apache-2.0
"""Capture stateless Cosmos3 GEN MLPs without capturing the UND cache."""

from contextlib import contextmanager
from functools import wraps

import torch

from sglang.multimodal_gen.runtime.breakable_cuda_graph.runner import (
    BaseBreakableCudaGraphRunner,
)
from sglang.multimodal_gen.runtime.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    RowParallelLinear,
    UnquantizedLinearMethod,
)
from sglang.multimodal_gen.runtime.models.dits.cosmos3video import (
    Cosmos3DenseMLP,
    Cosmos3GatedMLP,
)


class Cosmos3MLPCudaGraphs:
    """Request-scoped replay over resident, unquantized GEN MLPs.

    MLP inputs contain only visual/audio/action tokens. Their shape does not
    depend on the text length, and replay copies fresh inputs on every step.
    Attention and the request-owned UND cache remain eager.
    """

    def __init__(self, transformer, device):
        self.mlps = [layer.mlp for layer in transformer.gen_layers]
        self.runners = []
        self._enabled = False
        self._is_warmup = False
        self._weight_signature = None
        pool = torch.get_device_module(device).graph_pool_handle()
        for mlp in self.mlps:
            eager = mlp.forward
            runner = BaseBreakableCudaGraphRunner(eager, device, pool=pool)
            self.runners.append(runner)
            mlp.forward = self._wrap_forward(eager, runner)

    def _wrap_forward(self, eager, runner):
        @wraps(eager)
        def forward(x):
            if not self._enabled:
                return eager(x)
            if self._is_warmup:
                runner.capture(x=x)
            return runner(x=x)

        return forward

    def _get_weight_signature(self):
        signature = []
        for mlp in self.mlps:
            if type(mlp) is Cosmos3GatedMLP:
                projections = (mlp.gate_up_proj, mlp.down_proj)
            elif type(mlp) is Cosmos3DenseMLP:
                projections = (mlp.up_proj, mlp.down_proj)
            else:
                return None
            for projection in projections:
                # Dynamic LoRA wrappers and step-dependent quant methods must
                # execute their live Python dispatch, even after warmup.
                if (
                    type(projection)
                    not in (
                        ColumnParallelLinear,
                        MergedColumnParallelLinear,
                        RowParallelLinear,
                    )
                    or type(projection.quant_method) is not UnquantizedLinearMethod
                ):
                    return None
                weight = projection.weight
                if weight.device.type != "cuda" or weight.dtype not in (
                    torch.float16,
                    torch.bfloat16,
                ):
                    return None
                signature.append(
                    (
                        id(projection),
                        weight.data_ptr(),
                        weight.shape,
                        weight.stride(),
                        weight.dtype,
                        weight.device,
                    )
                )
        return tuple(signature)

    @contextmanager
    def activate(self, *, is_warmup):
        signature = self._get_weight_signature()
        if self._weight_signature is not None and signature != self._weight_signature:
            # Loading a new adapter or replacing weights must not replay a
            # graph referencing old storage. Serving never recaptures it.
            for runner in self.runners:
                runner.reset()
        self._weight_signature = signature
        self._enabled = signature is not None
        self._is_warmup = is_warmup
        try:
            yield
        finally:
            self._enabled = False
            self._is_warmup = False
