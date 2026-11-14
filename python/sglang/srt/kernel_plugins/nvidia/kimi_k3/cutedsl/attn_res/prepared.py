# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Prepared seven-CTA AttnRes consumers for ordinary or Lamport mailbox input."""

from functools import cache
from typing import Any

import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import make_fake_stream

from sglang.kernels.jit.kimi_k3.host import (
    architecture,
    dynamic_fake,
    dynamic_view,
    fake,
    stream,
)
from sglang.kernels.jit.kimi_k3.tensor_adapter import to_cute
from sglang.kernels.ops.layernorm.kimi_k3_specialized.attn_res import ClusterAttnRes


@cache
def compile_consumer(nb, delta, write, final, slots, cap, arch, deadline_ns, aux=False):
    row, weight, eps = dynamic_fake(2), fake((7168,), (1,)), cutlass.Float32(1e-6)
    source = fake((slots, cap, 7168), (cap * 7168, 7168, 1)) if slots else row
    kernel = ClusterAttnRes(
        nb,
        slots=slots,
        has_delta=delta,
        write=write,
        final=final,
        deadline_ns=deadline_ns,
    )
    args = (source, row, dynamic_fake(3), weight, weight, weight, row, eps, eps)
    args += (row,) * aux  # the updated prefix row, also stored to an aux stream
    return cute.compile(kernel, *args, make_fake_stream(), options=f"--gpu-arch={arch}")


class PreparedAttnRes:
    """Ordinary-input code plus the Lamport variant of its bound mailbox ring."""

    def __init__(self, weights, key, eps, deadline_ns):
        self.device = weights[0].device
        self.views = tuple(to_cute(w.reshape(-1)) for w in weights)
        self.key, self.eps = key, tuple(map(cutlass.Float32, eps))
        self.deadline_ns = deadline_ns
        self.code = self._compile(0, 0)
        self.ring: Any = None
        self.aux: Any = None

    def _compile(self, slots, cap, aux=False):
        arch = architecture(self.device)
        return compile_consumer(*self.key, slots, cap, arch, self.deadline_ns, aux)

    @classmethod
    def for_target(cls, target, *, deadline_ns, final=False, has_delta=True):
        if final:
            norm, query = target.output_attn_res_norm, target.output_attn_res_proj
            output, nb, write = target.norm, target.num_attn_res_blocks, -1
        else:
            norm = target.self_attention_res_norm
            query, output = target.self_attention_res_proj, target.input_layernorm
            nb = target.prev_valid_blocks
            write = target.block_write_idx if target.is_block_write_layer else -1
        weights = (norm.weight, query.weight, output.weight)
        eps = (norm.variance_epsilon, output.variance_epsilon)
        return cls(weights, (nb, has_delta, write, final), eps, deadline_ns)

    def bind_ring(self, mailbox):
        """Lamport input: sum the [slots, capacity, 7168] mailbox rows, then re-arm."""
        shape = tuple(mailbox.shape[:2])
        self.ring = self._compile(*shape), to_cute(mailbox), shape

    def bind_aux(self):
        """Variants that also store the updated prefix rows (a stock aux stream)."""
        ring = None
        if self.ring is not None:
            slots, cap = self.ring[2]
            ring = self._compile(slots, cap, True)
        self.aux = self._compile(0, 0, True), ring

    def run_ring(self, prefix, blocks, out, aux=None):
        code = self.ring[0] if aux is None else self.aux[1]
        return self._launch(code, self.ring[1], prefix, blocks, out, aux)

    def run(self, delta, prefix, blocks, out, aux=None):
        code = self.code if aux is None else self.aux[0]
        return self._launch(code, dynamic_view(delta), prefix, blocks, out, aux)

    def _launch(self, code, source, prefix, blocks, out, aux=None):
        rows = (prefix, blocks, out) + (() if aux is None else (aux,))
        prefix, blocks, view, *aux = map(dynamic_view, rows)
        args = *self.views, view, *self.eps, *aux
        code(source, prefix, blocks, *args, stream(self.device))
        return out
