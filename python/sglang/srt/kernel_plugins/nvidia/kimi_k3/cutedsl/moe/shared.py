# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""Fused shared expert for B>8: gate/up, SiTU, down and the routed tail's reduce-scatter
push in one persistent tcgen05 kernel, synchronized by DSMEM and one global-memory
barrier instead of kernel bounds.

120 CTAs of 160 threads in clusters of ks = 120 / tiles (TP8 5, TP16 10). Loader warps
0-3 feed a 128B-swizzled ring of W [64, 64] (TMA) and x [N, 64] (cp.async) K blocks to
warp 4, which issues each block's four M=64 kind::f16 MMAs warp-uniformly
(``mma_bf16_block``). Gate/up M-tile m (32 gate rows and the matching 32 up rows)
belongs to cluster m, whose CTAs split its K. Each CTA pushes its FP32 partial rows
over DSMEM to the owner of their h columns (``width`` columns per owner); an owner
thread per (column, token) sums the parts in K (rank) order, rounds gate and up to BF16
for SiTU, and the owner stores h rows as vectors. After the grid's h barrier, CTAs
below 112 run 64 down rows (the first ring's worth of weights already staged) and
peer-store them, -0 made +0, into the tail's RS mailboxes. The last CTA through the
barrier re-arms the counters."""

import functools

import cuda.bindings.driver as cuda
import cutlass as c
import cutlass.cute as cute
import torch
from cutlass import Float32, Int32, Int64
from cutlass.cute.runtime import make_fake_stream

from sglang.kernels.jit.kimi_k3.host import fake, tensor_maps
from sglang.kernels.ops.communication.kimi_k3_specialized.shared import (
    COUNTERS,
    FRONT_RING,
    SharedDown,
    SharedUmma,
)


def weight_maps(gate_up, down, front=None):
    """TMA maps (W [64, 32] boxes, 128B swizzle) of gate/up, down, then (optional) the
    MoE front GEMM's packed router / latent rows."""
    dtype = cuda.CUtensorMapDataType.CU_TENSOR_MAP_DATA_TYPE_BFLOAT16
    swizzle = cuda.CUtensorMapSwizzle.CU_TENSOR_MAP_SWIZZLE_128B
    l2 = cuda.CUtensorMapL2promotion.CU_TENSOR_MAP_L2_PROMOTION_L2_256B
    weights = (gate_up, down) if front is None else (gate_up, down, front)
    specs = [
        (w.data_ptr(), dtype, (w.shape[1], w.shape[0]), (2 * w.stride(0),))
        + ((64, 32), swizzle, l2)
        for w in weights
    ]
    return tensor_maps(specs, gate_up.device)


@functools.cache
def compile_shared(tp, batch, device, gate=False, front=None):
    """Arguments: x, ``weight_maps``, h, counters, RS peers, parity, rank, capacity,
    beta, linear beta. ``gate``: the split variant's gate/up + SiTU half; ``front``
    (rows, split, valid, mailbox row): with the MoE front GEMM's rows, whose results
    go to the front mailbox (more arguments: multicast address, logit slot, latent
    base; ``weight_maps`` with the front's)."""
    maps = fake((256 if front is None else 384,), None, c.Uint8, 64)
    args: tuple = (fake((batch, 7168)), maps, fake((batch, 6144 // tp)))
    args += (fake((COUNTERS,), None, Int32), fake((tp,), None, Int64))
    args += (Int32(0), Int32(0), Int32(0), Float32(4), Float32(25))
    if front is not None:
        args += (Int64(0), Int32(0), Int32(0))
    ring = 80 * 1024 if front is None else FRONT_RING[tp]
    kernel = SharedUmma(tp, batch, ring, gate, front)
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())


@functools.cache
def compile_down(tp, batch, device, sms):
    """Arguments: h, ``weight_maps``, RS peers, parity, rank, capacity."""
    args: tuple = (fake((batch, 6144 // tp)), fake((256,), None, c.Uint8, 64))
    args += (fake((tp,), None, Int64), Int32(0), Int32(0), Int32(0))
    kernel = SharedDown(tp, batch, sms)
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())


def scratch(device):
    """Model-owned fused shared-expert counters (zeroed; re-armed by each launch)."""
    return torch.zeros(COUNTERS, dtype=torch.int32, device=device)
