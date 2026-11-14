# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
"""MoE front: the weight-stationary [router shard; down shard] GEMM publishes logits
and the latent shard into a multicast mailbox row; the routing kernel polls all 896
logits, ranks the top16 (sigmoid + bias, renormalized) and Lamport-copies the latent."""

import functools

import cutlass as c
import cutlass.cute as cute
import torch
from cutlass import Int32
from cutlass.cute.runtime import make_fake_stream

from sglang.kernels.jit.kimi_k3.host import fake, stream
from sglang.kernels.jit.kimi_k3.policy import batch_table
from sglang.kernels.jit.kimi_k3.tensor_adapter import to_cute
from sglang.kernels.ops.communication.kimi_k3_specialized.experts import PLAN_BYTES
from sglang.kernels.ops.communication.kimi_k3_specialized.ws_gemm import WsGemm
from sglang.kernels.ops.moe.kimi_k3_specialized.front import (
    FRONT,
    FUSED,
    MERGED,
    SHARED_TACTICS,
    SPLIT,
    SPLITS,
    WIDTH,
    RouteSitu,
)
from sglang.srt.kernel_plugins.nvidia.kimi_k3.cutedsl.gemm.projection import (
    PreparedProjection,
)
from sglang.srt.kernel_plugins.nvidia.kimi_k3.cutedsl.moe.shared import (
    compile_down,
    compile_shared,
    weight_maps,
)
from sglang.srt.kernel_plugins.nvidia.kimi_k3.cutedsl.moe.tail_runtime import CAP


@functools.cache
def front_table(n):
    return batch_table(FRONT.get(n, ""))


def front_ks(batch, n):
    return front_table(n).get(batch, (4,))[0]


def front_geometry(batch, n, ks, kb=None, tg=None, rt=None, ch=None):
    """(rt, kq, staged, tg, ch) of the front GEMM over ``n`` rows. Unset arguments come
    from the table when ``ks`` is its K-split, else from the single pass; token groups
    whose x slice does not fit in SMEM fall back to the single pass."""
    entry = front_table(n).get(batch, ())
    table = entry[1:] if entry[:1] == (ks,) else (14, 0, 1, 0)
    kb, tg, rt, ch = (t if v is None else v for v, t in zip((kb, tg, rt, ch), table))
    kq = min(16, 224 // (ks * kb))
    kb, nt = 224 // (ks * kq), -(-batch // 8)
    staged = max(kb - 8, 0) if kq == 16 else 0
    used = (
        rt * kq * (nt * 544 + max(staged, 1) * 1024) + -(-nt * rt // ks) * ks * kq * 512
    )
    if nt * 8 * (7168 // ks * 2 + 64) > 220 * 1024 - used:
        tg = 0
    return rt, kq, staged, tg, ch


@functools.cache
def compile_route(
    batch,
    shared,
    renorm,
    roles,
    device,
    polled=False,
    plan=False,
    deadline_ns=0,
):
    kernel = RouteSitu(
        batch,
        shared,
        renorm,
        batch <= 2,
        roles,
        polled,
        plan,
        deadline_ns,
    )
    f32 = c.Float32
    args: tuple = (fake((batch, 896), None, f32), fake((896,), None, f32))
    args += (fake((batch, 16), None, c.Int32), fake((batch, 16)), fake((1, 64, WIDTH)))
    args += (fake((batch, 3584)), fake((batch, 2 * shared), None, f32))
    args += (fake((batch, shared)), *(f32(v) for v in (1, 4, 25)))
    args += (fake((PLAN_BYTES,), None, c.Uint8),) * plan + (make_fake_stream(),)
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args)


@functools.cache
def compile_publish(n, split, valid, batch, device, ks, geometry=None):
    """Front GEMM that only publishes; routing then runs in the polled kernel."""
    rt, kq, staged, tg, ch = geometry or front_geometry(batch, n, ks)
    kernel = WsGemm(n, 7168, batch, rt, kq, staged, split, valid, ks, tg, ch)
    args: tuple = (fake((batch, 7168)), fake((n, 7168)), fake((1, 64, WIDTH)))
    args += (c.Int64(0), c.Int32(0), c.Int32(0), fake((batch, 896), None, c.Float32))
    with torch.accelerator.device_index(device):
        return cute.compile(kernel, *args, make_fake_stream())


class SharedFront:
    def __init__(
        self, moe, packed, box, storage, batches, beta, linear_beta, plan=None,
        counters=None, q=1, deadline_ns=0,
    ):  # fmt: skip
        """``plan``: the experts' plan buffer, written by the routing kernel at B>8;
        ``counters``: the fused shared expert's model-owned barrier counters; ``q``: the
        deployment's verify length (its SPLITS entries)."""
        experts = moe.shared_experts
        gate_up, down = experts.gate_up_proj.weight, experts.down_proj.weight
        self.shared = shared = gate_up.shape[0] // 2
        tp = 6144 // shared
        shard, valid = 3584 // tp, 896 // tp
        self.device, self.packed, self.storage = packed.device, to_cute(packed), storage
        gu, dn = map(batch_table, SHARED_TACTICS[shared])
        self.gate_up = PreparedProjection(
            gate_up, {b: gu[b] for b in batches}, output_dtype=torch.float32
        ).prepare()
        self.down = PreparedProjection(down, {b: dn[b] for b in batches}).prepare()
        scale, renorm = moe.routed_scaling_factor, moe.moe_renormalize
        self.scalars = tuple(map(c.Float32, (scale, beta, linear_beta)))
        self.box: tuple = (to_cute(box.buffer), c.Int64(box.mc))
        self.box += (c.Int32(box.rank * valid), c.Int32(box.rank * shard))
        bias = to_cute(moe.gate.e_score_correction_bias)
        dev, n = packed.device.index, packed.shape[0]
        fused = batch_table(FUSED)
        fused = {b for b in batches if b > 8 and counters is not None and fused.get(b)}
        verify = q > 1
        split = batch_table(f"{SPLIT},{SPLITS[tp][verify]}")
        self.split = {b: split[b] for b in fused if split.get(b)}  # down SMs
        if fused:  # TMA maps and counters of the fused kernel
            self.maps = weight_maps(gate_up, down)
            self.fused = to_cute(self.maps, 64), to_cute(counters)
        merged = batch_table(MERGED[tp])
        self.merged = {b for b in self.split if merged.get(b)}
        if self.merged:  # gate/up, down and front maps; unused RS arguments
            self.merged_maps = weight_maps(gate_up, down, front=packed)
            # Kept alive here: a CuTe view does not hold its tensor.
            self.unused = torch.zeros(tp, dtype=torch.int64, device=packed.device)
            self.merged_args = to_cute(self.merged_maps, 64), to_cute(self.unused)
            geometry = (n, n - shard, valid, box.buffer.shape[-1])
        self.plans: dict[int, tuple] = {}
        for b in batches:
            views = tuple(t[:b] for t in storage)
            logits, gate, ids, weights, h, _, latent = map(to_cute, views)
            front = compile_publish(n, n - shard, valid, b, dev, front_ks(b, n))
            planned = plan is not None and b > 8
            roles = (True, True, False)
            route = compile_route(
                b, shared, renorm, roles, dev, True, planned, deadline_ns
            )
            situ = compile_route(
                b,
                shared,
                renorm,
                (False, False, True),
                dev,
                deadline_ns=deadline_ns,
            )
            args = (logits, bias, ids, weights, self.box[0], latent, gate, h)
            extra = (to_cute(plan),) if planned else ()
            code = None
            if b in self.merged:
                gate = compile_shared(tp, b, dev, True, geometry)
                code = gate, compile_down(tp, b, dev, self.split[b])
            elif b in self.split:
                gate = compile_shared(tp, b, dev, gate=True)
                code = gate, compile_down(tp, b, dev, self.split[b])
            elif b in fused:
                code = compile_shared(tp, b, dev)
            self.plans[b] = views, front, route, situ, args + self.scalars, extra, code

    def route(self, x):
        """Router logits + published down shard, then top16 and the latent copy (and
        at B>8 the experts' MXFP8 plan). Merged batches publish them from the gate/up
        kernel, which also leaves h for ``shared_expert``'s down rows."""
        views, front, route, _, args, extra, code = self.plans[x.shape[0]]
        s = stream(self.device)
        if x.shape[0] in self.merged:
            maps, unused = self.merged_args
            work = (to_cute(x), maps, args[7], self.fused[1], unused)
            work += (Int32(0), Int32(0), Int32(CAP), *self.scalars[1:], *self.box[1:])
            code[0](*work, s)
        else:
            front(to_cute(x), self.packed, *self.box, args[0], s)
        route(*args, *extra, s)
        return views[2], views[3], views[6]

    def merged_batch(self, batch):
        return batch in self.merged

    def planned(self, batch):
        return bool(self.plans[batch][5])

    def shared_expert(self, x, tail):
        """The shared expert's rows into the tail's RS mailboxes (``tail.push``); the
        fused kernel peer-stores them itself and leaves the returned rows unwritten.
        Returns (rows, the split's down launch to run beside the experts, or None)."""
        views, _, _, situ, args, _, code = self.plans[x.shape[0]]
        s = stream(self.device)
        if code is not None:
            maps, counters = self.fused
            mailbox = (tail.peers, *tail.ring, Int32(CAP))
            front = (to_cute(x), maps, args[7], counters, *mailbox, *self.scalars[1:])
            if x.shape[0] not in self.split:
                code(*front, s)
                return views[5], None
            gate, down = code
            if x.shape[0] not in self.merged:  # merged: h comes from ``route``
                gate(*front, s)
            return views[5], lambda: down(args[7], maps, *mailbox, stream(self.device))
        self.gate_up.run(x, views[1])
        situ(*args, s)
        shared = self.down.run(views[4], views[5])
        tail.push(shared)
        return shared, None
