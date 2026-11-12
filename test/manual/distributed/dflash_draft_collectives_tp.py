# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
"""Bitwise checks, deadline checks and timing of the DFLASH draft collectives on one node, under torchrun.

    python3 -m torch.distributed.run --nproc-per-node 4 dflash_draft_collectives_tp.py --mode check
    python3 -m torch.distributed.run --nproc-per-node 4 dflash_draft_collectives_tp.py --mode deadline --op ar
    python3 -m torch.distributed.run --nproc-per-node 4 dflash_draft_collectives_tp.py --mode bench --out DIR

``--rpg 4`` runs 16 ranks on 4 GPUs (4 per process). The references are independent of the code under test: every
process generates every rank's inputs from one seed and sums them in torch (rank order, FP32, one rounding, after
the shipped 1shot_push's zero-word rule) or concatenates them. Real-rank runs also compare with the shipped
1shot_push and the group's all-gather. Output: one PASS/FAIL line per case, then ``TP_RESULT rank=R failures=N``.

``--mode deadline`` runs one call in which the last process starts late (or never): every process waiting on it must
fail with a CUDA error (the kernel's trap) after the deadline, never return. A trap kills the CUDA context, so it is
one case per torchrun.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import time
import zlib
from pathlib import Path

import torch
import torch.distributed as dist

import sglang.kernels.jit as jit_pkg
import sglang.kernels.ops.communication.draft_collectives as draft_collectives_ops
import sglang.srt.distributed.parallel_state as ps
from sglang.kernels.ops.communication.all_reduce import AllReduceAlgo
from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
    CustomAllReduceV2,
)
from sglang.srt.runtime_context import get_parallel
from sglang.srt.speculative.dflash_draft_collectives import (
    DraftAllReduce,
    DraftTopkGather,
    DraftTopkGathers,
)

# K3.1 DFlash2 drafter at TP16, one request: 8 rows x 7168 all-reduces, 7 rows x top-16 gathers, vocab 163840.
HIDDEN = 7168
AR_ROWS = 8
G_ROWS = 7
TOPK = 16
VOCAB = 163840
# K31_DEADLINE_NS: a longer wait deadline for runs under compute-sanitizer, whose slowdown can outlast 2 s.
DEADLINE_NS = int(os.environ.get("K31_DEADLINE_NS", 2_000_000_000))
POISON_BF16 = 0x7FC1


def emit(line: str) -> None:
    # One write per line: ranks share the log, and unbuffered print splits text and newline.
    os.write(1, (line + "\n").encode())


class Ctx:
    def __init__(self) -> None:
        self.local_rank = int(os.environ["LOCAL_RANK"])
        self.world = int(os.environ["WORLD_SIZE"])
        self.device = torch.device("cuda", self.local_rank)
        torch.cuda.set_device(self.device)
        dist.init_process_group(backend="gloo")
        self.coord = ps.init_world_group(
            ranks=list(range(self.world)), local_rank=self.local_rank, backend="nccl"
        )
        ps._WORLD = self.coord
        get_parallel().override_permanently(world_group=self.coord)
        self.nccl = dist.new_group(backend="nccl", device_id=self.device)
        self.rank = self.coord.rank_in_group
        self.failures = 0
        self._shipped = None

    def shipped(self) -> CustomAllReduceV2:
        if self._shipped is None:
            comm = CustomAllReduceV2(
                self.coord.cpu_group,
                self.device,
                max_pull_size=0,
                max_push_size=4 * 1024 * 1024,
            )
            assert not comm.disabled, "shipped CustomAllReduceV2 is disabled here"
            comm.override_algo = AllReduceAlgo.ONE_SHOT_PUSH
            self._shipped = comm
        return self._shipped

    def any_rank(self, flag: bool) -> bool:
        t = torch.tensor([1 if flag else 0], dtype=torch.int32, device=self.device)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self.nccl)
        return bool(t.item())

    def report(self, case: str, bad: int, checked: int, iters: int) -> None:
        ok = bad == 0 and checked > 0
        word = "PASS" if ok else "FAIL"
        emit(f"{word} {case} rank={self.rank} mismatched={bad} checked={checked} iters={iters}")
        if not ok:
            self.failures += 1


def mutation_banner(rank: int) -> None:
    name = os.environ.get("K31_MUTATION")
    if not name:
        return
    path = Path(jit_pkg.__file__).parent / "csrc/distributed/draft_collectives.cuh"
    md5 = hashlib.md5(path.read_bytes()).hexdigest()
    if md5 == os.environ.get("K31_MUT_EXPECT_MD5") and md5 != os.environ.get("K31_BASE_MD5"):
        emit(f"K31_MUTATION {name} applied (rank {rank}, {path} md5 {md5})")
    else:
        emit(f"K31_MUTATION {name} NOT applied (rank {rank}, md5 {md5})")


# ---------------------------------------------------------------- references


def hosted(ctx: Ctx, rpg: int) -> list:
    """Global rank of each partial this process hosts (rank = gpu + q * num_gpus, as the kernels map them)."""
    return [ctx.rank + q * ctx.world for q in range(rpg)]


def ref_all_reduce(parts: torch.Tensor) -> torch.Tensor:
    """[S, rows, hidden] BF16 -> rank-order FP32 sum rounded once; zero words first become 0x00008000 (1shot_push)."""
    words = parts.contiguous().view(torch.int32)
    words = torch.where(words == 0, torch.full_like(words, 0x8000), words)
    vals = words.view(torch.bfloat16).float()
    acc = vals[0].clone()
    for r in range(1, vals.shape[0]):
        acc = acc + vals[r]
    return acc.to(torch.bfloat16)


def ar_parts(kind: str, num_slots: int, gen: torch.Generator, device) -> torch.Tensor:
    shape = (num_slots, AR_ROWS, HIDDEN)
    x = torch.randn(shape, generator=gen, device=device)
    if kind == "random":
        exp = torch.randint(-6, 7, shape, generator=gen, device=device).float()
        return (x * torch.exp2(exp)).to(torch.bfloat16)
    if kind == "cancel":
        # Rank 0 adds +L and the last rank -L with L ~ 2^16: the FP32 result depends on the summation order.
        big = (1 + torch.rand((AR_ROWS, HIDDEN), generator=gen, device=device)) * 65536.0
        x[0] += big
        x[-1] -= big
        return x.to(torch.bfloat16)
    if kind == "zeros":
        x = x.to(torch.bfloat16)
        words = x.view(torch.int32)
        all_zero = torch.rand((AR_ROWS, HIDDEN // 2), generator=gen, device=device) < 0.15
        some_zero = torch.rand(words.shape, generator=gen, device=device) < 0.2
        neg_zero = torch.rand(words.shape, generator=gen, device=device) < 0.05
        words[some_zero] = 0
        words[neg_zero] = torch.tensor(0x80008000 - (1 << 32), dtype=torch.int32, device=device)
        words[:, all_zero] = 0
        return x
    raise ValueError(kind)


def g_parts(num_slots: int, gen: torch.Generator, device, rows: int = G_ROWS):
    """Per-rank top-k: fp32 values (with -inf and ties) and global ids inside each rank's vocab shard."""
    shape = (num_slots, rows, TOPK)
    vals = torch.randn(shape, generator=gen, device=device)
    vals = torch.where(torch.rand(shape, generator=gen, device=device) < 0.05, float("-inf"), vals)
    vals = torch.where(torch.rand(shape, generator=gen, device=device) < 0.1, vals.round(), vals)
    shard = VOCAB // num_slots
    local = torch.randint(0, shard, shape, generator=gen, device=device)
    ids = local + torch.arange(num_slots, device=device).view(-1, 1, 1) * shard
    return vals.contiguous(), ids.contiguous()


def ref_gather(parts: torch.Tensor) -> torch.Tensor:
    """[S, rows, k] -> [rows, S * k]: all-gather along the last dim in rank order."""
    return parts.permute(1, 0, 2).reshape(parts.shape[1], -1).contiguous()


def bits_mismatch(a: torch.Tensor, b: torch.Tensor) -> int:
    view = {torch.bfloat16: torch.int16, torch.float32: torch.int32, torch.int64: torch.int64}[a.dtype]
    assert a.shape == b.shape and a.dtype == b.dtype, (a.shape, b.shape, a.dtype, b.dtype)
    return int((a.contiguous().view(view) != b.contiguous().view(view)).sum().item())


def sleep_cycles(ctx: Ctx) -> int:
    # Process-dependent delay before each call (staggered GPU-cycle sleeps): peers poll a late rank's slot early.
    return (ctx.rank * 3 % ctx.world) * 50_000


# ---------------------------------------------------------------- all-reduce checks


def check_ar(ctx: Ctx, kind: str, rpg: int, mode: str, iters: int, calls: int = 4) -> None:
    case = f"ar_{kind}_{mode}_rpg{rpg}"
    num_slots = ctx.world * rpg
    ar = DraftAllReduce(
        group=ctx.coord, rows=AR_ROWS, hidden_size=HIDDEN, deadline_ns=DEADLINE_NS, ranks_per_gpu=rpg
    )
    mine = hosted(ctx, rpg)
    # Same seed on every process: each one generates every rank's inputs.
    gen = torch.Generator(device=ctx.device).manual_seed(zlib.crc32(case.encode()))
    static_in = torch.zeros((calls, rpg, AR_ROWS, HIDDEN), dtype=torch.bfloat16, device=ctx.device)
    ar.reduce_partials(static_in[0])  # eager build of the mailboxes
    torch.cuda.synchronize()
    graph, outs = None, None
    if mode != "eager":
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs = []
            for c in range(calls):
                if mode == "skew" and sleep_cycles(ctx):
                    torch.cuda._sleep(sleep_cycles(ctx))
                outs.append(ar.reduce_partials(static_in[c]))
        torch.cuda.synchronize()
    bad = checked = 0
    it = 0
    for it in range(iters):
        parts = [ar_parts(kind, num_slots, gen, ctx.device) for _ in range(calls)]
        refs = [ref_all_reduce(p) for p in parts]
        for c in range(calls):
            static_in[c].copy_(parts[c][mine])
        if graph is not None:
            for o in outs:
                o.view(torch.int16).fill_(POISON_BF16)
            graph.replay()
            got = outs
        else:
            got = [ar.reduce_partials(static_in[c]) for c in range(calls)]
        torch.cuda.synchronize()
        for c in range(calls):
            bad += bits_mismatch(got[c], refs[c])
            checked += refs[c].numel()
            if kind == "zeros" and rpg == 1 and mode == "eager":
                shipped = ctx.shipped().custom_all_reduce(static_in[c][0])
                torch.cuda.synchronize()
                bad += bits_mismatch(got[c], shipped)
                checked += shipped.numel()
        if ctx.any_rank(bad > 0):
            break
    ctx.report(case, bad, checked, it + 1)


def check_ar_vs_shipped(ctx: Ctx, kind: str, iters: int) -> None:
    """Real ranks only: the new all-reduce is bitwise equal to the shipped 1shot_push on the same inputs."""
    case = f"ar_vs_shipped_{kind}"
    ar = DraftAllReduce(group=ctx.coord, rows=AR_ROWS, hidden_size=HIDDEN, deadline_ns=DEADLINE_NS)
    gen = torch.Generator(device=ctx.device).manual_seed(77)
    bad = checked = 0
    it = 0
    for it in range(iters):
        x = ar_parts(kind, ctx.world, gen, ctx.device)[ctx.rank].contiguous()
        got = ar(x)
        shipped = ctx.shipped().custom_all_reduce(x)
        torch.cuda.synchronize()
        bad += bits_mismatch(got, shipped)
        checked += got.numel()
        if ctx.any_rank(bad > 0):
            break
    ctx.report(case, bad, checked, it + 1)


# ---------------------------------------------------------------- gather checks


def check_gather(ctx: Ctx, rpg: int, mode: str, iters: int, calls: int = 1) -> None:
    case = f"gather_{mode}_rpg{rpg}"
    num_slots = ctx.world * rpg
    tg = DraftTopkGather(
        rows=G_ROWS, k=TOPK, vocab_size=VOCAB, deadline_ns=DEADLINE_NS, ranks_per_gpu=rpg
    )
    mine = hosted(ctx, rpg)
    gen = torch.Generator(device=ctx.device).manual_seed(4242 + rpg)
    s_vals = torch.zeros((calls, rpg, G_ROWS, TOPK), dtype=torch.float32, device=ctx.device)
    s_ids = torch.zeros((calls, rpg, G_ROWS, TOPK), dtype=torch.int64, device=ctx.device)
    tg.gather_partials(ctx.coord, s_vals[0], s_ids[0])  # eager build
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outs = []
        for c in range(calls):
            if mode == "skew" and sleep_cycles(ctx):
                torch.cuda._sleep(sleep_cycles(ctx))
            gv, gi = tg.gather_partials(ctx.coord, s_vals[c], s_ids[c])
            # candidate_topk's tail on the gathered candidates
            top_vals, sel = torch.topk(gv, TOPK, dim=-1)
            top_ids = torch.gather(gi, -1, sel)
            outs.append((gv, gi, top_vals, top_ids))
    torch.cuda.synchronize()
    bad = checked = 0
    it = 0
    for it in range(iters):
        data = [g_parts(num_slots, gen, ctx.device) for _ in range(calls)]
        for c in range(calls):
            s_vals[c].copy_(data[c][0][mine])
            s_ids[c].copy_(data[c][1][mine])
        for o in outs:
            for t in o:
                t.view(torch.uint8).fill_(0xA5)
        graph.replay()
        torch.cuda.synchronize()
        for c in range(calls):
            rv, ri = ref_gather(data[c][0]), ref_gather(data[c][1])
            ref_top_vals, ref_sel = torch.topk(rv, TOPK, dim=-1)
            ref_top_ids = torch.gather(ri, -1, ref_sel)
            refs = [rv, ri, ref_top_vals, ref_top_ids]
            if rpg == 1 and c == 0 and it < 50:
                # The shipped path: GroupCoordinator.all_gather along the last dim.
                refs[0] = ctx.coord.all_gather(data[c][0][ctx.rank].contiguous(), dim=-1)
                refs[1] = ctx.coord.all_gather(data[c][1][ctx.rank].contiguous(), dim=-1)
                bad += bits_mismatch(refs[0], rv) + bits_mismatch(refs[1], ri)
            for got_t, ref_t in zip(outs[c], refs):
                bad += bits_mismatch(got_t, ref_t)
                checked += ref_t.numel()
        if ctx.any_rank(bad > 0):
            break
    ctx.report(case, bad, checked, it + 1)


# ---------------------------------------------------------------- packed gather at several batch sizes

GB_MAX = 8
GB_GRAPH_NS = (1, 2, 5, 8)


def _count_packed(gs: DraftTopkGathers) -> dict:
    """Counts, per row count, the calls that took the packed path (DraftTopkGather.__call__ goes through
    gather_partials, so wrapping it counts both entry points)."""
    counts = {rows: 0 for rows in gs.by_rows}
    for rows, gather in gs.by_rows.items():

        def counted(group, vals, ids, _rows=rows, _inner=gather.gather_partials):
            counts[_rows] += 1
            return _inner(group, vals, ids)

        gather.gather_partials = counted
    return counts


def check_gathers(ctx: Ctx, rpg: int, calls: int, replays: int) -> None:
    """DraftTopkGathers (SGLANG_OPT_DFLASH_PACKED_TOPK_GATHER_MAX_BS): eager calls over a random sequence of batch
    sizes 1..GB_MAX, then one CUDA graph per batch size in GB_GRAPH_NS replayed in a random order with new inputs.
    Every output must equal the rank-order concatenation bit for bit, every batch size must take its own packed
    gather (counted), and a row count outside the set must take the all-gather and stay exact."""
    num_slots = ctx.world * rpg
    gs = DraftTopkGathers(
        rows=G_ROWS, max_requests=GB_MAX, k=TOPK, vocab_size=VOCAB, deadline_ns=DEADLINE_NS, ranks_per_gpu=rpg
    )
    counts = _count_packed(gs)
    mine = hosted(ctx, rpg)
    gen = torch.Generator(device=ctx.device).manual_seed(5150 + rpg)
    # The same seed on every process: every rank runs the same batch-size sequence.
    order = torch.Generator().manual_seed(99 + rpg)

    def call(vals, ids):
        if rpg == 1:
            return gs(ctx.coord, vals[0], ids[0])
        return gs.gather_partials(ctx.coord, vals, ids)

    case = f"gathers_eager_rpg{rpg}"
    bad = checked = 0
    seen = set()
    it = 0
    for it in range(calls):
        n = int(torch.randint(1, GB_MAX + 1, (1,), generator=order).item())
        vals, ids = g_parts(num_slots, gen, ctx.device, rows=G_ROWS * n)
        gv, gi = call(vals[mine].contiguous(), ids[mine].contiguous())
        rv, ri = ref_gather(vals), ref_gather(ids)
        if rpg == 1 and it < 20:
            # The shipped path: GroupCoordinator.all_gather along the last dim.
            bad += bits_mismatch(ctx.coord.all_gather(vals[ctx.rank].contiguous(), dim=-1), rv)
            bad += bits_mismatch(ctx.coord.all_gather(ids[ctx.rank].contiguous(), dim=-1), ri)
        torch.cuda.synchronize()
        bad += bits_mismatch(gv, rv) + bits_mismatch(gi, ri)
        checked += rv.numel() + ri.numel()
        seen.add(n)
        if it % 64 == 63 and ctx.any_rank(bad > 0):
            break
    ctx.report(case, bad, checked, it + 1)

    emit(f"GATHERS_COUNTS rpg={rpg} rank={ctx.rank} " + " ".join(f"bs{r // G_ROWS}:{c}" for r, c in sorted(counts.items())))
    missing = [n for n in sorted(seen) if counts.get(G_ROWS * n, 0) == 0]
    if missing:
        emit(f"GATHERS_NOT_PACKED rpg={rpg} rank={ctx.rank} batch_sizes={missing}")
    ctx.report(f"gathers_engaged_rpg{rpg}", len(missing), len(seen), 1)

    if rpg == 1:
        before = sum(counts.values())
        vals, ids = g_parts(num_slots, gen, ctx.device, rows=G_ROWS * (GB_MAX + 1))
        gv, gi = call(vals[mine].contiguous(), ids[mine].contiguous())
        torch.cuda.synchronize()
        bad = bits_mismatch(gv, ref_gather(vals)) + bits_mismatch(gi, ref_gather(ids)) + sum(counts.values()) - before
        ctx.report(f"gathers_fallback_rpg{rpg}", bad, gv.numel() + gi.numel(), 1)

    case = f"gathers_graph_rpg{rpg}"
    static, outs, graphs = {}, {}, {}
    for n in GB_GRAPH_NS:
        sv = torch.zeros((rpg, G_ROWS * n, TOPK), dtype=torch.float32, device=ctx.device)
        si = torch.zeros((rpg, G_ROWS * n, TOPK), dtype=torch.int64, device=ctx.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            outs[n] = call(sv, si)
        static[n], graphs[n] = (sv, si), graph
    torch.cuda.synchronize()
    bad = checked = 0
    it = 0
    for it in range(replays):
        n = GB_GRAPH_NS[int(torch.randint(0, len(GB_GRAPH_NS), (1,), generator=order).item())]
        vals, ids = g_parts(num_slots, gen, ctx.device, rows=G_ROWS * n)
        static[n][0].copy_(vals[mine])
        static[n][1].copy_(ids[mine])
        for t in outs[n]:
            t.view(torch.uint8).fill_(0xA5)
        graphs[n].replay()
        torch.cuda.synchronize()
        bad += bits_mismatch(outs[n][0], ref_gather(vals)) + bits_mismatch(outs[n][1], ref_gather(ids))
        checked += 2 * outs[n][0].numel()
        if it % 64 == 63 and ctx.any_rank(bad > 0):
            break
    ctx.report(case, bad, checked, it + 1)


# ---------------------------------------------------------------- deadline: timeouts fail closed


def run_deadline(ctx: Ctx, op: str, rpg: int, graph: bool, absent: bool, late_s: float, deadline_ms: float) -> None:
    """One on-time call (the deadline must not fire), then one call that the last process starts `late_s` late (or never).

    The waiting processes must fail that call with a CUDA error no sooner than the deadline; a returned call is a
    published partial result. The late process may only return the reference output, or fail too. No collective runs
    after the late call: each process reports, waits for the late one to finish, and exits."""
    case = f"deadline_{op}_{'graph' if graph else 'eager'}_rpg{rpg}_{'absent' if absent else 'late'}"
    late = ctx.world - 1
    num_slots = ctx.world * rpg
    mine = hosted(ctx, rpg)
    deadline_ns = int(deadline_ms * 1e6)
    gen = torch.Generator(device=ctx.device).manual_seed(zlib.crc32(case.encode()))
    if op == "ar":
        coll = DraftAllReduce(
            group=ctx.coord, rows=AR_ROWS, hidden_size=HIDDEN, deadline_ns=deadline_ns, ranks_per_gpu=rpg
        )
        x = torch.zeros((rpg, AR_ROWS, HIDDEN), dtype=torch.bfloat16, device=ctx.device)

        def call():
            return [coll.reduce_partials(x)]

        def draw():
            parts = ar_parts("random", num_slots, gen, ctx.device)
            x.copy_(parts[mine])
            return [ref_all_reduce(parts)]

    else:
        coll = DraftTopkGather(rows=G_ROWS, k=TOPK, vocab_size=VOCAB, deadline_ns=deadline_ns, ranks_per_gpu=rpg)
        v = torch.zeros((rpg, G_ROWS, TOPK), dtype=torch.float32, device=ctx.device)
        i = torch.zeros((rpg, G_ROWS, TOPK), dtype=torch.int64, device=ctx.device)

        def call():
            return list(coll.gather_partials(ctx.coord, v, i))

        def draw():
            vals, ids = g_parts(num_slots, gen, ctx.device)
            v.copy_(vals[mine])
            i.copy_(ids[mine])
            return [ref_gather(vals), ref_gather(ids)]

    def checked_call(fn) -> int:
        refs = draw()
        outs = fn()
        torch.cuda.synchronize()
        return sum(bits_mismatch(o, r) for o, r in zip(outs, refs))

    # JIT-load first: compile skew between processes must not reach the first deadline-bound call.
    draft_collectives_ops._module(num_slots, rpg)
    dist.barrier(group=ctx.coord.cpu_group)
    bad = checked_call(call)  # eager build of the mailboxes, every process on time
    run = call
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = call()
        torch.cuda.synchronize()
        dist.barrier(group=ctx.coord.cpu_group)

        def run():
            g.replay()
            return outs

        bad += checked_call(run)
    if bad:
        emit(f"FAIL {case} rank={ctx.rank} outcome=ONTIME_MISMATCH mismatched={bad}")
    refs = draw()
    dist.barrier(group=ctx.coord.cpu_group)
    me_late = ctx.rank == late
    if me_late and absent:
        outcome, elapsed_ms, got = "ABSENT", 0.0, None
    else:
        if me_late:
            time.sleep(late_s)
        t0 = time.monotonic()
        try:
            got = run()
            torch.cuda.synchronize()
            outcome = "RETURNED"
        except Exception as e:  # the trap surfaces as a CUDA error on the first sync
            msg = f"{type(e).__name__}: {str(e).splitlines()[0][:80]}"
            outcome, got = (f"TRAPPED({msg})" if "CUDA error" in msg else f"RAISED({msg})"), None
        elapsed_ms = (time.monotonic() - t0) * 1e3
    if me_late:
        ok = got is None or sum(bits_mismatch(o, r) for o, r in zip(got, refs)) == 0
    else:
        ok = outcome.startswith("TRAPPED") and elapsed_ms >= deadline_ms
    failures = int(bad > 0) + int(not ok)
    role = "late" if me_late else "waiting"
    emit(
        f"{'PASS' if ok else 'FAIL'} {case} rank={ctx.rank} role={role} outcome={outcome} "
        f"elapsed_ms={elapsed_ms:.1f} deadline_ms={deadline_ms} late_s={late_s}"
    )
    emit(f"TP_RESULT rank={ctx.rank} failures={failures}")
    if not me_late:
        time.sleep(late_s + 3.0)  # keep this process's mailbox mapped until the late process is done
    os._exit(0 if failures == 0 else 1)


# ---------------------------------------------------------------- timing


def time_graph(ctx: Ctx, fn, calls: int, reps: int, lead_in: int, warm: int = 10) -> list:
    """Per-call times of a graph of `calls` back-to-back calls. With lead_in > 0, a graph of lead_in calls replays first
    and the clock starts after it: the collectives in it line the ranks up, so host start skew stays out of the time."""
    fn()
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(calls):
            fn()
    lead = None
    if lead_in > 0:
        lead = torch.cuda.CUDAGraph()
        with torch.cuda.graph(lead):
            for _ in range(lead_in):
                fn()
    torch.cuda.synchronize()
    for _ in range(warm):
        if lead is not None:
            lead.replay()
        graph.replay()
    torch.cuda.synchronize()
    per_call = []
    for _ in range(reps):
        dist.barrier(group=ctx.coord.cpu_group)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        if lead is not None:
            lead.replay()
        start.record()
        graph.replay()
        end.record()
        end.synchronize()
        per_call.append(start.elapsed_time(end) * 1000.0 / calls)
    return per_call


def bench(ctx: Ctx, out_dir: str, reps: int, calls: int, order: str, lead_in: int) -> None:
    results = []

    def arm(name: str, rpg: int, fn) -> None:
        t = time_graph(ctx, fn, calls, reps, lead_in)
        t.sort()
        row = dict(
            arm=name, rpg=rpg, slots=ctx.world * rpg, rank=ctx.rank, calls=calls, reps=reps, lead_in=lead_in,
            p10=t[len(t) // 10], p50=statistics.median(t), p90=t[len(t) * 9 // 10],
        )
        results.append(row)
        emit("BENCH " + json.dumps(row))

    ars = {}

    def ar_fn(rpg: int):
        if rpg not in ars:
            ars[rpg] = (
                DraftAllReduce(
                    group=ctx.coord, rows=AR_ROWS, hidden_size=HIDDEN, deadline_ns=DEADLINE_NS, ranks_per_gpu=rpg
                ),
                torch.randn((rpg, AR_ROWS, HIDDEN), device=ctx.device).to(torch.bfloat16),
            )
        ar, x = ars[rpg]
        return lambda: ar.reduce_partials(x)

    x1 = torch.randn((AR_ROWS, HIDDEN), device=ctx.device).to(torch.bfloat16)
    shipped = ctx.shipped()

    def shipped_fn():
        # A push algorithm takes no graph-registered inputs, so capture needs no shipped.capture().
        return lambda: shipped.custom_all_reduce(x1)

    tgs = {}

    def gather_fn(rpg: int):
        if rpg not in tgs:
            tgs[rpg] = (
                DraftTopkGather(rows=G_ROWS, k=TOPK, vocab_size=VOCAB, deadline_ns=DEADLINE_NS, ranks_per_gpu=rpg),
                torch.randn((rpg, G_ROWS, TOPK), device=ctx.device),
                torch.randint(0, VOCAB, (rpg, G_ROWS, TOPK), device=ctx.device),
            )
        tg, v, i = tgs[rpg]
        return lambda: tg.gather_partials(ctx.coord, v, i)

    gv = torch.randn((G_ROWS, TOPK), device=ctx.device)
    gi = torch.randint(0, VOCAB, (G_ROWS, TOPK), device=ctx.device)
    gbs = {}

    def gathers_fn(rpg: int, n: int):
        if rpg not in gbs:
            gbs[rpg] = DraftTopkGathers(
                rows=G_ROWS, max_requests=GB_MAX, k=TOPK, vocab_size=VOCAB, deadline_ns=DEADLINE_NS, ranks_per_gpu=rpg
            )
        v = torch.randn((rpg, G_ROWS * n, TOPK), device=ctx.device)
        i = torch.randint(0, VOCAB, (rpg, G_ROWS * n, TOPK), device=ctx.device)
        return lambda: gbs[rpg].gather_partials(ctx.coord, v, i)

    def nccl_gathers_fn(n: int):
        v = torch.randn((G_ROWS * n, TOPK), device=ctx.device)
        i = torch.randint(0, VOCAB, (G_ROWS * n, TOPK), device=ctx.device)
        return lambda: (ctx.coord.all_gather(v, dim=-1), ctx.coord.all_gather(i, dim=-1))

    def nccl_gather_fn():
        return lambda: (ctx.coord.all_gather(gv, dim=-1), ctx.coord.all_gather(gi, dim=-1))

    def fused_candidate_ag_fn():
        # The launch sequence of models/dflash.py's _fused_candidate_all_gather (sgl-project/sglang#48414):
        # int32 payload staging, sp_collective.all_gather_direct into a multicast symmetric buffer, two reshapes.
        # Timing control only: its output layout is not checked here.
        from sglang.kernels.ops.communication import sp_collective
        from sglang.srt.distributed.device_communicators.custom_all_reduce_v2 import (
            _allocate_symmetric_memory,
        )

        comm = CustomAllReduceV2(
            ctx.coord.cpu_group, ctx.device, max_pull_size=1 << 20, max_push_size=1 << 20
        )
        sp_collective.register_comm(comm.obj)
        tp = ctx.world
        nbytes = G_ROWS * tp * 4 * TOPK * 2
        buf, symm = _allocate_symmetric_memory(nbytes, device=ctx.device, group=ctx.coord.cpu_group)
        out = buf.view(torch.bfloat16).view(G_ROWS, tp * 4 * TOPK)
        mc = int(symm.multicast_ptr)
        assert mc != 0, "no multicast"
        gids = gi.clone()

        def fn():
            payload = torch.empty((G_ROWS, 2 * TOPK), dtype=torch.int32, device=ctx.device)
            payload[:, :TOPK].view(torch.float32).copy_(gv)
            payload[:, TOPK:].copy_(gids)
            sp_collective.all_gather_direct(
                tp, payload.view(torch.bfloat16), out, output_mc_ptr=mc,
                tuning=sp_collective.Tuning(num_blocks=1, block_size=128),
            )
            g = out.view(torch.int32).view(tp, G_ROWS, 2 * TOPK).transpose(0, 1)
            return g[..., :TOPK].view(torch.float32).reshape(G_ROWS, tp * TOPK), g[..., TOPK:].reshape(
                G_ROWS, tp * TOPK
            )

        return fn

    plan = {
        "v16": lambda: arm("pk3_ar", 4, ar_fn(4)),
        "r4": lambda: arm("pk3_ar", 1, ar_fn(1)),
        "s4": lambda: arm("shipped_1shot_push", 1, shipped_fn()),
        "gv16": lambda: arm("packed_gather", 4, gather_fn(4)),
        "gr4": lambda: arm("packed_gather", 1, gather_fn(1)),
        "gn4": lambda: arm("nccl_gather_x2", 1, nccl_gather_fn()),
        "g115": lambda: arm("fused_candidate_ag", 1, fused_candidate_ag_fn()),
    }
    for n in (2, 4, 8):
        plan[f"gbv16_{n}"] = lambda n=n: arm(f"packed_gathers_bs{n}", 4, gathers_fn(4, n))
        plan[f"gbr4_{n}"] = lambda n=n: arm(f"packed_gathers_bs{n}", 1, gathers_fn(1, n))
        plan[f"gbn4_{n}"] = lambda n=n: arm(f"nccl_gather_x2_bs{n}", 1, nccl_gathers_fn(n))
    for key in order.split(","):
        try:
            plan[key]()
        except Exception as e:  # an informative arm that cannot be captured is reported, not fatal
            emit(f"BENCH_ARM_ERROR {key} rank={ctx.rank}: {type(e).__name__}: {e}")
            results.append(dict(arm=key, rank=ctx.rank, error=f"{type(e).__name__}: {e}"))
    gathered = [None] * ctx.world
    dist.all_gather_object(gathered, results, group=ctx.coord.cpu_group)
    if ctx.rank == 0:
        rows = [r for part in gathered for r in part]
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        with open(Path(out_dir) / "bench.jsonl", "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        # Per arm occurrence: median of the per-GPU medians.
        seen = {}
        for r in rows:
            if "p50" in r:
                seen.setdefault((r["arm"], r["rpg"]), []).append(r["p50"])
        for (name, rpg), p50s in seen.items():
            per_occ = [statistics.median(p50s[i : i + ctx.world]) for i in range(0, len(p50s), ctx.world)]
            emit(f"BENCH_SUMMARY arm={name} slots={ctx.world * rpg} median_of_gpu_medians_us={per_occ}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["check", "deadline", "bench", "gathers"], default="check")
    ap.add_argument("--op", choices=["ar", "gather"], default="ar")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--absent", action="store_true")
    ap.add_argument("--late-s", type=float, default=1.0)
    ap.add_argument("--deadline-ms", type=float, default=50.0)
    ap.add_argument("--rpg", default="1,4")
    ap.add_argument("--replays", type=int, default=1000)
    ap.add_argument("--out", default="")
    ap.add_argument("--reps", type=int, default=100)
    ap.add_argument("--calls", type=int, default=32)
    ap.add_argument("--order", default="v16,r4,s4,s4,r4,v16,gv16,gr4,gn4,v16")
    # Calls replayed before the timed graph in each repetition (0: timed from the barrier).
    ap.add_argument("--lead-in", type=int, default=0)
    # Two iterations per case (compute-sanitizer runs).
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    ctx = Ctx()
    mutation_banner(ctx.rank)

    def n(iters: int) -> int:
        return 2 if args.quick else iters

    with torch.inference_mode():
        if args.mode == "bench":
            bench(ctx, args.out, args.reps, args.calls, args.order, args.lead_in)
            emit(f"BENCH_DONE rank={ctx.rank}")
            return
        if args.mode == "gathers":
            for rpg in [int(x) for x in args.rpg.split(",")]:
                check_gathers(ctx, rpg, calls=n(600), replays=n(400))
            emit(f"TP_RESULT rank={ctx.rank} failures={ctx.failures}")
            return
        if args.mode == "deadline":
            rpg = int(args.rpg.split(",")[0])
            run_deadline(ctx, args.op, rpg, args.graph, args.absent, args.late_s, args.deadline_ms)
        rpgs = [int(x) for x in args.rpg.split(",")]
        if 1 in rpgs:
            for kind in ("random", "cancel", "zeros"):
                check_ar_vs_shipped(ctx, kind, iters=n(20))
        for rpg in rpgs:
            check_ar(ctx, "random", rpg, "graph", iters=n(200))
            check_ar(ctx, "cancel", rpg, "graph", iters=n(50))
            check_ar(ctx, "zeros", rpg, "eager", iters=n(20))
            check_ar(ctx, "random", rpg, "skew", iters=n(60), calls=6)
            check_gather(ctx, rpg, "graph", iters=n(args.replays))
            check_gather(ctx, rpg, "skew", iters=n(100), calls=4)
    emit(f"TP_RESULT rank={ctx.rank} failures={ctx.failures}")


if __name__ == "__main__":
    main()
