"""Race stress of the DeepSeek-V4.1 mono FFN launch's hand-off protocol, as SGLang runs it.

- One runner (one epoch, one scratch per width) serves a chain of ``--layers`` launches (default 40, as
  the model), each layer feeding the next: its MoE output (this rank's share) is the next attention
  partial; its residual and mixes are the next seam's. ``--sets`` weight sets are cycled over the layers.
- One CUDA graph per width (1, 2, ..., 48), captured largest first in one shared memory pool, as SGLang
  captures decode graphs; a second graph per width with a rank delay inside the chain.
- Width orders: ascending, descending, extremes alternating (48 -> 1 -> 48), small -> large jumps, long
  runs of one width, random. Between steps: eager chains, rank delays before a replay, and other
  collectives and work on the same GPUs (as a prefill step would run).
- Every step is checked against the width's eager reference, all layers and all five outputs. Outputs
  are poisoned (NaN) before each step, so a launch that writes nothing fails. Mismatches add into
  device counters that are never reset; the first bad step is kept.
- Then every graph is destroyed, the pool freed, the graphs captured again (smallest first), and the
  random order runs again.

Run on 4 GPUs: torchrun --nproc-per-node 4 stress_dsv41_mono_ffn.py [--minutes 25]
"""

import argparse
import gc
import os
import random
import time

import torch
import torch.distributed as dist
from test_dsv41_mono_ffn import (
    HC,
    HIDDEN,
    WIDTHS,
    make_weights,
    mono_weights,
    seam_inputs,
)

KINDS = ("out", "residual", "post", "comb", "pre")


def delay(us):
    """Hold this rank's stream for about ``us`` microseconds."""
    torch.cuda._sleep(int(us * 2400))  # ~2.4 GHz shader clock


class Width:
    """One step width's fixed inputs, its per-layer outputs, and their eager reference."""

    def __init__(self, M, L, tp, rank, dev):
        self.M = M
        self.inputs = seam_inputs(M, tp, rank, dev)
        f32, b16 = torch.float32, torch.bfloat16
        self.bufs = (
            torch.empty(L, M, HIDDEN, dtype=b16, device=dev),
            torch.empty(L, M, HC, HIDDEN, dtype=b16, device=dev),
            torch.empty(L, M, HC, 1, dtype=f32, device=dev),
            torch.empty(L, M, HC, HC, dtype=f32, device=dev),
            torch.empty(L, M, HC, dtype=f32, device=dev),
        )
        self.parts = torch.empty(L, M, HIDDEN, dtype=b16, device=dev)
        self.ref = None

    def poison(self):
        for b in self.bufs:
            b.fill_(float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--minutes", type=float, default=25.0, help="stress time, split over the phases"
    )
    ap.add_argument("--layers", type=int, default=40)
    ap.add_argument(
        "--sets",
        type=int,
        default=4,
        help="distinct weight sets cycled over the layers",
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sglang.srt.models.deepseek_common.amd.dsv41_mono.runner import DSV41MonoLayer

    def log(msg):
        if rank == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    L = args.layers
    sets = []
    for s in range(args.sets):
        w, _, _ = make_weights(tp, rank, dev, seed=s)
        mw = mono_weights(w)
        mw.check(tp)
        sets.append(mw)
        del w
        gc.collect()
        torch.cuda.empty_cache()
    runner = DSV41MonoLayer(tp, rank, cpu, dev)
    share = (rank + 1) / (
        tp * (tp + 1) / 2
    )  # this rank's share of the next layer's attention output
    st = {M: Width(M, L, tp, rank, dev) for M in WIDTHS}
    log(f"tp {tp}, {L} layers, {args.sets} weight sets, widths {WIDTHS}")

    def chain(wd, delay_rank=-1, delay_us=0):
        part, res, post, comb, pre = wd.inputs
        for i in range(L):
            if i == L // 2 and rank == delay_rank:
                delay(delay_us)
            outs = tuple(b[i] for b in wd.bufs)
            runner.ffn(sets[i % len(sets)], part, res, post, comb, pre, outs=outs)
            if i + 1 < L:
                torch.mul(wd.bufs[0][i], share, out=wd.parts[i])
                part = wd.parts[i]
                res, post, comb, pre = (b[i] for b in wd.bufs[1:])

    # eager references: two eager chains per width must agree bit for bit and be finite
    fails = []
    for M in sorted(
        WIDTHS, reverse=True
    ):  # SGLang warms up and captures the largest width first
        wd = st[M]
        chain(wd)
        chain(wd)
        torch.cuda.synchronize()
        first = [b.clone() for b in wd.bufs]
        wd.poison()
        chain(wd)
        torch.cuda.synchronize()
        if not all(torch.equal(a, b) for a, b in zip(first, wd.bufs)):
            fails.append(f"M={M} eager not deterministic")
        if not all(torch.isfinite(b).all() for b in first):
            fails.append(f"M={M} reference not finite")
        wd.ref = first
    log(f"eager references: {'ok' if not fails else fails}")
    # the check's own control: a step that writes nothing must count every element
    wd = st[4]
    wd.poison()
    n_bad = sum((g != r).sum().item() for g, r in zip(wd.bufs, wd.ref))
    n_all = sum(b.numel() for b in wd.bufs)
    log(f"check control (no launch after poison): {n_bad} of {n_all} elements flagged")
    if n_bad != n_all:
        fails.append("check control")

    def capture(order):
        pool = torch.cuda.graph_pool_handle()
        graphs = {}
        for M in order:
            for kind, dr in (("plain", -1), ("delay", M % tp)):
                g = torch.cuda.CUDAGraph()
                with torch.cuda.graph(g, pool=pool):
                    chain(st[M], delay_rank=dr, delay_us=200)
                graphs[M, kind] = g
        torch.cuda.synchronize()
        return graphs

    t0 = time.time()
    graphs = capture(sorted(WIDTHS, reverse=True))
    log(f"captured {len(graphs)} graphs in {time.time() - t0:.1f} s")

    # device-side tallies: mismatching elements per (kind, layer), never reset
    mism = torch.zeros(len(KINDS), L, dtype=torch.int64, device=dev)
    step_no = torch.zeros(1, dtype=torch.int64, device=dev)
    first_bad = torch.full((1,), -1, dtype=torch.int64, device=dev)
    big = torch.ones(
        16 << 20, dtype=torch.float32, device=dev
    )  # 64 MB: another collective's traffic
    mat = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    count = {
        "graph": 0,
        "graph_delay": 0,
        "eager": 0,
        "eager_delay": 0,
        "pre_delay": 0,
        "foreign": 0,
    }

    def check(wd):
        tot = torch.zeros(1, dtype=torch.int64, device=dev)
        for k, (got, ref) in enumerate(zip(wd.bufs, wd.ref)):
            d = (got != ref).flatten(1).sum(1)
            mism[k] += d
            tot += d.sum()
        first_bad.copy_(torch.where((first_bad < 0) & (tot > 0), step_no, first_bad))
        step_no.add_(1)

    def step(M, j, rng):
        wd = st[M]
        wd.poison()
        r = rng.random()
        if j % 11 == 10:
            dist.all_reduce(big)  # RCCL on the same ranks between mono steps
            big.mul_(1.0 / tp)
            torch.mm(mat, mat)
            count["foreign"] += 1
        if j % 5 == 4:
            if rank == j % tp:
                delay(rng.uniform(5, 500))
            else:
                rng.uniform(5, 500)
            count["pre_delay"] += 1
        else:
            rng.uniform(5, 500)  # keep every rank's generator in step
        if j % 7 == 6:
            if r < 0.5:
                chain(wd)
                count["eager"] += 1
            else:
                chain(wd, delay_rank=j % tp, delay_us=rng.uniform(5, 300))
                count["eager_delay"] += 1
        elif j % 3 == 2:
            graphs[M, "delay"].replay()
            count["graph_delay"] += 1
        else:
            graphs[M, "plain"].replay()
            count["graph"] += 1
        check(wd)

    def run(name, seq, rng):
        t = time.time()
        before = mism.sum().item()
        for j, M in enumerate(seq):
            step(M, j, rng)
            if j % 200 == 199:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        bad = mism.sum().item() - before
        log(
            f"phase {name}: {len(seq)} steps in {time.time() - t:.1f} s, mismatching elements {bad}"
        )
        return time.time() - t

    rng = random.Random(args.seed)  # the same sequence on every rank
    W = list(WIDTHS)
    # calibrate: steps a second, agreed by every rank (rank 0's measurement); after a warm-up, whose first
    # replays are about 10x slower and would shrink the budget
    run("warm-up (random)", [rng.choice(W) for _ in range(300)], rng)
    dt = run("calibrate (random)", [rng.choice(W) for _ in range(1000)], rng)
    rate = torch.tensor([1000 / dt], dtype=torch.float64)
    dist.broadcast(rate, 0, group=cpu)
    budget = int(rate.item() * args.minutes * 60)
    n = max(budget // 10, 100)
    log(f"{rate.item():.1f} steps/s; budget {budget} steps")
    run("ascending", (W * (n // len(W) + 1))[:n], rng)
    run("descending", (W[::-1] * (n // len(W) + 1))[:n], rng)
    run("extremes 48 <-> 1", [48 if j % 2 else 1 for j in range(n)], rng)
    jumps = [(1, 48), (2, 48), (8, 48), (48, 2), (12, 40), (4, 32), (48, 12), (24, 1)]
    run(
        "small <-> large jumps",
        [jumps[(j // 2) % len(jumps)][j % 2] for j in range(n)],
        rng,
    )
    runs = []
    while len(runs) < n:
        runs += [rng.choice(W)] * rng.randint(20, 80)
    run("long runs of one width", runs[:n], rng)
    run("random", [rng.choice(W) for _ in range(2 * n)], rng)

    # destroy every graph and its pool, then capture again in the other order
    for g in graphs.values():
        g.reset()
    del graphs
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    t0 = time.time()
    graphs = capture(sorted(WIDTHS))
    log(
        f"destroyed and captured again ({len(graphs)} graphs, smallest first) in {time.time() - t0:.1f} s"
    )
    run("after re-capture: random", [rng.choice(W) for _ in range(2 * n)], rng)
    run("after re-capture: extremes", [48 if j % 2 else 1 for j in range(n)], rng)

    # every rank's tallies; the outputs agree across ranks
    dist.all_reduce(mism)
    fbs = [torch.empty_like(first_bad) for _ in range(tp)]
    dist.all_gather(fbs, first_bad)
    fb = min((int(f.item()) for f in fbs if f.item() >= 0), default=-1)
    steps = step_no.item()
    ref = st[48].bufs[0].clone()
    dist.broadcast(ref, 0)
    agree = torch.tensor([int(torch.equal(ref, st[48].bufs[0]))], device=dev)
    dist.all_reduce(agree, op=dist.ReduceOp.MIN)
    total = mism.sum().item()
    log(f"steps {steps} ({count}); layer launches {steps * L}")
    for k, name in enumerate(KINDS):
        bad_layers = [i for i in range(L) if mism[k, i].item()]
        log(
            f"  {name}: mismatching elements {mism[k].sum().item()}, layers {bad_layers or 'none'}"
        )
    log(
        f"first bad step: {fb} (-1: none); ranks agree on the last output: {bool(agree.item())}"
    )
    if total or not agree.item():
        fails.append("stress")
    log("ALL PASS" if not fails else f"FAILED: {fails}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
