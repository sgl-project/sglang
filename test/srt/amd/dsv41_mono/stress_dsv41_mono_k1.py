"""Race stress of K1 + the mono FFN launch: stress_dsv41_mono_ffn.py's protocol with each layer's K1 (one epoch,
the runner's scratch) before its FFN launch, as the model runs a standard layer.

- A chain of ``--layers`` layers (default 30, the standard layers): K1 reads the previous layer's FFN outputs
  and feeds the FFN launch its residual and mixes; every layer has its own attention weights.
- One CUDA graph per width, plus one with a rank delay inside the chain; widths ``WIDTHS``.
- Width orders: ascending, extremes 48 <-> 1, small <-> large jumps, random; between steps eager chains,
  rank delays before a replay, and RCCL / GEMM work on the same GPUs. Then every graph is destroyed and
  captured again, and the random order runs again.
- Every step is checked against the width's eager reference: the five FFN outputs of every layer, the last
  layer's q and the KV records (outputs NaN-poisoned, records 0xFF-filled before each step).

Run on 4 GPUs: torchrun --nproc-per-node 4 stress_dsv41_mono_k1.py [--minutes 3]
"""

import argparse
import gc
import math
import os
import random
import time

import torch
import torch.distributed as dist
from test_dsv41_mono_ffn import HC, HIDDEN, make_weights, mono_weights, seam_inputs

WIDTHS = (1, 2, 4, 8, 12, 16, 24, 32, 40, 48)  # SGLang's captured decode widths
KINDS = ("out", "residual", "post", "comb", "pre", "q", "records")
RECORD, BLOCK, PAGES, POSITIONS = (
    584,
    64,
    64,
    8192,
)  # V4 SWA record bytes, page tokens, pages, rope positions


def delay(us):
    torch.cuda._sleep(int(us * 2400))  # ~2.4 GHz shader clock


def attn_weights(heads, dev, g):
    """Random K1 weights of one layer: e4m3 [N, K] with E8M0 32 x 32 block codes, unit norms."""

    def fp8(n, k):
        w = (torch.randn(n, k, generator=g, device=dev) * 0.5).to(torch.float8_e4m3fn)
        s = torch.full((n // 32, k // 32), 121, dtype=torch.uint8, device=dev)
        return w, s

    wqkv, wqkv_s = fp8(1792, HIDDEN)
    wqb, wqb_s = fp8(heads * 512, 1280)
    one = lambda n: torch.ones(n, dtype=torch.bfloat16, device=dev)  # noqa: E731
    return dict(
        wqkv=wqkv,
        wqkv_s=wqkv_s,
        q_norm=one(1280),
        kv_norm=one(512),
        wqb=wqb,
        wqb_s=wqb_s,
    )


def rope_table(dev):
    """K1's rope table, [POSITIONS, 64] f32: 32 cosines, then 32 sines."""
    j = torch.arange(32, device=dev, dtype=torch.float64)
    ang = torch.arange(POSITIONS, device=dev, dtype=torch.float64)[
        :, None
    ] * 10000.0 ** (-j / 32)
    return torch.cat([ang.cos(), ang.sin()], -1).float().contiguous()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=3.0)
    ap.add_argument("--layers", type=int, default=30)
    ap.add_argument(
        "--sets", type=int, default=2, help="FFN weight sets cycled over the layers"
    )
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    rank, tp = int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    dev = torch.device("cuda", rank)
    dist.init_process_group("nccl")
    cpu = dist.new_group(backend="gloo")
    from sglang.srt.models.deepseek_common.amd.dsv41_mono.layer import (
        MonoBuild,
        build_mono_k1,
    )
    from sglang.srt.models.deepseek_common.amd.dsv41_mono.runner import (
        MAX_TOKENS,
        DSV41MonoLayer,
    )

    def log(msg):
        if rank == 0:
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

    L = args.layers
    ffn = []
    for s in range(args.sets):
        w, _, _ = make_weights(tp, rank, dev, seed=s)
        ffn.append(mono_weights(w))
        ffn[-1].check(tp)
        del w
        gc.collect()
        torch.cuda.empty_cache()
    r = DSV41MonoLayer(tp, rank, cpu, dev)
    heads = r.q.shape[1]
    g = torch.Generator(device=dev).manual_seed(
        11
    )  # the same attention weights on every rank
    attn = [attn_weights(heads, dev, g) for _ in range(L)]
    cos_sin = rope_table(dev)
    stride = math.ceil(BLOCK * RECORD / 576) * 576  # SGLang's V4 page bytes
    dummy = torch.zeros(MAX_TOKENS * 128, dtype=torch.int32, device=dev)
    share = (rank + 1) / (
        tp * (tp + 1) / 2
    )  # this rank's share of the next layer's attention output

    class Width:
        def __init__(self, M):
            self.M = M
            self.k1 = build_mono_k1(MonoBuild(M, tp, 0))
            self.part, self.res, self.post, self.comb, self.pre = seam_inputs(
                M, tp, rank, dev
            )
            self.x = (torch.randn(M, HIDDEN, generator=g, device=dev) * 0.1).to(
                torch.bfloat16
            )
            self.pos = torch.randint(0, POSITIONS, (M,), generator=g, device=dev)
            self.slot = torch.randperm(PAGES * BLOCK, generator=g, device=dev)[:M]
            self.swa = torch.full((PAGES, stride), 255, dtype=torch.uint8, device=dev)
            b16, f32 = torch.bfloat16, torch.float32
            self.bufs = (
                torch.empty(L, M, HIDDEN, dtype=b16, device=dev),
                torch.empty(L, M, HC, HIDDEN, dtype=b16, device=dev),
                torch.empty(L, M, HC, 1, dtype=f32, device=dev),
                torch.empty(L, M, HC, HC, dtype=f32, device=dev),
                torch.empty(L, M, HC, dtype=f32, device=dev),
                torch.empty(M, heads, 512, dtype=b16, device=dev),
                self.swa,
            )
            self.parts = torch.empty(L, M, HIDDEN, dtype=b16, device=dev)
            self.ref = None

        def poison(self):
            for b in self.bufs[:6]:
                b.fill_(float("nan"))
            r.q.fill_(float("nan"))
            self.swa.fill_(255)

    def chain(wd, delay_rank=-1, delay_us=0):
        M = wd.M
        x, part, res, post, comb, pre = wd.x, wd.part, wd.res, wd.post, wd.comb, wd.pre
        for i in range(L):
            if i == L // 2 and rank == delay_rank:
                delay(delay_us)
            a, m = attn[i], ffn[i % len(ffn)]
            # the attention seam takes the set's mHC and norm weights too
            wd.k1(
                res.data_ptr(), x.data_ptr(), post.data_ptr(), comb.data_ptr(), pre.data_ptr(),
                m.hc_ffn_fn.data_ptr(), m.hc_ffn_scale.data_ptr(), m.hc_ffn_base.data_ptr(), m.ffn_norm.data_ptr(),
                r.res_mid.data_ptr(), r.post_a.data_ptr(), r.comb_a.data_ptr(), r.pre_a.data_ptr(),
                a["wqkv"].data_ptr(), a["wqkv_s"].data_ptr(), a["q_norm"].data_ptr(), a["kv_norm"].data_ptr(),
                a["wqb"].data_ptr(), a["wqb_s"].data_ptr(), cos_sin.data_ptr(), wd.pos.data_ptr(),
                wd.slot.data_ptr(), wd.swa.data_ptr(), wd.swa.stride(0), BLOCK, r.q.data_ptr(),
                dummy.data_ptr(), dummy.data_ptr(), 1, r._dummy.data_ptr(), 0, r._dummy.data_ptr(),
                dummy.data_ptr(), r.kt.data_ptr(), r.klen.data_ptr(), r.scratch(M).data_ptr(),
                r.epoch.data_ptr(), 0, stream=torch.cuda.current_stream(),
            )  # fmt: skip
            outs = tuple(b[i] for b in wd.bufs[:5])
            r.ffn(
                m,
                part,
                r.res_mid[:M],
                r.post_a[:M].view(M, HC, 1),
                r.comb_a[:M],
                r.pre_a[:M],
                outs=outs,
            )
            torch.mul(wd.bufs[0][i], share, out=wd.parts[i])
            x, part = wd.bufs[0][i], wd.parts[i]
            res, post, comb, pre = (b[i] for b in wd.bufs[1:5])
        wd.bufs[5].copy_(r.q[:M])

    st = {}
    fails = []
    t0 = time.time()
    for M in sorted(
        WIDTHS, reverse=True
    ):  # SGLang warms up and captures the largest width first
        wd = st[M] = Width(M)
        chain(wd)
        chain(wd)
        torch.cuda.synchronize()
        first = [b.clone() for b in wd.bufs]
        wd.poison()
        chain(wd)
        torch.cuda.synchronize()
        if not all(torch.equal(a, b) for a, b in zip(first, wd.bufs)):
            fails.append(f"M={M} eager not deterministic")
        if not all(torch.isfinite(b).all() for b in first[:6]):
            fails.append(f"M={M} reference not finite")
        wd.ref = first
    log(
        f"tp {tp}, {L} layers (K1 + FFN launch each), widths {WIDTHS}; built and eager references in "
        f"{time.time() - t0:.1f} s: {'ok' if not fails else fails}"
    )
    wd = st[4]
    wd.poison()
    n_bad = sum((g_ != r_).sum().item() for g_, r_ in zip(wd.bufs, wd.ref))
    n_all = sum(b.numel() for b in wd.bufs)
    flagged_rec = (wd.swa != wd.ref[6]).sum().item()
    log(
        f"check control (no launch after poison): {n_bad} of {n_all} elements flagged (records {flagged_rec})"
    )
    if n_bad == 0 or n_bad < sum(b.numel() for b in wd.bufs[:6]):
        fails.append("check control")

    def capture(order):
        pool = torch.cuda.graph_pool_handle()
        graphs = {}
        for M in order:
            for kind, dr in (("plain", -1), ("delay", M % tp)):
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr, pool=pool):
                    chain(st[M], delay_rank=dr, delay_us=200)
                graphs[M, kind] = gr
        torch.cuda.synchronize()
        return graphs

    graphs = capture(sorted(WIDTHS, reverse=True))
    mism = torch.zeros(len(KINDS), dtype=torch.int64, device=dev)
    big = torch.ones(16 << 20, dtype=torch.float32, device=dev)
    mat = torch.randn(4096, 4096, device=dev, dtype=torch.bfloat16)
    count = {
        "graph": 0,
        "graph_delay": 0,
        "eager": 0,
        "eager_delay": 0,
        "pre_delay": 0,
        "foreign": 0,
    }
    MODES = ("graph", "graph_delay", "eager", "eager_delay")
    phases = []
    # this rank's mismatches by phase, width, step mode, rank delay before, foreign work before, kind, layer
    det = torch.zeros(
        8, len(WIDTHS), len(MODES), 2, 2, len(KINDS), L, dtype=torch.int64, device=dev
    )

    def step(M, j, rng):
        wd = st[M]
        wd.poison()
        if j % 11 == 10:
            dist.all_reduce(big)
            big.mul_(1.0 / tp)
            torch.mm(mat, mat)
            count["foreign"] += 1
        u = rng.uniform(5, 500)  # every rank draws, so the generators stay in step
        if j % 5 == 4:
            if rank == j % tp:
                delay(u)
            count["pre_delay"] += 1
        if j % 7 == 6:
            if rng.random() < 0.5:
                chain(wd)
                mode = "eager"
            else:
                chain(wd, delay_rank=j % tp, delay_us=rng.uniform(5, 300))
                mode = "eager_delay"
        elif j % 3 == 2:
            graphs[M, "delay"].replay()
            mode = "graph_delay"
        else:
            graphs[M, "plain"].replay()
            mode = "graph"
        count[mode] += 1
        d = det[
            len(phases) - 1,
            WIDTHS.index(M),
            MODES.index(mode),
            int(j % 5 == 4),
            int(j % 11 == 10),
        ]
        for k, (got, ref) in enumerate(zip(wd.bufs, wd.ref)):
            bad = got != ref
            mism[k] += bad.sum()
            if k < 5:
                d[k] += bad.flatten(1).sum(1)
            else:
                d[k, 0] += bad.sum()

    def run(name, seq, rng):
        phases.append(name)
        t = time.time()
        before = mism.sum().item()
        for j, M in enumerate(seq):
            step(M, j, rng)
            if j % 200 == 199:
                torch.cuda.synchronize()
        torch.cuda.synchronize()
        log(
            f"phase {name}: {len(seq)} steps in {time.time() - t:.1f} s, mismatching elements {mism.sum().item() - before}"
        )
        return time.time() - t

    rng = random.Random(args.seed)
    W = list(WIDTHS)
    run("warm-up (random)", [rng.choice(W) for _ in range(100)], rng)
    dt = run("calibrate (random)", [rng.choice(W) for _ in range(300)], rng)
    rate = torch.tensor([300 / dt], dtype=torch.float64)
    dist.broadcast(rate, 0, group=cpu)
    n = max(int(rate.item() * args.minutes * 60) // 6, 50)
    log(f"{rate.item():.1f} steps/s; {n} steps a phase")
    run("ascending", (W * (n // len(W) + 1))[:n], rng)
    run("extremes 48 <-> 1", [48 if j % 2 else 1 for j in range(n)], rng)
    jumps = [(1, 48), (2, 40), (48, 4), (12, 32), (40, 1), (24, 2), (8, 48), (16, 12)]
    run(
        "small <-> large jumps",
        [jumps[(j // 2) % len(jumps)][j % 2] for j in range(n)],
        rng,
    )
    run("random", [rng.choice(W) for _ in range(n)], rng)
    for gr in graphs.values():
        gr.reset()
    del graphs
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    graphs = capture(sorted(WIDTHS))
    log(f"destroyed and captured again ({len(graphs)} graphs, smallest first)")
    run("after re-capture: random", [rng.choice(W) for _ in range(2 * n)], rng)

    torch.cuda.synchronize()
    for r_ in range(
        tp
    ):  # every rank prints its own mismatches (rank 0 logs only its own per phase)
        if r_ == rank:
            nz = det.nonzero().tolist()
            print(f"[rank {rank}] mismatch cells: {len(nz)}", flush=True)
            for ph, wi, mo, pd, fo, k, lay in nz[:80]:
                print(
                    f"[rank {rank}] phase '{phases[ph]}' M={WIDTHS[wi]} {MODES[mo]} pre_delay={pd} foreign={fo} "
                    f"{KINDS[k]} layer={lay}: {det[ph, wi, mo, pd, fo, k, lay].item()}",
                    flush=True,
                )
        dist.barrier()
    dist.all_reduce(mism)
    ref = st[48].bufs[0].clone()
    dist.broadcast(ref, 0)
    agree = torch.tensor([int(torch.equal(ref, st[48].bufs[0]))], device=dev)
    dist.all_reduce(agree, op=dist.ReduceOp.MIN)
    steps = sum(count.values()) - count["pre_delay"] - count["foreign"]
    log(f"steps {steps} ({count}); K1 launches {steps * L}, FFN launches {steps * L}")
    for k, name in enumerate(KINDS):
        log(f"  {name}: mismatching elements {mism[k].item()}")
    log(f"ranks agree on the last output: {bool(agree.item())}")
    if mism.sum().item() or not agree.item():
        fails.append("stress")
    log("ALL PASS" if not fails else f"FAILED: {fails}")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
