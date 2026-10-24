"""Latency of the gfx950 DSA indexer logits kernel (node 3 of the fused decode
indexer) on MTP verify grids: 6 rows per request sharing one page table.

Each sample is one CUDA graph replay of NUM_LAYERS launches, each reading a
different KV pool (as the layers of a decode step do), after a 1 GiB read that
evicts L2 and MALL; the reported time is graph time / NUM_LAYERS.  Each setup is
timed with the per-row kernel and with the shared-page kernel (share_pages);
the server selects the shared-page kernel only for <= 18 rows whose longest
context is >= 96k tokens.

    python benchmark/kernels/deepseek/benchmark_dsa_gfx950_logits_hist.py
"""

import argparse
import math
import statistics

import torch

FP8 = torch.float8_e4m3fn
PAGE_SIZE = 64
PAGE_BYTES = PAGE_SIZE * 132
K_BYTES = PAGE_SIZE * 128
POOL_PAGES = 62280  # ~526 MB per layer, larger than MALL
TABLE_PAGES = 16386  # GLM-5.2's captured page-table width


def make_pool(gen):
    u8 = torch.empty(POOL_PAGES, PAGE_BYTES, dtype=torch.uint8, device="cuda")
    for s in range(0, POOL_PAGES, 8192):
        e = min(POOL_PAGES, s + 8192)
        n = e - s
        u8[s:e, :K_BYTES] = (
            (0.7 * torch.randn(n, K_BYTES, generator=gen, device="cuda"))
            .to(FP8)
            .view(torch.uint8)
        )
        scale = torch.exp2(
            -8 + 2 * torch.rand(n, PAGE_SIZE, generator=gen, device="cuda")
        )
        u8[s:e, K_BYTES:] = scale.view(torch.uint8)
    return u8.view(FP8)


def bench(mod, bpr, share, ctxs, pools, gen, flush, iters):
    rows = 6 * len(ctxs)
    perm = torch.randperm(POOL_PAGES, generator=torch.Generator().manual_seed(1))
    table = torch.zeros(rows, TABLE_PAGES, dtype=torch.int32)
    seqlens, off = [], 0
    for i, c in enumerate(ctxs):
        n = math.ceil(c / PAGE_SIZE)
        table[6 * i : 6 * i + 6, :n] = perm[
            torch.arange(off, off + n) % POOL_PAGES
        ].int()
        off += n
        seqlens += [c - 5 + j for j in range(6)]
    table = table.cuda()
    seqlens = torch.tensor(seqlens, dtype=torch.int32, device="cuda")
    q = (2.0 * torch.randn(rows, 32, 128, generator=gen, device="cuda")).to(FP8)
    gate = torch.rand(rows, 32, generator=gen, device="cuda") * 0.02
    out = torch.empty(rows, TABLE_PAGES * PAGE_SIZE, dtype=torch.float32, device="cuda")
    ghist = torch.zeros(rows, 4096 + 64, dtype=torch.int32, device="cuda")

    def run():
        for kv in pools:
            mod.logits_hist(q, kv, gate, seqlens, table, out, ghist, bpr, share)

    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        run()
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        run()
    sink = torch.empty((), device="cuda")
    times = []
    for _ in range(iters):
        torch.sum(flush, dim=0, out=sink)
        a, b = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        a.record()
        graph.replay()
        b.record()
        b.synchronize()
        times.append(a.elapsed_time(b) * 1e3 / len(pools))
    return statistics.median(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument(
        "--ctx",
        type=int,
        nargs="+",
        default=[4096, 32768, 65536, 131072, 262144, 524288],
    )
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--iters", type=int, default=200)
    a = ap.parse_args()

    from sglang.kernels.ops.attention.dsa.hip_gfx950 import loader

    mod = loader.modules()[2]
    gen = torch.Generator(device="cuda").manual_seed(0)
    pools = [make_pool(gen) for _ in range(a.layers)]
    flush = torch.ones((1 << 30) // 4, dtype=torch.float32, device="cuda")
    print(
        f"{'requests':>8} {'rows':>4} {'ctx':>8} {'per-row us':>10} "
        f"{'shared us':>10} {'speedup':>8}"
    )
    for reqs in a.requests:
        for c in a.ctx:
            us = [
                bench(
                    mod,
                    loader.LOGITS_BLOCKS_PER_ROW,
                    share,
                    [c] * reqs,
                    pools,
                    gen,
                    flush,
                    a.iters,
                )
                for share in (0, 1)
            ]
            print(
                f"{reqs:>8} {6 * reqs:>4} {c:>8} {us[0]:>10.2f} {us[1]:>10.2f} "
                f"{us[0] / us[1]:>7.2f}x"
            )


if __name__ == "__main__":
    main()
