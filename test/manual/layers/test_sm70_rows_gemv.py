"""SM70 row-exact batched GEMV: each of 1-4 rows is bitwise the one-row launch
for every (threads, lanes) config, error vs fp64, and a config sweep against
torch on the GLM-5.3-Flash TP8 shapes."""

import sys

import torch

from sglang.kernels.ops.gemm import sm70_rows_gemv as rg

dev = torch.device("cuda")
g = torch.Generator(device=dev).manual_seed(0)

# (name, batch, N, K, weight dtype)
SHAPES = [
    ("indexer wk / compress gate", 1, 128, 4096, torch.float16),
    ("indexer weights_proj", 1, 32, 4096, torch.float32),
    ("mla w_kc", 8, 512, 256, torch.float16),
    ("mla w_vc", 8, 256, 512, torch.float16),
    ("kda fg_b", 2, 1024, 128, torch.float16),
]
CONFIGS = [(nt, l) for nt in (64, 128, 256) for l in (8, 16, 32, 64, 128, 256) if l <= nt]


def timeit(fn, iters=200):
    for _ in range(10):
        fn()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        for _ in range(20):
            fn()
    s, e = torch.cuda.Event(True), torch.cuda.Event(True)
    s.record()
    for _ in range(iters // 20):
        graph.replay()
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / iters * 1e3


def torch_ref(x, w):
    return torch.bmm(x.to(w.dtype), w.transpose(1, 2))


def check_shape(name, B, N, K, dtype, sweep):
    w = (torch.randn(B, N, K, device=dev, generator=g) * 0.05).to(dtype)
    # Strided x like the transposed MLA / KDA views: [B, M, K] with row stride B * K.
    x4 = torch.randn(4, B, K, device=dev, generator=g).half().transpose(0, 1)
    ref = (x4.double() @ w.double().transpose(1, 2))
    best = None
    for nt, l in CONFIGS:
        one = torch.cat([rg.bmm_with(x4[:, m:m + 1], w, nt, l) for m in range(4)], dim=1)
        for M in (2, 3, 4):
            many = rg.bmm_with(x4[:, :M], w, nt, l)
            assert torch.equal(many, one[:, :M]), f"{name} nt={nt} l={l} M={M} differs from one row"
        err = ((one.double() - ref).abs().max() / ref.abs().max()).item()
        assert err < 2e-3, f"{name} nt={nt} l={l} err {err}"
        if sweep:
            t1 = timeit(lambda: rg.bmm_with(x4[:, :1], w, nt, l))
            t4 = timeit(lambda: rg.bmm_with(x4, w, nt, l))
            if best is None or t1 + t4 < best[0]:
                best = (t1 + t4, nt, l, t1, t4)
    line = f"{name:28s} B={B} N={N:5d} K={K:5d} {str(dtype)[6:]:8s} all configs row-exact"
    if sweep:
        c1 = timeit(lambda: torch_ref(x4[:, :1], w))
        c4 = timeit(lambda: torch_ref(x4, w))
        _, nt, l, t1, t4 = best
        line += f"; best ({nt},{l}) {t1:.1f}/{t4:.1f}us vs torch {c1:.1f}/{c4:.1f}us (1/4 rows)"
    print(line)


if __name__ == "__main__":
    sweep = "--sweep" in sys.argv
    for shape in SHAPES:
        check_shape(*shape, sweep)
    print("OK")
