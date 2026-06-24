"""Benchmark FP4 quantize: sglang jit_kernel vs flashinfer.

Compares ``sglang.jit_kernel.nvfp4.scaled_fp4_quant`` against
``flashinfer.fp4_quantize`` over a sweep of (M, K) shapes.

Timing uses ``flashinfer.testing.bench_gpu_time`` (CUDA-graph based with
rotating-buffer cold-L2).
"""

import argparse
import itertools

import numpy as np
import torch
from flashinfer import fp4_quantize as flashinfer_fp4_quantize
from flashinfer.testing import bench_gpu_time

from sglang.jit_kernel.nvfp4 import scaled_fp4_quant

Ms = [1, 8, 32, 128, 512, 1024, 2048, 4096, 8192, 16384, 32768]
Ks = [128, 256, 384, 512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 8192, 16384]


def _bench(fn, input_args) -> float:
    times = bench_gpu_time(
        fn=fn,
        input_args=input_args,
        use_cuda_graph=True,
        dry_run_time_ms=25,
        repeat_time_ms=100,
    )
    return float(np.median(times))


def benchmark(M: int, K: int, dtype: torch.dtype, device: str):
    x = torch.randn(M, K, device=device, dtype=dtype)
    global_scale = torch.ones(1, device=device, dtype=torch.float32)

    sglang_ms = _bench(
        lambda x, gs: scaled_fp4_quant(x, gs),
        input_args=(x, global_scale),
    )
    flashinfer_ms = _bench(
        lambda x, gs: flashinfer_fp4_quantize(x, gs, backend="cute-dsl"),
        input_args=(x, global_scale),
    )

    return sglang_ms, flashinfer_ms


def plot_speedup(rows, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Ms_unique = sorted({int(r[0]) for r in rows})
    Ks_unique = sorted({int(r[1]) for r in rows})
    grid = np.full((len(Ms_unique), len(Ks_unique)), np.nan)
    m_idx = {m: i for i, m in enumerate(Ms_unique)}
    k_idx = {k: i for i, k in enumerate(Ks_unique)}
    for M, K, _, _, sp in rows:
        grid[m_idx[int(M)], k_idx[int(K)]] = float(sp)

    fig, ax = plt.subplots(figsize=(12, 8))
    vmax = max(2.0, np.nanmax(grid))
    vmin = min(0.5, np.nanmin(grid))
    im = ax.imshow(
        grid,
        aspect="auto",
        cmap="RdYlGn",
        vmin=vmin,
        vmax=vmax,
        origin="lower",
    )
    ax.set_xticks(range(len(Ks_unique)))
    ax.set_xticklabels(Ks_unique, rotation=45)
    ax.set_yticks(range(len(Ms_unique)))
    ax.set_yticklabels(Ms_unique)
    ax.set_xlabel("K")
    ax.set_ylabel("M")
    ax.set_title("Speedup: flashinfer / sglang  (>1 means sglang faster)")
    for i in range(len(Ms_unique)):
        for j in range(len(Ks_unique)):
            v = grid[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, ax=ax, label="speedup")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"Saved plot to {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--csv", type=str, default=None)
    parser.add_argument("--plot", type=str, default=None)
    args = parser.parse_args()

    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    rows = []
    header = (
        f"{'M':>8} {'K':>8} {'sglang(us)':>12} {'flashinfer(us)':>16} {'speedup':>10}"
    )
    print(header)
    print("-" * len(header))

    for M, K in itertools.product(Ms, Ks):
        try:
            sglang_ms, flashinfer_ms = benchmark(M, K, dtype, args.device)
        except Exception as e:
            print(f"{M:>8} {K:>8}  skipped: {e}")
            continue
        sglang_us = sglang_ms * 1e3
        flashinfer_us = flashinfer_ms * 1e3
        speedup = flashinfer_us / sglang_us
        print(
            f"{M:>8} {K:>8} {sglang_us:>12.3f} {flashinfer_us:>16.3f} {speedup:>10.3f}"
        )
        rows.append((M, K, sglang_us, flashinfer_us, speedup))

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("M,K,sglang_us,flashinfer_us,speedup_flashinfer_over_sglang\n")
            for M, K, s, fi, sp in rows:
                f.write(f"{M},{K},{s:.6f},{fi:.6f},{sp:.6f}\n")
        print(f"Saved CSV to {args.csv}")

    if args.plot:
        plot_speedup(rows, args.plot)


if __name__ == "__main__":
    main()
