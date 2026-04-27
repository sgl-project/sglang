"""Kernel comparison: cuBLAS BF16 bmm vs SGLang Triton fused_moe vs Marlin W4A16.

Companion to test_efficiency.py. Where test_efficiency only races the kernels
SGLang ships, this script also adds:

  - cuBLAS bmm BF16 baseline (no MoE routing — upper bound for BF16 grouped GEMM)
  - Per-M Triton tile autotune (mini grid search instead of the default-fallback
    config that ships when no JSON exists for the (E, N, GPU) triple)
  - Optional Triton-with-default-tile row, to show the under-tuning regression

Why this matters: if SGLang has no autotuned config JSON at
  python/sglang/srt/layers/moe/fused_moe_triton/configs/triton_<ver>/E=<E>,N=<N>,device_name=<gpu>.json
the BF16 path falls back to a 64×64×32, 4w, 2s tile that is terrible for
Ampere BF16. test_efficiency.py's "Marlin always wins" result on A100 is an
artifact of that fallback. With a per-M tuned tile, BF16 wins above M/e ≈ 128
exactly as the compute-bound roofline predicts.

Run directly:
    python bench_kernel_compare.py                          # full sweep + figure
    python bench_kernel_compare.py --mode isolate           # single M, no figure
    python bench_kernel_compare.py --m-max 1024 --no-figure # quick sweep
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# Bootstrap: make the test_heter_moe package importable when this file is run
# directly (pytest sets rootdir=test/, but `python bench_kernel_compare.py`
# from anywhere else won't).
# --------------------------------------------------------------------------- #
_THIS = Path(__file__).resolve()
_TEST_ROOT = _THIS.parents[2]  # .../sglang/test
if str(_TEST_ROOT) not in sys.path:
    sys.path.insert(0, str(_TEST_ROOT))

from test_heter_moe.util import CUDA_AVAILABLE, init_mock_server_args  # noqa: E402

# Initialise the Triton kernel's config-lookup path before importing it.
init_mock_server_args()

from sglang.srt.layers.moe.fused_moe_triton import fused_moe as _fm  # noqa: E402
from sglang.srt.layers.moe.fused_moe_triton import (  # noqa: E402
    fused_moe_triton_config as _cfgmod,
)
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (  # noqa: E402
    fused_marlin_moe,
)
from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (  # noqa: E402
    outplace_fused_experts,
)

# --------------------------------------------------------------------------- #
# Shape (matches test_efficiency.py KERN_* constants).
# --------------------------------------------------------------------------- #
KERN_E, KERN_K, KERN_N, KERN_TOP_K = 128, 2048, 1536, 8
KERN_GROUP_SIZE = 128
KERN_NUM_BITS = 4

A100_BF16_PEAK_TFLOPS = 312.0  # dense BF16 tensor-core peak

# --------------------------------------------------------------------------- #
# Triton tile candidates for the per-M mini-autotune.
#   - SMALL_M_CFGS: aimed at M ≤ ~64 (memory-bound; small tiles win)
#   - LARGE_M_CFGS: aimed at M ≥ 128 (compute-bound; need bigger tiles)
# Each entry must contain BLOCK_SIZE_M/N/K, GROUP_SIZE_M, num_warps, num_stages.
# --------------------------------------------------------------------------- #
SMALL_M_CFGS = [
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 32,  "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
    {"BLOCK_SIZE_M": 16, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 1, "num_warps": 4, "num_stages": 4},
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
    {"BLOCK_SIZE_M": 32, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
    {"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64,  "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
]
LARGE_M_CFGS = [
    {"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 4},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 4},
    {"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 256, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3},
    {"BLOCK_SIZE_M": 64,  "BLOCK_SIZE_N": 256, "BLOCK_SIZE_K": 64, "GROUP_SIZE_M": 8, "num_warps": 8, "num_stages": 3},
]

# Default tile that ships when no JSON config exists for (E, N, GPU).
# Lifted from get_default_config() in fused_moe_triton_config.py:186-191.
DEFAULT_FALLBACK_CFG = {
    "BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32,
    "GROUP_SIZE_M": 8, "num_warps": 4, "num_stages": 2,
}

# --------------------------------------------------------------------------- #
# Tile injection: replaces try_get_optimal_moe_config so we can dictate the
# tile per measurement instead of letting it read JSON / fall back.
# --------------------------------------------------------------------------- #
_FORCED_CFG = None


def _patched_get_cfg(*args, **kwargs):
    if kwargs.get("return_down_config", False):
        return _FORCED_CFG, (_FORCED_CFG, _FORCED_CFG["BLOCK_SIZE_M"])
    return _FORCED_CFG


def install_tile_patch():
    _cfgmod.try_get_optimal_moe_config = _patched_get_cfg
    _fm.try_get_optimal_moe_config = _patched_get_cfg


def set_forced_tile(cfg):
    global _FORCED_CFG
    _FORCED_CFG = dict(cfg)


# --------------------------------------------------------------------------- #
# Timing harness — same recipe as test_efficiency.py:_bench (warmup, L2 flush,
# median of N timed iters). CUDA-graph capture is OFF here because the
# monkey-patched config function and the per-iter L2 flush both interact
# poorly with capture.
# --------------------------------------------------------------------------- #
_L2_FLUSH_BUF = None


def _flush_l2(device):
    global _L2_FLUSH_BUF
    if _L2_FLUSH_BUF is None or _L2_FLUSH_BUF.device != device:
        _L2_FLUSH_BUF = torch.empty(50 * 1024 * 1024, dtype=torch.int8, device=device)
    _L2_FLUSH_BUF.zero_()


def bench(fn, device, warmup=8, iters=15):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(iters):
        _flush_l2(device)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


# --------------------------------------------------------------------------- #
# Tensor factories.
# --------------------------------------------------------------------------- #
def make_bf16_bmm_tensors(M, device):
    """Pre-gathered per-expert BF16 tensors for cuBLAS bmm."""
    x_w13 = torch.randn(KERN_E, M, KERN_K, dtype=torch.bfloat16, device=device)
    w13 = torch.randn(KERN_E, KERN_K, 2 * KERN_N, dtype=torch.bfloat16, device=device)
    x_w2 = torch.randn(KERN_E, M, KERN_N, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(KERN_E, KERN_N, KERN_K, dtype=torch.bfloat16, device=device)
    return x_w13, w13, x_w2, w2


def make_moe_inputs(M, device):
    """MoE inputs: M_global tokens routed to top_k=8 experts uniformly across E."""
    M_global = M * KERN_E // KERN_TOP_K
    x = torch.randn(M_global, KERN_K, dtype=torch.bfloat16, device=device)
    topk_w = torch.ones(M_global, KERN_TOP_K, dtype=torch.bfloat16, device=device) / KERN_TOP_K
    all_ids = torch.arange(KERN_E, device=device).repeat(M)
    all_ids = all_ids[torch.randperm(len(all_ids), device=device)]
    topk_ids = all_ids.reshape(M_global, KERN_TOP_K)
    gating = torch.randn(M_global, KERN_E, dtype=torch.bfloat16, device=device)
    return x, topk_w, topk_ids, gating


def make_bf16_moe_weights(device):
    w13 = torch.randn(KERN_E, 2 * KERN_N, KERN_K, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(KERN_E, KERN_K, KERN_N, dtype=torch.bfloat16, device=device)
    return w13, w2


def make_int4_marlin_weights(device):
    w1 = torch.randint(
        0, 2**31, (KERN_E, KERN_K // 16, 2 * KERN_N * (KERN_NUM_BITS // 2)),
        dtype=torch.int32, device=device,
    )
    w2 = torch.randint(
        0, 2**31, (KERN_E, KERN_N // 16, KERN_K * (KERN_NUM_BITS // 2)),
        dtype=torch.int32, device=device,
    )
    s1 = torch.ones(KERN_E, KERN_K // KERN_GROUP_SIZE, 2 * KERN_N,
                    dtype=torch.bfloat16, device=device) * 0.01
    s2 = torch.ones(KERN_E, KERN_N // KERN_GROUP_SIZE, KERN_K,
                    dtype=torch.bfloat16, device=device) * 0.01
    return w1, w2, s1, s2


def flops_for_M(M):
    """Total FLOPs for the W13 + W2 grouped-GEMM pair."""
    return 2 * KERN_E * M * KERN_K * (2 * KERN_N) + 2 * KERN_E * M * KERN_N * KERN_K


# --------------------------------------------------------------------------- #
# Per-row measurement.
# --------------------------------------------------------------------------- #
def measure_one_M(M, device, *, include_default_tile=True,
                  bf16_bmm_tensors=None, bf16_moe_weights=None,
                  marlin_weights=None):
    """Time all kernels at a given M_per_expert. Returns a dict row."""
    M_global = M * KERN_E // KERN_TOP_K
    flops_total = flops_for_M(M)

    # --- (A) cuBLAS bmm ---
    if bf16_bmm_tensors is None:
        x_w13, w13_bf, x_w2, w2_bf = make_bf16_bmm_tensors(M, device)
    else:
        x_w13_max, w13_bf, x_w2_max, w2_bf = bf16_bmm_tensors
        x_w13 = x_w13_max[:, :M, :].contiguous()
        x_w2 = x_w2_max[:, :M, :].contiguous()
    ms_w13 = bench(lambda: torch.bmm(x_w13, w13_bf), device, warmup=5, iters=10)
    ms_w2 = bench(lambda: torch.bmm(x_w2, w2_bf), device, warmup=5, iters=10)
    ms_cublas = ms_w13 + ms_w2

    # --- MoE inputs (shared by Triton + Marlin) ---
    x_moe, topk_w, topk_ids, gating = make_moe_inputs(M, device)
    if bf16_moe_weights is None:
        w13_moe, w2_moe = make_bf16_moe_weights(device)
    else:
        w13_moe, w2_moe = bf16_moe_weights
    if marlin_weights is None:
        int4_w1, int4_w2, int4_s1, int4_s2 = make_int4_marlin_weights(device)
    else:
        int4_w1, int4_w2, int4_s1, int4_s2 = marlin_weights

    # --- (C) Marlin W4A16 ---
    ms_marlin = bench(
        lambda: fused_marlin_moe(
            x_moe, int4_w1, int4_w2, int4_s1, int4_s2,
            gating, topk_w, topk_ids,
            num_bits=KERN_NUM_BITS, is_k_full=True,
        ),
        device, warmup=5, iters=10,
    )

    # --- (B1) Triton fused_moe BF16 with default fallback tile ---
    ms_triton_default = None
    if include_default_tile:
        set_forced_tile(DEFAULT_FALLBACK_CFG)
        ms_triton_default = bench(
            lambda: outplace_fused_experts(x_moe, w13_moe, w2_moe, topk_w, topk_ids),
            device, warmup=5, iters=10,
        )

    # --- (B2) Triton fused_moe BF16 with per-M autotuned tile ---
    candidates = SMALL_M_CFGS if M <= 64 else (SMALL_M_CFGS + LARGE_M_CFGS)
    best_ms, best_cfg = float("inf"), None
    for cfg in candidates:
        set_forced_tile(cfg)
        try:
            ms = bench(
                lambda: outplace_fused_experts(x_moe, w13_moe, w2_moe, topk_w, topk_ids),
                device, warmup=4, iters=8,
            )
        except Exception:
            continue
        if ms < best_ms:
            best_ms, best_cfg = ms, cfg

    return {
        "M_per_expert": M,
        "M_global": M_global,
        "flops_T": flops_total / 1e12,
        "ms_cublas_bmm": ms_cublas,
        "ms_triton_tuned": best_ms,
        "ms_triton_default": ms_triton_default,
        "ms_marlin_w4a16": ms_marlin,
        "tflops_cublas_bmm": flops_total / 1e12 / (ms_cublas / 1e3),
        "tflops_triton_tuned": flops_total / 1e12 / (best_ms / 1e3),
        "tflops_triton_default": (
            flops_total / 1e12 / (ms_triton_default / 1e3)
            if ms_triton_default is not None else None
        ),
        "tflops_marlin_w4a16": flops_total / 1e12 / (ms_marlin / 1e3),
        "best_triton_tile": (
            f"{best_cfg['BLOCK_SIZE_M']}x{best_cfg['BLOCK_SIZE_N']}x{best_cfg['BLOCK_SIZE_K']}"
            f",{best_cfg['num_warps']}w,{best_cfg['num_stages']}s"
        ),
    }


# --------------------------------------------------------------------------- #
# Output: table, CSV, figure.
# --------------------------------------------------------------------------- #
def print_table(rows, include_default_tile):
    cols = ["M/e", "M_glob", "cuBLAS", "Triton(t)", "Marlin"]
    if include_default_tile:
        cols.insert(4, "Triton(d)")
    cols += ["cuBLAS_TF", "Marlin_TF", "TritonTile"]
    widths = [5, 7, 10, 10, 10, 10, 9, 9, 22] if include_default_tile else \
             [5, 7, 10, 10, 10, 9, 9, 22]
    line_w = sum(widths) + 3 * len(widths)
    print("=" * line_w)
    print("  ".join(f"{c:>{w}}" for c, w in zip(cols, widths)))
    print("-" * line_w)
    for r in rows:
        cells = [
            r["M_per_expert"], r["M_global"],
            f"{r['ms_cublas_bmm']:.3f}", f"{r['ms_triton_tuned']:.3f}",
        ]
        if include_default_tile:
            cells.append(f"{r['ms_triton_default']:.3f}")
        cells += [
            f"{r['ms_marlin_w4a16']:.3f}",
            f"{r['tflops_cublas_bmm']:.1f}", f"{r['tflops_marlin_w4a16']:.1f}",
            r["best_triton_tile"],
        ]
        print("  ".join(f"{c:>{w}}" for c, w in zip(cells, widths)))
    print("=" * line_w)


def write_csv(rows, path):
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def make_figure(rows, path, include_default_tile):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ms = [r["M_per_expert"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    ax = axes[0]
    ax.plot(ms, [r["ms_cublas_bmm"] for r in rows], "o-",
            label="cuBLAS BF16 bmm (no MoE routing)", linewidth=2)
    ax.plot(ms, [r["ms_triton_tuned"] for r in rows], "s-",
            label="Triton fused_moe BF16 (per-M tuned tile)", linewidth=2)
    if include_default_tile:
        ax.plot(ms, [r["ms_triton_default"] for r in rows], "x--",
                label="Triton fused_moe BF16 (default fallback tile)",
                color="gray", alpha=0.7)
    ax.plot(ms, [r["ms_marlin_w4a16"] for r in rows], "^-",
            label="Marlin W4A16", linewidth=2)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("Tokens per expert")
    ax.set_ylabel("Latency (ms, log)")
    ax.set_title("Latency")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="upper left")

    ax = axes[1]
    ax.plot(ms, [r["tflops_cublas_bmm"] for r in rows], "o-",
            label="cuBLAS BF16 bmm (upper bound)", linewidth=2)
    ax.plot(ms, [r["tflops_triton_tuned"] for r in rows], "s-",
            label="Triton fused_moe BF16 (tuned)", linewidth=2)
    if include_default_tile:
        ax.plot(ms, [r["tflops_triton_default"] for r in rows], "x--",
                label="Triton fused_moe BF16 (default)",
                color="gray", alpha=0.7)
    ax.plot(ms, [r["tflops_marlin_w4a16"] for r in rows], "^-",
            label="Marlin W4A16", linewidth=2)
    ax.axhline(A100_BF16_PEAK_TFLOPS, linestyle=":", color="red", alpha=0.6,
               label=f"A100 BF16 peak ({A100_BF16_PEAK_TFLOPS:.0f})")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Tokens per expert")
    ax.set_ylabel("Achieved TFLOPS")
    ax.set_title("Throughput (BF16-equivalent FLOPs/s)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9, loc="lower right")

    fig.suptitle(
        f"MoE kernel comparison on {torch.cuda.get_device_name(0)}  "
        f"(E={KERN_E}, K={KERN_K}, N={KERN_N}, top_k={KERN_TOP_K})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")


# --------------------------------------------------------------------------- #
# Entry point.
# --------------------------------------------------------------------------- #
DEFAULT_M_LIST = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


def main():
    if not CUDA_AVAILABLE:
        print("CUDA not available — skipping.")
        return

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=["sweep", "isolate"], default="sweep",
                        help="sweep = M ∈ DEFAULT_M_LIST; isolate = single --m-isolate")
    parser.add_argument("--m-isolate", type=int, default=8192,
                        help="M_per_expert for --mode isolate")
    parser.add_argument("--m-max", type=int, default=4096,
                        help="upper bound on M_per_expert for sweep mode")
    parser.add_argument("--no-default-tile", action="store_true",
                        help="skip the default-fallback Triton baseline (faster)")
    parser.add_argument("--no-figure", action="store_true",
                        help="skip the matplotlib figure")
    parser.add_argument("--out-dir", default="/tmp",
                        help="where to write CSV + PNG")
    args = parser.parse_args()

    install_tile_patch()
    device = torch.device("cuda")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    include_default_tile = not args.no_default_tile

    if args.mode == "isolate":
        print(f"GPU: {torch.cuda.get_device_name(0)}  "
              f"(BF16 peak {A100_BF16_PEAK_TFLOPS:.0f} TFLOPS)")
        print(f"Shape: E={KERN_E}, M/e={args.m_isolate}, K={KERN_K}, N={KERN_N}, "
              f"top_k={KERN_TOP_K}\n")
        row = measure_one_M(args.m_isolate, device,
                            include_default_tile=include_default_tile)
        print_table([row], include_default_tile)
        return

    # ---- sweep mode ----
    M_list = [m for m in DEFAULT_M_LIST if m <= args.m_max]
    M_max = max(M_list)
    print(f"GPU: {torch.cuda.get_device_name(0)}  "
          f"(BF16 peak {A100_BF16_PEAK_TFLOPS:.0f} TFLOPS)")
    print(f"Shape: E={KERN_E}, K={KERN_K}, N={KERN_N}, top_k={KERN_TOP_K}")
    print(f"Sweep: M_per_expert ∈ {M_list}\n")

    # Pre-allocate weights / max-M activation tensors so we don't reallocate
    # per row (save a few seconds on the long sweep).
    bmm_max = make_bf16_bmm_tensors(M_max, device)
    moe_w = make_bf16_moe_weights(device)
    marlin_w = make_int4_marlin_weights(device)

    rows = []
    t_start = time.time()
    print(f"{'M/e':>5} {'M_glob':>7} | running...")
    for M in M_list:
        row = measure_one_M(
            M, device,
            include_default_tile=include_default_tile,
            bf16_bmm_tensors=bmm_max,
            bf16_moe_weights=moe_w,
            marlin_weights=marlin_w,
        )
        rows.append(row)
        msg = (f"{M:>5} {row['M_global']:>7} | "
               f"cuBLAS={row['ms_cublas_bmm']:7.2f}ms  "
               f"Triton(t)={row['ms_triton_tuned']:7.2f}ms")
        if include_default_tile:
            msg += f"  Triton(d)={row['ms_triton_default']:7.2f}ms"
        msg += (f"  Marlin={row['ms_marlin_w4a16']:7.2f}ms  "
                f"tile={row['best_triton_tile']}")
        print(msg)

    print(f"\nSweep wall-time: {time.time() - t_start:.1f}s\n")
    print_table(rows, include_default_tile)

    csv_path = out_dir / "bench_kernel_compare.csv"
    write_csv(rows, csv_path)
    print(f"\nCSV: {csv_path}")
    if not args.no_figure:
        png_path = out_dir / "bench_kernel_compare.png"
        make_figure(rows, png_path, include_default_tile)
        print(f"PNG: {png_path}")


if __name__ == "__main__":
    main()
