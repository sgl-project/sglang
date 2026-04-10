"""Verify HeterMoE hypothesis: mixed precision lies between the two extremes.

Tests:
  1. Pure kernels: a16w4 < a8w8 < a16w16 (a8w8 deprecated, shown for reference)
  2. mix{a16w4 cold, a16w16 hot} lies between a16w4 and a16w16
  M = per-expert batch size (uniform distribution across 128 experts)

Usage: PYTHONPATH=python CUDA_VISIBLE_DEVICES=4 python3 scripts/heter_moe_verify_hypothesis.py
"""

import csv
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "python"))

import sglang.srt.server_args as sa

if sa._global_server_args is None:
    mock = object.__new__(sa.ServerArgs)
    mock.enable_deterministic_inference = False
    mock.disable_moe_autotuning = False
    sa._global_server_args = mock

from sglang.srt.layers.moe.fused_moe_triton.fused_moe import outplace_fused_experts
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import fused_marlin_moe
from sglang.srt.layers.moe.heter_policy import TokenCountPolicy

K, N, E, TOP_K = 2048, 768, 128, 8
GROUP_SIZE = 128
NUM_BITS = 4
COLD_RATIO = 0.8
WARMUP, ITERS = 50, 200
PER_EXPERT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
OUT_DIR = "/data/heter-moe/profiles/groupgemm"

_L2_FLUSH_SIZE = 50 * 1024 * 1024
_l2_flush_buf = None


def get_l2_flush_buf(device):
    global _l2_flush_buf
    if _l2_flush_buf is None or _l2_flush_buf.device != device:
        _l2_flush_buf = torch.empty(_L2_FLUSH_SIZE, dtype=torch.int8, device=device)
    return _l2_flush_buf


def flush_l2(device):
    buf = get_l2_flush_buf(device)
    buf.zero_()


def make_bf16_weights(device):
    w13 = torch.randn(E, 2 * N, K, dtype=torch.bfloat16, device=device)
    w2 = torch.randn(E, K, N, dtype=torch.bfloat16, device=device)
    return w13, w2


def make_int8_weights(device):
    w13 = torch.randint(-128, 127, (E, 2 * N, K), dtype=torch.int8, device=device)
    w2 = torch.randint(-128, 127, (E, K, N), dtype=torch.int8, device=device)
    s13 = torch.rand(E, 2 * N, 1, dtype=torch.float32, device=device) * 0.01
    s2 = torch.rand(E, K, 1, dtype=torch.float32, device=device) * 0.01
    return w13, w2, s13, s2


def make_int4_weights(device):
    w1 = torch.randint(
        0,
        2**31,
        (E, K // 16, 2 * N * (NUM_BITS // 2)),
        dtype=torch.int32,
        device=device,
    )
    w2 = torch.randint(
        0, 2**31, (E, N // 16, K * (NUM_BITS // 2)), dtype=torch.int32, device=device
    )
    s1 = (
        torch.ones(E, K // GROUP_SIZE, 2 * N, dtype=torch.bfloat16, device=device)
        * 0.01
    )
    s2 = torch.ones(E, N // GROUP_SIZE, K, dtype=torch.bfloat16, device=device) * 0.01
    return w1, w2, s1, s2


def make_inputs(m_per_expert, device):
    """Each expert gets exactly m_per_expert tokens.
    Global M = m_per_expert * E / TOP_K.
    """
    M_global = m_per_expert * E // TOP_K
    x = torch.randn(M_global, K, dtype=torch.bfloat16, device=device)
    topk_w = torch.ones(M_global, TOP_K, dtype=torch.bfloat16, device=device) / TOP_K
    all_expert_ids = torch.arange(E, device=device).repeat(m_per_expert)
    all_expert_ids = all_expert_ids[torch.randperm(len(all_expert_ids), device=device)]
    topk_ids = all_expert_ids.reshape(M_global, TOP_K)
    gating = torch.randn(M_global, E, dtype=torch.bfloat16, device=device)
    return x, topk_w, topk_ids, gating, M_global


def bench(fn, device, warmup=WARMUP, iters=ITERS, use_cuda_graph=True):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    graph = None
    if use_cuda_graph:
        try:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()
            for _ in range(warmup):
                graph.replay()
            torch.cuda.synchronize()
        except Exception:
            graph = None

    if graph is None:
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        flush_l2(device)
        torch.cuda.synchronize()
        s = torch.cuda.Event(enable_timing=True)
        e = torch.cuda.Event(enable_timing=True)
        s.record()
        if graph is not None:
            graph.replay()
        else:
            fn()
        e.record()
        torch.cuda.synchronize()
        times.append(s.elapsed_time(e))
    times.sort()
    return times[len(times) // 2]


def run_all(device):
    bf16_w13, bf16_w2 = make_bf16_weights(device)
    int8_w13, int8_w2, int8_s13, int8_s2 = make_int8_weights(device)
    int4_w1, int4_w2, int4_s1, int4_s2 = make_int4_weights(device)
    policy = TokenCountPolicy()

    rows = []
    header = (
        "M_per_expert",
        "M_global",
        "a16w16",
        "a16w4",
        "a8w8_deprecated",
        "mix_w4_w16",
    )

    for M in PER_EXPERT_BATCH_SIZES:
        x, topk_w, topk_ids, gating, M_global = make_inputs(M, device)

        lat_bf16 = bench(
            lambda: outplace_fused_experts(x, bf16_w13, bf16_w2, topk_w, topk_ids),
            device,
        )

        lat_int4 = bench(
            lambda: fused_marlin_moe(
                x,
                int4_w1,
                int4_w2,
                int4_s1,
                int4_s2,
                gating,
                topk_w,
                topk_ids,
                num_bits=4,
                is_k_full=True,
            ),
            device,
        )

        # a8w8: kept to show Triton INT8 is broken on A100 (deprecated)
        lat_int8 = bench(
            lambda: outplace_fused_experts(
                x,
                int8_w13,
                int8_w2,
                topk_w,
                topk_ids,
                use_int8_w8a8=True,
                per_channel_quant=True,
                w1_scale=int8_s13,
                w2_scale=int8_s2,
            ),
            device,
        )

        # mix{a16w4 cold, a16w16 hot}: full weight tensors, zero non-group weights
        plan = policy.assign(topk_ids, E, [COLD_RATIO, 1.0 - COLD_RATIO])
        cold_expert_list = plan.group_assignments[0]
        hot_expert_list = plan.group_assignments[1]

        cold_mask = torch.zeros(E, dtype=torch.bool, device=device)
        cold_mask[cold_expert_list] = True
        hot_mask = torch.zeros(E, dtype=torch.bool, device=device)
        hot_mask[hot_expert_list] = True

        cold_w = topk_w * cold_mask[topk_ids].to(topk_w.dtype)
        hot_w = topk_w * hot_mask[topk_ids].to(topk_w.dtype)

        def mix_w4_w16():
            fused_marlin_moe(
                x,
                int4_w1,
                int4_w2,
                int4_s1,
                int4_s2,
                gating,
                cold_w,
                topk_ids,
                num_bits=4,
                is_k_full=True,
            )
            outplace_fused_experts(x, bf16_w13, bf16_w2, hot_w, topk_ids)

        lat_mix = bench(mix_w4_w16, device, use_cuda_graph=True)

        row = (M, M_global, lat_bf16, lat_int4, lat_int8, lat_mix)
        rows.append(row)

        in_range = "✓" if lat_int4 <= lat_mix <= lat_bf16 else "✗"
        print(
            f"M/e={M:>5} (global={M_global:>6})  "
            f"a16w16={lat_bf16:7.3f}  a16w4={lat_int4:7.3f}  "
            f"a8w8*={lat_int8:7.3f}  mix={lat_mix:7.3f}  "
            f"[a16w4 < mix < a16w16: {in_range}]"
        )

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "hypothesis_verify.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r[0], r[1]] + [f"{v:.3f}" for v in r[2:]])
    print(f"\nCSV: {csv_path}")

    print(f"\n{'=' * 90}")
    print(
        f"{'M/e':>5} {'Mglob':>6} | {'a16w16':>8} {'a16w4':>8} {'a8w8*':>8} | "
        f"{'mix4+16':>8} | {'in range':>8}"
    )
    print("-" * 90)
    for r in rows:
        M, M_g, bf16, int4, int8, mix = r
        in_range = "✓" if int4 <= mix <= bf16 else "✗"
        print(
            f"{M:>5} {M_g:>6} | {bf16:8.3f} {int4:8.3f} {int8:8.3f} | "
            f"{mix:8.3f} | {in_range:>8}"
        )
    print("=" * 90)
    print("(* a8w8 deprecated: Triton INT8 ~6% A100 peak, shown for reference)")
    print(
        f"(mix = {{a16w4 cold {COLD_RATIO:.0%}, a16w16 hot {1 - COLD_RATIO:.0%}}}, "
        f"both kernels in single CUDA graph)"
    )

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        ms = PER_EXPERT_BATCH_SIZES
        ax.plot(
            ms, [r[2] for r in rows], "o-", label="a16w16", color="#1f77b4", linewidth=2
        )
        ax.plot(
            ms, [r[3] for r in rows], "s-", label="a16w4", color="#2ca02c", linewidth=2
        )
        ax.plot(
            ms,
            [r[4] for r in rows],
            "x--",
            label="a8w8 (deprecated)",
            color="#ff7f0e",
            linewidth=1,
            alpha=0.5,
        )
        ax.plot(
            ms,
            [r[5] for r in rows],
            "D-",
            label=f"mix{{w4,w16}} ({COLD_RATIO:.0%}/{1 - COLD_RATIO:.0%})",
            color="#9467bd",
            linewidth=2,
        )
        ax.set_xlabel("Tokens per Expert (M/e)")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"MoE Kernel Latency (E={E}, K={K}, N={N}, CUDA graph)")
        ax.set_xscale("log", base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(OUT_DIR, "hypothesis_verify.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Plot: {plot_path}")
    except ImportError:
        print("matplotlib not installed, skipping plot")

    os.makedirs("/data/heter-moe/results", exist_ok=True)
    with open("/data/heter-moe/results/kernel_comparison_summary.txt", "w") as f:
        f.write(
            f"{'M/e':>5} {'Mglob':>6} | {'a16w16':>8} {'a16w4':>8} {'a8w8*':>8} | "
            f"{'mix4+16':>8} | {'in range':>8}\n"
        )
        f.write("-" * 90 + "\n")
        for r in rows:
            M, M_g, bf16, int4, int8, mix = r
            in_range = "Y" if int4 <= mix <= bf16 else "N"
            f.write(
                f"{M:>5} {M_g:>6} | {bf16:8.3f} {int4:8.3f} {int8:8.3f} | "
                f"{mix:8.3f} | {in_range:>8}\n"
            )
        f.write(f"\n* a8w8 deprecated: Triton INT8 ~6% of A100 peak\n")
        f.write(
            f"mix = {{a16w4 cold {COLD_RATIO:.0%}, a16w16 hot {1 - COLD_RATIO:.0%}}}\n"
        )
        f.write(f"Both mix kernels captured in single CUDA graph\n")
    print("Summary: /data/heter-moe/results/kernel_comparison_summary.txt")

    return rows


if __name__ == "__main__":
    run_all(torch.device("cuda"))
