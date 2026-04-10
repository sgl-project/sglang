"""Verify HeterMoE hypothesis: mixed precision is between the two extremes.

Tests three claims:
  1. Low batch (memory-bound):  a16w4 < mix{a16w4,a16w16} < a16w16
  2. Large batch (compute-bound): a8w8 < mix{a8w8,a16w16} < a16w16
  3. Optimal mixed: mix{a16w4 cold, a8w8 hot} across all batch sizes

Usage: PYTHONPATH=python CUDA_VISIBLE_DEVICES=7 python3 scripts/heter_moe_verify_hypothesis.py
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
# M = per-expert batch size (tokens per expert, uniform distribution)
# Global batch = M * E / TOP_K
PER_EXPERT_BATCH_SIZES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
OUT_DIR = "/data/heter-moe/profiles/groupgemm"

# L2 flush: allocate a buffer larger than L2 cache (A100 = 40MB L2)
_L2_FLUSH_SIZE = 50 * 1024 * 1024  # 50MB > 40MB L2
_l2_flush_buf = None


def get_l2_flush_buf(device):
    global _l2_flush_buf
    if _l2_flush_buf is None or _l2_flush_buf.device != device:
        _l2_flush_buf = torch.empty(_L2_FLUSH_SIZE, dtype=torch.int8, device=device)
    return _l2_flush_buf


def flush_l2(device):
    """Write to a large buffer to evict all L2 cache lines."""
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
    """Create inputs where each expert gets exactly m_per_expert tokens.

    Global M = m_per_expert * E / TOP_K (each token routed to TOP_K experts).
    topk_ids constructed so expert i appears exactly m_per_expert times.
    """
    M_global = m_per_expert * E // TOP_K
    x = torch.randn(M_global, K, dtype=torch.bfloat16, device=device)
    topk_w = torch.ones(M_global, TOP_K, dtype=torch.bfloat16, device=device) / TOP_K

    # Build topk_ids so each expert gets exactly m_per_expert tokens:
    # Total slots = M_global * TOP_K = m_per_expert * E
    # Fill with E experts, each repeated m_per_expert times, then reshape
    all_expert_ids = torch.arange(E, device=device).repeat(m_per_expert)
    all_expert_ids = all_expert_ids[torch.randperm(len(all_expert_ids), device=device)]
    topk_ids = all_expert_ids.reshape(M_global, TOP_K)

    gating = torch.randn(M_global, E, dtype=torch.bfloat16, device=device)
    return x, topk_w, topk_ids, gating, M_global


def mask_weights(topk_w, topk_ids, active_experts):
    mask = torch.zeros(E, dtype=torch.bool, device=topk_w.device)
    mask[active_experts] = True
    return topk_w * mask[topk_ids].to(topk_w.dtype)


def bench(fn, device, warmup=WARMUP, iters=ITERS, use_cuda_graph=True):
    # Warmup (eager)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()

    graph = None
    if use_cuda_graph:
        try:
            # Capture CUDA graph to eliminate launch overhead
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                fn()
            torch.cuda.synchronize()
            # Warmup the graph replay
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


def run_all(device, use_graph=False):
    bf16_w13, bf16_w2 = make_bf16_weights(device)
    int8_w13, int8_w2, int8_s13, int8_s2 = make_int8_weights(device)
    int4_w1, int4_w2, int4_s1, int4_s2 = make_int4_weights(device)
    policy = TokenCountPolicy()

    graph_label = "graph=ON" if use_graph else "graph=OFF"
    print(f"\n{'=' * 80}")
    print(
        f"Running with CUDA graph: {use_graph}  (all kernels same setting for fairness)"
    )
    print(f"{'=' * 80}\n")

    rows = []
    header = (
        "M_per_expert",
        "M_global",
        "a16w16",
        "a16w4",
        "a8w8",
        "mix_w4_w16",
        "mix_w8_w16",
        "mix_w4_w8",
    )

    for M in PER_EXPERT_BATCH_SIZES:
        x, topk_w, topk_ids, gating, M_global = make_inputs(M, device)

        # Pure a16w16
        lat_bf16 = bench(
            lambda: outplace_fused_experts(x, bf16_w13, bf16_w2, topk_w, topk_ids),
            device,
            use_cuda_graph=use_graph,
        )

        # Pure a16w4
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
            use_cuda_graph=use_graph,
        )

        # Pure a8w8
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
            use_cuda_graph=use_graph,
        )

        # Mixed assignments via policy
        plan = policy.assign(topk_ids, E, [COLD_RATIO, 1.0 - COLD_RATIO])
        cold_ids = torch.tensor(plan.group_assignments[0], device=device)
        hot_ids = torch.tensor(plan.group_assignments[1], device=device)
        cold_w = mask_weights(topk_w, topk_ids, cold_ids)
        hot_w = mask_weights(topk_w, topk_ids, hot_ids)

        # mix{a16w4 cold, a16w16 hot}
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

        lat_mix_w4_w16 = bench(mix_w4_w16, device, use_cuda_graph=False)

        # mix{a8w8 cold, a16w16 hot}
        def mix_w8_w16():
            outplace_fused_experts(
                x,
                int8_w13,
                int8_w2,
                cold_w,
                topk_ids,
                use_int8_w8a8=True,
                per_channel_quant=True,
                w1_scale=int8_s13,
                w2_scale=int8_s2,
            )
            outplace_fused_experts(x, bf16_w13, bf16_w2, hot_w, topk_ids)

        lat_mix_w8_w16 = bench(mix_w8_w16, device, use_cuda_graph=False)

        # mix{a16w4 cold, a8w8 hot}
        def mix_w4_w8():
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
            outplace_fused_experts(
                x,
                int8_w13,
                int8_w2,
                hot_w,
                topk_ids,
                use_int8_w8a8=True,
                per_channel_quant=True,
                w1_scale=int8_s13,
                w2_scale=int8_s2,
            )

        lat_mix_w4_w8 = bench(mix_w4_w8, device, use_cuda_graph=False)

        row = (
            M,
            M_global,
            lat_bf16,
            lat_int4,
            lat_int8,
            lat_mix_w4_w16,
            lat_mix_w8_w16,
            lat_mix_w4_w8,
        )
        rows.append(row)
        print(
            f"M/e={M:>5} (global={M_global:>6})  "
            f"bf16={lat_bf16:7.3f}  int4={lat_int4:7.3f}  int8={lat_int8:7.3f}  "
            f"mix_w4w16={lat_mix_w4_w16:7.3f}  mix_w8w16={lat_mix_w8_w16:7.3f}  "
            f"mix_w4w8={lat_mix_w4_w8:7.3f}"
        )

    # Save CSV
    os.makedirs(OUT_DIR, exist_ok=True)
    suffix = "graph_on" if use_graph else "graph_off"
    csv_path = os.path.join(OUT_DIR, f"hypothesis_verify_{suffix}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(header)
        for r in rows:
            w.writerow([r[0], r[1]] + [f"{v:.3f}" for v in r[2:]])
    print(f"\nCSV: {csv_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print(
        f"{'M/e':>5} {'Mglob':>6} | {'a16w16':>8} {'a16w4':>8} {'a8w8':>8} | "
        f"{'mix4+16':>8} {'mix8+16':>8} {'mix4+8':>8} | {'best':>8}"
    )
    print("-" * 100)
    for r in rows:
        M, M_g, bf16, int4, int8, m4_16, m8_16, m4_8 = r
        all_lats = {
            "a16w16": bf16,
            "a16w4": int4,
            "a8w8": int8,
            "mix4+16": m4_16,
            "mix8+16": m8_16,
            "mix4+8": m4_8,
        }
        best = min(all_lats, key=all_lats.get)
        print(
            f"{M:>5} {M_g:>6} | {bf16:8.3f} {int4:8.3f} {int8:8.3f} | "
            f"{m4_16:8.3f} {m8_16:8.3f} {m4_8:8.3f} | {best:>8}"
        )
    print("=" * 100)
    print(f"(M/e = tokens per expert, uniform distribution, {graph_label})")

    # Plot
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 6))
        labels = ["a16w16", "a16w4", "a8w8", "mix{w4,w16}", "mix{w8,w16}", "mix{w4,w8}"]
        colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#d62728", "#8c564b"]
        for i, label in enumerate(labels):
            vals = [r[i + 2] for r in rows]
            ax.plot(
                PER_EXPERT_BATCH_SIZES,
                vals,
                "o-",
                label=label,
                color=colors[i],
                linewidth=2,
            )
        ax.set_xlabel("Tokens per Expert (M/e)")
        ax.set_ylabel("Latency (ms)")
        ax.set_title(f"MoE Kernel Latency ({graph_label}, E={E}, K={K}, N={N})")
        ax.set_xscale("log", base=2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plot_path = os.path.join(OUT_DIR, f"hypothesis_verify_{suffix}.png")
        plt.savefig(plot_path, dpi=150)
        print(f"Plot: {plot_path}")
    except ImportError:
        print("matplotlib not installed, skipping plot")

    return rows


if __name__ == "__main__":
    device = torch.device("cuda")

    # Run both modes for fair comparison
    rows_no_graph = run_all(device, use_graph=False)
    rows_graph = run_all(device, use_graph=True)

    # Save combined summary
    os.makedirs("/data/heter-moe/results", exist_ok=True)
    summary_path = "/data/heter-moe/results/kernel_comparison_summary.txt"
    with open(summary_path, "w") as f:
        for label, rows in [
            ("NO CUDA GRAPH (fair)", rows_no_graph),
            ("WITH CUDA GRAPH", rows_graph),
        ]:
            f.write(f"\n{'=' * 100}\n{label}\n{'=' * 100}\n")
            f.write(
                f"{'M/e':>5} {'Mglob':>6} | {'a16w16':>8} {'a16w4':>8} {'a8w8':>8} | "
                f"{'mix4+16':>8} {'mix8+16':>8} {'mix4+8':>8} | {'best':>8}\n"
            )
            f.write("-" * 100 + "\n")
            for r in rows:
                M, M_g, bf16, int4, int8, m4_16, m8_16, m4_8 = r
                all_lats = {
                    "a16w16": bf16,
                    "a16w4": int4,
                    "a8w8": int8,
                    "mix4+16": m4_16,
                    "mix8+16": m8_16,
                    "mix4+8": m4_8,
                }
                best = min(all_lats, key=all_lats.get)
                f.write(
                    f"{M:>5} {M_g:>6} | {bf16:8.3f} {int4:8.3f} {int8:8.3f} | "
                    f"{m4_16:8.3f} {m8_16:8.3f} {m4_8:8.3f} | {best:>8}\n"
                )
            f.write(f"(M/e = tokens per expert, uniform, cold_ratio={COLD_RATIO})\n")
    print(f"\nSummary: {summary_path}")
