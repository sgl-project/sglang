"""Task 1: characterize INT4 (cold) Marlin path latency.

Sweeps:
  n_cold       in {64, 72, 80, 88, 96, 104, 112, 120, 128}    (= 128 - x)
  m_per_expert in {8, 16, 24, 32, 48, 64, 96, 128}

For each (n_cold, m_per_expert):
  - Build inputs at uniform load (m_per_expert tokens per active expert).
  - Build sparse-active dispatch (only n_cold of 128 active, sentinel = E
    for inactive).
  - Run fused_marlin_moe (Marlin uses its own internal block_size_m
    heuristic — no autotune file consulted).
  - Time with median-of-50 + L2 flush + CUDA graph.

Output: results/int4_table.csv

Sharding: pass --n-cold to run a single n_cold value; the dispatcher in
run.sh splits n_cold values across GPUs 0..7.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _utils import (  # noqa: E402
    KERN_E,
    KERN_NUM_BITS,
    bench,
    make_int4_weights,
    make_uniform_inputs,
    sparse_active_dispatch,
    write_csv,
)

N_COLD_VALUES = [64, 72, 80, 88, 96, 104, 112, 120, 128]
# Extended grid to cover Zipf-induced loads: at M_global=8192 with x=0, all
# 128 experts active uniformly → bse ≈ 512. Extending past for safety.
M_PER_EXPERT_VALUES = [8, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384, 512]


def run_one_cell(n_cold: int, m_per_expert: int, w1, w2, s1, s2, device):
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
        fused_marlin_moe,
    )

    x, topk_w, topk_ids, gating = make_uniform_inputs(m_per_expert, device)
    M_global = x.shape[0]

    # Pick the first n_cold expert IDs as the "active" cold set. Choice of
    # which n_cold IDs is irrelevant under uniform load.
    active = torch.arange(n_cold, device=device, dtype=topk_ids.dtype)
    cold_ids, cold_w = sparse_active_dispatch(topk_ids, topk_w, active, "marlin")

    def fn():
        fused_marlin_moe(
            x, w1, w2, s1, s2, gating, cold_w, cold_ids,
            num_bits=KERN_NUM_BITS, is_k_full=True,
        )

    lat = bench(fn, device)
    return M_global, lat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-cold", type=int, default=None,
                        help="run only one n_cold value (for sharding)")
    parser.add_argument("--out", type=str,
                        default="test/test_heter_moe/unittest/kernel_profile/results/int4_table.csv")
    args = parser.parse_args()

    assert torch.cuda.is_available(), "CUDA required"
    device = torch.device("cuda")

    n_cold_set = [args.n_cold] if args.n_cold is not None else N_COLD_VALUES

    print(f"[task1] device={torch.cuda.get_device_name(0)} n_cold={n_cold_set}")
    t0 = time.perf_counter()

    w1, w2, s1, s2 = make_int4_weights(device, seed=0)

    rows = []
    for n_cold in n_cold_set:
        for m_pe in M_PER_EXPERT_VALUES:
            M_global, lat = run_one_cell(n_cold, m_pe, w1, w2, s1, s2, device)
            rows.append([n_cold, m_pe, M_global, f"{lat:.4f}"])
            print(f"  n_cold={n_cold:>3} m_pe={m_pe:>3} M={M_global:>5} lat={lat:.4f}ms")

    # Output: append-friendly per-shard files
    if args.n_cold is not None:
        out = args.out.replace(".csv", f".n{args.n_cold}.csv")
    else:
        out = args.out
    write_csv(out, ["n_cold", "m_per_expert", "M_global", "lat_ms"], rows)
    elapsed = time.perf_counter() - t0
    print(f"[task1] wrote {out} ({len(rows)} rows) in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
