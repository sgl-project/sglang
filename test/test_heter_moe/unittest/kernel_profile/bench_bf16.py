"""Task 2.bench: characterize BF16 (hot) Triton path latency at sparse activation,
using autotuned tiles from Task 2.prelim (bf16_sparse_configs.json).

Sweeps:
  n_hot        in {8, 16, 24, 32, 40, 48, 56, 64}
  m_per_expert in {32, 48, 64, ..., 512} step 16  (31 values)

For each (n_hot, m_per_expert):
  - Hierarchical-nearest lookup: nearest n_active in JSON, then nearest bse.
  - Inject tile via override_config(...) and run outplace_fused_experts on
    sparse-active dispatch (only n_hot of 128 active, sentinel = -1 for
    inactive).
  - Time with median-of-50 + L2 flush + CUDA graph.

Output: results/bf16_table.csv (columns: n_hot, m_per_expert, M_global,
        tile_key_used, lat_ms).

Sharding: --n-hot for one value at a time.
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
    bench,
    hierarchical_lookup,
    make_bf16_weights,
    make_uniform_inputs,
    read_json,
    sparse_active_dispatch,
    write_csv,
)

N_HOT_VALUES = [8, 16, 24, 32, 40, 48, 56, 64]
# Extended grid: under Zipf, hot experts can see thousands of tokens (e.g. at
# M=8192, x=8, top-8 capture ~47% of routing → ~3.8k tokens/hot-expert).
M_PER_EXPERT_VALUES = (
    list(range(32, 512 + 1, 16))            # 32, 48, ..., 512  (31)
    + [640, 768, 1024, 1280, 1536, 2048, 3072, 4096]  # extension (8)
)


def run_one_cell(n_hot: int, m_per_expert: int, configs, w13, w2, device):
    from sglang.srt.layers.moe.fused_moe_triton import override_config
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
        outplace_fused_experts,
    )

    x, topk_w, topk_ids, _ = make_uniform_inputs(m_per_expert, device, seed=0)
    M_global = x.shape[0]
    active = torch.arange(n_hot, device=device, dtype=topk_ids.dtype)
    hot_ids, hot_w = sparse_active_dispatch(topk_ids, topk_w, active, "triton")

    key, tile = hierarchical_lookup(configs, n_hot, m_per_expert)
    # Strip metadata fields (the autotune script writes _lat_ms / _tune_seconds).
    tile_pure = {k: v for k, v in tile.items() if not k.startswith("_")}

    def fn():
        with override_config(tile_pure):
            outplace_fused_experts(x, w13, w2, hot_w, hot_ids)

    lat = bench(fn, device)
    return M_global, key, lat


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-hot", type=int, default=None,
                        help="run only one n_hot value (for sharding)")
    parser.add_argument("--configs", type=str,
                        default="test/test_heter_moe/unittest/kernel_profile/results/bf16_sparse_configs.json")
    parser.add_argument("--out", type=str,
                        default="test/test_heter_moe/unittest/kernel_profile/results/bf16_table.csv")
    args = parser.parse_args()

    assert torch.cuda.is_available()
    device = torch.device("cuda")
    configs = read_json(args.configs)
    print(f"[task2.bench] device={torch.cuda.get_device_name(0)} "
          f"loaded {len(configs)} tile cells from {args.configs}")

    n_hot_set = [args.n_hot] if args.n_hot is not None else N_HOT_VALUES

    t0 = time.perf_counter()
    w13, w2 = make_bf16_weights(device, seed=0)

    rows = []
    for n_hot in n_hot_set:
        for m_pe in M_PER_EXPERT_VALUES:
            M_global, key, lat = run_one_cell(n_hot, m_pe, configs, w13, w2, device)
            rows.append([n_hot, m_pe, M_global, key, f"{lat:.4f}"])
            print(f"  n_hot={n_hot:>3} m_pe={m_pe:>3} M={M_global:>5} "
                  f"tile={key:<12} lat={lat:.4f}ms")

    if args.n_hot is not None:
        out = args.out.replace(".csv", f".n{args.n_hot}.csv")
    else:
        out = args.out
    write_csv(out, ["n_hot", "m_per_expert", "M_global", "tile_key", "lat_ms"], rows)
    elapsed = time.perf_counter() - t0
    print(f"[task2.bench] wrote {out} ({len(rows)} rows) in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
