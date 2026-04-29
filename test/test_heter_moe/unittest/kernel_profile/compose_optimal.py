"""Task 3: compose Tasks 1+2 microbench tables, find x*(M_global), and validate.

For each M_global ∈ M_GLOBAL_VALUES (24 points spanning decode→heavy prefill):
  1. Build a Zipf-distributed routing for that M_global (synthetic; matches
     the shape used in scripts/heter_moe_collect_routing.py).
  2. For each x ∈ {0, 8, 16, 24, 32, 40, 48, 56, 64}:
     - Promotion policy: top-x experts by routing frequency are BF16 (hot);
       rest are INT4 (cold).
     - Look up t_int4(128-x, bse_cold) from int4_table.csv (bilinear in
       (n_cold, m_per_expert)).
     - Look up t_bf16(x, bse_hot) from bf16_table.csv (same bilinear).
     - t_pred(M_global, x) = t_int4 + t_bf16.
  3. Pick x* = argmin_x t_pred.
  4. Translate to threshold T*: per-expert-load cutoff that separates hot
     from cold at x*. Snap to nearest multiple of 8.
  5. Validation: actually run the paired sparse-active kernel call (cold
     INT4 on 128-x* experts + hot BF16 on x* experts) with the *real* Zipf
     routing AND PIN the autotuned BF16 tile via override_config. Measure
     t_meas.
  6. Compute agreement = |t_meas - t_pred|/t_pred. Pass if ≤ 10%.
  7. At lowest and highest M_global, also report a random-promotion baseline.

Output: results/optimal_assignment.csv + a printed table.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from _utils import (  # noqa: E402
    KERN_E,
    KERN_NUM_BITS,
    KERN_TOP_K,
    bench,
    hierarchical_lookup,
    make_bf16_weights,
    make_int4_weights,
    make_zipf_routing,
    override_split_config,
    read_csv,
    read_json,
    sparse_active_dispatch,
    write_csv,
)

# 11 M_global values that match exact-tuned keys in the production
# E=128,N=768,device_name=NVIDIA_A100-SXM4-80GB.json. Anywhere else, the
# nearest-neighbor lookup makes the comparison apples-to-oranges (production
# falls back to a tile tuned for a different M).
M_GLOBAL_VALUES = [
    32, 64, 96, 128, 256, 512, 1024, 1536, 2048, 3072, 4096,
]
# 17 x candidates: x=0 (pure INT4 baseline) + 16 BF16 promotion levels
# matching the autotune column grid {4, 8, ..., 64}.
X_CANDIDATES = [0] + list(range(4, 64 + 1, 4))
TOLERANCE = 0.10


def load_int4_table(path: str) -> np.ndarray:
    header, rows = read_csv(path)
    # columns: n_cold, m_per_expert, M_global, lat_ms
    out = {}
    for r in rows:
        n_cold = int(r[0]); m_pe = int(r[1]); lat = float(r[3])
        out.setdefault(n_cold, {})[m_pe] = lat
    return out


def load_bf16_table(path: str) -> Dict[int, Dict[int, Tuple[str, float]]]:
    header, rows = read_csv(path)
    # columns: n_hot, m_per_expert, M_global, tile_key, lat_ms
    out = {}
    for r in rows:
        n_hot = int(r[0]); m_pe = int(r[1]); tile_key = r[3]; lat = float(r[4])
        out.setdefault(n_hot, {})[m_pe] = (tile_key, lat)
    return out


def lookup_lat(table: dict, n: int, bse: int, with_tile: bool = False):
    """Bilinear-on-grid nearest-neighbor lookup. Tables are sparse but
    monotonic enough that nearest-neighbor is fine for prediction."""
    if n not in table:
        n_keys = sorted(table.keys())
        n = min(n_keys, key=lambda nn: abs(nn - n))
    bse_keys = sorted(table[n].keys())
    bse_nn = min(bse_keys, key=lambda mm: abs(mm - bse))
    val = table[n][bse_nn]
    if with_tile:
        return val  # (tile_key, lat)
    return val if not isinstance(val, tuple) else val[1]


def derive_loads(expert_freq: torch.Tensor, x: int) -> Tuple[float, float, list]:
    """Given per-expert routing counts (sum over batch), return:
      bse_cold = mean tokens per cold (INT4) expert
      bse_hot  = mean tokens per hot (BF16) expert
      hot_set  = list of expert IDs promoted to BF16 (top-x by frequency)
    """
    counts = expert_freq.cpu().numpy()
    order = np.argsort(-counts)  # descending
    hot_set = order[:x].tolist()
    cold_set = order[x:].tolist()
    bse_hot = (counts[hot_set].mean()) if x > 0 else 0.0
    bse_cold = (counts[cold_set].mean()) if (KERN_E - x) > 0 else 0.0
    return float(bse_cold), float(bse_hot), hot_set


def predict(int4_tab, bf16_tab, x: int, bse_cold: float, bse_hot: float):
    n_cold = KERN_E - x
    if x == 0:
        t_bf16 = 0.0
        tile_key = "n/a"
    else:
        tile_key, t_bf16 = lookup_lat(bf16_tab, x, int(round(bse_hot)), with_tile=True)
    if n_cold == 0:
        t_int4 = 0.0
    else:
        t_int4 = lookup_lat(int4_tab, n_cold, int(round(bse_cold)))
    return t_int4 + t_bf16, t_int4, t_bf16, tile_key


def measure(M_global: int, x: int, bf16_configs, hot_set, cold_set,
            bf16_w13, bf16_w2, int4_w1, int4_w2, int4_s1, int4_s2,
            device, seed=42):
    from sglang.srt.layers.moe.fused_moe_triton import override_config
    from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
        fused_marlin_moe,
    )
    from sglang.srt.layers.moe.fused_moe_triton.fused_moe import (
        outplace_fused_experts,
    )

    x_, topk_w, topk_ids, gating, expert_freq = make_zipf_routing(
        M_global, device, seed=seed
    )

    # Cold (INT4) — sparse activation on cold_set
    cold_active = torch.tensor(cold_set, device=device, dtype=topk_ids.dtype)
    cold_ids, cold_w = sparse_active_dispatch(topk_ids, topk_w, cold_active, "marlin")

    # Hot (BF16) — sparse activation on hot_set, with PINNED autotuned tile
    # (separated up + down). For the new bf16_sparse_configs_sep.json each
    # cell has both "up" and "down" tile dicts; we use override_split_config
    # to inject both into try_get_optimal_moe_config.
    use_split = False
    up_tile = down_tile = None
    if x > 0:
        hot_active = torch.tensor(hot_set, device=device, dtype=topk_ids.dtype)
        hot_ids, hot_w = sparse_active_dispatch(topk_ids, topk_w, hot_active, "triton")
        bse_hot = float(expert_freq[hot_active].float().mean().item())
        tile_key, tile_meta = hierarchical_lookup(bf16_configs, x, int(round(bse_hot)))
        if isinstance(tile_meta, dict) and "up" in tile_meta and "down" in tile_meta:
            up_tile = {k: v for k, v in tile_meta["up"].items() if not k.startswith("_")}
            down_tile = {k: v for k, v in tile_meta["down"].items() if not k.startswith("_")}
            use_split = True
        else:
            tile_pure = {k: v for k, v in tile_meta.items() if not k.startswith("_")}
    else:
        tile_key = "n/a"
        hot_ids = hot_w = None

    if x > 0 and (KERN_E - x) > 0:
        if use_split:
            def fn():
                fused_marlin_moe(
                    x_, int4_w1, int4_w2, int4_s1, int4_s2, gating, cold_w, cold_ids,
                    num_bits=KERN_NUM_BITS, is_k_full=True,
                )
                with override_split_config(up_tile, down_tile):
                    outplace_fused_experts(x_, bf16_w13, bf16_w2, hot_w, hot_ids)
        else:
            def fn():
                fused_marlin_moe(
                    x_, int4_w1, int4_w2, int4_s1, int4_s2, gating, cold_w, cold_ids,
                    num_bits=KERN_NUM_BITS, is_k_full=True,
                )
                with override_config(tile_pure):
                    outplace_fused_experts(x_, bf16_w13, bf16_w2, hot_w, hot_ids)
    elif x == 0:
        def fn():
            fused_marlin_moe(
                x_, int4_w1, int4_w2, int4_s1, int4_s2, gating, cold_w, cold_ids,
                num_bits=KERN_NUM_BITS, is_k_full=True,
            )
    else:  # x == E (all bf16; not in our X_CANDIDATES but covered defensively)
        if use_split:
            def fn():
                with override_split_config(up_tile, down_tile):
                    outplace_fused_experts(x_, bf16_w13, bf16_w2, hot_w, hot_ids)
        else:
            def fn():
                with override_config(tile_pure):
                    outplace_fused_experts(x_, bf16_w13, bf16_w2, hot_w, hot_ids)

    return bench(fn, device), tile_key


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--int4-table", type=str,
        default="test/test_heter_moe/unittest/kernel_profile/results/int4_table.csv")
    parser.add_argument(
        "--bf16-table", type=str,
        default="test/test_heter_moe/unittest/kernel_profile/results/bf16_table.csv")
    parser.add_argument(
        "--bf16-configs", type=str,
        default="test/test_heter_moe/unittest/kernel_profile/results/bf16_sparse_configs_sep.json")
    parser.add_argument(
        "--out", type=str,
        default="test/test_heter_moe/unittest/kernel_profile/results/optimal_assignment.csv")
    parser.add_argument(
        "--no-validate", action="store_true",
        help="Skip the actual kernel measurement (predict-only)")
    parser.add_argument(
        "--measure-all-x", action="store_true",
        help="In addition to predicted x*, measure every x candidate at every "
             "M_global to get the ground-truth optimum.")
    parser.add_argument(
        "--m-global", type=int, default=None,
        help="Run only one M_global value (for sharding measure-all-x).")
    args = parser.parse_args()

    int4_tab = load_int4_table(args.int4_table)
    bf16_tab = load_bf16_table(args.bf16_table)
    bf16_configs = read_json(args.bf16_configs)
    print(f"[task3] int4 cells={sum(len(v) for v in int4_tab.values())} | "
          f"bf16 cells={sum(len(v) for v in bf16_tab.values())} | "
          f"bf16 tile cells={len(bf16_configs)}")

    if not args.no_validate:
        assert torch.cuda.is_available()
        device = torch.device("cuda")
        print(f"[task3] device={torch.cuda.get_device_name(0)}")
        bf16_w13, bf16_w2 = make_bf16_weights(device, seed=0)
        int4_w1, int4_w2, int4_s1, int4_s2 = make_int4_weights(device, seed=0)
    else:
        device = None
        bf16_w13 = bf16_w2 = int4_w1 = int4_w2 = int4_s1 = int4_s2 = None

    M_set = [args.m_global] if args.m_global is not None else M_GLOBAL_VALUES
    all_x_rows = []  # ground-truth latency at every (M_global, x)

    rows = []
    for M_global in M_set:
        # 1. Routing distribution at this M_global
        # We always seed=42 so the SAME distribution is used for predict
        # AND validation — that's the point.
        x_, topk_w, topk_ids, gating, expert_freq = make_zipf_routing(
            M_global, torch.device("cuda" if device else "cpu"), seed=42
        )

        # 2. Predict for every x candidate
        preds = []
        for x in X_CANDIDATES:
            bse_cold, bse_hot, hot_set = derive_loads(expert_freq, x)
            t_pred, t_int4, t_bf16, tile_key = predict(
                int4_tab, bf16_tab, x, bse_cold, bse_hot)
            preds.append((x, t_pred, t_int4, t_bf16, bse_cold, bse_hot,
                          hot_set, tile_key))

        # 3. Best x
        x_star, t_pred_star, t_int4_star, t_bf16_star, \
            bse_cold_star, bse_hot_star, hot_set_star, tile_key_star = min(
                preds, key=lambda p: p[1])
        cold_set_star = [i for i in range(KERN_E) if i not in set(hot_set_star)]

        # 4. Threshold T* (raw and snapped to multiple of 8)
        counts = expert_freq.cpu().numpy()
        order = np.argsort(-counts)
        if x_star == 0:
            T_raw = float(counts.max() + 1)  # nothing promoted
        elif x_star == KERN_E:
            T_raw = 0.0
        else:
            T_raw = (counts[order[x_star - 1]] + counts[order[x_star]]) / 2.0
        T_snap = int(round(T_raw / 8) * 8)

        # 5. Validate (measure) — sweep all x if requested
        if not args.no_validate:
            if args.measure_all_x:
                meas_by_x = {}
                for x in X_CANDIDATES:
                    bse_cold_x, bse_hot_x, hot_set_x = derive_loads(expert_freq, x)
                    cold_set_x = [i for i in range(KERN_E) if i not in set(hot_set_x)]
                    t_x, tile_x = measure(
                        M_global, x, bf16_configs, hot_set_x, cold_set_x,
                        bf16_w13, bf16_w2, int4_w1, int4_w2, int4_s1, int4_s2,
                        device)
                    meas_by_x[x] = t_x
                    all_x_rows.append([M_global, x, f"{t_x:.4f}", tile_x])
                    print(f"    [meas-all-x] M={M_global:>5} x={x:>2} "
                          f"t={t_x:.4f}ms tile={tile_x}")
                # True optimum from measurements
                x_meas_star = min(meas_by_x, key=meas_by_x.get)
                t_meas = meas_by_x[x_star]
                t_meas_true_star = meas_by_x[x_meas_star]
            else:
                t_meas, tile_key_meas = measure(
                    M_global, x_star, bf16_configs, hot_set_star, cold_set_star,
                    bf16_w13, bf16_w2, int4_w1, int4_w2, int4_s1, int4_s2, device)
                x_meas_star = x_star
                t_meas_true_star = t_meas
            agreement = abs(t_meas - t_pred_star) / t_pred_star * 100
            passed = "Y" if agreement <= TOLERANCE * 100 else "N"
        else:
            t_meas = float("nan"); agreement = float("nan"); passed = "?"
            tile_key_meas = tile_key_star
            x_meas_star = x_star
            t_meas_true_star = float("nan")

        # 6. Random baseline at extremes
        if M_global in (M_GLOBAL_VALUES[0], M_GLOBAL_VALUES[-1]) and not args.no_validate:
            rng = np.random.default_rng(7)
            rand_perm = rng.permutation(KERN_E)
            rand_hot = rand_perm[:x_star].tolist()
            rand_cold = rand_perm[x_star:].tolist()
            counts_arr = expert_freq.cpu().numpy()
            bse_hot_rand = float(counts_arr[rand_hot].mean()) if x_star > 0 else 0.0
            bse_cold_rand = (
                float(counts_arr[rand_cold].mean()) if (KERN_E - x_star) > 0 else 0.0
            )
            t_pred_rand, *_ = predict(
                int4_tab, bf16_tab, x_star, bse_cold_rand, bse_hot_rand)
            t_meas_rand, _ = measure(
                M_global, x_star, bf16_configs, rand_hot, rand_cold,
                bf16_w13, bf16_w2, int4_w1, int4_w2, int4_s1, int4_s2, device)
        else:
            t_pred_rand = float("nan"); t_meas_rand = float("nan")

        rows.append([
            M_global, x_star, f"{T_raw:.1f}", T_snap,
            f"{t_pred_star:.4f}", f"{t_meas:.4f}",
            f"{agreement:.2f}", passed,
            tile_key_star,
            f"{t_pred_rand:.4f}", f"{t_meas_rand:.4f}",
            x_meas_star, f"{t_meas_true_star:.4f}",
        ])
        print(f"  M={M_global:>5} x*={x_star:>2} T*={T_snap:>4} "
              f"t_pred={t_pred_star:.4f}ms t_meas={t_meas:.4f}ms "
              f"({agreement:5.2f}% {passed}) tile={tile_key_star} "
              f"| x_meas*={x_meas_star} t_meas*={t_meas_true_star:.4f}")

    if args.m_global is not None:
        out = args.out.replace(".csv", f".M{args.m_global}.csv")
    else:
        out = args.out
    write_csv(out, [
        "M_global", "x_star_pred", "T_star_raw", "T_star_snapped",
        "t_pred_ms", "t_meas_at_pred_x_ms", "agreement_pct", "pass",
        "tile_key",
        "t_pred_random_ms", "t_meas_random_ms",
        "x_star_meas", "t_meas_at_meas_x_ms",
    ], rows)
    print(f"[task3] wrote {out}")

    if all_x_rows:
        ax_out = out.replace(".csv", ".all_x.csv")
        write_csv(ax_out, ["M_global", "x", "lat_ms", "tile_key"], all_x_rows)
        print(f"[task3] wrote {ax_out} ({len(all_x_rows)} rows)")

    # Pretty-print table
    print()
    print("=" * 130)
    print(f"{'M_global':>8} {'x*pred':>6} {'T*':>4} {'t_pred':>9} {'t@pred':>9} "
          f"{'agree%':>7} {'pass':>4} {'tile':<14} "
          f"{'x*meas':>6} {'t@meas':>9}")
    print("-" * 130)
    for r in rows:
        print(f"{r[0]:>8} {r[1]:>6} {r[3]:>4} {r[4]:>9} {r[5]:>9} "
              f"{r[6]:>7} {r[7]:>4} {r[8]:<14} "
              f"{r[11]:>6} {r[12]:>9}")
    print("=" * 130)


if __name__ == "__main__":
    main()
