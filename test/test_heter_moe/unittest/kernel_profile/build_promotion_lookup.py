"""Build a single per-M lookup JSON for EfficiencyPromotionPolicy.

Input:
  - x_star_curve.csv (or optimal_assignment.all_x.csv): M_global -> winner_x
  - bf16_sparse_configs_sep.json: (n_active, bse) -> (up_tile, down_tile)

Output: efficiency_promotion_lookup.json mapping
  str(M_global) -> {
    "x_runtime":    int  (number of BF16-promoted experts at this M)
    "T":            int  (per-expert-load threshold; counts >= T → BF16)
    "m_per_expert": int  (the bse used for the tile lookup)
    "up_tile":      {BLOCK_SIZE_M, BLOCK_SIZE_N, ...} (or None if x_runtime==0)
    "down_tile":    {...} (or None if x_runtime==0)
  }

Threshold T derived from Zipf-expected per-expert counts at the chosen x*
boundary, NOT from per-step counts at runtime — runtime dispatch is then a
single integer compare with no GPU sort.

Generated values for x_runtime, T are derived assuming the synthetic Zipf
routing (alpha=1.1, seed=42) the curve was measured against.
"""

import csv
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _utils import KERN_E, make_zipf_routing  # noqa: E402

OUT_DIR = "test/test_heter_moe/unittest/kernel_profile/results"
ALL_X_CSV = f"{OUT_DIR}/optimal_assignment.all_x.csv"
SEP_JSON = f"{OUT_DIR}/bf16_sparse_configs_sep.json"
OUT_JSON = f"{OUT_DIR}/efficiency_promotion_lookup.json"


def derive_winner_per_M():
    """Per M_global, pick the x with smallest measured latency from
    optimal_assignment.all_x.csv (the median-of-3-runs ground truth)."""
    rows = {}
    with open(ALL_X_CSV) as f:
        r = csv.reader(f); next(r)
        for row in r:
            M = int(row[0]); x = int(row[1]); lat = float(row[2])
            rows.setdefault(M, []).append((x, lat))
    winner = {}
    for M, xs in rows.items():
        x_best, lat_best = min(xs, key=lambda p: p[1])
        winner[M] = x_best
    return winner


def derive_threshold(M_global, x_runtime):
    """T = midpoint of Zipf-expected per-expert counts at the x_runtime
    boundary. Snapped to nearest multiple of 8 (matches the runtime
    bf16_promotion_threshold convention).
    """
    if x_runtime == 0:
        return 10**9   # no promotion
    if x_runtime >= KERN_E:
        return 0       # everyone promoted
    _, _, _, _, expert_freq = make_zipf_routing(M_global, torch.device("cpu"), seed=42)
    counts = expert_freq.cpu().numpy()
    order = np.argsort(-counts)
    T_raw = (counts[order[x_runtime - 1]] + counts[order[x_runtime]]) / 2.0
    T_snap = int(round(T_raw / 8) * 8)
    return T_snap


def lookup_tile(sep_cfg: dict, n_active: int, m_per_expert: int):
    """Hierarchical-nearest lookup in the separated tune JSON. Returns
    (up_tile_dict, down_tile_dict, key_used). Pure-Python, no kwargs."""
    if n_active == 0:
        return None, None, "n/a"
    parsed = {}
    for k, v in sep_cfg.items():
        n_str, m_str = k.split("_")
        n = int(n_str[1:]); m = int(m_str[3:])
        parsed.setdefault(n, {})[m] = v
    n_keys = sorted(parsed.keys())
    n_match = n_keys[0]
    n_dist = abs(n_match - n_active)
    for nn in n_keys[1:]:
        d = abs(nn - n_active)
        if d < n_dist:
            n_dist = d; n_match = nn
    bse_keys = sorted(parsed[n_match].keys())
    m_match = bse_keys[0]
    m_dist = abs(m_match - m_per_expert)
    for mm in bse_keys[1:]:
        d = abs(mm - m_per_expert)
        if d < m_dist:
            m_dist = d; m_match = mm
    cell = parsed[n_match][m_match]
    up = {kk: vv for kk, vv in cell["up"].items() if not kk.startswith("_")}
    down = {kk: vv for kk, vv in cell["down"].items() if not kk.startswith("_")}
    return up, down, f"n{n_match}_bse{m_match}"


def main():
    winner = derive_winner_per_M()
    sep_cfg = json.load(open(SEP_JSON))

    out = {}
    for M, x in sorted(winner.items()):
        T = derive_threshold(M, x)
        # m_per_expert under expected Zipf at this M, x
        if x > 0:
            _, _, _, _, expert_freq = make_zipf_routing(
                M, torch.device("cpu"), seed=42)
            counts = expert_freq.cpu().numpy()
            order = np.argsort(-counts)
            m_per_expert = int(round(counts[order[:x]].mean()))
        else:
            m_per_expert = 0
        up, down, tile_key = lookup_tile(sep_cfg, x, m_per_expert)
        out[str(M)] = {
            "x_runtime": x,
            "T": T,
            "m_per_expert": m_per_expert,
            "tile_key": tile_key,
            "up_tile": up,
            "down_tile": down,
        }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {OUT_JSON} ({len(out)} entries)")
    for M, e in out.items():
        print(f"  M={M:>5}: x={e['x_runtime']:>3}, T={e['T']:>5}, "
              f"bse={e['m_per_expert']:>4}, tile={e['tile_key']}")


if __name__ == "__main__":
    main()
