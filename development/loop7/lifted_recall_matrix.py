"""Loop-7 lifted-budget served-recall matrix (DS-default top_k=2048 vs
DS-lifted lifted_budget_top_k=4096), same node, EAGER, NIAH 4K.

Reuses the Clopper-Pearson exact-binomial CI + the up-front DIRECTIONAL
materiality rule from niah_recall_matrix.py: a lifted UPLIFT over the DS-default
baseline is MATERIAL only when the lifted recall point EXCEEDS the DS-default
baseline's 95% CP CI HIGH (h > base_hi). Both servers are measured EAGER on the
same node so the comparison isolates the BUDGET (2048 vs 4096), not the
eager-vs-graph upstream-numerics gap.

Usage:
    python development/loop7/lifted_recall_matrix.py \
        --ds-default development/loop7/niah_ds_default2048_eager.json \
        --ds-lifted  development/loop7/niah_ds_lifted4096.json \
        --out development/loop7/ds_lifted_vs_default_recall_4k.json
"""

from __future__ import annotations

import argparse
import json

from scipy.stats import beta


def cp_ci(hits, n, alpha=0.05):
    """Clopper-Pearson exact binomial CI. Returns (lo, hi) in [0,1]."""
    if n == 0:
        return (None, None)
    lo = 0.0 if hits == 0 else beta.ppf(alpha / 2, hits, n - hits + 1)
    hi = 1.0 if hits == n else beta.ppf(1 - alpha / 2, hits + 1, n - hits)
    return (round(float(lo), 4), round(float(hi), 4))


def load(path):
    with open(path) as fh:
        d = json.load(fh)
    out = {}
    for row in d.get("lengths", []):
        out[int(row["length_words"])] = {
            "hits": int(row["recall_hits"]),
            "served": int(row["served"]),
            "admission_fail": int(row.get("admission_fail", 0)),
            "prompt_tokens_min": row.get("prompt_tokens_min"),
            "prompt_tokens_max": row.get("prompt_tokens_max"),
        }
    return out, d.get("op_point", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds-default", required=True)
    ap.add_argument("--ds-lifted", required=True)
    ap.add_argument("--out", default="development/loop7/ds_lifted_vs_default_recall_4k.json")
    args = ap.parse_args()

    dfl, dfl_op = load(args.ds_default)
    lft, lft_op = load(args.ds_lifted)

    matrix = {
        "comparison": "DS-default top_k=2048 vs DS-lifted lifted_budget_top_k=4096",
        "mode": "EAGER (--disable-cuda-graph) on BOTH, same node — isolates the "
        "selection BUDGET (2048 vs 4096), not the eager-vs-graph numerics gap",
        "ci": "Clopper-Pearson exact 95%",
        "materiality_rule": (
            "A lifted UPLIFT over the DS-default baseline is MATERIAL only when the "
            "lifted recall point EXCEEDS the DS-default baseline 95% CP CI HIGH "
            "(directional, lifted > base_hi). N=20 makes one needle = 5pp."
        ),
        "op_points": {"ds_default": dfl_op, "ds_lifted": lft_op},
        "lengths": {},
    }

    for L in sorted(set(dfl) | set(lft)):
        d, l = dfl.get(L), lft.get(L)
        cell = {}
        if d:
            d_rec = d["hits"] / d["served"] if d["served"] else None
            d_ci = cp_ci(d["hits"], d["served"])
            cell["ds_default"] = {
                "hits": d["hits"], "served": d["served"],
                "admission_fail": d["admission_fail"], "recall": d_rec, "ci95": d_ci,
                "prompt_tokens": [d["prompt_tokens_min"], d["prompt_tokens_max"]],
            }
        if l:
            l_rec = l["hits"] / l["served"] if l["served"] else None
            l_ci = cp_ci(l["hits"], l["served"])
            cell["ds_lifted"] = {
                "hits": l["hits"], "served": l["served"],
                "admission_fail": l["admission_fail"], "recall": l_rec, "ci95": l_ci,
                "prompt_tokens": [l["prompt_tokens_min"], l["prompt_tokens_max"]],
            }
        if d and l and d["served"] and l["served"]:
            base_hi = cell["ds_default"]["ci95"][1]
            lifted_pt = cell["ds_lifted"]["recall"]
            cell["uplift_pp"] = round((lifted_pt - cell["ds_default"]["recall"]) * 100, 1)
            cell["material_uplift"] = bool(lifted_pt > base_hi)
            cell["rule"] = f"lifted {lifted_pt:.2f} > base_hi {base_hi:.4f} -> {cell['material_uplift']}"
        matrix["lengths"][str(L)] = cell

    with open(args.out, "w") as fh:
        json.dump(matrix, fh, indent=2)
    print(json.dumps(matrix, indent=2))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
