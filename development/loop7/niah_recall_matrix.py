"""Loop-7 AC-2 — assemble the binding DS-vs-DSA same-node served-recall matrix
with Clopper-Pearson 95% binomial CIs and the up-front materiality rule.

Consumes the per-config NIAH JSONs written by `niah_ds_baseline.py` (the
server-agnostic served-recall driver: served vs admission separated, recall via
`_niah_recall_hits`). Each input is `{"lengths": [{length_words, served,
admission_fail, recall_hits, served_recall, ...}, ...]}`.

Materiality rule (stated BEFORE any uplift claim, directional): an uplift of a
DS-variant over the DS-default baseline at a length counts as **material** only
when the variant recall point EXCEEDS the baseline's 95% Clopper-Pearson CI HIGH
(`h > base_hi`). A below-CI point is reported separately and a one-/two-needle
floor move at small N is NOT a material regression. DSA is the recall ceiling
reference; the gap is `DSA_recall - DS_recall`.
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
    # length_words -> (hits, served, admission_fail)
    out = {}
    for row in d.get("lengths", []):
        out[int(row["length_words"])] = (
            int(row["recall_hits"]),
            int(row["served"]),
            int(row.get("admission_fail", 0)),
        )
    return out, d.get("op_point", path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dsa", required=True, help="DSA reference NIAH json")
    ap.add_argument("--ds-default", required=True, help="DS default-scorer NIAH json")
    ap.add_argument("--ds-hybrid", required=True, help="DS hybrid-scorer (Tier-2.B) NIAH json")
    ap.add_argument("--out", default="development/loop7/ds_vs_dsa_recall_matrix.json")
    args = ap.parse_args()

    dsa, dsa_op = load(args.dsa)
    dsd, dsd_op = load(args.ds_default)
    dsh, dsh_op = load(args.ds_hybrid)

    lengths = sorted(set(dsa) | set(dsd) | set(dsh))
    matrix = {
        "materiality_rule": (
            "An UPLIFT of a DS variant over the DS-default baseline is MATERIAL "
            "only when the variant recall point EXCEEDS the DS-default baseline's "
            "95% Clopper-Pearson CI HIGH (directional, h > base_hi). A below-CI "
            "point is reported separately (hybrid_below_baseline_CI) and a sub-2-"
            "needle floor move is NOT a material regression. DSA is the recall "
            "ceiling; gap = DSA - DS."
        ),
        "op_points": {"dsa": dsa_op, "ds_default": dsd_op, "ds_hybrid": dsh_op},
        "ci": "Clopper-Pearson exact 95%",
        "lengths": {},
    }

    def cell(d, L):
        if L not in d:
            return None
        hits, served, adm = d[L]
        rec = (hits / served) if served else None
        return {
            "hits": hits,
            "served": served,
            "admission_fail": adm,
            "recall": round(rec, 4) if rec is not None else None,
            "ci95": cp_ci(hits, served),
        }

    for L in lengths:
        c_dsa, c_dsd, c_dsh = cell(dsa, L), cell(dsd, L), cell(dsh, L)
        entry = {"dsa": c_dsa, "ds_default": c_dsd, "ds_hybrid": c_dsh}
        # Tier-2.B uplift vs DS-default, judged against the baseline CI. Per the
        # plan, a *material uplift* requires the variant point to EXCEED the
        # baseline CI upward (h > base_hi). A below-CI point is reported
        # separately (a 1-needle move at the floor, e.g. default 1/20 vs hybrid
        # 0/20, is below the degenerate lower bound but is sampling noise, NOT a
        # material regression).
        if c_dsd and c_dsh and c_dsd["served"] and c_dsh["served"]:
            base_lo, base_hi = c_dsd["ci95"]
            h = c_dsh["recall"]
            entry["hybrid_uplift_vs_default"] = round(c_dsh["recall"] - c_dsd["recall"], 4)
            entry["hybrid_material_uplift_vs_default_CI"] = bool(
                h is not None and base_hi is not None and h > base_hi
            )
            entry["hybrid_below_baseline_CI"] = bool(
                h is not None and base_lo is not None and h < base_lo
            )
        if c_dsa and c_dsd and c_dsa["recall"] is not None and c_dsd["recall"] is not None:
            entry["gap_dsa_minus_ds_default"] = round(c_dsa["recall"] - c_dsd["recall"], 4)
        if c_dsa and c_dsh and c_dsa["recall"] is not None and c_dsh["recall"] is not None:
            entry["gap_dsa_minus_ds_hybrid"] = round(c_dsa["recall"] - c_dsh["recall"], 4)
        matrix["lengths"][str(L)] = entry

    with open(args.out, "w") as fh:
        json.dump(matrix, fh, indent=2)

    # human-readable table
    print(f"{'len':>7} | {'DSA':>16} | {'DS-default':>16} | {'DS-hybrid':>16} | uplift  material  gap(DSA-hyb)")
    print("-" * 100)
    for L in lengths:
        e = matrix["lengths"][str(L)]
        def fmt(c):
            if not c:
                return f"{'-':>16}"
            ci = c["ci95"]
            return f"{c['recall']*100:5.1f}% [{ci[0]*100:4.0f},{ci[1]*100:4.0f}]"
        up = e.get("hybrid_uplift_vs_default")
        mat = e.get("hybrid_material_uplift_vs_default_CI")
        gap = e.get("gap_dsa_minus_ds_hybrid")
        print(
            f"{L:>7} | {fmt(e['dsa']):>16} | {fmt(e['ds_default']):>16} | {fmt(e['ds_hybrid']):>16} | "
            f"{('%+.0f' % (up*100)) if up is not None else '  -':>5}pp  {str(mat):>5}    "
            f"{('%.0f' % (gap*100)) if gap is not None else '-':>3}pp"
        )
    print(f"\nwrote -> {args.out}")


if __name__ == "__main__":
    main()
