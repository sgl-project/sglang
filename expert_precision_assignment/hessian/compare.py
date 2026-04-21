"""Compare Hessian-based expert ranking against the current hybrid score.

The current hybrid score from ``assign_experts._composite_scores`` is:
    score(L, E) = max(0, ppl_increase[L]) * l2[L][E] / sum_E l2[L]

This script loads both the existing per-layer PPL-delta and per-expert L2
summaries and rebuilds the hybrid score, then correlates it against the
new Hessian-based ``½·dᵀHd`` score from ``hessian_score.py``.

Outputs Spearman correlation across all experts, top-K overlap at
{10,25,50}%, and the 20 experts with the largest rank delta between the
two rankings (candidate damping/amplification cases).
"""

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np


def load_hybrid_scores(
    ppl_path: str, l2_path: str,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], int]]:
    """Rebuild the _composite_scores formula from the existing summaries."""
    with open(ppl_path) as f:
        ppl_d = json.load(f)
    with open(l2_path) as f:
        l2_d = json.load(f)

    scores: Dict[Tuple[int, int], float] = {}
    token_counts: Dict[Tuple[int, int], int] = {}

    for L_str, layer_d in l2_d["per_layer"].items():
        L = int(L_str)
        layer_ppl = max(0.0, float(ppl_d["per_layer"][L_str]["ppl_increase"]))
        experts = layer_d["experts"]
        sum_l2 = sum(float(v["sensitivity"]) for v in experts.values())
        for E_str, e_d in experts.items():
            E = int(E_str)
            tc = int(e_d["token_count"])
            token_counts[(L, E)] = tc
            if sum_l2 <= 0 or layer_ppl <= 0 or tc == 0:
                scores[(L, E)] = 0.0
            else:
                scores[(L, E)] = layer_ppl * float(e_d["sensitivity"]) / sum_l2
    return scores, token_counts


def load_hessian_scores(
    path: str,
) -> Tuple[Dict[Tuple[int, int], float], Dict[Tuple[int, int], float], Dict[Tuple[int, int], float]]:
    with open(path) as f:
        d = json.load(f)

    score: Dict[Tuple[int, int], float] = {}
    fo: Dict[Tuple[int, int], float] = {}
    d_norm: Dict[Tuple[int, int], float] = {}

    for L_str, layer_d in d["per_layer"].items():
        L = int(L_str)
        for E_str, e_d in layer_d["experts"].items():
            E = int(E_str)
            score[(L, E)] = float(e_d["hessian_score"])
            fo[(L, E)] = float(e_d["first_order_score"])
            d_norm[(L, E)] = float(e_d["d_norm_sq"])
    return score, fo, d_norm


def spearman(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson on ranks (ties broken by first occurrence)."""
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    c = np.corrcoef(ra, rb)[0, 1]
    return float(c) if np.isfinite(c) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hessian", default="results/hessian_scores.json")
    ap.add_argument(
        "--ppl_summary",
        default="../sensitivity/per_moe_layer/results/summary.json",
    )
    ap.add_argument(
        "--l2_summary",
        default="../sensitivity/per_expert/results/summary.json",
    )
    ap.add_argument("--out", default="results/comparison.json")
    args = ap.parse_args()

    script_dir = Path(__file__).resolve().parent
    hessian_path = (script_dir / args.hessian).resolve() if not Path(args.hessian).is_absolute() else Path(args.hessian)
    ppl_path = (script_dir / args.ppl_summary).resolve() if not Path(args.ppl_summary).is_absolute() else Path(args.ppl_summary)
    l2_path = (script_dir / args.l2_summary).resolve() if not Path(args.l2_summary).is_absolute() else Path(args.l2_summary)
    out_path = (script_dir / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)

    print(f"  hessian: {hessian_path}")
    print(f"  ppl:     {ppl_path}")
    print(f"  l2:      {l2_path}")

    hybrid, tc = load_hybrid_scores(str(ppl_path), str(l2_path))
    hessian, fo, d_norm = load_hessian_scores(str(hessian_path))

    # Intersect keys. Hybrid uses 128 experts × 48 layers from its coverage check;
    # Hessian may have a subset if nsamples was tiny.
    keys = sorted(set(hybrid.keys()) & set(hessian.keys()))
    print(f"  comparing {len(keys)} experts")

    h_arr = np.array([hybrid[k] for k in keys], dtype=np.float64)
    he_arr = np.array([hessian[k] for k in keys], dtype=np.float64)
    fo_arr = np.array([fo[k] for k in keys], dtype=np.float64)
    d_arr = np.array([d_norm[k] for k in keys], dtype=np.float64)

    # Hessian may be negative in pathological cases (shouldn't be at a minimum,
    # but with N=8 and first-order not truly zero, anything is possible). We
    # use the signed score for ranking since positive = loss goes up = important.
    # For experts with zero tokens routed, hessian may be ~0 as well.

    rho = spearman(h_arr, he_arr)
    rho_abs = spearman(np.abs(h_arr), np.abs(he_arr))

    # Top-K overlap by score magnitude.
    N = len(keys)
    top_k_overlap = {}
    for frac in [0.10, 0.25, 0.50]:
        K = max(1, int(N * frac))
        h_top = set(np.argsort(-h_arr)[:K].tolist())
        he_top = set(np.argsort(-he_arr)[:K].tolist())
        top_k_overlap[f"{frac:.2f}"] = len(h_top & he_top) / K

    # First-order sanity: ‖g·d‖ vs ‖½·dᵀHd‖. At a trained minimum, first-order
    # should be small relative to second-order.
    fo_abs_mean = float(np.abs(fo_arr).mean())
    he_abs_mean = float(np.abs(he_arr).mean())
    fo_to_he_ratio = fo_abs_mean / he_abs_mean if he_abs_mean > 0 else float("inf")

    # Biggest rank disagreements.
    h_ranks = np.argsort(np.argsort(-h_arr))
    he_ranks = np.argsort(np.argsort(-he_arr))
    rank_deltas = np.abs(h_ranks - he_ranks)
    top20_idx = np.argsort(-rank_deltas)[:20]

    disagreements = []
    for idx in top20_idx:
        L, E = keys[idx]
        disagreements.append({
            "layer": int(L),
            "expert": int(E),
            "hybrid_score": float(h_arr[idx]),
            "hessian_score": float(he_arr[idx]),
            "first_order_score": float(fo_arr[idx]),
            "d_norm_sq": float(d_arr[idx]),
            "token_count": int(tc.get((L, E), -1)),
            "hybrid_rank": int(h_ranks[idx]),
            "hessian_rank": int(he_ranks[idx]),
            "rank_delta": int(rank_deltas[idx]),
        })

    # Score distribution summary.
    def summary(arr):
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "nonzero_frac": float((arr != 0).mean()),
        }

    result = {
        "num_experts_compared": N,
        "spearman_corr": rho,
        "spearman_corr_abs_scores": rho_abs,
        "top_k_overlap": top_k_overlap,
        "first_order_sanity": {
            "mean_abs_first_order": fo_abs_mean,
            "mean_abs_hessian": he_abs_mean,
            "ratio_fo_to_hessian": fo_to_he_ratio,
            "note": (
                "At a trained minimum, first-order ≈ 0 so ratio should be small. "
                "Large ratio suggests N is too small for the expectation to converge."
            ),
        },
        "hybrid_score_stats": summary(h_arr),
        "hessian_score_stats": summary(he_arr),
        "biggest_disagreements": disagreements,
        "note": (
            "Spearman / top-K overlap at low N is noisy. If N < 32, a ratio "
            "|first-order|/|½·dᵀHd| > 0.5 means the first-order term hasn't "
            "averaged out yet — scale N up before drawing conclusions."
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== Comparison ===")
    print(f"  N (experts):            {N}")
    print(f"  Spearman (signed):      {rho:+.4f}")
    print(f"  Spearman (|scores|):    {rho_abs:+.4f}")
    print(f"  Top-10% overlap:        {top_k_overlap['0.10']:.3f}")
    print(f"  Top-25% overlap:        {top_k_overlap['0.25']:.3f}")
    print(f"  Top-50% overlap:        {top_k_overlap['0.50']:.3f}")
    print(f"  |FO| / |Hess|:          {fo_to_he_ratio:.3f}  (low = good; high = noisy)")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
