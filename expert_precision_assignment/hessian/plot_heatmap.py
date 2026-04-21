"""Small heatmap of per-expert Hessian scores (layers × experts).

Rows = layers (0..47), cols = experts (0..127). Diverging cmap centered at
zero: red = positive (INT4 hurts, keep BF16), blue = negative (INT4 neutral
or mildly helpful). Color range clipped to the symmetric 99th percentile
so a few outliers don't flatten the middle.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main():
    here = Path(__file__).resolve().parent
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=str(here / "results" / "hessian_scores.json"))
    ap.add_argument("--out", default=str(here / "results" / "hessian_heatmap.png"))
    ap.add_argument("--figw", type=float, default=5.0)
    ap.add_argument("--figh", type=float, default=2.2)
    args = ap.parse_args()

    with open(args.inp) as f:
        d = json.load(f)
    per_layer = d["per_layer"]

    L = max(int(k) for k in per_layer) + 1
    E = max(int(e) for ld in per_layer.values() for e in ld["experts"]) + 1
    M = np.zeros((L, E), dtype=np.float64)
    fo_abs: list[float] = []
    for L_str, ld in per_layer.items():
        for E_str, ed in ld["experts"].items():
            M[int(L_str), int(E_str)] = float(ed["hessian_score"])
            fo_abs.append(abs(float(ed["first_order_score"])))

    # Intensity = |hessian| above the noise floor. Anything at or below
    # |fo|_mean reads as white (no evidence of importance); darker = more
    # important. Sign is not encoded here — importance is magnitude.
    fo_mean = float(np.mean(fo_abs))
    M_imp = np.maximum(np.abs(M) - fo_mean, 0.0)
    vmax = float(np.percentile(M_imp[M_imp > 0], 99)) if (M_imp > 0).any() else 1.0

    fig, ax = plt.subplots(figsize=(args.figw, args.figh), dpi=200)
    im = ax.imshow(
        M_imp, cmap="Reds", vmin=0.0, vmax=vmax,
        aspect="auto", interpolation="nearest", origin="lower",
    )
    ax.set_xlabel("expert", fontsize=7)
    ax.set_ylabel("layer", fontsize=7)
    ax.tick_params(labelsize=6)
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cb.ax.tick_params(labelsize=6)
    cb.set_label("max(|½·dᵀHd| − |fo|_mean, 0)", fontsize=6)
    fig.tight_layout(pad=0.3)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    print(f"{L}×{E} heatmap → {out} (|fo|_mean={fo_mean:.2e}, vmax={vmax:.2e})")


if __name__ == "__main__":
    main()
