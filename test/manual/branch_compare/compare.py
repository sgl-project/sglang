"""Compute per-step divergence metrics from two branch_compare artifacts.

Usage (run from sglang-source/test/manual/):
    python -m branch_compare.compare \\
        --record-dir DIR --branch-dir DIR --out-dir DIR

Per (prompt, step) the comparator finds the intersection of main's top-K
token IDs with branch's top-K, looks up the logprob each side assigned to
each common token, and computes:
  - cosine similarity (log-prob vector dot product / norm product)
  - mean absolute difference
  - mean relative difference (denominator: max(|main|, eps))

These per-step scalars are aggregated into mean/p50/p90/p99/min/max.
The intersection size (`common_topk_count`) per step is also reported - if
it dips below K, the rank ordering itself shifted.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from branch_compare import artifacts


def _per_step_metrics(
    main_top_ids: np.ndarray,  # int32 [K]
    main_top_lps: np.ndarray,  # float32 [K]
    branch_top_ids: np.ndarray,  # int32 [K]
    branch_top_lps: np.ndarray,  # float32 [K]
) -> Tuple[float, float, float, int]:
    """Return (cos, mean_abs, mean_rel, common_count)."""
    branch_lookup: Dict[int, float] = {
        int(t): float(lp) for t, lp in zip(branch_top_ids, branch_top_lps)
    }
    common_main_lps: List[float] = []
    common_branch_lps: List[float] = []
    for tid, lp in zip(main_top_ids, main_top_lps):
        tid_int = int(tid)
        if tid_int in branch_lookup:
            common_main_lps.append(float(lp))
            common_branch_lps.append(branch_lookup[tid_int])

    common = len(common_main_lps)
    if common < 2:
        return float("nan"), float("nan"), float("nan"), common

    m = np.asarray(common_main_lps, dtype=np.float64)
    b = np.asarray(common_branch_lps, dtype=np.float64)
    nm, nb = np.linalg.norm(m), np.linalg.norm(b)
    if nm == 0.0 or nb == 0.0:
        cos = 1.0
    else:
        cos = float(np.dot(m, b) / (nm * nb))

    abs_d = np.abs(m - b)
    mean_abs = float(abs_d.mean())
    mean_rel = float((abs_d / np.maximum(np.abs(m), 1e-9)).mean())
    return cos, mean_abs, mean_rel, common


def _aggregate(values: List[float]) -> Dict[str, float]:
    arr = np.asarray([v for v in values if not np.isnan(v)], dtype=np.float64)
    if arr.size == 0:
        return {
            "mean": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "n": 0,
        }
    return {
        "mean": float(arr.mean()),
        "p50": float(np.quantile(arr, 0.50)),
        "p90": float(np.quantile(arr, 0.90)),
        "p99": float(np.quantile(arr, 0.99)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "n": int(arr.size),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--record-dir", required=True)
    parser.add_argument("--branch-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    record_meta = artifacts.read_meta(args.record_dir)
    branch_meta = artifacts.read_meta(args.branch_dir)

    record_by_idx = {p["idx"]: p for p in record_meta["prompts"]}

    cos_all: List[float] = []
    abs_all: List[float] = []
    rel_all: List[float] = []
    common_all: List[float] = []
    per_prompt: List[Dict[str, Any]] = []

    for entry in branch_meta["prompts"]:
        idx = entry["idx"]
        rec = record_by_idx.get(idx)
        if rec is None:
            print(f"[branch_compare] no record entry for idx={idx}, skipping")
            continue

        main_a = artifacts.read_prompt_artifact(args.record_dir, idx)
        branch_a = artifacts.read_prompt_artifact(args.branch_dir, idx)

        # Steps must match (verify is forced to record's length). Compare
        # the shorter prefix if not (e.g. forcing-mismatch case).
        n_steps = min(
            main_a["top_k_token_ids"].shape[0],
            branch_a["top_k_token_ids"].shape[0],
        )

        # Convert bf16 -> float32 for numpy.
        main_top_lps_f32 = (
            main_a["top_k_logprobs"].to(dtype=__import__("torch").float32).numpy()
        )
        branch_top_lps_f32 = (
            branch_a["top_k_logprobs"].to(dtype=__import__("torch").float32).numpy()
        )
        main_top_ids = main_a["top_k_token_ids"].numpy()
        branch_top_ids = branch_a["top_k_token_ids"].numpy()

        prompt_cos: List[float] = []
        prompt_abs: List[float] = []
        prompt_rel: List[float] = []
        for s in range(n_steps):
            cos, mabs, mrel, common = _per_step_metrics(
                main_top_ids[s],
                main_top_lps_f32[s],
                branch_top_ids[s],
                branch_top_lps_f32[s],
            )
            cos_all.append(cos)
            abs_all.append(mabs)
            rel_all.append(mrel)
            common_all.append(float(common))
            prompt_cos.append(cos)
            prompt_abs.append(mabs)
            prompt_rel.append(mrel)

        per_prompt.append(
            {
                "idx": idx,
                "n_steps": n_steps,
                "cosine": _aggregate(prompt_cos),
                "abs_diff": _aggregate(prompt_abs),
                "rel_diff": _aggregate(prompt_rel),
            }
        )

    summary: Dict[str, Any] = {
        "record_commit": record_meta.get("git_commit"),
        "branch_commit": branch_meta.get("git_commit"),
        "record_server_args": record_meta.get("server_args"),
        "branch_server_args": branch_meta.get("server_args"),
        "topk": record_meta.get("topk_logprobs"),
        "cosine_similarity": _aggregate(cos_all),
        "mean_abs_diff": _aggregate(abs_all),
        "mean_rel_diff": _aggregate(rel_all),
        "common_topk_count": _aggregate(common_all),
        "failed_steps": branch_meta.get("failed_steps", []),
        "per_prompt": per_prompt,
    }

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[branch_compare] wrote {summary_path}")

    # Terminal histograms (also archived to histograms.txt). Three panels
    # share the same width / bin count for easy visual comparison.
    hist_text = _render_histograms(
        [
            ("Cosine similarity", cos_all, summary["cosine_similarity"]),
            ("Mean abs diff", abs_all, summary["mean_abs_diff"]),
            ("Mean relative diff", rel_all, summary["mean_rel_diff"]),
            ("Common top-K count", common_all, summary["common_topk_count"]),
        ],
        n_bins=24,
        width=48,
    )
    print(hist_text)
    hist_path = os.path.join(args.out_dir, "histograms.txt")
    with open(hist_path, "w") as f:
        f.write(hist_text)
    print(f"[branch_compare] wrote {hist_path}")


# --- ASCII histogram rendering ----------------------------------------------

_BLOCKS = " ▏▎▍▌▋▊▉█"  # 0/8, 1/8, ..., 8/8 of a cell


def _bar(fraction: float, width: int) -> str:
    """Render a horizontal bar of `fraction` (0..1) using sub-block chars."""
    fraction = max(0.0, min(1.0, fraction))
    total_eighths = int(round(fraction * width * 8))
    full = total_eighths // 8
    rem = total_eighths % 8
    out = "█" * full
    if rem and full < width:
        out += _BLOCKS[rem]
    return out.ljust(width)


def _fmt(v: float) -> str:
    """Format a float for the bin-edge label column."""
    if not np.isfinite(v):
        return "nan"
    av = abs(v)
    if av == 0.0:
        return "0"
    if av < 1e-3 or av >= 1e4:
        return f"{v:.3e}"
    return f"{v:.6g}"


def _ascii_histogram(
    title: str,
    values: List[float],
    aggregates: Dict[str, float],
    *,
    n_bins: int = 24,
    width: int = 48,
) -> str:
    """Render one labeled histogram + summary line as ASCII."""
    clean = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    lines: List[str] = []
    lines.append(f"=== {title} (n={clean.size}) ===")

    if clean.size == 0:
        lines.append("  (no data)")
        return "\n".join(lines)

    lo = float(clean.min())
    hi = float(clean.max())
    if lo == hi:
        # Degenerate: every value identical. Print one row.
        lines.append(f"  {_fmt(lo):>12s}  {_bar(1.0, width)} {clean.size}")
    else:
        edges = np.linspace(lo, hi, n_bins + 1)
        counts, _ = np.histogram(clean, bins=edges)
        max_count = int(counts.max()) if counts.size else 0
        for i, c in enumerate(counts):
            edge_left = edges[i]
            frac = (c / max_count) if max_count else 0.0
            lines.append(f"  {_fmt(edge_left):>12s}  {_bar(frac, width)} {int(c)}")
        lines.append(f"  {_fmt(edges[-1]):>12s}  (upper edge)")

    s = aggregates
    lines.append(
        f"  mean={_fmt(s['mean'])}  p50={_fmt(s['p50'])}  "
        f"p90={_fmt(s['p90'])}  p99={_fmt(s['p99'])}  "
        f"min={_fmt(s['min'])}  max={_fmt(s['max'])}"
    )
    return "\n".join(lines)


def _render_histograms(
    panels: List[Tuple[str, List[float], Dict[str, float]]],
    *,
    n_bins: int,
    width: int,
) -> str:
    return "\n\n".join(
        _ascii_histogram(title, vals, aggs, n_bins=n_bins, width=width)
        for title, vals, aggs in panels
    )


if __name__ == "__main__":
    main()
