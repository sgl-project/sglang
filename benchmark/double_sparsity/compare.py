"""Read bench_decode.py JSONs (single or concurrency-sweep) and print the
visible-win comparison table.

Single-config usage (back-compat with v0 single-point JSONs):
  python benchmark/double_sparsity/compare.py \
      --main results_main_dense.json \
      --branch-off results_branch_ds_off.json \
      --branch-on  results_branch_ds_on.json

Sweep usage (new): JSONs may contain a list of per-concurrency results.
The script renders a per-concurrency table and reports the BEST DS-on
point (decode_tok_per_s) against the matching DS-off point.

The two-tier gate (Llama-3.1-70B, H200 TP=8):
  * **VISIBLE_WIN**: at least one concurrency satisfies
        decode_tok_per_s(on)  >= 1.10 * decode_tok_per_s(off)
     OR tbt_ms_p50(on)        <= 0.90 * tbt_ms_p50(off)
    Plus the quality guard at the best point:
        niah_accuracy(on)     >= niah_accuracy(off) - 0.02
  * **STRETCH_1_30X**: ≥1.30x at the best point.
  * Branch DS-off regression vs main (structural sanity, optional):
      decode_tok_per_s(branch_ds_off) within 2% of decode_tok_per_s(main_dense)
      at the lowest shared concurrency.
"""

import argparse
import json
from typing import Dict, List, Optional

VISIBLE_WIN_SPEEDUP_THRESHOLD = 1.10
VISIBLE_WIN_TBT_RATIO_THRESHOLD = 0.90
STRETCH_SPEEDUP_THRESHOLD = 1.30
QUALITY_GUARD_NIAH_DELTA_MIN = -0.02
DENSE_REGRESSION_MAX = 0.02


def _load_results(p: str) -> List[Dict]:
    """Return a list of result dicts — handles single-point and sweep JSONs."""
    with open(p) as f:
        payload = json.load(f)
    if isinstance(payload, dict) and "results" in payload:
        return payload["results"]
    return [payload]


def _by_concurrency(results: List[Dict]) -> Dict[int, Dict]:
    return {int(r["concurrency"]): r for r in results}


def _fmt(v: Optional[float], spec: str) -> str:
    return "-" if v is None else format(v, spec)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", required=False, default=None, help="results_main_dense.json (optional)")
    p.add_argument("--branch-off", required=True, help="results_branch_ds_off.json")
    p.add_argument("--branch-on", required=True, help="results_branch_ds_on.json")
    args = p.parse_args()

    off_list = _load_results(args.branch_off)
    on_list = _load_results(args.branch_on)
    main_list = _load_results(args.main) if args.main else []

    off = _by_concurrency(off_list)
    on = _by_concurrency(on_list)
    shared = sorted(set(off.keys()) & set(on.keys()))
    if not shared:
        raise SystemExit(
            f"No matching concurrency points: off={sorted(off.keys())}, on={sorted(on.keys())}"
        )

    sample = off[shared[0]]
    print(
        f"# Double Sparsity benchmark — model={sample['model']} "
        f"context={sample['context_len']} output={sample['output_len']}\n"
    )

    # Per-concurrency table
    print(
        "| conc | DS-off tok/s | DS-on tok/s | speedup | "
        "DS-off TBT p50 | DS-on TBT p50 | TBT ratio | NIAH off | NIAH on |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    rows = []
    for c in shared:
        o = off[c]
        n = on[c]
        speedup = (
            n["decode_tok_per_s"] / o["decode_tok_per_s"]
            if o["decode_tok_per_s"]
            else 0.0
        )
        on_tbt = n.get("tbt_ms_p50") or float("inf")
        off_tbt = o.get("tbt_ms_p50") or float("inf")
        tbt_ratio = on_tbt / off_tbt if off_tbt else float("inf")
        rows.append((c, o, n, speedup, tbt_ratio))
        print(
            f"| {c} "
            f"| {_fmt(o.get('decode_tok_per_s'), '.2f')} "
            f"| {_fmt(n.get('decode_tok_per_s'), '.2f')} "
            f"| {speedup:.2f}x "
            f"| {_fmt(off_tbt, '.2f')} ms "
            f"| {_fmt(on_tbt, '.2f')} ms "
            f"| {tbt_ratio:.2f}x "
            f"| {_fmt(o.get('niah_accuracy'), '.3f')} "
            f"| {_fmt(n.get('niah_accuracy'), '.3f')} "
            f"|"
        )

    # Best concurrency point: largest decode_tok_per_s ratio.
    best = max(rows, key=lambda r: r[3])
    best_c, best_off, best_on, best_speedup, best_tbt_ratio = best
    visible_win_decode = best_speedup >= VISIBLE_WIN_SPEEDUP_THRESHOLD
    visible_win_tbt = best_tbt_ratio <= VISIBLE_WIN_TBT_RATIO_THRESHOLD
    visible_win = visible_win_decode or visible_win_tbt
    stretch = best_speedup >= STRETCH_SPEEDUP_THRESHOLD
    on_niah = best_on.get("niah_accuracy")
    off_niah = best_off.get("niah_accuracy")
    niah_delta = (
        on_niah - off_niah
        if on_niah is not None and off_niah is not None
        else None
    )
    if niah_delta is None:
        quality_guard_label = "UNKNOWN (NIAH not measured on both legs)"
        quality_pass = False
    else:
        quality_pass = niah_delta >= QUALITY_GUARD_NIAH_DELTA_MIN
        quality_guard_label = "PASS" if quality_pass else "FAIL"

    print("\n## Best concurrency point\n")
    print(f"BEST_CONCURRENCY:      {best_c}{'  (diagnostic-only is conc=1)' if best_c == 1 else ''}")
    print(f"VISIBLE_WIN:           {'PASS' if visible_win else 'FAIL'}")
    print(
        f"  decode_tok_s_speedup: {best_speedup:.3f}x  threshold: "
        f">={VISIBLE_WIN_SPEEDUP_THRESHOLD:.2f}x  "
        f"{'PASS' if visible_win_decode else 'FAIL'}"
    )
    print(
        f"  p50_tbt_ratio:        {best_tbt_ratio:.3f}x  threshold: "
        f"<={VISIBLE_WIN_TBT_RATIO_THRESHOLD:.2f}x  "
        f"{'PASS' if visible_win_tbt else 'FAIL'}"
    )
    print(f"STRETCH_1_30X:         {'YES' if stretch else 'NO'}")
    print(
        f"quality_guard:         {quality_guard_label}"
        + (
            f"  (niah_on - niah_off = {niah_delta:+.3f}, min "
            f"{QUALITY_GUARD_NIAH_DELTA_MIN:+.2f})"
            if niah_delta is not None
            else ""
        )
    )
    calib_mode = best_on.get("calibration_mode")
    print(
        "calibration_mode:      "
        + (calib_mode if calib_mode is not None else "(unset)")
        + (
            "  WARN: synthetic — do not cite as headline"
            if calib_mode == "synthetic"
            else ""
        )
    )
    if best_c == 1 and any(r[3] > best_speedup for r in rows):
        # Shouldn't happen since best is max-speedup, but defensive.
        pass
    if best_c == 1:
        print(
            "\nNOTE: concurrency=1 is the diagnostic point. Long-context DS "
            "decode throughput wins normally live at concurrency>=4 — sweep "
            "higher concurrencies if the visible-win threshold isn't met here."
        )

    # Structural sanity: dense path regression vs main (only at concurrency=1
    # if a main_dense was provided).
    if main_list:
        main = _by_concurrency(main_list)
        c_for_sanity = next(
            (c for c in shared if c in main), None
        )
        if c_for_sanity is not None:
            md = main[c_for_sanity]["decode_tok_per_s"]
            od = off[c_for_sanity]["decode_tok_per_s"]
            regression = (od - md) / md if md else 0.0
            print(
                f"\n## Structural sanity (concurrency={c_for_sanity})\n"
                f"- Branch DS-off vs main regression: {regression:+.2%}  "
                f"(target |x| <= {DENSE_REGRESSION_MAX:.0%})  "
                f"{'PASS' if abs(regression) <= DENSE_REGRESSION_MAX else 'FAIL'}"
            )

    import sys
    sys.exit(0 if (visible_win and (niah_delta is None or quality_pass)) else 1)


if __name__ == "__main__":
    main()
