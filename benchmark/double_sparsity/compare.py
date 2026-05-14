"""Read 3 bench_decode.py JSONs and print the visible-win comparison table.

Usage:
  python benchmark/double_sparsity/compare.py \
      --main results_main_dense.json \
      --branch-off results_branch_ds_off.json \
      --branch-on  results_branch_ds_on.json

The two-tier gate (Llama-3.1-70B, H200 TP=8):
  * **VISIBLE_WIN (current goal)**: PASS when EITHER
      decode_tok_per_s(branch_ds_on)  >= 1.10 * decode_tok_per_s(branch_ds_off)
      OR  tbt_ms_p50(branch_ds_on)    <= 0.90 * tbt_ms_p50(branch_ds_off)
    Plus the quality guard:
      niah_accuracy(branch_ds_on)     >= niah_accuracy(branch_ds_off) - 0.02
  * **STRETCH_1_30X (original ship-gate)**:
      decode_tok_per_s(branch_ds_on)  >= 1.30 * decode_tok_per_s(branch_ds_off)
  * Branch DS-off regression vs main (structural sanity, optional):
      decode_tok_per_s(branch_ds_off) within 2% of decode_tok_per_s(main_dense)
"""

import argparse
import json


VISIBLE_WIN_SPEEDUP_THRESHOLD = 1.10
VISIBLE_WIN_TBT_RATIO_THRESHOLD = 0.90
STRETCH_SPEEDUP_THRESHOLD = 1.30
QUALITY_GUARD_NIAH_DELTA_MIN = -0.02
DENSE_REGRESSION_MAX = 0.02


def _load(p):
    with open(p) as f:
        return json.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", required=True, help="results_main_dense.json")
    p.add_argument("--branch-off", required=True, help="results_branch_ds_off.json")
    p.add_argument("--branch-on", required=True, help="results_branch_ds_on.json")
    args = p.parse_args()

    main_r = _load(args.main)
    off_r = _load(args.branch_off)
    on_r = _load(args.branch_on)

    rows = [
        ("decode tok/s (per-req, decode-only)", "decode_tok_per_s", "{:.2f}"),
        ("aggregate tok/s (system, wall-clock)", "aggregate_tok_per_s", "{:.2f}"),
        ("e2e latency (s)", "e2e_latency_s", "{:.2f}"),
        ("TTFT p50 (ms)", "ttft_ms_p50", "{:.1f}"),
        ("TTFT p95 (ms)", "ttft_ms_p95", "{:.1f}"),
        ("TBT p50 (ms)", "tbt_ms_p50", "{:.2f}"),
        ("TBT p95 (ms)", "tbt_ms_p95", "{:.2f}"),
        ("NIAH accuracy", "niah_accuracy", "{:.3f}"),
    ]

    def _val(r, key, fmt):
        v = r.get(key)
        if v is None:
            return "-"
        return fmt.format(v)

    print(
        f"# Double Sparsity benchmark — model={main_r['model']} "
        f"context={main_r['context_len']} output={main_r['output_len']}\n"
    )
    print("| metric | main dense | branch DS off | branch DS on | DS speedup |")
    print("|---|---|---|---|---|")
    for label, key, fmt in rows:
        on_val = on_r.get(key)
        off_val = off_r.get(key)
        speedup = ""
        if key == "decode_tok_per_s" and on_val and off_val:
            speedup = f"{on_val / off_val:.2f}x"
        elif key == "tbt_ms_p50" and on_val and off_val:
            # TBT lower-is-better: print on/off ratio so reader sees both sides.
            speedup = f"{on_val / off_val:.2f}x"
        print(
            f"| {label} | {_val(main_r, key, fmt)} | "
            f"{_val(off_r, key, fmt)} | {_val(on_r, key, fmt)} | {speedup} |"
        )

    print("\n## Visible-win + stretch gate\n")
    on_decode = on_r["decode_tok_per_s"]
    off_decode = off_r["decode_tok_per_s"]
    main_decode = main_r["decode_tok_per_s"]
    on_tbt = on_r.get("tbt_ms_p50") or float("inf")
    off_tbt = off_r.get("tbt_ms_p50") or float("inf")

    speedup = on_decode / off_decode if off_decode else 0.0
    tbt_ratio = on_tbt / off_tbt if off_tbt else float("inf")
    regression = (
        (off_decode - main_decode) / main_decode if main_decode else 0.0
    )
    on_niah = on_r.get("niah_accuracy")
    off_niah = off_r.get("niah_accuracy")
    niah_delta = (
        on_niah - off_niah
        if on_niah is not None and off_niah is not None
        else None
    )

    visible_win_decode = speedup >= VISIBLE_WIN_SPEEDUP_THRESHOLD
    visible_win_tbt = tbt_ratio <= VISIBLE_WIN_TBT_RATIO_THRESHOLD
    visible_win = visible_win_decode or visible_win_tbt
    stretch = speedup >= STRETCH_SPEEDUP_THRESHOLD
    # Quality guard passes when NIAH was measured AND delta >= -0.02; if NIAH
    # was not measured at all, we report UNKNOWN rather than silently PASS.
    if niah_delta is None:
        quality_guard_label = "UNKNOWN (NIAH not measured on both legs)"
        quality_pass = False
    else:
        quality_pass = niah_delta >= QUALITY_GUARD_NIAH_DELTA_MIN
        quality_guard_label = "PASS" if quality_pass else "FAIL"

    # Exact output block the plan specifies. Format is machine-grep-friendly
    # so scripts can extract per-field values; checkmarks are kept off this
    # block intentionally (they go in the optional structural-sanity section).
    print(f"VISIBLE_WIN:           {'PASS' if visible_win else 'FAIL'}")
    print(
        f"  decode_tok_s_speedup: {speedup:.3f}x  threshold: "
        f">={VISIBLE_WIN_SPEEDUP_THRESHOLD:.2f}x  "
        f"{'PASS' if visible_win_decode else 'FAIL'}"
    )
    print(
        f"  p50_tbt_ratio:        {tbt_ratio:.3f}x  threshold: "
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

    # Calibration-mode cite check: a README-cited DS-on result must not be
    # backed by synthetic calibration.
    calib_mode = on_r.get("calibration_mode")
    print(
        "calibration_mode:      "
        + (calib_mode if calib_mode is not None else "(unset)")
        + ("  WARN: synthetic — do not cite as headline" if calib_mode == "synthetic" else "")
    )

    print("\n## Structural sanity (dense-path regression vs main)\n")
    print(
        f"- Branch DS-off vs main regression: {regression:.2%}  "
        f"(target |x| <= {DENSE_REGRESSION_MAX:.0%})  "
        f"{'PASS' if abs(regression) <= DENSE_REGRESSION_MAX else 'FAIL'}"
    )

    # Non-zero exit when the visible-win or quality_guard checks fail, so
    # this can drive scripted gates if the team wants.
    import sys
    sys.exit(0 if (visible_win and (niah_delta is None or quality_pass)) else 1)


if __name__ == "__main__":
    main()
