"""Read 3 bench_decode.py JSONs and print the v1 ship-gate comparison table.

Usage:
  python benchmark/double_sparsity/compare.py \
      --main results_main_dense.json \
      --branch-off results_branch_ds_off.json \
      --branch-on  results_branch_ds_on.json

The v1 ship-gate (Llama-3.1-70B, H200):
  * decode_tok_per_s(branch_ds_on)  >= 1.3 * decode_tok_per_s(branch_ds_off)
  * decode_tok_per_s(branch_ds_off) within 2% of decode_tok_per_s(main_dense)
  * niah_accuracy(branch_ds_on) >= niah_accuracy(branch_ds_off) - 0.02
"""

import argparse
import json


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
        ("decode tok/s", "decode_tok_per_s", "{:.2f}"),
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
        print(
            f"| {label} | {_val(main_r, key, fmt)} | "
            f"{_val(off_r, key, fmt)} | {_val(on_r, key, fmt)} | {speedup} |"
        )

    print("\n## Ship-gate checks\n")
    on_decode = on_r["decode_tok_per_s"]
    off_decode = off_r["decode_tok_per_s"]
    main_decode = main_r["decode_tok_per_s"]
    speedup = on_decode / off_decode if off_decode else 0.0
    regression = (off_decode - main_decode) / main_decode if main_decode else 0.0
    on_niah = on_r.get("niah_accuracy") or 0.0
    off_niah = off_r.get("niah_accuracy") or 0.0

    print(
        f"- Speedup (DS on / DS off): {speedup:.2f}x  (target >= 1.30x)  "
        f"{'✓' if speedup >= 1.30 else '✗'}"
    )
    print(
        f"- Branch DS-off vs main regression: {regression:.2%}  (target |x| <= 2%)  "
        f"{'✓' if abs(regression) <= 0.02 else '✗'}"
    )
    if on_r.get("niah_accuracy") is not None:
        delta = on_niah - off_niah
        print(
            f"- NIAH accuracy delta (DS on - DS off): {delta:+.3f}  (target >= -0.02)  "
            f"{'✓' if delta >= -0.02 else '✗'}"
        )


if __name__ == "__main__":
    main()
