"""Loop-7 M0 DS-only NIAH served-recall baseline driver.

Reuses the established harness helpers (``_make_niah_prompt``, ``_niah_needle``,
``_generate``, ``_niah_recall_hits``) so the methodology matches the AC-12 NIAH
gate exactly. Measures DS served recall + the real tokenized length + admission
status at the Loop-7 op-point (int8 / mem 0.7), separating *served-but-missed*
from *admission/HTTP failure*.

Usage:
    DS_BASE_URL=http://127.0.0.1:30000 python development/loop7/niah_ds_baseline.py \
        --lengths 4096 16384 --num 5 --out development/loop7/ds_niah_baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "test", "manual"))
import test_double_sparsity_v32 as h  # noqa: E402


def measure_length(base_url: str, length_words: int, num: int, max_new_tokens: int):
    needles, responses = [], []
    served, admission_fail = 0, 0
    prompt_tokens_seen = []
    errors = []
    for idx in range(num):
        needle = h._niah_needle(length_words, idx)
        prompt = h._make_niah_prompt(length_words, seed=1000 + idx, needle=needle)
        try:
            text, ptoks = h._generate(
                base_url, prompt, max_new_tokens=max_new_tokens, use_chat=True
            )
            served += 1
            needles.append(needle)
            responses.append(text or "")
            if ptoks is not None:
                prompt_tokens_seen.append(ptoks)
        except Exception as exc:  # HTTP 400 / admission / OOM rejection
            admission_fail += 1
            errors.append(f"idx{idx}: {type(exc).__name__}: {str(exc)[:160]}")
    hits = h._niah_recall_hits(needles, responses) if served else 0
    return {
        "length_words": length_words,
        "num_requested": num,
        "served": served,
        "admission_fail": admission_fail,
        "recall_hits": hits,
        "served_recall": (hits / served) if served else None,
        "prompt_tokens_min": min(prompt_tokens_seen) if prompt_tokens_seen else None,
        "prompt_tokens_max": max(prompt_tokens_seen) if prompt_tokens_seen else None,
        "errors": errors[:5],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lengths", type=int, nargs="+", default=[4096, 16384, 65536])
    ap.add_argument("--num", type=int, default=int(os.environ.get("AC12_NIAH_NUM_PROMPTS", 5)))
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--out", default="development/loop7/ds_niah_baseline.json")
    ap.add_argument(
        "--op-point",
        default="DS int8 / mem_fraction_static=0.7 / TP=8 / page64 / fp8-KV",
        help="accurate op-point label for the artifact (e.g. native-NSA/no-DS for DSA).",
    )
    args = ap.parse_args()

    base_url = os.environ.get("DS_BASE_URL", "http://127.0.0.1:30000")
    result = {
        "op_point": args.op_point,
        "base_url": base_url,
        "num_per_length": args.num,
        "max_new_tokens": args.max_new_tokens,
        "index_topk": 2048,
        "timestamp": time.strftime("%Y%m%dT%H%M%SZ", time.gmtime()),
        "lengths": [],
    }
    for L in args.lengths:
        t0 = time.time()
        r = measure_length(base_url, L, args.num, args.max_new_tokens)
        r["elapsed_s"] = round(time.time() - t0, 1)
        result["lengths"].append(r)
        print(
            f"[{L:>6} words] served={r['served']}/{r['num_requested']} "
            f"admission_fail={r['admission_fail']} recall_hits={r['recall_hits']} "
            f"served_recall={r['served_recall']} "
            f"prompt_tokens={r['prompt_tokens_min']}..{r['prompt_tokens_max']} "
            f"({r['elapsed_s']}s)",
            flush=True,
        )
        if r["errors"]:
            for e in r["errors"]:
                print(f"        ! {e}", flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nwrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
