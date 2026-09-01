"""Verify the emulated TP=2 SGLang server and time a few generations.

Usage: python verify_and_bench.py [--port 30000] [--label emulated-tp2]
Prints one JSON line: label, correctness snippet, and tok/s over N requests.
"""

import argparse
import json
import time

import requests


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--label", default="emulated-tp2")
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=128)
    args = ap.parse_args()

    url = f"http://127.0.0.1:{args.port}/generate"
    prompt = (
        "Explain in one paragraph why tensor parallelism normally requires "
        "multiple GPUs, and what changes when ranks share one device."
    )

    # correctness: one deterministic generation
    r = requests.post(
        url,
        json={
            "text": prompt,
            "sampling_params": {"temperature": 0, "max_new_tokens": 64},
        },
        timeout=120,
    )
    r.raise_for_status()
    text = r.json()["text"]
    assert len(text.strip()) > 20, "empty generation"

    # throughput: N requests, count output tokens from meta_info
    total_tokens = 0
    t0 = time.perf_counter()
    for _ in range(args.n):
        r = requests.post(
            url,
            json={
                "text": prompt,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": args.max_new_tokens,
                },
            },
            timeout=120,
        )
        r.raise_for_status()
        total_tokens += r.json()["meta_info"]["completion_tokens"]
    dt = time.perf_counter() - t0

    print(
        json.dumps(
            {
                "label": args.label,
                "sample": text.strip()[:160],
                "requests": args.n,
                "output_tokens": total_tokens,
                "seconds": round(dt, 2),
                "output_tok_per_s": round(total_tokens / dt, 1),
            }
        )
    )


if __name__ == "__main__":
    main()
