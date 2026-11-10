"""Shrink the vectorized_gather_kernel crash to the smallest request sequence.

What is known so far:
  - batch_invariance.py kills the engine reproducibly (3x, on both GPU sets)
  - it dies during the SERIAL pass, at decode batch size 1, so concurrency is
    not required (GPQA ran #running-req: 4 2,658 times without incident)
  - it dies around request ~12 of 40, i.e. shortly after the prompt list wraps
    and requests start HITTING the prefix cache rather than filling it
  - the last prefill before each crash reports #cached-token: 64

So the suspect is a prefix-cache hit on a repeated prompt. This sends each
prompt twice, back to back, strictly serially, and prints before every request
so the log identifies the exact failing one even though the process on the
other end dies mid-request.

Prompts are the batch_invariance set, in its order, because that set is known
to trigger; anything that survives here narrows the trigger to the prompt text.

Usage: minimal_repro.py --base-url http://127.0.0.1:11436 [--repeats 2]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.request

PROMPTS = [
    "Count from 1 to 60, separated by commas, then stop.",
    "List the first 40 prime numbers, separated by commas.",
    "Recite the alphabet, then the alphabet backwards, then stop.",
    "Write the multiplication table for 7, from 7x1 to 7x25, one per line.",
    "Name the planets in order from the Sun, with one sentence each.",
    "List the days of the week and the months of the year, then stop.",
    "Write the squares of 1 through 30, one per line, as 'n: n^2'.",
    "Explain in exactly ten numbered steps how to make a cup of tea.",
    "List the 26 letters of the alphabet, each with a common word starting with it.",
    "Write the Fibonacci sequence up to the 30th term, comma separated.",
]


def generate(base_url: str, prompt: str, max_tokens: int) -> tuple:
    body = {
        "model": "qwen38next-nvfp4",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
        "chat_template_kwargs": {"thinking": False},
    }
    req = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=900) as r:
        out = json.load(r)
    usage = out.get("usage") or {}
    return (
        usage.get("completion_tokens"),
        (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True)
    ap.add_argument("--repeats", type=int, default=2, help="serial sends per prompt")
    ap.add_argument("--max-tokens", type=int, default=256)
    args = ap.parse_args()

    for i, prompt in enumerate(PROMPTS):
        for rep in range(args.repeats):
            tag = f"prompt[{i}] rep{rep}"
            print(f"{tag:22s} -> sending... ", end="", flush=True)
            t0 = time.time()
            try:
                comp, cached = generate(args.base_url, prompt, args.max_tokens)
            except Exception as e:
                print(f"\n\nDIED ON {tag}: {type(e).__name__}: {e}")
                print(f"prompt text: {prompt!r}")
                sys.exit(1)
            print(
                f"ok  completion={comp} cached={cached}  ({time.time() - t0:.1f}s)",
                flush=True,
            )

    print(f"\nAll {len(PROMPTS)} prompts x {args.repeats} survived.")


if __name__ == "__main__":
    main()
