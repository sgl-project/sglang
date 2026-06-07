"""Capture greedy output token-IDs from a running server for a fixed prompt set.

Used for the DS-off byte-identity check: run against the candidate (HEAD) and the
baseline (parent-commit worktree) GLM-5.1 DSA-native (DS disabled) servers under
one fixed tuple, then diff the JSON outputs. Token IDs come from
`meta_info.output_token_logprobs[i][1]` (return_logprob=True).

Usage:
    python development/loop8/capture_ds_off_token_ids.py --base-url http://127.0.0.1:30000 \
        --out runs/.../glm_dsoff_head_ids.json --max-new-tokens 48
"""

from __future__ import annotations

import argparse
import json
import sys
import urllib.request

# Fixed prompt set (held constant across baseline + candidate).
PROMPTS = [
    "The capital of France is",
    "List the first five prime numbers:",
    "Explain in one sentence what a transformer neural network is.",
    "Translate to French: Good morning, how are you?",
    "Q: What is 17 multiplied by 23? A:",
    "Write a haiku about the ocean.",
]


def _generate_ids(base_url: str, prompt: str, max_new_tokens: int) -> list:
    body = json.dumps(
        {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": 0.0,
            },
            "return_logprob": True,
            "logprob_start_len": 0,
        }
    ).encode()
    req = urllib.request.Request(
        base_url + "/generate",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=600) as r:
        data = json.loads(r.read().decode())
    otl = data.get("meta_info", {}).get("output_token_logprobs")
    if not otl:
        raise RuntimeError(f"no output_token_logprobs for prompt {prompt!r}")
    # each entry is [logprob, token_id, (token_text)]
    return [int(e[1]) for e in otl]


def main(argv) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:30000")
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args(argv)

    result = {
        "base_url": args.base_url,
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0,
        "prompts": {},
    }
    for p in PROMPTS:
        ids = _generate_ids(args.base_url, p, args.max_new_tokens)
        result["prompts"][p] = ids
        print(f"[{len(ids)} tok] {p!r} -> {ids[:12]}{'...' if len(ids) > 12 else ''}")

    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
