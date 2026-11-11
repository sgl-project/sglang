#!/usr/bin/env python3
"""Compare against an already running SGLang server; does not modify or inject
model adapters. Use --candidate for a server started with
--engram-mnnvl-config.
"""

import argparse
import json
import math
import os
from pathlib import Path
import time
from urllib.request import Request, urlopen


DEFAULT_PROMPTS = [
    "Explain what an immutable lookup table is.",
    "Briefly describe the difference between NVLink and RDMA.",
]


def generate(endpoint, prompt, max_tokens, timeout):
    payload = {
        "text": prompt,
        "sampling_params": {"temperature": 0, "max_new_tokens": max_tokens},
        "return_logprob": True,
    }
    headers = {"Content-Type": "application/json"}
    if token := os.environ.get("SGLANG_API_KEY"):
        headers["Authorization"] = "Bearer " + token
    started = time.perf_counter()
    with urlopen(
        Request(
            endpoint.rstrip("/") + "/generate",
            data=json.dumps(payload).encode(),
            headers=headers,
        ),
        timeout=timeout,
    ) as response:
        result = json.load(response)
    elapsed = time.perf_counter() - started
    rows = result.get("meta_info", {}).get("output_token_logprobs")
    if not isinstance(rows, list) or not rows:
        raise ValueError(
            "Response is missing output_token_logprobs; token equality cannot "
            "be judged from text alone"
        )
    tokens, probabilities = [], []
    for row in rows:
        if not isinstance(row, list) or len(row) < 2 or not isinstance(row[1], int):
            raise ValueError("Invalid token logprob structure")
        value = float(row[0])
        if not math.isfinite(value):
            raise ValueError("Response contains a non-finite logprob")
        tokens.append(row[1])
        probabilities.append(value)
    return tokens, probabilities, elapsed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference", required=True, help="native/local server root URL"
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="root URL of the MNNVL Engram server under test",
    )
    parser.add_argument("--prompts", type=Path, help="optional JSON string array")
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--logprob-atol", type=float, default=0.0)
    args = parser.parse_args()
    prompts = json.loads(args.prompts.read_text()) if args.prompts else DEFAULT_PROMPTS
    if (
        not isinstance(prompts, list)
        or not prompts
        or not all(isinstance(p, str) for p in prompts)
    ):
        parser.error("prompts must be a nonempty string array")
    if args.max_tokens < 1 or not math.isfinite(args.timeout) or args.timeout <= 0:
        parser.error("max-tokens and timeout must be positive")
    if not math.isfinite(args.logprob_atol) or args.logprob_atol < 0:
        parser.error("logprob-atol must be a finite nonnegative number")
    reports = []
    for index, prompt in enumerate(prompts):
        reference, r_probs, r_seconds = generate(
            args.reference,
            prompt,
            args.max_tokens,
            args.timeout,
        )
        candidate, c_probs, c_seconds = generate(
            args.candidate,
            prompt,
            args.max_tokens,
            args.timeout,
        )
        equal = reference == candidate
        delta = max(abs(a - b) for a, b in zip(r_probs, c_probs)) if equal else None
        reports.append(
            {
                "request": index,
                "tokens_equal": equal,
                "max_logprob_delta": delta,
                "passed": equal and delta <= args.logprob_atol,
                "reference_seconds": r_seconds,
                "candidate_seconds": c_seconds,
            }
        )
    print(
        json.dumps(
            {
                "results": reports,
                "note": (
                    "Outputs only; confirm the actual EGM path from the "
                    "deployment config/logs. Non-streaming latency is not "
                    "TTFT/TPOT."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0 if all(report["passed"] for report in reports) else 1


if __name__ == "__main__":
    raise SystemExit(main())
