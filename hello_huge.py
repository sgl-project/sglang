#!/usr/bin/env python3
"""Send a large "number chain" prompt to the chat completions endpoint.

Mimics hello.sh, but builds a long prompt of the form:
    请帮我接龙，0 1 2 3 ... (n-1)。
"""

import argparse
import json
import sys

import requests

URL = "http://127.0.0.1:30000/v1/chat/completions"
MODEL = "sgl-project/DeepSeek-V4-Flash-FP8"


def build_prompt(n: int) -> str:
    numbers = " ".join(str(i) for i in range(n))
    return f"请帮我接龙，{numbers}。直到 {n + 100}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Send a huge number-chain prompt.")
    parser.add_argument(
        "-n", type=int, default=100, help="number of integers 0..n-1 in the prompt"
    )
    parser.add_argument("--url", default=URL, help="chat completions endpoint")
    parser.add_argument("--model", default=MODEL, help="model name")
    args = parser.parse_args()

    prompt = build_prompt(args.n)
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "max_tokens": 1024,
    }

    try:
        resp = requests.post(
            args.url,
            headers={"Content-Type": "application/json"},
            data=json.dumps(payload),
            timeout=600,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        print(f"request failed: {e}", file=sys.stderr)
        sys.exit(1)

    print(resp.text)


if __name__ == "__main__":
    main()
