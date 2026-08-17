#!/usr/bin/env python
"""GSM8K eval via sgl-eval against an sglang server.

Usage:
    HF_ENDPOINT=https://hf-mirror.com python run_sgl_eval_gsm8k.py [--num 200]
"""
import argparse

from sgl_eval.registry import get
from sgl_eval.sampler import ChatCompletionSampler
from sgl_eval.types import GenConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:30000/v1")
    ap.add_argument("--model", default="Qwen3-4B")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--threads", type=int, default=64)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    sampler = ChatCompletionSampler(
        base_url=args.base_url, model=args.model, api_key="EMPTY"
    )
    result = get("gsm8k").run(
        sampler=sampler,
        gen=GenConfig(max_tokens=args.max_tokens, temperature=0),
        n_repeats=1,
        num_examples=args.num,
        num_threads=args.threads,
    )
    print(f"\n===== sgl-eval GSM8K {args.num} questions (BF16 KV cache) =====")
    for k, v in result.aggregate.items():
        print(f"{k}: {v}")
    return result


if __name__ == "__main__":
    main()
