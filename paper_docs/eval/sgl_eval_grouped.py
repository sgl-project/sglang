#!/usr/bin/env python
"""GSM8K eval via sgl-eval, with truncated/completed results reported separately.

Groups per-example results by ``finish_reason``: ``stop`` = generation ran to
completion, ``length`` = hit max_tokens and was truncated (answer may be
incomplete), ``error`` = request failed.
"""
import argparse
from collections import Counter

from sgl_eval.registry import get
from sgl_eval.sampler import ChatCompletionSampler
from sgl_eval.types import GenConfig


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:30000/v1")
    ap.add_argument("--model", default="Qwen3-4B")
    ap.add_argument("--num", type=int, default=10)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument(
        "--out",
        default="/sgl-workspace/sglang/paper_docs/results/result.jsonl",
    )
    args = ap.parse_args()

    sampler = ChatCompletionSampler(
        base_url=args.base_url, model=args.model, api_key="EMPTY"
    )
    result = get("gsm8k").run(
        sampler=sampler,
        gen=GenConfig(
            max_tokens=args.max_tokens,
            temperature=0,
            chat_template_kwargs={"enable_thinking": False},  # sglang 0.5.2 Qwen3 模板认 enable_thinking
        ),
        n_repeats=1,
        num_examples=args.num,
        num_threads=args.threads,
    )

    stats = {"stop": [0, 0], "length": [0, 0], "error": [0, 0]}
    for r in result.per_example:
        s = r.samples[0]
        fr = s.finish_reason if s.finish_reason in stats else "error"
        stats[fr][0] += 1
        stats[fr][1] += int(r.scores[0] == 1.0)

    n = result.num_examples
    print(f"\n===== sgl-eval GSM8K {n} questions (BF16 KV cache) =====")
    print(f"max_tokens     : {args.max_tokens}")
    print(f"aggregate      : {result.aggregate}")
    print(f"total latency  : {result.latency:.1f}s")
    print(f"\nfinish_reason breakdown:")
    for fr, (cnt, correct) in stats.items():
        acc = correct / cnt if cnt else 0.0
        print(f"  {fr:8s}: {cnt:3d}  correct={correct:3d}  acc={acc:.4f}")
    print(f"  ---     total: {n}  correct={sum(c for _, c in stats.values())}")

    # dump details for inspection
    if args.out:
        import json
        with open(args.out, "w") as f:
            for r in result.per_example:
                s = r.samples[0]
                f.write(json.dumps({
                    "id": r.example.id,
                    "question": r.example.inputs.get("problem"),
                    "target": r.example.target,
                    "pred": r.extracted[0],
                    "correct": r.scores[0] == 1.0,
                    "finish_reason": s.finish_reason,
                    "completion_tokens": s.completion_tokens,
                    "output": s.text,
                }, ensure_ascii=False) + "\n")
        print(f"details -> {args.out}")


if __name__ == "__main__":
    main()
