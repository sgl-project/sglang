#!/usr/bin/env python
"""GSM8K evaluation against an sglang server (OpenAI-compatible API).

Usage:
    python gsm8k_eval.py --base-url http://localhost:30000/v1 \
        --num 200 --workers 16 --out result.jsonl

Uses an 8-shot CoT prompt built from the first 8 examples of the train split
(standard GSM8K evaluation protocol). Accuracy + per-request latency are
written to stdout and --out jsonl.
"""
import argparse
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
from openai import OpenAI

FEWSHOT_N = 8
MODEL = "Qwen3-4B"


def load_fewshot(train_path, n=FEWSHOT_N):
    df = pd.read_parquet(train_path)
    return "\n\n".join(f"Q: {row['question']}\nA: {row['answer']}"
                       for _, row in df.head(n).iterrows())


def build_prompt(question, fewshot):
    return f"{fewshot}\n\nQ: {question}\nA:"


def extract_answer(text):
    """Last integer after the final '####' marker."""
    m = re.findall(r"####\s*(-?\d[\d,]*)", text)
    if not m:
        return None
    return int(m[-1].replace(",", ""))


def ask(client, row, fewshot):
    prompt = build_prompt(row["question"], fewshot)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=1024,
    )
    elapsed = time.time() - t0
    out = resp.choices[0].message.content
    gold = extract_answer(row["answer"])
    pred = extract_answer(out)
    ok = gold is not None and pred == gold
    return {
        "idx": row["idx"],
        "question": row["question"],
        "gold": gold,
        "pred": pred,
        "correct": ok,
        "latency_s": round(elapsed, 2),
        "output": out,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://localhost:30000/v1")
    ap.add_argument("--num", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--test", default="/home/xubowen/gsm8k/test.parquet")
    ap.add_argument("--train", default="/home/xubowen/gsm8k/train.parquet")
    ap.add_argument("--out", default="/home/xubowen/gsm8k/result_bf16.jsonl")
    args = ap.parse_args()

    fewshot = load_fewshot(args.train)
    df = pd.read_parquet(args.test).head(args.num).reset_index(drop=True)
    df["idx"] = df.index

    client = OpenAI(base_url=args.base_url, api_key="EMPTY")
    results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(ask, client, row, fewshot) for row in df.to_dict("records")]
        done = 0
        for f in futs:
            r = f.result()
            done += 1
            if done % 20 == 0 or done == args.num:
                acc = sum(1 for x in results if x["correct"]) / len(results)
                print(f"[{done}/{args.num}] acc={acc:.4f}", flush=True)
            results.append(r)

    results.sort(key=lambda r: r["idx"])
    n = len(results)
    acc = sum(1 for r in results if r["correct"]) / n
    lat = [r["latency_s"] for r in results]
    lat.sort()
    med = lat[n // 2] if n % 2 else (lat[n // 2 - 1] + lat[n // 2]) / 2
    print(f"\n===== GSM8K {n} questions (BF16 KV cache) =====")
    print(f"Accuracy : {acc:.4f} ({sum(1 for r in results if r['correct'])}/{n})")
    print(f"Latency  : mean={sum(lat)/n:.2f}s  median={med:.2f}s  "
          f"p90={lat[int(n*0.9)-1]:.2f}s")
    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Saved to {args.out}")


if __name__ == "__main__":
    main()
