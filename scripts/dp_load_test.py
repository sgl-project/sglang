#!/usr/bin/env python3
import argparse
import json
import random
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional
from urllib import request as urlrequest
from urllib.error import URLError, HTTPError

PROMPTS = [
    "用一句话介绍东京",
    "请用两点总结机器学习的主要挑战",
    "解释一下量子纠缠是什么",
    "列出三种节能的小技巧",
    "用一句话描述富士山",
    "给出一个关于排序算法的直观解释",
    "简单介绍一下太阳风",
    "说出两种常见的数据库索引",
    "解释微服务的核心思想",
    "一句话解释区块链的不可篡改性",
    "简要说明为什么天空是蓝色的",
    "用一句话概括“缓存命中率”的意义",
]


def build_payload(
    model: str,
    prompt: str,
    max_tokens: int,
    rid: str,
    data_parallel_rank: Optional[int],
):
    payload = {
        "model": model,
        "temperature": 0,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": False,
        "rid": rid,
    }
    if data_parallel_rank is not None:
        payload["data_parallel_rank"] = data_parallel_rank
    return payload


def send_one(
    endpoint: str,
    model: str,
    prompt: str,
    max_tokens: int,
    rid: str,
    data_parallel_rank: Optional[int],
    timeout: float,
):
    payload = build_payload(model, prompt, max_tokens, rid, data_parallel_rank)
    data = json.dumps(payload).encode("utf-8")
    req = urlrequest.Request(
        endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.time()
    ok = False
    status = None
    err = None
    try:
        with urlrequest.urlopen(req, timeout=timeout) as resp:
            status = resp.status
            _ = resp.read()
            ok = status == 200
    except HTTPError as e:
        status = e.code
        err = f"HTTPError: {e}"
    except URLError as e:
        err = f"URLError: {e}"
    except Exception as e:
        err = f"Error: {e}"
    latency = time.time() - start
    return {
        "rid": rid,
        "data_parallel_rank": data_parallel_rank,
        "ok": ok,
        "status": status,
        "latency": latency,
        "error": err,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="DP load test for SGLang")
    parser.add_argument("--endpoint", default="http://127.0.0.1:30000/v1/chat/completions")
    parser.add_argument("--model", default="/models/deepseek-r1-1.5b")
    parser.add_argument("--duration", type=int, default=120, help="seconds")
    parser.add_argument("--interval", type=float, default=1.0, help="seconds between requests")
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--force-dp",
        action="store_true",
        help="alternate requests across dp ranks using data_parallel_rank",
    )
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--csv", default=None, help="write per-request metrics to CSV")
    return parser.parse_args()


def main():
    args = parse_args()
    end_time = time.time() + args.duration
    futures = []
    results = []
    next_time = time.time()
    i = 0

    def pick_prompt(idx: int) -> str:
        base = random.choice(PROMPTS)
        return f"{base}（问题编号 {idx}）"

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        while time.time() < end_time:
            now = time.time()
            if now >= next_time:
                rid = f"dp_test_{i}_{uuid.uuid4().hex[:8]}"
                dp_rank = (i % args.dp_size) if args.force_dp else None
                prompt = pick_prompt(i)
                fut = pool.submit(
                    send_one,
                    args.endpoint,
                    args.model,
                    prompt,
                    args.max_tokens,
                    rid,
                    dp_rank,
                    args.timeout,
                )
                futures.append(fut)
                i += 1
                next_time += args.interval
            else:
                time.sleep(min(0.02, next_time - now))

        for fut in as_completed(futures):
            res = fut.result()
            results.append(res)
            if args.verbose:
                print(res)

    ok_lat = [r["latency"] for r in results if r["ok"]]
    ok_count = sum(1 for r in results if r["ok"])
    total = len(results)

    avg = statistics.mean(ok_lat) if ok_lat else 0.0
    p95 = statistics.quantiles(ok_lat, n=20)[-1] if len(ok_lat) >= 2 else avg

    print("=== Summary ===")
    print(f"total_requests={total}")
    print(f"success={ok_count}")
    print(f"success_rate={(ok_count / total * 100) if total else 0:.2f}%")
    print(f"avg_latency={avg:.3f}s")
    print(f"p95_latency={p95:.3f}s")

    if args.csv:
        import csv

        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["rid", "data_parallel_rank", "ok", "status", "latency", "error"],
            )
            writer.writeheader()
            for r in results:
                writer.writerow(r)


if __name__ == "__main__":
    main()
