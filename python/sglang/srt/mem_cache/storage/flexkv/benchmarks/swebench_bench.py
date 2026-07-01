"""SWE-bench Lite (oracle) workload driver for an OpenAI-compatible endpoint.

Loads prompts from ``princeton-nlp/SWE-bench_Lite_oracle``, sends them through
``--concurrency`` async workers paced at ``--qps``, and records TTFT / E2E /
generated token counts. Supports a two-pass mode: pass 1 populates the host
KV cache; pass 2 is the measured run and is what appears in the summary.

Requires ``pip install datasets transformers aiohttp``. ``MODEL_PATH`` env var
must point at a local checkpoint (used for tokenizer-based length filtering
only; requests still hit ``--base-url``).

Usage
-----
    export MODEL_PATH=/path/to/Qwen3-8B
    python swebench_bench.py \
        --base-url http://127.0.0.1:30000 \
        --model Qwen/Qwen3-8B \
        --label flexkv \
        --num-prompts 120 --qps 2.0 --concurrency 24 \
        --max-input-tokens 28000 --max-new-tokens 32 \
        --passes 2 --out flexkv.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import statistics
import time
from typing import List

import aiohttp

PROMPT_TEMPLATE = (
    "You are an expert software engineer. Solve the following GitHub issue by "
    "producing a unified diff patch. Output ONLY the patch in standard diff "
    "format inside <patch> ... </patch> tags.\n\n"
    "{text}"
)


def load_prompts(max_input_tokens: int, n: int, seed: int):
    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset("princeton-nlp/SWE-bench_Lite_oracle", split="test")
    tok = AutoTokenizer.from_pretrained(
        os.environ["MODEL_PATH"], trust_remote_code=True
    )
    items = []
    for ex in ds:
        body = PROMPT_TEMPLATE.format(text=ex["text"])
        ids = tok.encode(body)
        if len(ids) > max_input_tokens:
            continue
        items.append(
            {
                "instance_id": ex["instance_id"],
                "repo": ex["repo"],
                "prompt": body,
                "n_tokens": len(ids),
            }
        )
    items.sort(key=lambda x: (x["repo"], x["instance_id"]))
    rng = random.Random(seed)
    by_repo: dict = {}
    for it in items:
        by_repo.setdefault(it["repo"], []).append(it)
    out = []
    for _, lst in sorted(by_repo.items()):
        rng.shuffle(lst)
        out.extend(lst)
    if n > 0:
        out = out[:n]
    return out


async def one_request(session, base_url, model, item, max_new_tokens, results):
    url = f"{base_url}/v1/completions"
    payload = {
        "model": model,
        "prompt": item["prompt"],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    t_send = time.perf_counter()
    ttft = None
    n_out = 0
    text_out = ""
    err = None
    try:
        async with session.post(
            url,
            json=payload,
            headers=headers,
            timeout=aiohttp.ClientTimeout(total=600),
        ) as resp:
            buf = b""
            done = False
            async for chunk in resp.content.iter_any():
                if not chunk:
                    continue
                buf += chunk
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    line = line.strip()
                    if not line or not line.startswith(b"data: "):
                        continue
                    data = line[6:].strip()
                    if data == b"[DONE]":
                        done = True
                        break
                    if ttft is None:
                        ttft = time.perf_counter() - t_send
                    try:
                        obj = json.loads(data)
                    except Exception:
                        continue
                    ch = obj.get("choices") or []
                    if ch:
                        piece = ch[0].get("text") or ""
                        if piece:
                            text_out += piece
                            n_out += 1
                if done:
                    break
    except Exception as e:
        err = str(e)
    t_end = time.perf_counter()
    results.append(
        {
            "instance_id": item["instance_id"],
            "repo": item["repo"],
            "n_in": item["n_tokens"],
            "ttft": ttft,
            "e2e": t_end - t_send,
            "n_out": n_out,
            "text": text_out,
            "err": err,
        }
    )


async def producer(items, qps, queue, n_workers):
    interval = 1.0 / qps if qps > 0 else 0.0
    t_start = time.perf_counter()
    for i, it in enumerate(items):
        target = t_start + i * interval
        delay = target - time.perf_counter()
        if delay > 0:
            await asyncio.sleep(delay)
        await queue.put(it)
    for _ in range(n_workers):
        await queue.put(None)


async def worker(session, base_url, model, max_new_tokens, queue, results):
    while True:
        it = await queue.get()
        if it is None:
            return
        await one_request(session, base_url, model, it, max_new_tokens, results)


async def run_pass(items, args, session, label):
    results: List[dict] = []
    q: asyncio.Queue = asyncio.Queue(maxsize=args.concurrency * 2)
    prod = asyncio.create_task(producer(items, args.qps, q, args.concurrency))
    workers = [
        asyncio.create_task(
            worker(
                session,
                args.base_url,
                args.model,
                args.max_new_tokens,
                q,
                results,
            )
        )
        for _ in range(args.concurrency)
    ]
    t0 = time.perf_counter()
    await prod
    await asyncio.gather(*workers)
    wall = time.perf_counter() - t0
    ok = sum(1 for r in results if r["err"] is None and r["ttft"] is not None)
    print(f"[swebench:{label}] wall={wall:.1f}s ok={ok}/{len(results)}")
    return results, wall


async def run(args):
    items = load_prompts(args.max_input_tokens, args.num_prompts, args.seed)
    print(
        f"[swebench] loaded {len(items)} prompts; "
        f"p50_in={statistics.median(x['n_tokens'] for x in items):.0f}, "
        f"max_in={max(x['n_tokens'] for x in items)}"
    )
    timeout = aiohttp.ClientTimeout(total=None, sock_connect=10, sock_read=600)
    conn = aiohttp.TCPConnector(limit=args.concurrency * 2)
    cold_results = None
    async with aiohttp.ClientSession(timeout=timeout, connector=conn) as session:
        if args.passes > 1:
            cold_args = argparse.Namespace(**vars(args))
            if not args.same_output_for_cold:
                cold_args.max_new_tokens = max(16, args.max_new_tokens // 4)
            cold_results, _ = await run_pass(items, cold_args, session, "cold")
        results, wall = await run_pass(items, args, session, "warm")

    ok = [r for r in results if r["err"] is None and r["ttft"] is not None]
    fail = len(results) - len(ok)
    ttfts = sorted(r["ttft"] for r in ok)
    e2es = sorted(r["e2e"] for r in ok)
    n_out = sum(r["n_out"] for r in ok)
    n_in = sum(r["n_in"] for r in ok)

    def pct(xs, p):
        return xs[max(0, min(len(xs) - 1, int(len(xs) * p)))]

    summary = {
        "config": args.label,
        "wall_s": wall,
        "n_requests": len(items),
        "n_ok": len(ok),
        "n_fail": fail,
        "throughput_reqs_s": len(ok) / wall if wall else 0,
        "input_tps": n_in / wall if wall else 0,
        "output_tps": n_out / wall if wall else 0,
        "ttft_avg": statistics.mean(ttfts) if ttfts else None,
        "ttft_p50": pct(ttfts, 0.5) if ttfts else None,
        "ttft_p90": pct(ttfts, 0.9) if ttfts else None,
        "ttft_p99": pct(ttfts, 0.99) if ttfts else None,
        "e2e_avg": statistics.mean(e2es) if e2es else None,
        "e2e_p50": pct(e2es, 0.5) if e2es else None,
        "e2e_p90": pct(e2es, 0.9) if e2es else None,
    }
    print("=" * 60)
    print(f"config             = {summary['config']}")
    print(f"requests ok / fail = {summary['n_ok']} / {summary['n_fail']}")
    print(f"wall               = {summary['wall_s']:.1f}s")
    print(f"throughput         = {summary['throughput_reqs_s']:.3f} req/s")
    print(f"input tok/s        = {summary['input_tps']:.1f}")
    print(f"output tok/s       = {summary['output_tps']:.1f}")
    print(
        f"TTFT avg/p50/p90/p99 = {summary['ttft_avg']:.2f} / "
        f"{summary['ttft_p50']:.2f} / {summary['ttft_p90']:.2f} / "
        f"{summary['ttft_p99']:.2f} s"
    )
    print(
        f"E2E  avg/p50/p90     = {summary['e2e_avg']:.2f} / "
        f"{summary['e2e_p50']:.2f} / {summary['e2e_p90']:.2f} s"
    )
    if args.out:
        payload = {"summary": summary, "results": results}
        if cold_results is not None:
            payload["cold_results"] = cold_results
        with open(args.out, "w") as fh:
            json.dump(payload, fh)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base-url", default="http://127.0.0.1:30000")
    p.add_argument("--model", required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--num-prompts", type=int, default=120)
    p.add_argument("--max-input-tokens", type=int, default=28000)
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--qps", type=float, default=2.0)
    p.add_argument("--concurrency", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--passes",
        type=int,
        default=2,
        help=(
            "If >1, send each prompt as a cold pass first "
            "(short output) to populate the host cache; the "
            "final pass is the measured one."
        ),
    )
    p.add_argument(
        "--same-output-for-cold",
        action="store_true",
        help=(
            "Use the same max_new_tokens for cold pass as warm; "
            "needed if you want to byte-compare cold vs warm output."
        ),
    )
    p.add_argument("--out", default="")
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
