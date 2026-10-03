"""Fixed-workload serving experiment with checked streaming responses.

Independent client processes use identical prompts for each TP/EP pair. Requests
used for warmup have different token IDs. Full per-request metrics are retained.
"""

import argparse
import asyncio
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import aiohttp


def make_prompts(n, length, seed):
    rng = random.Random(seed)
    return [[rng.randint(1000, 150000) for _ in range(length)] for _ in range(n)]


async def request(session, url, ids, output_len):
    start = time.perf_counter()
    first = None
    completion = 0
    payload = dict(
        input_ids=ids,
        stream=True,
        sampling_params=dict(max_new_tokens=output_len, temperature=0, ignore_eos=True),
    )
    async with session.post(url, json=payload) as response:
        response.raise_for_status()
        async for raw in response.content:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            event = json.loads(data)
            if "error" in event:
                raise RuntimeError(event["error"])
            count = event.get("meta_info", {}).get("completion_tokens", 0)
            if count > 0 and first is None:
                first = time.perf_counter()
            completion = max(count, completion)
    end = time.perf_counter()
    if first is None or completion != output_len:
        raise RuntimeError(f"incomplete response: {completion}/{output_len} tokens")
    return dict(
        ttft_ms=(first - start) * 1000,
        total_ms=(end - start) * 1000,
        tpot_ms=(end - first) * 1000 / (completion - 1) if completion > 1 else None,
        output_tokens=completion,
    )


async def run(args):
    prompts = make_prompts(args.requests, args.input_len, args.seed)
    warmup = make_prompts(max(2, args.concurrency), args.input_len, args.seed + 100000)
    sem = asyncio.Semaphore(args.concurrency)
    url = f"http://127.0.0.1:{args.port}/generate"
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(total=600)
    ) as session:

        async def guarded(ids):
            async with sem:
                return await request(session, url, ids, args.output_len)

        await asyncio.gather(*(guarded(ids) for ids in warmup))
        start = time.perf_counter()
        results = await asyncio.gather(*(guarded(ids) for ids in prompts))
        wall = time.perf_counter() - start
    report = dict(
        input_len=args.input_len,
        output_len=args.output_len,
        requests=args.requests,
        concurrency=args.concurrency,
        seed=args.seed,
        warmup_requests=len(warmup),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        input_sha256=hashlib.sha256(json.dumps(prompts).encode()).hexdigest(),
        wall_seconds=wall,
        output_tokens=sum(r["output_tokens"] for r in results),
        throughput=sum(r["output_tokens"] for r in results) / wall,
        per_request=results,
    )
    for key in ("ttft_ms", "total_ms", "tpot_ms"):
        values = sorted(r[key] for r in results if r[key] is not None)
        report[key + "_median"] = statistics.median(values) if values else None
        report[key + "_p95"] = (
            values[min(len(values) - 1, int(len(values) * 0.95))] if values else None
        )
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in report.items() if k != "per_request"}), flush=True
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=30198)
    p.add_argument("--input-len", type=int, required=True)
    p.add_argument("--output-len", type=int, default=64)
    p.add_argument("--requests", type=int, default=16)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--seed", type=int, default=20260922)
    p.add_argument("--out", required=True)
    asyncio.run(run(p.parse_args()))


if __name__ == "__main__":
    main()
