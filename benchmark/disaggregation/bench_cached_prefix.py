"""Measure PD TTFT with a shared cached prefix and unique request suffixes.

Start a PD router and workers before running this client. Both worker caches are
flushed after warmup; the shared prefix is then warmed on prefill. Decode radix
caching should be disabled to exercise transfer of the entire missing KV range.
Saves per-request responses and client-observed TTFT to the requested JSON file.
"""

import argparse
import asyncio
import json
import random
import time
from pathlib import Path

import aiohttp
import numpy as np


async def main(args):
    rng = random.Random(35762)
    prefix = [rng.randrange(1000, 30000) for _ in range(args.prefix)]
    prompts = [
        prefix + [rng.randrange(1000, 30000) for _ in range(args.unique)]
        for _ in range(args.requests)
    ]
    timeout = aiohttp.ClientTimeout(total=1800)
    async with aiohttp.ClientSession(timeout=timeout) as session:

        async def generate(ids, stream=True):
            start = time.perf_counter()
            first = None
            result = None
            body = {
                "input_ids": ids,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": args.output,
                    "ignore_eos": True,
                },
                "stream": stream,
            }
            async with session.post(args.url + "/generate", json=body) as response:
                response.raise_for_status()
                if stream:
                    async for line in response.content:
                        if not line.startswith(b"data:"):
                            continue
                        payload = line[5:].strip()
                        if payload == b"[DONE]":
                            continue
                        result = json.loads(payload)
                        if first is None:
                            first = time.perf_counter() - start
                else:
                    result = await response.json()
            elapsed = time.perf_counter() - start
            if result is None or "error" in result:
                raise RuntimeError(result)
            return {"ttft_s": first, "elapsed_s": elapsed, "response": result}

        # Warm both cold-prefill and cached-prefix/concurrent shapes. Clear both
        # sides afterwards so measured requests never reuse a unique suffix.
        await generate(prompts[0])
        await asyncio.gather(*(generate(p) for p in prompts[1 : args.concurrency + 1]))
        for url in args.workers:
            async with session.post(url + "/flush_cache", params={"timeout": 30}) as r:
                r.raise_for_status()
        if prefix:
            await generate(prefix)
        sem = asyncio.Semaphore(args.concurrency)

        async def run(ids):
            async with sem:
                return await generate(ids)

        start = time.perf_counter()
        records = await asyncio.gather(*(run(ids) for ids in prompts))
        elapsed = time.perf_counter() - start
    ttfts = [r["ttft_s"] for r in records]
    summary = {
        "args": vars(args),
        "elapsed_s": elapsed,
        "requests_per_s": len(records) / elapsed,
        "ttft_mean_ms": float(np.mean(ttfts) * 1000),
        "ttft_p50_ms": float(np.percentile(ttfts, 50) * 1000),
        "ttft_p99_ms": float(np.percentile(ttfts, 99) * 1000),
        "records": records,
    }
    Path(args.result).write_text(json.dumps(summary, indent=2))
    print(json.dumps({k: v for k, v in summary.items() if k != "records"}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--workers",
        nargs="+",
        default=["http://127.0.0.1:30001", "http://127.0.0.1:30002"],
    )
    parser.add_argument("--prefix", type=int, default=83264)
    parser.add_argument("--unique", type=int, default=6720)
    parser.add_argument("--output", type=int, default=1)
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--result", required=True)
    args = parser.parse_args()
    if args.prefix < 0 or args.unique <= 0:
        parser.error("--prefix must be nonnegative and --unique must be positive")
    if min(args.output, args.requests, args.concurrency) <= 0:
        parser.error("--output, --requests and --concurrency must be positive")
    asyncio.run(main(args))
