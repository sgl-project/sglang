#!/usr/bin/env python3
"""LPLB e2e functional test — verifies correctness under concurrent load."""
import asyncio
import aiohttp
import json
import sys
import time

SERVER = "http://localhost:30000"
PROMPT = "The capital of France is"


async def send_one(session, sem, prompt, max_tokens=1, timeout=120):
    """Send one completion request, return (status, latency, error)."""
    async with sem:
        t0 = time.perf_counter()
        try:
            async with session.post(
                f"{SERVER}/v1/completions",
                json={"model": "default", "prompt": prompt, "max_tokens": max_tokens},
                timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                body = await resp.json()
                dt = time.perf_counter() - t0
                if resp.status == 200:
                    return {"ok": True, "latency": dt, "tokens": body["usage"]["prompt_tokens"]}
                else:
                    return {"ok": False, "latency": dt, "error": f"HTTP {resp.status}: {body}"}
        except Exception as e:
            return {"ok": False, "latency": time.perf_counter() - t0, "error": str(e)}


async def send_batch(prompts, concurrency, max_tokens=1, timeout=120):
    """Send a batch of requests with bounded concurrency."""
    sem = asyncio.Semaphore(concurrency)
    async with aiohttp.ClientSession() as session:
        tasks = [send_one(session, sem, p, max_tokens, timeout) for p in prompts]
        return await asyncio.gather(*tasks)


def report(label, results):
    ok = [r for r in results if r["ok"]]
    fail = [r for r in results if not r["ok"]]
    latencies = [r["latency"] for r in ok]
    avg_lat = sum(latencies) / len(latencies) if latencies else 0
    print(f"  {label}: {len(ok)}/{len(results)} OK, "
          f"avg_lat={avg_lat:.3f}s, "
          f"errors={len(fail)}")
    if fail:
        for f in fail[:3]:
            print(f"    ERROR: {f['error'][:100]}")
    return len(fail) == 0


def main():
    print("=" * 60)
    print("LPLB E2E Functional Test")
    print("=" * 60)
    all_pass = True

    # Test 1: Sequential requests
    print("\n[1/3] Sequential requests (5 requests, concurrency=1)")
    prompts = [f"{PROMPT} {i}" for i in range(5)]
    results = asyncio.run(send_batch(prompts, concurrency=1, timeout=300))
    if not report("sequential", results):
        all_pass = False

    # Test 2: Moderate concurrency
    print("\n[2/3] Concurrent requests (50 requests, concurrency=16)")
    prompts = [f"Question {i}: What is {i}+{i}?" for i in range(50)]
    results = asyncio.run(send_batch(prompts, concurrency=16, timeout=120))
    if not report("conc=16", results):
        all_pass = False

    # Test 3: High concurrency (stress DP-attention all-reduce)
    print("\n[3/3] High concurrency (50 requests, concurrency=64)")
    prompts = [f"Prompt {i}: Explain concept {i} briefly." for i in range(50)]
    results = asyncio.run(send_batch(prompts, concurrency=64, timeout=120))
    if not report("conc=64", results):
        all_pass = False

    print("\n" + "=" * 60)
    if all_pass:
        print("RESULT: ALL TESTS PASSED")
    else:
        print("RESULT: SOME TESTS FAILED")
    print("=" * 60)
    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
