#!/usr/bin/env python3
"""c=1 sequential streaming /generate driver for the rust-TM TTFT waterfall.

Sends requests one at a time (fresh connection each) with client-chosen rids,
and records client-side CLOCK_MONOTONIC stamps per request as JSONL. Same-host
client, rust stamps, and scheduler stamps all share CLOCK_MONOTONIC on Linux,
so postprocess.py can join them with no clock calibration.
"""

import argparse
import asyncio
import json
import os
import random
import time
import uuid

import aiohttp


def build_prompt(tokenizer, target_len: int, seed: int) -> str:
    """Random-token prompt of <= target_len tokens (trimmed via re-encode)."""
    rng = random.Random(seed)
    if target_len <= 1:
        return "a"
    ids = [rng.randint(256, tokenizer.vocab_size - 1) for _ in range(target_len)]
    text = tokenizer.decode(ids, skip_special_tokens=True)
    for _ in range(4):
        re_ids = tokenizer.encode(text, add_special_tokens=True)
        if len(re_ids) <= target_len:
            break
        text = tokenizer.decode(re_ids[:target_len], skip_special_tokens=True)
    return text


async def run_one(url: str, prompt: str, rid: str, output_len: int) -> dict:
    payload = {
        "text": prompt,
        "rid": rid,
        "stream": True,
        "sampling_params": {
            "max_new_tokens": output_len,
            "temperature": 0.0,
            "ignore_eos": True,
        },
    }
    t_first_ns = None
    last_meta = None
    connector = aiohttp.TCPConnector(force_close=True, limit=1)
    timeout = aiohttp.ClientTimeout(total=1200)
    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as sess:
        t_send_ns = time.perf_counter_ns()
        async with sess.post(url, json=payload) as resp:
            assert resp.status == 200, f"HTTP {resp.status}: {await resp.text()}"
            buf = b""
            async for chunk in resp.content.iter_any():
                now = time.perf_counter_ns()
                buf += chunk
                while b"\n\n" in buf:
                    event, buf = buf.split(b"\n\n", 1)
                    if not event.startswith(b"data:"):
                        continue
                    data = event[len(b"data:") :].strip()
                    if data == b"[DONE]":
                        continue
                    if t_first_ns is None:
                        t_first_ns = now
                    last_meta = json.loads(data).get("meta_info", {})
        t_done_ns = time.perf_counter_ns()
    assert t_first_ns is not None, "no data frame received"
    return {
        "rid": rid,
        "prompt_tokens": (last_meta or {}).get("prompt_tokens"),
        "t_send_ns": t_send_ns,
        "t_first_ns": t_first_ns,
        "t_done_ns": t_done_ns,
        "ttft_ms": (t_first_ns - t_send_ns) / 1e6,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=30000)
    ap.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    ap.add_argument("--input-len", type=int, required=True)
    ap.add_argument("--num-requests", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--output-len", type=int, default=8)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-cache", help="JSON file to save/load prompts")
    args = ap.parse_args()

    url = f"http://{args.host}:{args.port}/generate"
    n_total = args.warmup + args.num_requests

    if args.prompt_cache and os.path.exists(args.prompt_cache):
        prompts = json.load(open(args.prompt_cache))[:n_total]
        assert len(prompts) >= n_total, "prompt cache too small"
    else:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.model)
        prompts = [
            build_prompt(tokenizer, args.input_len, seed=1000 * args.input_len + i)
            for i in range(n_total)
        ]
        if args.prompt_cache:
            json.dump(prompts, open(args.prompt_cache, "w"))

    results = []
    for i, prompt in enumerate(prompts):
        is_warmup = i < args.warmup
        tag = "warm" if is_warmup else "meas"
        rid = f"ttft-{args.input_len}-{tag}-{i:03d}-{uuid.uuid4().hex[:8]}"
        r = await run_one(url, prompt, rid, args.output_len)
        r["input_len_target"] = args.input_len
        print(
            f"[{tag} {i + 1}/{n_total}] in={args.input_len} rid={rid} "
            f"prompt_tokens={r['prompt_tokens']} ttft={r['ttft_ms']:.2f}ms",
            flush=True,
        )
        if not is_warmup:
            results.append(r)

    with open(args.out, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    ttfts = sorted(r["ttft_ms"] for r in results)
    n = len(ttfts)
    print(
        f"DONE in={args.input_len} n={n} mean={sum(ttfts) / n:.2f} "
        f"p50={ttfts[n // 2]:.2f} p99={ttfts[min(n - 1, int(n * 0.99))]:.2f} (ms)"
    )


if __name__ == "__main__":
    asyncio.run(main())
