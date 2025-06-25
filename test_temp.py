"""
Concurrent batched benchmarking.

Example
-------
python3 send_batch_many.py \
    --host localhost --port 30000 \
    --batch-size 32  --concurrency 8 --num-batches 100
"""

import argparse
import asyncio
import json
import random
import time
from dataclasses import dataclass, fields

import aiohttp

# ---------------------------- CLI args ---------------------------- #


@dataclass
class Args:
    host: str = "localhost"
    port: int = 30000
    batch_size: int = 16  # prompts per request
    num_batches: int = 2  # how many batch requests in total
    concurrency: int = 2  # simultaneous in-flight requests
    temperature: float = 0.0
    max_new_tokens: int = 16
    stream: bool = False

    @staticmethod
    def add_cli_args(p: argparse.ArgumentParser):
        for f in fields(Args):
            flag = f"--{f.name.replace('_', '-')}"
            if isinstance(f.default, bool):
                p.add_argument(flag, action="store_true")
            else:
                p.add_argument(flag, type=type(f.default), default=f.default)

    @classmethod
    def from_cli(cls, ns):
        return cls(**{f.name: getattr(ns, f.name) for f in fields(cls)})


# ---------------------------- prompt pool ---------------------------- #

PROMPTS = [f"Human: {n}+{n}=?\n\nAssistant:" for n in range(1, 201)]  # 200 小算术


# ---------------------------- core helpers ---------------------------- #


async def one_batch(session: aiohttp.ClientSession, cfg: Args, batch_id: int):
    offset = (batch_id * cfg.batch_size) % len(PROMPTS)
    prompts = [PROMPTS[(offset + i) % len(PROMPTS)] for i in range(cfg.batch_size)]
    payload = {
        "text": prompts,  # <-- 列表！
        "image_data": None,
        "sampling_params": {
            "temperature": cfg.temperature,
            "max_new_tokens": cfg.max_new_tokens,
            "stop": ["Question", "Assistant:", "<|separator|>", "<|eos|>"],
        },
        "return_logprob": False,
        "stream": cfg.stream,
    }

    url = f"http://{cfg.host}:{cfg.port}/generate"
    t0 = time.perf_counter()

    async with session.post(url, json=payload, timeout=None) as resp:
        if cfg.stream:
            # 逐行读取流式返回，直到 [DONE]
            chunks = []
            async for line in resp.content:
                if line.startswith(b"data: [DONE]"):
                    break
                if line.startswith(b"data:"):
                    chunks.append(json.loads(line[5:].strip()))
            data = chunks[-1]  # 最后一条包含完整 batch
        else:
            data = await resp.json()

    latency = time.perf_counter() - t0
    meta = data[0]["meta_info"]
    speed = meta["completion_tokens"] / latency if latency else 0

    # ---- 打印每条回答 ---- #
    for i, item in enumerate(data):
        q = prompts[i].splitlines()[0]
        a = item["text"].strip()
        print(f"[B{batch_id:03d}|{i:02d}] {q} -> {a}")

    return latency, speed, cfg.batch_size


async def runner(cfg: Args):
    sem = asyncio.Semaphore(cfg.concurrency)
    lats, spds, cnts = [], [], []

    async with aiohttp.ClientSession() as session:

        async def task(bid: int):
            async with sem:
                try:
                    l, s, c = await one_batch(session, cfg, bid)
                    lats.append(l)
                    spds.append(s)
                    cnts.append(c)
                except Exception as e:
                    print(f"[B{bid:03d}] ERROR: {e}")

        await asyncio.gather(*(task(b) for b in range(cfg.num_batches)))

    if lats:
        p95 = sorted(lats)[int(0.95 * len(lats)) - 1]
        total_tokens = sum(s * l for s, l in zip(spds, lats))
        total_time = sum(lats)
        print("\n===== summary =====")
        print(f"Batches completed  : {len(lats)}")
        print(f"Batch size         : {cfg.batch_size}")
        print(f"Avg latency / batch: {sum(lats)/len(lats):.2f} s")
        print(f"P95 latency        : {p95:.2f} s")
        print(f"Aggregate throughput: {total_tokens/total_time:.2f} tok/s")


# ---------------------------- entry ---------------------------- #

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    Args.add_cli_args(parser)
    cfg = Args.from_cli(parser.parse_args())
    asyncio.run(runner(cfg))
