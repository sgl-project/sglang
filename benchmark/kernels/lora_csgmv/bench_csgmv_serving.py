"""Measure a running SGLang server with loaded LoRA adapters.

Start the normal server with --enable-lora --lora-backend csgmv. Set
SGLANG_CSGMV_SPLIT_K=1 in its environment to enable the fork's split-K policy.
Kernel settings belong to the server; restart it to change them.

python benchmark/kernels/lora_csgmv/bench_csgmv_serving.py \
    --model Qwen/Qwen3.5-9B --lora-names adapter0 adapter1 \
    --input-len 512 --output-lens 4096 8192 --concurrency 32 --flush-cache

Cache flushing affects every request on the server; use a dedicated server.
"""

import argparse
import asyncio
import json
import math
import statistics
import time
from pathlib import Path

import httpx
from transformers import AutoTokenizer


async def run_batch(client, prompts, adapters, output_len):
    async def request(i, prompt):
        started = time.perf_counter()
        response = await client.post(
            "/generate",
            json={
                "input_ids": prompt,
                "lora_path": adapters[i % len(adapters)],
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": output_len,
                    "ignore_eos": True,
                },
                "stream": False,
            },
        )
        response.raise_for_status()
        body = response.json()
        meta = body["meta_info"]
        if meta.get("finish_reason", {}).get("type") == "abort":
            raise RuntimeError(f"Request {i} aborted: {meta['finish_reason']}")
        if meta["completion_tokens"] != output_len:
            raise RuntimeError(
                f"Request {i} returned {meta['completion_tokens']} tokens; expected {output_len}"
            )
        return {
            "request": i,
            "adapter": adapters[i % len(adapters)],
            "latency_s": time.perf_counter() - started,
            "meta": meta,
        }

    started = time.perf_counter()
    records = await asyncio.gather(
        *(request(i, prompt) for i, prompt in enumerate(prompts))
    )
    wall = time.perf_counter() - started
    latencies = sorted(r["latency_s"] for r in records)
    return {
        "input_tokens": len(prompts[0]),
        "output_tokens": output_len,
        "concurrency": len(prompts),
        "batch_wall_s": wall,
        "output_tokens_per_second": len(prompts) * output_len / wall,
        "request_latency_median_s": statistics.median(latencies),
        "request_latency_p95_s": latencies[math.ceil(0.95 * len(latencies)) - 1],
        "requests": records,
    }


async def benchmark(args, prompts):
    limits = httpx.Limits(
        max_connections=args.concurrency, max_keepalive_connections=args.concurrency
    )
    async with httpx.AsyncClient(
        base_url=args.base_url.rstrip("/"), timeout=args.timeout, limits=limits
    ) as client:
        response = await client.get("/get_server_info")
        response.raise_for_status()
        result = {
            "label": args.label,
            "model": args.model,
            "input_len": args.input_len,
            "output_lens": args.output_lens,
            "concurrency": args.concurrency,
            "flush_cache": args.flush_cache,
            "lora_names": args.lora_names,
            "server_info": response.json(),
            "measurements": [],
        }
        await run_batch(client, prompts, args.lora_names, args.warmup_output_len)
        for repeat in range(args.repeats):
            lengths = (
                args.output_lens
                if repeat % 2 == 0
                else list(reversed(args.output_lens))
            )
            for output_len in lengths:
                if args.flush_cache:
                    response = await client.post("/flush_cache")
                    response.raise_for_status()
                measurement = await run_batch(
                    client, prompts, args.lora_names, output_len
                )
                measurement["repeat"] = repeat
                result["measurements"].append(measurement)
                print(
                    json.dumps(
                        {k: v for k, v in measurement.items() if k != "requests"}
                    ),
                    flush=True,
                )
                if args.output:
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument(
        "--model", required=True, help="Model/tokenizer path matching the server"
    )
    parser.add_argument(
        "--lora-names",
        nargs="+",
        required=True,
        help="Adapter names already loaded by the server",
    )
    parser.add_argument("--input-len", type=int, default=512)
    parser.add_argument("--output-lens", type=int, nargs="+", default=[4096, 8192])
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--warmup-output-len", type=int, default=128)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--flush-cache", action="store_true")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--label",
        default="",
        help="Label for this server configuration in the saved results",
    )
    args = parser.parse_args()
    if (
        min(
            args.input_len,
            *args.output_lens,
            args.concurrency,
            args.repeats,
            args.warmup_output_len,
            args.timeout,
        )
        <= 0
    ):
        parser.error("Lengths, concurrency, repeats, and timeout must be positive")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    prose = "Explain external merge sort, including memory limits, stable ordering, disk access, and correctness. "
    body = tokenizer.encode(prose, add_special_tokens=False)
    if not body:
        parser.error("Tokenizer returned an empty prompt")
    prompts = []
    for i in range(args.concurrency):
        prefix = tokenizer.encode(f"Problem {i}. ", add_special_tokens=False)
        ids = prefix + body * math.ceil(args.input_len / len(body))
        prompts.append(ids[: args.input_len])
    asyncio.run(benchmark(args, prompts))


if __name__ == "__main__":
    main()
