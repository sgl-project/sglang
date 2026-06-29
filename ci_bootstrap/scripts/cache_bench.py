#!/usr/bin/env python3
"""Prefix-cache serving benchmark for sglang.

Measures how serving performance scales as a function of the shared-prefix ratio.
Total input per request is always `total_tokens` (default 70,000).
The shared prefix (identical across all requests in a batch) grows from 0%
to 100%, and the unique suffix (fresh random tokens per request) shrinks.

Uses sglang's native /generate endpoint with streaming enabled so TTFT, ITL,
and TPOT are measured similarly to sglang.bench_serving.
"""

import argparse
import asyncio
import csv
import json
import random
import time
from dataclasses import dataclass, field

import aiohttp
import numpy as np
import requests
from transformers import AutoTokenizer


BENCH_AIOHTTP_TIMEOUT_SECONDS = 600
BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024**2


@dataclass
class RequestResult:
    success: bool = False
    error: str = ""
    e2e: float = 0.0
    ttft: float = 0.0
    itl: list[float] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    generated_text: str = ""


def remove_prefix(text: str, prefix: str) -> str:
    if text.startswith(prefix):
        return text[len(prefix):]
    return text


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(values, q))


def mean(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(values))


def std(values: list[float]) -> float:
    if not values:
        return 0.0
    return float(np.std(values))


async def send_request(
    session: aiohttp.ClientSession,
    url: str,
    input_ids: list[int],
    prompt_len: int,
    output_len: int,
    semaphore: asyncio.Semaphore,
) -> RequestResult:
    payload = {
        "input_ids": input_ids,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_len,
            "ignore_eos": True,
        },
        "stream": True,
    }

    result = RequestResult(input_tokens=prompt_len)

    async with semaphore:
        st = time.perf_counter()
        most_recent_timestamp = st
        last_output_len = 0

        try:
            async with session.post(url, json=payload) as resp:
                if resp.status != 200:
                    body = await resp.text()
                    result.error = f"HTTP {resp.status}: {body}"
                    return result

                async for chunk_bytes in resp.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue

                    chunk = chunk_bytes.decode("utf-8")
                    chunk = remove_prefix(chunk, "data: ")
                    chunk = remove_prefix(chunk, "data:")

                    if chunk == "[DONE]":
                        continue

                    data = json.loads(chunk)

                    # Mirror sglang native streaming behavior:
                    # only act on chunks that carry text.
                    if "text" not in data or not data["text"]:
                        continue

                    timestamp = time.perf_counter()
                    latency = timestamp - st

                    result.generated_text = data["text"]
                    current_output_len = data["meta_info"]["completion_tokens"]

                    # First token.
                    if result.ttft == 0.0:
                        result.ttft = timestamp - st
                    else:
                        # Some chunks may advance multiple tokens; distribute
                        # the gap evenly across those newly produced tokens.
                        num_new_tokens = current_output_len - last_output_len
                        if num_new_tokens == 0:
                            continue

                        chunk_gap = timestamp - most_recent_timestamp
                        adjust_itl = chunk_gap / num_new_tokens
                        result.itl.extend([adjust_itl] * num_new_tokens)

                    most_recent_timestamp = timestamp
                    last_output_len = current_output_len

                    result.success = True
                    result.e2e = latency
                    result.output_tokens = current_output_len

        except Exception as exc:
            result.error = str(exc)

    return result


async def run_batch(
    base_url: str,
    prompts: list[dict],
    output_len: int,
    max_concurrency: int,
) -> list[RequestResult]:
    sem = asyncio.Semaphore(max_concurrency)
    url = f"{base_url}/generate"
    timeout = aiohttp.ClientTimeout(total=BENCH_AIOHTTP_TIMEOUT_SECONDS)

    async with aiohttp.ClientSession(
        timeout=timeout,
        read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES,
    ) as session:
        tasks = [
            asyncio.create_task(
                send_request(
                    session=session,
                    url=url,
                    input_ids=p["input_ids"],
                    prompt_len=p["prompt_len"],
                    output_len=output_len,
                    semaphore=sem,
                )
            )
            for p in prompts
        ]
        return await asyncio.gather(*tasks)


def flush_cache(base_url: str) -> None:
    resp = requests.post(f"{base_url}/flush_cache", timeout=30)
    resp.raise_for_status()


def gen_token_ids(vocab_ids: list[int], token_num: int, rng: random.Random) -> list[int]:
    """Generate a list of exactly `token_num` random token IDs."""
    if token_num <= 0:
        return []
    return rng.choices(vocab_ids, k=token_num)


def build_prompts(
    vocab_ids: list[int],
    total_tokens: int,
    shared_pct: int,
    num_prompts: int,
    rng: random.Random,
) -> list[dict]:
    prefix_len = total_tokens * shared_pct // 100
    suffix_len = total_tokens - prefix_len

    shared_prefix = gen_token_ids(vocab_ids, prefix_len, rng)

    prompts = []
    for _ in range(num_prompts):
        suffix = gen_token_ids(vocab_ids, suffix_len, rng)
        input_ids = shared_prefix + suffix
        prompts.append({"input_ids": input_ids, "prompt_len": len(input_ids)})
    return prompts


def calculate_metrics(results: list[RequestResult], elapsed: float) -> dict:
    ok = [r for r in results if r.success]

    total_input = sum(r.input_tokens for r in ok)
    total_output = sum(r.output_tokens for r in ok)

    ttfts_ms = [r.ttft * 1000.0 for r in ok if r.ttft > 0.0]
    tpots_ms = [
        ((r.e2e - r.ttft) / (r.output_tokens - 1)) * 1000.0
        for r in ok
        if r.output_tokens > 1 and r.e2e >= r.ttft
    ]
    itls_ms = [itl * 1000.0 for r in ok for itl in r.itl]
    e2e_ms = [r.e2e * 1000.0 for r in ok]

    request_throughput = len(ok) / elapsed if elapsed > 0 else 0.0
    input_throughput = total_input / elapsed if elapsed > 0 else 0.0
    output_throughput = total_output / elapsed if elapsed > 0 else 0.0
    total_throughput = (total_input + total_output) / elapsed if elapsed > 0 else 0.0
    concurrency = (sum(r.e2e for r in ok) / elapsed) if elapsed > 0 else 0.0

    return {
        "completed": len(ok),
        "elapsed_s": elapsed,
        "total_input_tokens": total_input,
        "total_output_tokens": total_output,
        "request_throughput_req_s": request_throughput,
        "input_throughput_tok_s": input_throughput,
        "output_throughput_tok_s": output_throughput,
        "total_token_throughput_tok_s": total_throughput,
        "mean_ttft_ms": mean(ttfts_ms),
        "median_ttft_ms": percentile(ttfts_ms, 50),
        "std_ttft_ms": std(ttfts_ms),
        "p99_ttft_ms": percentile(ttfts_ms, 99),
        "mean_tpot_ms": mean(tpots_ms),
        "median_tpot_ms": percentile(tpots_ms, 50),
        "std_tpot_ms": std(tpots_ms),
        "p99_tpot_ms": percentile(tpots_ms, 99),
        "mean_itl_ms": mean(itls_ms),
        "median_itl_ms": percentile(itls_ms, 50),
        "std_itl_ms": std(itls_ms),
        "p95_itl_ms": percentile(itls_ms, 95),
        "p99_itl_ms": percentile(itls_ms, 99),
        "max_itl_ms": max(itls_ms) if itls_ms else 0.0,
        "mean_e2e_latency_ms": mean(e2e_ms),
        "median_e2e_latency_ms": percentile(e2e_ms, 50),
        "std_e2e_latency_ms": std(e2e_ms),
        "p90_e2e_latency_ms": percentile(e2e_ms, 90),
        "p99_e2e_latency_ms": percentile(e2e_ms, 99),
        "concurrency": concurrency,
    }


def print_benchmark_result(metrics: dict, tp_size: int) -> None:
    tpm_per_gpu = (
        metrics["total_token_throughput_tok_s"] * 60.0 / tp_size if tp_size > 0 else 0.0
    )

    print("  Benchmark serving result:")
    print(f"    Successful requests:            {metrics['completed']}")
    print(f"    Benchmark duration (s):         {metrics['elapsed_s']:.2f}")
    print(f"    Total input tokens:             {metrics['total_input_tokens']}")
    print(f"    Total generated tokens:         {metrics['total_output_tokens']}")
    print(f"    Request throughput (req/s):     {metrics['request_throughput_req_s']:.2f}")
    print(f"    Input token throughput (tok/s): {metrics['input_throughput_tok_s']:.2f}")
    print(f"    Output token throughput (tok/s): {metrics['output_throughput_tok_s']:.2f}")
    print(f"    Total token throughput (tok/s): {metrics['total_token_throughput_tok_s']:.2f}")
    print(f"    TPM per GPU:                    {tpm_per_gpu:.2f}")
    print(f"    Concurrency:                    {metrics['concurrency']:.2f}")
    print(f"    Mean TTFT (ms):                 {metrics['mean_ttft_ms']:.2f}")
    print(f"    Median TTFT (ms):               {metrics['median_ttft_ms']:.2f}")
    print(f"    P99 TTFT (ms):                  {metrics['p99_ttft_ms']:.2f}")
    print(f"    Mean TPOT (ms):                 {metrics['mean_tpot_ms']:.2f}")
    print(f"    Median TPOT (ms):               {metrics['median_tpot_ms']:.2f}")
    print(f"    P99 TPOT (ms):                  {metrics['p99_tpot_ms']:.2f}")
    print(f"    Mean ITL (ms):                  {metrics['mean_itl_ms']:.2f}")
    print(f"    Median ITL (ms):                {metrics['median_itl_ms']:.2f}")
    print(f"    P95 ITL (ms):                   {metrics['p95_itl_ms']:.2f}")
    print(f"    P99 ITL (ms):                   {metrics['p99_itl_ms']:.2f}")
    print(f"    Max ITL (ms):                   {metrics['max_itl_ms']:.2f}")
    print(f"    Mean E2E latency (ms):          {metrics['mean_e2e_latency_ms']:.2f}")
    print(f"    Median E2E latency (ms):        {metrics['median_e2e_latency_ms']:.2f}")
    print(f"    P90 E2E latency (ms):           {metrics['p90_e2e_latency_ms']:.2f}")
    print(f"    P99 E2E latency (ms):           {metrics['p99_e2e_latency_ms']:.2f}")


async def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--total-tokens",
        type=int,
        default=70000,
        help="Total input tokens per request (shared + unique)",
    )
    parser.add_argument("--output-len", type=int, default=200)
    parser.add_argument("--num-prompts", type=int, default=64)
    parser.add_argument("--max-concurrency", type=int, default=4)
    parser.add_argument("--csv-out", default="bench_results.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model path or HF repo (used to load tokenizer)",
    )
    parser.add_argument(
        "--pcts",
        type=str,
        default="0,10,20,30,40,50,60,70,80,90,92,95,97,99",
        help="Comma-separated shared-prefix percentages to sweep",
    )
    parser.add_argument(
        "--tp-size",
        type=int,
        default=4,
        help="Tensor parallel size used by the server (for TPM/GPU calc)",
    )
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    pcts = [int(p) for p in args.pcts.split(",")]
    rng = random.Random(args.seed)

    print(f"Loading tokenizer from {args.model} ...")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    vocab_ids = list(tokenizer.get_vocab().values())
    print(f"Tokenizer loaded (vocab_size={len(vocab_ids)})")

    csv_fields = [
        "shared_prefix_pct",
        "prefix_len",
        "suffix_len",
        "num_prompts",
        "succeeded",
        "elapsed_s",
        "request_throughput_req_s",
        "total_input_tokens",
        "total_output_tokens",
        "total_token_throughput_tok_s",
        "input_throughput_tok_s",
        "output_throughput_tok_s",
        "concurrency",
        "mean_ttft_ms",
        "median_ttft_ms",
        "p99_ttft_ms",
        "mean_tpot_ms",
        "median_tpot_ms",
        "p99_tpot_ms",
        "mean_itl_ms",
        "median_itl_ms",
        "p95_itl_ms",
        "p99_itl_ms",
        "max_itl_ms",
        "mean_e2e_latency_ms",
        "median_e2e_latency_ms",
        "p90_e2e_latency_ms",
        "p99_e2e_latency_ms",
        "tpm_per_gpu",
    ]

    with open(args.csv_out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields)
        writer.writeheader()

    for pct in pcts:
        prefix_len = args.total_tokens * pct // 100
        suffix_len = args.total_tokens - prefix_len

        print(f"\n{'=' * 70}")
        print(
            f"shared_prefix={pct}%  prefix_len={prefix_len}  "
            f"suffix_len={suffix_len}  total={prefix_len + suffix_len}"
        )
        print(f"{'=' * 70}")

        print("  Flushing KV cache ...")
        flush_cache(base_url)
        time.sleep(1)

        print(f"  Building {args.num_prompts} prompts ...")
        prompts = build_prompts(
            vocab_ids,
            args.total_tokens,
            pct,
            args.num_prompts,
            rng,
        )

        if prefix_len > 0:
            print(f"  Warming cache with shared prefix ({prefix_len} tokens) ...")
            warmup_prompt = [{
                "input_ids": prompts[0]["input_ids"][:prefix_len],
                "prompt_len": prefix_len,
            }]
            warmup_results = await run_batch(base_url, warmup_prompt, 1, 1)
            if not warmup_results[0].success:
                raise RuntimeError(f"Warmup failed: {warmup_results[0].error}")

        print(f"  Sending requests (max_concurrency={args.max_concurrency}) ...")
        t0 = time.perf_counter()
        results = await run_batch(
            base_url,
            prompts,
            args.output_len,
            args.max_concurrency,
        )
        elapsed = time.perf_counter() - t0

        ok = [r for r in results if r.success]
        failed = len(results) - len(ok)

        if failed:
            print(f"  WARNING: {failed}/{len(results)} requests failed")
            for r in results:
                if not r.success:
                    print(f"    {r.error[:160]}")

        if not ok:
            print("  ERROR: all requests failed, skipping")
            continue

        metrics = calculate_metrics(results, elapsed)
        tpm_per_gpu = (
            metrics["total_token_throughput_tok_s"] * 60.0 / args.tp_size
            if args.tp_size > 0 else 0.0
        )

        row = {
            "shared_prefix_pct": pct,
            "prefix_len": prefix_len,
            "suffix_len": suffix_len,
            "num_prompts": len(results),
            "succeeded": metrics["completed"],
            "elapsed_s": f"{metrics['elapsed_s']:.2f}",
            "request_throughput_req_s": f"{metrics['request_throughput_req_s']:.2f}",
            "total_input_tokens": metrics["total_input_tokens"],
            "total_output_tokens": metrics["total_output_tokens"],
            "total_token_throughput_tok_s": f"{metrics['total_token_throughput_tok_s']:.2f}",
            "input_throughput_tok_s": f"{metrics['input_throughput_tok_s']:.2f}",
            "output_throughput_tok_s": f"{metrics['output_throughput_tok_s']:.2f}",
            "concurrency": f"{metrics['concurrency']:.2f}",
            "mean_ttft_ms": f"{metrics['mean_ttft_ms']:.2f}",
            "median_ttft_ms": f"{metrics['median_ttft_ms']:.2f}",
            "p99_ttft_ms": f"{metrics['p99_ttft_ms']:.2f}",
            "mean_tpot_ms": f"{metrics['mean_tpot_ms']:.2f}",
            "median_tpot_ms": f"{metrics['median_tpot_ms']:.2f}",
            "p99_tpot_ms": f"{metrics['p99_tpot_ms']:.2f}",
            "mean_itl_ms": f"{metrics['mean_itl_ms']:.2f}",
            "median_itl_ms": f"{metrics['median_itl_ms']:.2f}",
            "p95_itl_ms": f"{metrics['p95_itl_ms']:.2f}",
            "p99_itl_ms": f"{metrics['p99_itl_ms']:.2f}",
            "max_itl_ms": f"{metrics['max_itl_ms']:.2f}",
            "mean_e2e_latency_ms": f"{metrics['mean_e2e_latency_ms']:.2f}",
            "median_e2e_latency_ms": f"{metrics['median_e2e_latency_ms']:.2f}",
            "p90_e2e_latency_ms": f"{metrics['p90_e2e_latency_ms']:.2f}",
            "p99_e2e_latency_ms": f"{metrics['p99_e2e_latency_ms']:.2f}",
            "tpm_per_gpu": f"{tpm_per_gpu:.2f}",
        }

        with open(args.csv_out, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=csv_fields)
            writer.writerow(row)

        print_benchmark_result(metrics, args.tp_size)

    print(f"\nResults saved to {args.csv_out}")


if __name__ == "__main__":
    asyncio.run(main())
