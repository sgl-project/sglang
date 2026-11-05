"""SGLang Rerank Benchmark Script

This script benchmarks the cross-encoder /v1/rerank API with synthetic input.

Features:
- HTTP-only implementation
- Configurable input and query lengths, documents, and concurrency
- Validates rerank results and server-side prompt token counts
- Reports throughput and request latency metrics
- Excludes input generation and warmup from measured timing

Usage:
- Ensure the SGLang server is running
- Disable server prefix caching for this repeated-input workload
- Run: python benchmark/prefill_only/bench_rerank.py \\
    --model /localdisk2/huggingface/bge-reranker-base \\
    --input-len 256 --query-len 32 --documents 1 \\
    --concurrency 24 --num-requests 1000 --warmup-requests 48

Input length includes the query, document, and pair special tokens. This
benchmark measures performance, not ranking accuracy.
"""

import argparse
import asyncio
import json
import math
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
from sglang.srt.utils.hf_transformers import get_tokenizer


def build_payload(args: argparse.Namespace) -> dict:
    tokenizer = get_tokenizer(args.model)
    document_len = (
        args.input_len - args.query_len - tokenizer.num_special_tokens_to_add(pair=True)
    )
    if document_len < 1:
        raise ValueError("--input-len must leave space for document and special tokens")
    if args.input_len > tokenizer.model_max_length:
        raise ValueError(
            f"--input-len exceeds tokenizer limit {tokenizer.model_max_length}"
        )
    query = " ".join(["test"] * args.query_len)
    document = " ".join(["the"] * document_len)
    query_len = len(tokenizer.encode(query, add_special_tokens=False))
    pair_len = len(
        tokenizer([[query, document]], return_token_type_ids=True)["input_ids"][0]
    )
    if query_len != args.query_len or pair_len != args.input_len:
        raise ValueError(
            "Synthetic text does not produce the requested lengths with this "
            f"tokenizer: query={query_len}, pair={pair_len}"
        )
    return {
        "model": args.model,
        "query": query,
        "documents": [document] * args.documents,
        "return_documents": False,
    }


def validate_response(data: object, documents: int, input_len: int) -> None:
    if not isinstance(data, list) or len(data) != documents:
        raise ValueError(f"Expected {documents} rerank results, got {str(data)[:500]}")
    indices = set()
    for item in data:
        if not isinstance(item, dict):
            raise TypeError(f"Invalid rerank result: {item}")
        index = item.get("index")
        score = item.get("score")
        if (
            type(index) is not int
            or not 0 <= index < documents
            or index in indices
            or type(score) not in (int, float)
            or not math.isfinite(score)
        ):
            raise ValueError(f"Invalid rerank result: {item}")
        meta_info = item.get("meta_info")
        if (
            not isinstance(meta_info, dict)
            or meta_info.get("prompt_tokens") != input_len
        ):
            raise ValueError(
                f"Expected {input_len} server-side prompt tokens per pair, "
                f"got meta_info={meta_info}"
            )
        indices.add(index)


async def run_requests(
    *,
    session: aiohttp.ClientSession,
    url: str,
    payload: dict,
    count: int,
    concurrency: int,
    input_len: int,
) -> tuple[list[float], list[str], float]:
    requests = iter(range(count))
    latencies: list[float] = []
    errors: list[str] = []

    async def worker() -> None:
        for index in requests:
            start = time.perf_counter()
            try:
                async with session.post(url, json=payload) as response:
                    body = await response.text()
                    if response.status != 200:
                        raise ValueError(f"HTTP {response.status}: {body[:500]}")
                    validate_response(
                        json.loads(body), len(payload["documents"]), input_len
                    )
                latencies.append(time.perf_counter() - start)
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                TypeError,
                ValueError,
            ) as exc:
                error = f"Request {index}: {type(exc).__name__}: {exc}"
                print(error, file=sys.stderr, flush=True)
                errors.append(error)

    start = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(min(count, concurrency))))
    return latencies, errors, time.perf_counter() - start


async def benchmark(args: argparse.Namespace) -> bool:
    payload = build_payload(args)
    url = args.base_url.rstrip("/") + "/v1/rerank"
    print(
        f"URL: {url}\n"
        f"Pair tokens: {args.input_len}, documents/request: {args.documents}, "
        f"concurrency: {args.concurrency}",
        flush=True,
    )
    timeout = aiohttp.ClientTimeout(total=args.timeout)
    connector = aiohttp.TCPConnector(limit=args.concurrency)
    async with aiohttp.ClientSession(
        timeout=timeout, connector=connector, trust_env=False
    ) as session:
        _, warmup_errors, _ = await run_requests(
            session=session,
            url=url,
            payload=payload,
            count=args.warmup_requests,
            concurrency=args.concurrency,
            input_len=args.input_len,
        )
        if warmup_errors:
            raise RuntimeError("Warmup failed; benchmark was not run")
        print("Warmup complete; starting measured requests.", flush=True)
        latencies, errors, duration = await run_requests(
            session=session,
            url=url,
            payload=payload,
            count=args.num_requests,
            concurrency=args.concurrency,
            input_len=args.input_len,
        )

    completed = len(latencies)
    result = {
        "model": args.model,
        "base_url": args.base_url,
        "input_tokens_per_pair": args.input_len,
        "query_tokens": args.query_len,
        "documents_per_request": args.documents,
        "concurrency": args.concurrency,
        "warmup_requests": args.warmup_requests,
        "num_requests": args.num_requests,
        "completed": completed,
        "failed": len(errors),
        "duration_s": duration,
        "requests_per_s": completed / duration,
        "pairs_per_s": completed * args.documents / duration,
        "input_tokens_per_s": completed * args.documents * args.input_len / duration,
        "latency_ms": (
            {
                "mean": float(np.mean(latencies) * 1000),
                **{
                    f"p{p}": float(np.percentile(latencies, p) * 1000)
                    for p in (50, 90, 95, 99)
                },
            }
            if latencies
            else None
        ),
        "errors": errors,
    }
    output = json.dumps(result, indent=2)
    print(output)
    if args.output_file:
        args.output_file.write_text(output + "\n")
    return not errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Local tokenizer/model path")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--input-len", type=int, default=256)
    parser.add_argument("--query-len", type=int, default=32)
    parser.add_argument("--documents", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=24)
    parser.add_argument("--num-requests", type=int, default=1000)
    parser.add_argument("--warmup-requests", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args()
    for name in (
        "input_len",
        "query_len",
        "documents",
        "concurrency",
        "num_requests",
        "timeout",
    ):
        value = vars(args)[name]
        if not math.isfinite(value) or value <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive and finite")
    if args.warmup_requests < 0:
        parser.error("--warmup-requests must be non-negative")
    return args


if __name__ == "__main__":
    sys.exit(0 if asyncio.run(benchmark(parse_args())) else 1)
