"""Measure external KV reuse on an isolated, already running SGLang server.

The benchmark flushes the engine's GPU cache between request waves. For PD,
send generation through the router and set --prefill-url to the prefill server.
Enable deterministic inference when checking cold/warm output equality.
"""

import argparse
import json
import math
import statistics
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
from transformers import AutoTokenizer


def generate(base_url, tokens, salt, output_len):
    start = time.perf_counter()
    first_token = None
    result = None
    with requests.post(
        base_url + "/generate",
        json={
            "input_ids": tokens,
            "cache_salt": salt,
            "stream": True,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": output_len,
                "ignore_eos": True,
            },
        },
        stream=True,
        timeout=180,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1, decode_unicode=True):
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            result = json.loads(line[6:])
            if first_token is None and result.get("output_ids"):
                first_token = time.perf_counter()
    if first_token is None or result is None:
        raise RuntimeError("Server returned no streamed tokens")
    if len(result["output_ids"]) != output_len:
        raise RuntimeError(f"Incomplete generation: {result}")
    details = result["meta_info"].get("cached_tokens_details") or {}
    return {
        "ttft_ms": (first_token - start) * 1000,
        "elapsed_ms": (time.perf_counter() - start) * 1000,
        "storage_hits": details.get("storage", 0),
        "device_hits": details.get("device", 0),
        "output_ids": result["output_ids"],
    }


def benchmark(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    filler = tokenizer.encode("Berlin is the capital of Germany. " * args.input_len)
    question = tokenizer.encode(
        "\nWhat is the capital of Germany?\nAnswer:", add_special_tokens=False
    )
    tokens = filler[: args.input_len - len(question)] + question
    if len(tokens) != args.input_len:
        raise ValueError("Input length is shorter than the benchmark question")
    expected_hit = (len(tokens) - 1) // args.chunk_size * args.chunk_size
    if args.expected_storage_hit is not None:
        expected_hit = args.expected_storage_hit
    elif expected_hit == 0:
        raise ValueError("Input length must extend beyond one cache chunk")
    base_url = args.base_url.rstrip("/")
    flush_url = (args.prefill_url or args.base_url).rstrip("/") + "/flush_cache"
    prefix = f"reuse-{uuid.uuid4().hex}"
    salts = [f"{prefix}-{index}" for index in range(args.concurrency)]
    records = []
    wave_seconds = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:

        def wave():
            return list(
                pool.map(
                    lambda salt: generate(base_url, tokens, salt, args.output_len),
                    salts,
                )
            )

        # Salt-separated requests cannot satisfy one another from GPU radix KV.
        references = [row["output_ids"] for row in wave()]
        for index in range(args.warmup + args.waves):
            requests.post(
                flush_url, params={"timeout": 60}, timeout=90
            ).raise_for_status()
            start = time.perf_counter()
            rows = wave()
            elapsed = time.perf_counter() - start
            for slot, (row, reference) in enumerate(zip(rows, references, strict=True)):
                if row["output_ids"] != reference:
                    raise RuntimeError(f"Output mismatch in wave {index}, slot {slot}")
                if row["device_hits"] != 0 or row["storage_hits"] != expected_hit:
                    raise RuntimeError(
                        f"Expected {expected_hit} storage hits and zero device hits: {row}"
                    )
                if index >= args.warmup:
                    records.append(dict(row, wave=index - args.warmup, slot=slot))
            if index >= args.warmup:
                wave_seconds.append(elapsed)
    ttft = sorted(row["ttft_ms"] for row in records)
    summary = {
        "requests": len(records),
        "ttft_median_ms": statistics.median(ttft),
        "ttft_p95_ms": ttft[math.ceil(0.95 * len(ttft)) - 1],
        "elapsed_mean_ms": statistics.mean(row["elapsed_ms"] for row in records),
        "wave_requests_per_second": len(records) / sum(wave_seconds),
        "wave_output_tokens_per_second": len(records)
        * args.output_len
        / sum(wave_seconds),
    }
    report = {
        "model": args.model,
        "input_len": args.input_len,
        "output_len": args.output_len,
        "chunk_size": args.chunk_size,
        "concurrency": args.concurrency,
        "warmup_waves": args.warmup,
        "measured_waves": args.waves,
        "expected_storage_hit": expected_hit,
        "summary": summary,
        "wave_seconds": wave_seconds,
        "records": records,
    }
    if args.output:
        Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:30000")
    parser.add_argument("--prefill-url", help="PD prefill endpoint for GPU-cache flush")
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument("--input-len", type=int, default=3073)
    parser.add_argument("--output-len", type=int, default=64)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--expected-storage-hit",
        type=int,
        help="Expected external hit tokens; default full prefix. Use 0 for recompute comparisons.",
    )
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--warmup", type=int, default=4)
    parser.add_argument("--waves", type=int, default=16)
    parser.add_argument("--output", help="Save raw measurements and summary as JSON")
    args = parser.parse_args()
    if (
        min(
            args.input_len,
            args.output_len,
            args.chunk_size,
            args.concurrency,
            args.waves,
        )
        <= 0
    ):
        parser.error("Lengths, chunk size, concurrency and waves must be positive")
    if args.warmup < 0:
        parser.error("Warmup waves must be non-negative")
    if args.expected_storage_hit is not None and args.expected_storage_hit < 0:
        parser.error("Expected storage hit must be non-negative")
    benchmark(args)


if __name__ == "__main__":
    main()
