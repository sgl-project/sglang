"""
Self-contained HTTP benchmark for auto-spec (adaptive speculative decoding).

Sends concurrent requests to an SGLang server's /v1/completions endpoint,
measures throughput, latency, and TTFT. No dependency on sglang benchmark
internals -- just asyncio + aiohttp.

Usage:
    # Basic usage
    python bench_auto_spec.py --port 30000 --dataset-path mix_spec_dataset.jsonl

    # Control concurrency and prompts
    python bench_auto_spec.py --port 30000 --dataset-path mix_spec_dataset.jsonl \
        --max-concurrency 16 --num-prompts 100

    # Filter by source dataset
    python bench_auto_spec.py --port 30000 --dataset-path mix_spec_dataset.jsonl \
        --source-filter gsm8k,humaneval

    # Sweep over batch sizes (controls --max-concurrency per run)
    python bench_auto_spec.py --port 30000 --dataset-path mix_spec_dataset.jsonl \
        --batch-sizes 1,4,8,16,32 --num-prompts-each 32

    # Fixed output length
    python bench_auto_spec.py --port 30000 --dataset-path mix_spec_dataset.jsonl \
        --output-len 256
"""

import argparse
import asyncio
import json
import statistics
import sys
import time
from typing import Dict, List, Optional, Tuple


def load_dataset(
    dataset_path: str,
    num_prompts: Optional[int] = None,
    source_filter: Optional[List[str]] = None,
) -> List[Dict]:
    """Load mix-spec JSONL dataset directly.

    Each line is a JSON object with keys: prompt, expected_output_len, source.
    """
    records = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if source_filter and record.get("source") not in source_filter:
                continue
            records.append(record)

    if num_prompts is not None and num_prompts < len(records):
        records = records[:num_prompts]

    return records


async def send_request(
    session,
    url: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
) -> Dict:
    """Send a single completion request and measure timing."""
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
    }

    start_time = time.perf_counter()
    first_token_time = None
    output_tokens = 0
    full_text = ""

    try:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                return {
                    "request_id": request_id,
                    "success": False,
                    "error": f"HTTP {resp.status}: {error_text[:200]}",
                    "latency": 0,
                    "ttft": 0,
                    "output_tokens": 0,
                }

            async for line in resp.content:
                decoded = line.decode("utf-8").strip()
                if not decoded or not decoded.startswith("data:"):
                    continue
                data_str = decoded[len("data:") :].strip()
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    choices = data.get("choices", [])
                    if choices:
                        text = choices[0].get("text", "")
                        if text and first_token_time is None:
                            first_token_time = time.perf_counter()
                        full_text += text
                except json.JSONDecodeError:
                    continue

    except Exception as e:
        return {
            "request_id": request_id,
            "success": False,
            "error": str(e),
            "latency": 0,
            "ttft": 0,
            "output_tokens": 0,
        }

    end_time = time.perf_counter()
    latency = end_time - start_time
    ttft = (first_token_time - start_time) if first_token_time else latency

    # Estimate output tokens from text length (rough: 1 token ~ 4 chars)
    # This is approximate; exact count would need tokenizer
    output_tokens = max(len(full_text) // 4, 1)

    return {
        "request_id": request_id,
        "success": True,
        "latency": latency,
        "ttft": ttft,
        "output_tokens": output_tokens,
        "output_len": len(full_text),
    }


async def run_benchmark(
    host: str,
    port: int,
    records: List[Dict],
    max_concurrency: int,
    output_len: Optional[int] = None,
) -> Tuple[List[Dict], float]:
    """Run benchmark with controlled concurrency."""
    try:
        import aiohttp
    except ImportError:
        print("ERROR: aiohttp is required. Install with: pip install aiohttp")
        sys.exit(1)

    url = f"http://{host}:{port}/v1/completions"
    semaphore = asyncio.Semaphore(max_concurrency)
    results = []

    async def bounded_request(session, record, idx):
        async with semaphore:
            max_tokens = (
                output_len if output_len else record.get("expected_output_len", 256)
            )
            max_tokens = min(max_tokens, 4096)  # cap at 4096
            result = await send_request(session, url, record["prompt"], max_tokens, idx)
            result["source"] = record.get("source", "unknown")
            return result

    connector = aiohttp.TCPConnector(limit=max_concurrency + 10)
    timeout = aiohttp.ClientTimeout(total=600)

    print(f"Sending {len(records)} requests (max_concurrency={max_concurrency})...")
    bench_start = time.perf_counter()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout) as session:
        tasks = [
            bounded_request(session, record, i) for i, record in enumerate(records)
        ]
        results = await asyncio.gather(*tasks)

    bench_end = time.perf_counter()
    wall_time = bench_end - bench_start

    return list(results), wall_time


def print_summary(results: List[Dict], wall_time: float, label: str = ""):
    """Print benchmark summary statistics."""
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if not successful:
        print(f"\n{'='*60}")
        if label:
            print(f"  {label}")
        print(f"  All {len(results)} requests failed!")
        for r in failed[:5]:
            print(f"    Request {r['request_id']}: {r['error']}")
        return

    latencies = [r["latency"] for r in successful]
    ttfts = [r["ttft"] for r in successful]
    output_tokens = sum(r["output_tokens"] for r in successful)

    throughput = output_tokens / wall_time if wall_time > 0 else 0
    req_throughput = len(successful) / wall_time if wall_time > 0 else 0

    print(f"\n{'='*60}")
    if label:
        print(f"  {label}")
    print(f"{'='*60}")
    print(f"  Requests:        {len(successful)} succeeded, {len(failed)} failed")
    print(f"  Wall time:       {wall_time:.2f}s")
    print(f"  Throughput:      {throughput:.1f} tok/s  ({req_throughput:.1f} req/s)")
    print(f"  Output tokens:   {output_tokens}")
    print(
        f"  Latency (s):     avg={statistics.mean(latencies):.3f}  "
        f"p50={statistics.median(latencies):.3f}  "
        f"p99={sorted(latencies)[int(len(latencies)*0.99)]:.3f}  "
        f"max={max(latencies):.3f}"
    )
    print(
        f"  TTFT (s):        avg={statistics.mean(ttfts):.3f}  "
        f"p50={statistics.median(ttfts):.3f}  "
        f"p99={sorted(ttfts)[int(len(ttfts)*0.99)]:.3f}  "
        f"max={max(ttfts):.3f}"
    )

    # Per-source breakdown
    sources = set(r["source"] for r in successful)
    if len(sources) > 1:
        print(f"\n  Per-source breakdown:")
        for source in sorted(sources):
            source_results = [r for r in successful if r["source"] == source]
            source_tokens = sum(r["output_tokens"] for r in source_results)
            source_lat = [r["latency"] for r in source_results]
            print(
                f"    {source:12s}: {len(source_results):4d} reqs, "
                f"{source_tokens:6d} tokens, "
                f"avg_lat={statistics.mean(source_lat):.3f}s"
            )

    if failed:
        print(f"\n  Failed requests ({len(failed)}):")
        for r in failed[:3]:
            print(f"    Request {r['request_id']}: {r['error']}")
        if len(failed) > 3:
            print(f"    ... and {len(failed) - 3} more")

    print(f"{'='*60}")


async def check_server(host: str, port: int) -> bool:
    """Check if the server is ready."""
    try:
        import aiohttp
    except ImportError:
        print("ERROR: aiohttp is required. Install with: pip install aiohttp")
        sys.exit(1)

    url = f"http://{host}:{port}/health"
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as resp:
                return resp.status == 200
    except Exception:
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Self-contained HTTP benchmark for auto-spec speculative decoding"
    )
    parser.add_argument(
        "--host", type=str, default="127.0.0.1", help="Server host. Default: 127.0.0.1"
    )
    parser.add_argument(
        "--port", type=int, default=30000, help="Server port. Default: 30000"
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        required=True,
        help="Path to mix-spec JSONL dataset file.",
    )
    parser.add_argument(
        "--num-prompts",
        type=int,
        default=None,
        help="Total number of prompts to use. Default: all in dataset.",
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=16,
        help="Maximum number of concurrent requests. Default: 16.",
    )
    parser.add_argument(
        "--output-len",
        type=int,
        default=None,
        help="Fixed output length for all requests. Default: use dataset's expected_output_len.",
    )
    parser.add_argument(
        "--source-filter",
        type=str,
        default=None,
        help="Comma-separated list of source datasets to include (e.g., 'gsm8k,humaneval').",
    )
    parser.add_argument(
        "--num-prompts-each",
        type=int,
        default=None,
        help="Number of prompts from each source dataset. Overrides --num-prompts.",
    )
    parser.add_argument(
        "--batch-sizes",
        type=str,
        default=None,
        help="Comma-separated list of batch sizes (concurrency levels) to sweep. "
        "Runs benchmark once per batch size. E.g., '1,4,8,16,32'.",
    )

    args = parser.parse_args()

    # Parse source filter
    source_filter = None
    if args.source_filter:
        source_filter = [s.strip() for s in args.source_filter.split(",")]

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    all_records = load_dataset(args.dataset_path, source_filter=source_filter)
    print(f"  Total records after filtering: {len(all_records)}")

    if not all_records:
        print("ERROR: No records loaded. Check dataset path and source filter.")
        sys.exit(1)

    # If --num-prompts-each is set, sample from each source
    if args.num_prompts_each:
        sources = {}
        for r in all_records:
            src = r.get("source", "unknown")
            if src not in sources:
                sources[src] = []
            sources[src].append(r)

        records = []
        for src in sorted(sources.keys()):
            src_records = sources[src][: args.num_prompts_each]
            records.extend(src_records)
            print(f"  {src}: {len(src_records)} prompts")
        all_records = records
    elif args.num_prompts:
        all_records = all_records[: args.num_prompts]

    # Source distribution
    source_counts = {}
    for r in all_records:
        src = r.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    print(f"  Source distribution: {source_counts}")
    print(f"  Total prompts: {len(all_records)}")

    # Check server
    if not asyncio.run(check_server(args.host, args.port)):
        print(f"\nWARNING: Server at {args.host}:{args.port} is not responding.")
        print("Make sure the server is running. Proceeding anyway...\n")

    # Run benchmark
    if args.batch_sizes:
        # Sweep over batch sizes
        batch_sizes = [int(b.strip()) for b in args.batch_sizes.split(",")]
        all_summaries = []

        for bs in batch_sizes:
            print(f"\n{'#'*60}")
            print(f"  Batch size (concurrency): {bs}")
            print(f"{'#'*60}")

            results, wall_time = asyncio.run(
                run_benchmark(args.host, args.port, all_records, bs, args.output_len)
            )
            print_summary(results, wall_time, label=f"Batch Size = {bs}")

            successful = [r for r in results if r["success"]]
            if successful:
                output_tokens = sum(r["output_tokens"] for r in successful)
                throughput = output_tokens / wall_time if wall_time > 0 else 0
                avg_lat = statistics.mean([r["latency"] for r in successful])
                avg_ttft = statistics.mean([r["ttft"] for r in successful])
                all_summaries.append(
                    {
                        "batch_size": bs,
                        "throughput": throughput,
                        "avg_latency": avg_lat,
                        "avg_ttft": avg_ttft,
                        "succeeded": len(successful),
                        "failed": len(results) - len(successful),
                    }
                )

        # Print comparison table
        if all_summaries:
            print(f"\n{'='*80}")
            print(f"  BATCH SIZE COMPARISON")
            print(f"{'='*80}")
            print(
                f"  {'BS':>4s}  {'Throughput':>12s}  {'Avg Latency':>12s}  "
                f"{'Avg TTFT':>10s}  {'OK/Fail':>8s}"
            )
            print(f"  {'-'*4}  {'-'*12}  {'-'*12}  {'-'*10}  {'-'*8}")
            for s in all_summaries:
                print(
                    f"  {s['batch_size']:>4d}  "
                    f"{s['throughput']:>9.1f} t/s  "
                    f"{s['avg_latency']:>10.3f}s  "
                    f"{s['avg_ttft']:>8.3f}s  "
                    f"{s['succeeded']:>3d}/{s['failed']:<3d}"
                )
            print(f"{'='*80}")

    else:
        # Single run
        results, wall_time = asyncio.run(
            run_benchmark(
                args.host, args.port, all_records, args.max_concurrency, args.output_len
            )
        )
        print_summary(results, wall_time)


if __name__ == "__main__":
    main()
