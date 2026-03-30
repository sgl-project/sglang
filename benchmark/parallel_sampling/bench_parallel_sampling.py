"""
Benchmark: SGLang Parallel Sampling (n > 1) Performance Analysis
================================================================

SGLang does NOT support beam search. Instead, it provides "parallel sampling"
via the `n` parameter, which generates multiple independent completions.

This benchmark measures:
1. Latency & throughput impact of increasing `n` (1, 2, 4, 8, 16)
2. Comparison: single request with n=K vs K independent requests
3. Prefix cache hit rate impact
4. Memory pressure under different n values
5. Streaming vs non-streaming with parallel sampling

Usage:
    # Start SGLang server first (on GPU node):
    python -m sglang.launch_server --model-path meta-llama/Llama-3.1-8B-Instruct --port 30000

    # Run benchmark:
    python bench_parallel_sampling.py --host http://localhost --port 30000
    python bench_parallel_sampling.py --host http://localhost --port 30000 --test all
    python bench_parallel_sampling.py --host http://localhost --port 30000 --test latency
    python bench_parallel_sampling.py --host http://localhost --port 30000 --test comparison
    python bench_parallel_sampling.py --host http://localhost --port 30000 --test streaming
"""

import argparse
import asyncio
import json
import os
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import aiohttp
import numpy as np

# ============================================================================
# Configuration
# ============================================================================

PROMPTS_SHORT = [
    "Explain quantum computing in simple terms.",
    "Write a Python function to sort a list using merge sort.",
    "What are the main differences between TCP and UDP?",
    "Describe the process of photosynthesis step by step.",
]

PROMPTS_MEDIUM = [
    "Write a detailed technical blog post about the advantages and disadvantages of microservices architecture compared to monolithic applications. Include real-world examples and best practices.",
    "Explain the mathematical foundations of transformer neural networks, including the self-attention mechanism, positional encoding, and multi-head attention. Provide the key equations.",
    "Design a complete REST API for a task management application. Include endpoint definitions, request/response schemas, authentication strategy, and error handling approach.",
    "Compare and contrast the programming languages Rust, Go, and C++ for systems programming. Discuss memory safety, concurrency models, ecosystem, and performance characteristics.",
]

PROMPTS_LONG = [
    (
        "You are a world-class software architect. Design a complete distributed system for a real-time "
        "collaborative document editor (like Google Docs). Your design should cover:\n"
        "1. System architecture with all major components\n"
        "2. Data model and conflict resolution strategy (CRDT vs OT)\n"
        "3. Real-time synchronization protocol\n"
        "4. Scaling strategy for millions of concurrent users\n"
        "5. Persistence and backup strategy\n"
        "6. Security considerations\n"
        "7. Performance optimization techniques\n"
        "Please provide detailed technical explanations for each point."
    ),
]


@dataclass
class BenchmarkResult:
    """Result of a single benchmark run."""
    test_name: str
    n_value: int
    num_requests: int
    prompt_length: str  # "short", "medium", "long"
    max_new_tokens: int
    temperature: float

    # Timing
    total_time_s: float = 0.0
    avg_latency_s: float = 0.0
    p50_latency_s: float = 0.0
    p90_latency_s: float = 0.0
    p99_latency_s: float = 0.0
    min_latency_s: float = 0.0
    max_latency_s: float = 0.0

    # Throughput
    total_output_tokens: int = 0
    output_tokens_per_sec: float = 0.0
    requests_per_sec: float = 0.0
    completions_per_sec: float = 0.0  # n * requests_per_sec

    # Per-completion stats
    avg_output_tokens_per_completion: float = 0.0

    # Errors
    error_count: int = 0
    errors: List[str] = field(default_factory=list)

    # Streaming specific
    avg_ttft_s: float = 0.0  # time to first token
    p50_ttft_s: float = 0.0
    p90_ttft_s: float = 0.0

    def to_dict(self) -> dict:
        d = {k: v for k, v in self.__dict__.items() if k != "errors"}
        if self.errors:
            d["sample_errors"] = self.errors[:3]
        return d


# ============================================================================
# HTTP Client
# ============================================================================

class SGLangClient:
    """Async client for SGLang server."""

    def __init__(self, base_url: str, timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.timeout = aiohttp.ClientTimeout(total=timeout)

    async def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        n: int = 1,
        stream: bool = False,
        top_p: float = 0.95,
    ) -> dict:
        """Send a generate request to SGLang's native /generate endpoint."""
        payload = {
            "text": prompt,
            "sampling_params": {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "n": n,
            },
            "stream": stream,
        }

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            start_time = time.perf_counter()

            if stream:
                return await self._generate_stream(session, payload, start_time)

            async with session.post(
                f"{self.base_url}/generate", json=payload
            ) as resp:
                latency = time.perf_counter() - start_time
                if resp.status != 200:
                    text = await resp.text()
                    return {"error": f"HTTP {resp.status}: {text}", "latency": latency}
                result = await resp.json()
                result["latency"] = latency
                return result

    async def _generate_stream(
        self, session: aiohttp.ClientSession, payload: dict, start_time: float
    ) -> dict:
        """Handle streaming response."""
        ttft = None
        chunks = []
        async with session.post(
            f"{self.base_url}/generate", json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                latency = time.perf_counter() - start_time
                return {"error": f"HTTP {resp.status}: {text}", "latency": latency}

            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    if ttft is None:
                        ttft = time.perf_counter() - start_time
                    chunks.append(chunk)
                except json.JSONDecodeError:
                    pass

            latency = time.perf_counter() - start_time
            return {
                "latency": latency,
                "ttft": ttft or latency,
                "num_chunks": len(chunks),
                "stream": True,
            }

    async def completions_v1(
        self,
        prompt: str,
        max_tokens: int = 128,
        temperature: float = 0.8,
        n: int = 1,
        stream: bool = False,
        top_p: float = 0.95,
        model: str = "default",
    ) -> dict:
        """Send request via OpenAI-compatible /v1/completions endpoint."""
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "n": n,
            "stream": stream,
        }

        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            start_time = time.perf_counter()

            if stream:
                return await self._completions_stream(session, payload, start_time)

            async with session.post(
                f"{self.base_url}/v1/completions", json=payload
            ) as resp:
                latency = time.perf_counter() - start_time
                if resp.status != 200:
                    text = await resp.text()
                    return {"error": f"HTTP {resp.status}: {text}", "latency": latency}
                result = await resp.json()
                result["latency"] = latency
                return result

    async def _completions_stream(
        self, session: aiohttp.ClientSession, payload: dict, start_time: float
    ) -> dict:
        """Handle OpenAI streaming response."""
        ttft = None
        total_tokens = 0
        n_choices = payload.get("n", 1)

        async with session.post(
            f"{self.base_url}/v1/completions", json=payload
        ) as resp:
            if resp.status != 200:
                text = await resp.text()
                latency = time.perf_counter() - start_time
                return {"error": f"HTTP {resp.status}: {text}", "latency": latency}

            async for line in resp.content:
                line = line.decode("utf-8").strip()
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[len("data:"):].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    if ttft is None:
                        ttft = time.perf_counter() - start_time
                    for choice in chunk.get("choices", []):
                        text = choice.get("text", "")
                        if text:
                            total_tokens += 1  # approximate
                except json.JSONDecodeError:
                    pass

            latency = time.perf_counter() - start_time
            return {
                "latency": latency,
                "ttft": ttft or latency,
                "total_tokens_approx": total_tokens,
                "n_choices": n_choices,
                "stream": True,
            }

    async def get_model_info(self) -> dict:
        """Get model info from the server."""
        async with aiohttp.ClientSession(timeout=self.timeout) as session:
            try:
                async with session.get(f"{self.base_url}/get_model_info") as resp:
                    if resp.status == 200:
                        return await resp.json()
            except Exception:
                pass
        return {}

    async def health_check(self) -> bool:
        """Check if the server is healthy."""
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=10)) as session:
            try:
                async with session.get(f"{self.base_url}/health") as resp:
                    return resp.status == 200
            except Exception:
                return False


# ============================================================================
# Benchmark Functions
# ============================================================================

async def warmup(client: SGLangClient, num_requests: int = 3):
    """Warm up the server with a few requests."""
    print("  Warming up server...")
    tasks = []
    for i in range(num_requests):
        tasks.append(
            client.generate("Hello, how are you?", max_new_tokens=16, n=1)
        )
    await asyncio.gather(*tasks)
    print("  Warmup complete.")


async def bench_latency_vs_n(
    client: SGLangClient,
    n_values: List[int],
    prompts: List[str],
    prompt_label: str,
    max_new_tokens: int,
    temperature: float,
    num_repeats: int,
) -> List[BenchmarkResult]:
    """
    Test 1: Measure latency and throughput as n increases.
    Sends the same prompt with varying n values.
    """
    results = []

    for n in n_values:
        print(f"\n  Testing n={n} with {prompt_label} prompts, max_new_tokens={max_new_tokens}...")
        latencies = []
        total_output_tokens = 0
        errors = []

        for repeat in range(num_repeats):
            for prompt in prompts:
                try:
                    resp = await client.completions_v1(
                        prompt=prompt,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        n=n,
                    )
                    if "error" in resp:
                        errors.append(resp["error"])
                        continue

                    latencies.append(resp["latency"])
                    # Count output tokens from usage
                    usage = resp.get("usage", {})
                    total_output_tokens += usage.get("completion_tokens", 0)

                except Exception as e:
                    errors.append(str(e))

        if not latencies:
            print(f"    All requests failed! Errors: {errors[:3]}")
            results.append(BenchmarkResult(
                test_name="latency_vs_n",
                n_value=n,
                num_requests=num_repeats * len(prompts),
                prompt_length=prompt_label,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                error_count=len(errors),
                errors=errors[:3],
            ))
            continue

        arr = np.array(latencies)
        total_time = sum(latencies)
        num_requests = len(latencies)
        total_completions = num_requests * n

        result = BenchmarkResult(
            test_name="latency_vs_n",
            n_value=n,
            num_requests=num_requests,
            prompt_length=prompt_label,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            total_time_s=total_time,
            avg_latency_s=float(np.mean(arr)),
            p50_latency_s=float(np.percentile(arr, 50)),
            p90_latency_s=float(np.percentile(arr, 90)),
            p99_latency_s=float(np.percentile(arr, 99)),
            min_latency_s=float(np.min(arr)),
            max_latency_s=float(np.max(arr)),
            total_output_tokens=total_output_tokens,
            output_tokens_per_sec=total_output_tokens / total_time if total_time > 0 else 0,
            requests_per_sec=num_requests / total_time if total_time > 0 else 0,
            completions_per_sec=total_completions / total_time if total_time > 0 else 0,
            avg_output_tokens_per_completion=(
                total_output_tokens / total_completions if total_completions > 0 else 0
            ),
            error_count=len(errors),
            errors=errors[:3],
        )
        results.append(result)

        print(f"    avg_latency={result.avg_latency_s:.3f}s  "
              f"p50={result.p50_latency_s:.3f}s  p90={result.p90_latency_s:.3f}s  "
              f"output_tok/s={result.output_tokens_per_sec:.1f}  "
              f"errors={result.error_count}")

    return results


async def bench_n_vs_independent_requests(
    client: SGLangClient,
    n_values: List[int],
    prompt: str,
    prompt_label: str,
    max_new_tokens: int,
    temperature: float,
    num_repeats: int,
) -> List[BenchmarkResult]:
    """
    Test 2: Compare n=K in one request vs K independent requests with n=1.

    This tests whether SGLang's parallel sampling is more efficient than
    just sending K separate requests (which it should be due to prefix cache).
    """
    results = []

    for n in n_values:
        if n == 1:
            continue

        # --- Method A: Single request with n=K ---
        print(f"\n  [Method A] Single request with n={n}...")
        latencies_a = []
        tokens_a = 0
        errors_a = []

        for _ in range(num_repeats):
            try:
                resp = await client.completions_v1(
                    prompt=prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    n=n,
                )
                if "error" in resp:
                    errors_a.append(resp["error"])
                    continue
                latencies_a.append(resp["latency"])
                tokens_a += resp.get("usage", {}).get("completion_tokens", 0)
            except Exception as e:
                errors_a.append(str(e))

        if latencies_a:
            arr_a = np.array(latencies_a)
            result_a = BenchmarkResult(
                test_name=f"single_request_n={n}",
                n_value=n,
                num_requests=len(latencies_a),
                prompt_length=prompt_label,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                total_time_s=sum(latencies_a),
                avg_latency_s=float(np.mean(arr_a)),
                p50_latency_s=float(np.percentile(arr_a, 50)),
                p90_latency_s=float(np.percentile(arr_a, 90)),
                p99_latency_s=float(np.percentile(arr_a, 99)),
                min_latency_s=float(np.min(arr_a)),
                max_latency_s=float(np.max(arr_a)),
                total_output_tokens=tokens_a,
                output_tokens_per_sec=tokens_a / sum(latencies_a) if latencies_a else 0,
                error_count=len(errors_a),
            )
            results.append(result_a)
            print(f"    avg_latency={result_a.avg_latency_s:.3f}s  "
                  f"output_tok/s={result_a.output_tokens_per_sec:.1f}")

        # --- Method B: K independent requests with n=1, sent concurrently ---
        print(f"  [Method B] {n} concurrent independent requests with n=1...")
        latencies_b = []
        tokens_b = 0
        errors_b = []

        for _ in range(num_repeats):
            try:
                tasks = [
                    client.completions_v1(
                        prompt=prompt,
                        max_tokens=max_new_tokens,
                        temperature=temperature,
                        n=1,
                    )
                    for _ in range(n)
                ]
                start = time.perf_counter()
                resps = await asyncio.gather(*tasks, return_exceptions=True)
                wall_time = time.perf_counter() - start

                for resp in resps:
                    if isinstance(resp, Exception):
                        errors_b.append(str(resp))
                    elif "error" in resp:
                        errors_b.append(resp["error"])
                    else:
                        tokens_b += resp.get("usage", {}).get("completion_tokens", 0)

                latencies_b.append(wall_time)
            except Exception as e:
                errors_b.append(str(e))

        if latencies_b:
            arr_b = np.array(latencies_b)
            result_b = BenchmarkResult(
                test_name=f"independent_requests_k={n}",
                n_value=n,
                num_requests=len(latencies_b),
                prompt_length=prompt_label,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                total_time_s=sum(latencies_b),
                avg_latency_s=float(np.mean(arr_b)),
                p50_latency_s=float(np.percentile(arr_b, 50)),
                p90_latency_s=float(np.percentile(arr_b, 90)),
                p99_latency_s=float(np.percentile(arr_b, 99)),
                min_latency_s=float(np.min(arr_b)),
                max_latency_s=float(np.max(arr_b)),
                total_output_tokens=tokens_b,
                output_tokens_per_sec=tokens_b / sum(latencies_b) if latencies_b else 0,
                error_count=len(errors_b),
            )
            results.append(result_b)
            print(f"    avg_latency={result_b.avg_latency_s:.3f}s  "
                  f"output_tok/s={result_b.output_tokens_per_sec:.1f}")

        # --- Comparison ---
        if latencies_a and latencies_b:
            speedup = np.mean(arr_b) / np.mean(arr_a)
            print(f"  => Single-request n={n} is {speedup:.2f}x vs {n} independent requests")

    return results


async def bench_streaming_parallel(
    client: SGLangClient,
    n_values: List[int],
    prompt: str,
    prompt_label: str,
    max_new_tokens: int,
    temperature: float,
    num_repeats: int,
) -> List[BenchmarkResult]:
    """
    Test 3: Streaming latency with parallel sampling.
    Measures TTFT (time to first token) and total latency.
    """
    results = []

    for n in n_values:
        print(f"\n  Testing streaming n={n}...")
        latencies = []
        ttfts = []
        errors = []

        for _ in range(num_repeats):
            try:
                resp = await client.completions_v1(
                    prompt=prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    n=n,
                    stream=True,
                )
                if "error" in resp:
                    errors.append(resp["error"])
                    continue
                latencies.append(resp["latency"])
                if "ttft" in resp:
                    ttfts.append(resp["ttft"])
            except Exception as e:
                errors.append(str(e))

        if not latencies:
            results.append(BenchmarkResult(
                test_name="streaming_parallel",
                n_value=n,
                num_requests=num_repeats,
                prompt_length=prompt_label,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                error_count=len(errors),
                errors=errors[:3],
            ))
            continue

        arr = np.array(latencies)
        result = BenchmarkResult(
            test_name="streaming_parallel",
            n_value=n,
            num_requests=len(latencies),
            prompt_length=prompt_label,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            total_time_s=sum(latencies),
            avg_latency_s=float(np.mean(arr)),
            p50_latency_s=float(np.percentile(arr, 50)),
            p90_latency_s=float(np.percentile(arr, 90)),
            p99_latency_s=float(np.percentile(arr, 99)),
            min_latency_s=float(np.min(arr)),
            max_latency_s=float(np.max(arr)),
            error_count=len(errors),
        )

        if ttfts:
            arr_ttft = np.array(ttfts)
            result.avg_ttft_s = float(np.mean(arr_ttft))
            result.p50_ttft_s = float(np.percentile(arr_ttft, 50))
            result.p90_ttft_s = float(np.percentile(arr_ttft, 90))

        results.append(result)
        print(f"    avg_latency={result.avg_latency_s:.3f}s  "
              f"avg_ttft={result.avg_ttft_s:.3f}s  errors={result.error_count}")

    return results


async def bench_concurrent_load(
    client: SGLangClient,
    n_values: List[int],
    prompts: List[str],
    prompt_label: str,
    max_new_tokens: int,
    temperature: float,
    concurrency: int,
) -> List[BenchmarkResult]:
    """
    Test 4: Concurrent load test — multiple users sending n>1 requests simultaneously.
    Simulates realistic serving conditions.
    """
    results = []

    for n in n_values:
        print(f"\n  Concurrent load test: {concurrency} clients, n={n}...")
        all_latencies = []
        total_tokens = 0
        errors = []

        sem = asyncio.Semaphore(concurrency)

        async def send_one(prompt: str):
            async with sem:
                return await client.completions_v1(
                    prompt=prompt,
                    max_tokens=max_new_tokens,
                    temperature=temperature,
                    n=n,
                )

        # Send all prompts concurrently (repeat to fill concurrency slots)
        expanded_prompts = (prompts * ((concurrency // len(prompts)) + 1))[:concurrency]

        start = time.perf_counter()
        responses = await asyncio.gather(
            *[send_one(p) for p in expanded_prompts],
            return_exceptions=True,
        )
        wall_time = time.perf_counter() - start

        for resp in responses:
            if isinstance(resp, Exception):
                errors.append(str(resp))
            elif "error" in resp:
                errors.append(resp["error"])
            else:
                all_latencies.append(resp["latency"])
                total_tokens += resp.get("usage", {}).get("completion_tokens", 0)

        if not all_latencies:
            results.append(BenchmarkResult(
                test_name="concurrent_load",
                n_value=n,
                num_requests=concurrency,
                prompt_length=prompt_label,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                error_count=len(errors),
                errors=errors[:3],
            ))
            continue

        arr = np.array(all_latencies)
        num_ok = len(all_latencies)

        result = BenchmarkResult(
            test_name="concurrent_load",
            n_value=n,
            num_requests=num_ok,
            prompt_length=prompt_label,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            total_time_s=wall_time,
            avg_latency_s=float(np.mean(arr)),
            p50_latency_s=float(np.percentile(arr, 50)),
            p90_latency_s=float(np.percentile(arr, 90)),
            p99_latency_s=float(np.percentile(arr, 99)),
            min_latency_s=float(np.min(arr)),
            max_latency_s=float(np.max(arr)),
            total_output_tokens=total_tokens,
            output_tokens_per_sec=total_tokens / wall_time if wall_time > 0 else 0,
            requests_per_sec=num_ok / wall_time if wall_time > 0 else 0,
            completions_per_sec=(num_ok * n) / wall_time if wall_time > 0 else 0,
            error_count=len(errors),
        )
        results.append(result)
        print(f"    wall_time={wall_time:.3f}s  avg_latency={result.avg_latency_s:.3f}s  "
              f"throughput={result.output_tokens_per_sec:.1f} tok/s  errors={result.error_count}")

    return results


# ============================================================================
# Report
# ============================================================================

def print_report(all_results: Dict[str, List[BenchmarkResult]], model_info: dict):
    """Print a formatted benchmark report."""
    print("\n" + "=" * 100)
    print("  SGLang Parallel Sampling (n > 1) Benchmark Report")
    print("=" * 100)

    if model_info:
        print(f"\n  Model: {model_info.get('model_path', 'unknown')}")

    for test_name, results in all_results.items():
        print(f"\n{'─' * 100}")
        print(f"  Test: {test_name}")
        print(f"{'─' * 100}")

        if not results:
            print("  No results.")
            continue

        # Table header
        header = (
            f"  {'n':>4}  {'Avg Lat':>9}  {'P50':>9}  {'P90':>9}  {'P99':>9}  "
            f"{'Out Tok/s':>10}  {'Req/s':>7}  {'Comp/s':>7}  {'Errors':>6}  "
            f"{'TTFT':>8}"
        )
        print(header)
        print("  " + "-" * (len(header) - 2))

        baseline_latency = None
        for r in results:
            if baseline_latency is None:
                baseline_latency = r.avg_latency_s

            slowdown = (
                f" ({r.avg_latency_s / baseline_latency:.1f}x)"
                if baseline_latency and baseline_latency > 0 and r.avg_latency_s > 0
                else ""
            )
            ttft_str = f"{r.avg_ttft_s:.3f}s" if r.avg_ttft_s > 0 else "N/A"

            print(
                f"  {r.n_value:>4}  "
                f"{r.avg_latency_s:>8.3f}s  "
                f"{r.p50_latency_s:>8.3f}s  "
                f"{r.p90_latency_s:>8.3f}s  "
                f"{r.p99_latency_s:>8.3f}s  "
                f"{r.output_tokens_per_sec:>10.1f}  "
                f"{r.requests_per_sec:>7.2f}  "
                f"{r.completions_per_sec:>7.2f}  "
                f"{r.error_count:>6}  "
                f"{ttft_str:>8}"
                f"{slowdown}"
            )

    print(f"\n{'=' * 100}")
    print("  NOTE: SGLang does NOT implement beam search.")
    print("  The 'n' parameter produces independent parallel samples, not beam candidates.")
    print("  'best_of' is defined in the OpenAI protocol but is silently ignored.")
    print(f"{'=' * 100}\n")


def save_results(all_results: Dict[str, List[BenchmarkResult]], output_path: str):
    """Save results to JSON."""
    data = {}
    for test_name, results in all_results.items():
        data[test_name] = [r.to_dict() for r in results]

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Results saved to: {output_path}")


# ============================================================================
# Main
# ============================================================================

async def main(args):
    base_url = f"{args.host}:{args.port}"
    client = SGLangClient(base_url, timeout=args.timeout)

    # Health check
    print(f"\nConnecting to SGLang server at {base_url}...")
    if not await client.health_check():
        print("ERROR: Server is not responding. Please start the server first:")
        print("  python -m sglang.launch_server --model-path <model> --port 30000")
        return

    model_info = await client.get_model_info()
    print(f"  Model: {model_info.get('model_path', 'unknown')}")
    print(f"  Tests to run: {args.test}")

    n_values = [int(x) for x in args.n_values.split(",")]
    all_results: Dict[str, List[BenchmarkResult]] = {}

    # Warmup
    await warmup(client)

    tests_to_run = args.test.split(",") if args.test != "all" else [
        "latency", "comparison", "streaming", "concurrent"
    ]

    # Test 1: Latency vs n
    if "latency" in tests_to_run:
        print("\n" + "=" * 80)
        print("  Test 1: Latency & Throughput vs Parallel Sample Count (n)")
        print("=" * 80)

        for prompts, label in [
            (PROMPTS_SHORT, "short"),
            (PROMPTS_MEDIUM, "medium"),
        ]:
            results = await bench_latency_vs_n(
                client=client,
                n_values=n_values,
                prompts=prompts,
                prompt_label=label,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                num_repeats=args.num_repeats,
            )
            all_results[f"latency_vs_n_{label}"] = results

    # Test 2: n=K vs K independent requests
    if "comparison" in tests_to_run:
        print("\n" + "=" * 80)
        print("  Test 2: Single Request (n=K) vs K Independent Concurrent Requests")
        print("=" * 80)

        results = await bench_n_vs_independent_requests(
            client=client,
            n_values=n_values,
            prompt=PROMPTS_MEDIUM[0],
            prompt_label="medium",
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_repeats=args.num_repeats,
        )
        all_results["n_vs_independent"] = results

    # Test 3: Streaming with parallel sampling
    if "streaming" in tests_to_run:
        print("\n" + "=" * 80)
        print("  Test 3: Streaming Latency with Parallel Sampling")
        print("=" * 80)

        results = await bench_streaming_parallel(
            client=client,
            n_values=n_values,
            prompt=PROMPTS_SHORT[0],
            prompt_label="short",
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            num_repeats=args.num_repeats,
        )
        all_results["streaming_parallel"] = results

    # Test 4: Concurrent load
    if "concurrent" in tests_to_run:
        print("\n" + "=" * 80)
        print("  Test 4: Concurrent Load with Parallel Sampling")
        print("=" * 80)

        for concurrency in [4, 16, 32]:
            results = await bench_concurrent_load(
                client=client,
                n_values=n_values,
                prompts=PROMPTS_SHORT,
                prompt_label="short",
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                concurrency=concurrency,
            )
            all_results[f"concurrent_load_c={concurrency}"] = results

    # Print report
    print_report(all_results, model_info)

    # Save results
    if args.output:
        save_results(all_results, args.output)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark SGLang parallel sampling (n > 1) performance"
    )
    parser.add_argument("--host", type=str, default="http://localhost",
                        help="Server host URL")
    parser.add_argument("--port", type=int, default=30000,
                        help="Server port")
    parser.add_argument("--test", type=str, default="all",
                        help="Tests to run: all, latency, comparison, streaming, concurrent "
                             "(comma-separated)")
    parser.add_argument("--n-values", type=str, default="1,2,4,8,16",
                        help="Comma-separated list of n values to test")
    parser.add_argument("--max-new-tokens", type=int, default=256,
                        help="Max new tokens per completion")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature")
    parser.add_argument("--num-repeats", type=int, default=5,
                        help="Number of repeats per test case")
    parser.add_argument("--timeout", type=int, default=300,
                        help="Request timeout in seconds")
    parser.add_argument("--output", type=str, default=None,
                        help="Path to save JSON results (optional)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    asyncio.run(main(args))
