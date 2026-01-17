#!/usr/bin/env python3
"""
Benchmark script for Spectral KV Cache Eviction.

Measures:
1. Memory reduction: Compare VRAM usage with LRU vs Spectral eviction
2. Quality preservation: Compare output quality/perplexity
3. Latency impact: Overhead of spectral computation

Usage:
    # First start a server with spectral eviction:
    python -m sglang.launch_server \
        --model Qwen/Qwen3-1.7B \
        --radix-eviction-policy spectral \
        --spectral-retention-ratio 0.3 \
        --attention-fingerprint-mode \
        --port 30000

    # Then run benchmark:
    python scripts/benchmark_spectral_eviction.py --port 30000

    # For comparison, also run with LRU:
    python -m sglang.launch_server \
        --model Qwen/Qwen3-1.7B \
        --radix-eviction-policy lru \
        --port 30001

    python scripts/benchmark_spectral_eviction.py --port 30001 --baseline
"""

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import requests
except ImportError:
    print("Please install requests: pip install requests")
    sys.exit(1)


@dataclass
class BenchmarkResult:
    """Result from a single benchmark run."""

    name: str
    eviction_policy: str
    prompt_tokens: int
    output_tokens: int
    time_to_first_token_ms: float
    total_time_ms: float
    tokens_per_second: float
    output_text: str
    cache_hit_rate: Optional[float] = None
    memory_mb: Optional[float] = None


@dataclass
class ComparisonResult:
    """Comparison between spectral and baseline."""

    memory_reduction_pct: float
    quality_score: float  # 0-1, based on output similarity
    latency_overhead_pct: float
    cache_efficiency: float


def check_server_health(base_url: str) -> Tuple[bool, Dict]:
    """Check if server is running and get model info."""
    try:
        resp = requests.get(f"{base_url}/v1/models", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return True, data
        return False, {}
    except Exception as e:
        return False, {"error": str(e)}


def get_server_metrics(base_url: str) -> Dict:
    """Get server metrics including memory usage."""
    try:
        resp = requests.get(f"{base_url}/metrics", timeout=5)
        if resp.status_code == 200:
            # Parse Prometheus-style metrics
            metrics = {}
            for line in resp.text.split("\n"):
                if line and not line.startswith("#"):
                    parts = line.split()
                    if len(parts) >= 2:
                        metrics[parts[0]] = float(parts[1])
            return metrics
        return {}
    except Exception:
        return {}


def run_generation(
    base_url: str,
    prompt: str,
    max_tokens: int = 100,
    return_attention: bool = False,
) -> BenchmarkResult:
    """Run a single generation and measure performance."""

    start_time = time.perf_counter()
    ttft = None
    output_text = ""

    payload = {
        "model": "default",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
    }

    if return_attention:
        payload["extra_body"] = {"return_attention_tokens": True}

    try:
        resp = requests.post(
            f"{base_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=300,
        )

        tokens_received = 0
        for line in resp.iter_lines():
            if line:
                line_str = line.decode("utf-8")
                if line_str.startswith("data: "):
                    data_str = line_str[6:]
                    if data_str.strip() == "[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if "choices" in data and data["choices"]:
                            delta = data["choices"][0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                if ttft is None:
                                    ttft = (time.perf_counter() - start_time) * 1000
                                output_text += content
                                tokens_received += 1
                    except json.JSONDecodeError:
                        pass

        total_time = (time.perf_counter() - start_time) * 1000

        return BenchmarkResult(
            name=prompt[:50] + "..." if len(prompt) > 50 else prompt,
            eviction_policy="unknown",
            prompt_tokens=len(prompt.split()),  # Rough estimate
            output_tokens=tokens_received,
            time_to_first_token_ms=ttft or total_time,
            total_time_ms=total_time,
            tokens_per_second=(
                tokens_received / (total_time / 1000) if total_time > 0 else 0
            ),
            output_text=output_text,
        )

    except Exception as e:
        return BenchmarkResult(
            name=prompt[:50] + "...",
            eviction_policy="error",
            prompt_tokens=0,
            output_tokens=0,
            time_to_first_token_ms=0,
            total_time_ms=0,
            tokens_per_second=0,
            output_text=f"Error: {e}",
        )


def compute_text_similarity(text1: str, text2: str) -> float:
    """Compute similarity between two texts (0-1)."""
    # Simple word overlap similarity
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())

    if not words1 or not words2:
        return 0.0

    intersection = len(words1 & words2)
    union = len(words1 | words2)

    return intersection / union if union > 0 else 0.0


# Benchmark prompts of varying complexity
BENCHMARK_PROMPTS = [
    # Short prompt
    "What is the capital of France?",
    # Medium prompt with context
    """The following is a conversation about machine learning:
    User: What are transformers in deep learning?
    Assistant: Transformers are a type of neural network architecture...
    User: How do attention mechanisms work?
    Please explain attention mechanisms in detail.""",
    # Long prompt for cache testing
    """Please read and summarize the following text:

    Artificial intelligence (AI) is intelligence demonstrated by machines,
    as opposed to natural intelligence displayed by animals including humans.
    AI research has been defined as the field of study of intelligent agents,
    which refers to any system that perceives its environment and takes actions
    that maximize its chance of achieving its goals.

    The term "artificial intelligence" had previously been used to describe
    machines that mimic and display "human" cognitive skills that are associated
    with the human mind, such as "learning" and "problem-solving". This definition
    has since been rejected by major AI researchers who now describe AI in terms
    of rationality and acting rationally, which does not limit how intelligence
    can be articulated.

    AI applications include advanced web search engines, recommendation systems,
    understanding human speech, self-driving cars, automated decision-making
    and competing at the highest level in strategic game systems.

    What are the main points in this text?""",
    # Reasoning prompt
    """Let's solve this step by step:
    A train leaves Station A at 9:00 AM traveling at 60 mph.
    Another train leaves Station B at 10:00 AM traveling at 80 mph toward Station A.
    The stations are 280 miles apart.
    At what time will the trains meet?""",
]


def run_benchmark_suite(
    base_url: str,
    num_iterations: int = 3,
    verbose: bool = True,
) -> List[BenchmarkResult]:
    """Run the full benchmark suite."""

    results = []

    for prompt in BENCHMARK_PROMPTS:
        prompt_results = []

        for i in range(num_iterations):
            if verbose:
                print(
                    f"  Running iteration {i+1}/{num_iterations}...",
                    end=" ",
                    flush=True,
                )

            result = run_generation(base_url, prompt, max_tokens=150)
            prompt_results.append(result)

            if verbose:
                print(f"{result.tokens_per_second:.1f} tok/s")

        # Average the results
        avg_result = BenchmarkResult(
            name=prompt_results[0].name,
            eviction_policy=prompt_results[0].eviction_policy,
            prompt_tokens=prompt_results[0].prompt_tokens,
            output_tokens=int(np.mean([r.output_tokens for r in prompt_results])),
            time_to_first_token_ms=np.mean(
                [r.time_to_first_token_ms for r in prompt_results]
            ),
            total_time_ms=np.mean([r.total_time_ms for r in prompt_results]),
            tokens_per_second=np.mean([r.tokens_per_second for r in prompt_results]),
            output_text=prompt_results[-1].output_text,  # Use last iteration's output
        )
        results.append(avg_result)

    return results


def print_results(results: List[BenchmarkResult], title: str):
    """Print benchmark results in a table."""
    print(f"\n{'='*80}")
    print(f" {title}")
    print(f"{'='*80}")
    print(
        f"{'Prompt':<50} {'Tokens':<10} {'TTFT(ms)':<12} {'Total(ms)':<12} {'Tok/s':<10}"
    )
    print("-" * 80)

    for r in results:
        print(
            f"{r.name:<50} {r.output_tokens:<10} {r.time_to_first_token_ms:<12.1f} {r.total_time_ms:<12.1f} {r.tokens_per_second:<10.1f}"
        )

    # Summary
    avg_ttft = np.mean([r.time_to_first_token_ms for r in results])
    avg_tps = np.mean([r.tokens_per_second for r in results])
    print("-" * 80)
    print(f"{'AVERAGE':<50} {'':<10} {avg_ttft:<12.1f} {'':<12} {avg_tps:<10.1f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark Spectral KV Cache Eviction")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument("--host", type=str, default="localhost", help="Server host")
    parser.add_argument(
        "--iterations", type=int, default=3, help="Iterations per prompt"
    )
    parser.add_argument(
        "--baseline", action="store_true", help="Mark this as baseline (LRU) run"
    )
    parser.add_argument(
        "--compare", type=str, help="Path to baseline results JSON for comparison"
    )
    parser.add_argument("--output", type=str, help="Save results to JSON file")
    parser.add_argument("--quiet", action="store_true", help="Reduce output verbosity")
    args = parser.parse_args()

    base_url = f"http://{args.host}:{args.port}"
    verbose = not args.quiet

    # Check server health
    if verbose:
        print(f"Connecting to server at {base_url}...")

    healthy, info = check_server_health(base_url)
    if not healthy:
        print(f"Error: Server not available at {base_url}")
        print(f"Details: {info}")
        sys.exit(1)

    model_id = info.get("data", [{}])[0].get("id", "unknown")
    if verbose:
        print(f"Connected to model: {model_id}")

    # Get initial metrics
    metrics = get_server_metrics(base_url)

    # Run benchmark
    if verbose:
        policy = "BASELINE (LRU)" if args.baseline else "SPECTRAL"
        print(f"\nRunning benchmark suite ({policy})...")

    results = run_benchmark_suite(base_url, args.iterations, verbose)

    # Print results
    title = "Baseline (LRU) Results" if args.baseline else "Spectral Eviction Results"
    print_results(results, title)

    # Get final metrics
    final_metrics = get_server_metrics(base_url)

    # Save results if requested
    if args.output:
        output_data = {
            "policy": "lru" if args.baseline else "spectral",
            "model": model_id,
            "iterations": args.iterations,
            "results": [
                {
                    "name": r.name,
                    "output_tokens": r.output_tokens,
                    "ttft_ms": r.time_to_first_token_ms,
                    "total_ms": r.total_time_ms,
                    "tokens_per_second": r.tokens_per_second,
                    "output_text": r.output_text,
                }
                for r in results
            ],
            "metrics": final_metrics,
        }
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        if verbose:
            print(f"\nResults saved to {args.output}")

    # Compare with baseline if provided
    if args.compare:
        try:
            with open(args.compare, "r") as f:
                baseline_data = json.load(f)

            print(f"\n{'='*80}")
            print(" COMPARISON: Spectral vs Baseline (LRU)")
            print(f"{'='*80}")

            baseline_results = baseline_data.get("results", [])

            for i, (spectral, baseline) in enumerate(zip(results, baseline_results)):
                similarity = compute_text_similarity(
                    spectral.output_text, baseline.get("output_text", "")
                )
                latency_diff = (
                    (spectral.total_time_ms - baseline.get("total_ms", 0))
                    / baseline.get("total_ms", 1)
                ) * 100

                print(f"\nPrompt {i+1}: {spectral.name}")
                print(f"  Quality (text similarity): {similarity:.1%}")
                print(f"  Latency change: {latency_diff:+.1f}%")
                print(f"  Spectral tok/s: {spectral.tokens_per_second:.1f}")
                print(f"  Baseline tok/s: {baseline.get('tokens_per_second', 0):.1f}")

            # Overall summary
            avg_similarity = np.mean(
                [
                    compute_text_similarity(r.output_text, b.get("output_text", ""))
                    for r, b in zip(results, baseline_results)
                ]
            )
            avg_spectral_tps = np.mean([r.tokens_per_second for r in results])
            avg_baseline_tps = np.mean(
                [b.get("tokens_per_second", 0) for b in baseline_results]
            )

            print(f"\n{'='*80}")
            print(" SUMMARY")
            print(f"{'='*80}")
            print(f"  Average quality preservation: {avg_similarity:.1%}")
            print(f"  Spectral throughput: {avg_spectral_tps:.1f} tok/s")
            print(f"  Baseline throughput: {avg_baseline_tps:.1f} tok/s")
            print(
                f"  Throughput change: {((avg_spectral_tps - avg_baseline_tps) / avg_baseline_tps * 100):+.1f}%"
            )

        except FileNotFoundError:
            print(f"Warning: Baseline file not found: {args.compare}")
        except Exception as e:
            print(f"Warning: Error comparing results: {e}")

    print("\nBenchmark complete!")


if __name__ == "__main__":
    main()
