#!/usr/bin/env python3
"""
Memory Benchmark for Spectral KV Cache Eviction

Measures memory usage under different eviction policies.
"""

import json
import subprocess
import time

import requests


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=memory.used,memory.total",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
    )
    used, total = map(int, result.stdout.strip().split(", "))
    return used / 1024, total / 1024  # Convert to GB


def wait_for_server(port, timeout=120):
    """Wait for server to be ready."""
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(f"http://localhost:{port}/health", timeout=2)
            if r.status_code == 200:
                return True
        except:
            pass
        time.sleep(2)
    return False


def run_inference(port, prompt, max_tokens=100):
    """Run a single inference request."""
    try:
        r = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json={
                "model": "Qwen/Qwen3-4B",
                "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens,
            },
            timeout=60,
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}


def generate_long_context(base_length=2000):
    """Generate a long context prompt."""
    context = "Here is important background information:\n\n"
    # Generate paragraphs of context
    paragraphs = [
        "Machine learning is a subset of artificial intelligence that enables systems to learn from data. "
        "Deep learning uses neural networks with many layers to model complex patterns. "
        "Transformers revolutionized NLP with their attention mechanism. ",
        "The attention mechanism allows models to focus on relevant parts of the input. "
        "Self-attention computes relationships between all positions in a sequence. "
        "Multi-head attention runs multiple attention operations in parallel. ",
        "Large language models are trained on vast amounts of text data. "
        "They learn to predict the next token given previous context. "
        "Fine-tuning adapts pre-trained models to specific tasks. ",
    ]

    # Repeat to reach desired length
    while len(context) < base_length:
        for p in paragraphs:
            context += p + "\n\n"
            if len(context) >= base_length:
                break

    context += "\nBased on the above context, answer this question: What is attention in transformers?"
    return context


def run_benchmark(policy, port=30000):
    """Run memory benchmark for a given eviction policy."""
    print(f"\n{'='*60}")
    print(f"Benchmarking: {policy} eviction policy")
    print(f"{'='*60}")

    # Kill any existing server
    subprocess.run(["pkill", "-f", "sglang.launch_server"], capture_output=True)
    time.sleep(3)

    # Build server command
    cmd = [
        "python",
        "-m",
        "sglang.launch_server",
        "--model-path",
        "Qwen/Qwen3-4B",
        "--radix-eviction-policy",
        policy,
        "--port",
        str(port),
        "--host",
        "0.0.0.0",
    ]

    if policy == "spectral":
        cmd.extend(
            [
                "--attention-backend",
                "triton",
                "--return-attention-tokens",
                "--attention-fingerprint-mode",
                "--spectral-retention-ratio",
                "0.3",
            ]
        )

    # Start server
    print(f"Starting server with {policy} policy...")
    log_file = open(f"/tmp/sglang_{policy}_bench.log", "w")
    proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)

    if not wait_for_server(port):
        print(f"ERROR: Server failed to start")
        proc.terminate()
        return None

    print("Server ready!")

    # Get baseline memory
    mem_baseline, mem_total = get_gpu_memory()
    print(f"Baseline GPU memory: {mem_baseline:.2f} GB / {mem_total:.2f} GB")

    results = {
        "policy": policy,
        "baseline_memory_gb": mem_baseline,
        "total_memory_gb": mem_total,
        "requests": [],
    }

    # Run requests with increasing context
    context_sizes = [1000, 2000, 4000, 8000]

    for ctx_size in context_sizes:
        print(f"\n--- Context size: ~{ctx_size} chars ---")
        prompt = generate_long_context(ctx_size)

        # Run multiple requests to fill cache
        for i in range(5):
            start = time.time()
            resp = run_inference(port, prompt + f" (variant {i})", max_tokens=50)
            latency = time.time() - start

            mem_used, _ = get_gpu_memory()

            req_info = {
                "context_size": ctx_size,
                "request_num": i,
                "latency_s": latency,
                "memory_gb": mem_used,
                "success": "error" not in resp,
            }
            results["requests"].append(req_info)

            print(f"  Request {i+1}: {latency:.2f}s, Memory: {mem_used:.2f} GB")

        # Memory after batch
        mem_after, _ = get_gpu_memory()
        print(
            f"  Memory after batch: {mem_after:.2f} GB (delta: {mem_after - mem_baseline:.2f} GB)"
        )

    # Final memory
    mem_final, _ = get_gpu_memory()
    results["final_memory_gb"] = mem_final
    results["memory_delta_gb"] = mem_final - mem_baseline

    print(
        f"\nFinal memory: {mem_final:.2f} GB (delta: {mem_final - mem_baseline:.2f} GB)"
    )

    # Cleanup
    proc.terminate()
    proc.wait()
    log_file.close()

    return results


def main():
    print("=" * 60)
    print("Memory Benchmark: Spectral vs LRU Eviction")
    print("=" * 60)

    results = {}

    # Benchmark LRU (baseline)
    results["lru"] = run_benchmark("lru")
    time.sleep(5)

    # Benchmark Spectral
    results["spectral"] = run_benchmark("spectral")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    if results["lru"] and results["spectral"]:
        lru_delta = results["lru"]["memory_delta_gb"]
        spectral_delta = results["spectral"]["memory_delta_gb"]

        print(f"\nLRU eviction:")
        print(f"  Memory delta: {lru_delta:.2f} GB")

        print(f"\nSpectral eviction:")
        print(f"  Memory delta: {spectral_delta:.2f} GB")

        if lru_delta > 0:
            savings = (1 - spectral_delta / lru_delta) * 100
            print(f"\nMemory savings with spectral: {savings:.1f}%")

    # Save results
    with open("/tmp/memory_benchmark_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nResults saved to /tmp/memory_benchmark_results.json")


if __name__ == "__main__":
    main()
