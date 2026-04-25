"""TRAIL Embedding Collection Script.

Sends prompts from the Alpaca dataset to a running SGLang server
(launched with --trail-collect-embeddings) using concurrent requests.
Embeddings are auto-saved by the scheduler to --trail-embedding-save-dir.

Usage:
    # First, launch the server:
    python -m sglang.launch_server --model-path /path/to/model \
        --trail-collect-embeddings --trail-embedding-save-dir /tmp/trail_embeddings

    # Then run this script:
    python collect_embeddings.py --server-url http://localhost:30000 \
        --dataset alpaca --max-samples 5000 --concurrent 16
"""

import argparse
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests


def load_alpaca_dataset(max_samples: int = 5000):
    """Load Alpaca dataset from HuggingFace datasets or local cache."""
    try:
        from datasets import load_dataset
        ds = load_dataset("tatsu-lab/alpaca", split="train")
        samples = []
        for i, item in enumerate(ds):
            if i >= max_samples:
                break
            # Build prompt from instruction + input
            instruction = item.get("instruction", "")
            inp = item.get("input", "")
            if inp:
                prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{inp}\n\n### Response:\n"
            else:
                prompt = f"### Instruction:\n{instruction}\n\n### Response:\n"
            samples.append(prompt)
        return samples
    except ImportError:
        raise RuntimeError("Please install datasets: pip install datasets")


def send_request(server_url: str, prompt: str, max_tokens: int = 512):
    """Send a completion request to the SGLang server."""
    url = f"{server_url}/v1/completions"
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    try:
        resp = requests.post(url, json=payload, timeout=120)
        resp.raise_for_status()
        result = resp.json()
        usage = result.get("usage", {})
        return {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
    except Exception as e:
        return {"error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="TRAIL Embedding Collection")
    parser.add_argument("--server-url", type=str, default="http://localhost:30000")
    parser.add_argument("--dataset", type=str, default="alpaca", choices=["alpaca"])
    parser.add_argument("--max-samples", type=int, default=5000)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--concurrent", type=int, default=16)
    args = parser.parse_args()

    print(f"Loading {args.dataset} dataset (max {args.max_samples} samples)...")
    prompts = load_alpaca_dataset(args.max_samples)
    print(f"Loaded {len(prompts)} prompts")

    # Verify server is healthy
    try:
        health = requests.get(f"{args.server_url}/health", timeout=10)
        health.raise_for_status()
        print("Server is healthy")
    except Exception as e:
        print(f"Server not reachable at {args.server_url}: {e}")
        return

    print(f"Sending {len(prompts)} requests with concurrency={args.concurrent}...")
    start_time = time.time()
    completed = 0
    errors = 0
    total_completion_tokens = 0

    with ThreadPoolExecutor(max_workers=args.concurrent) as pool:
        futures = {
            pool.submit(send_request, args.server_url, p, args.max_tokens): i
            for i, p in enumerate(prompts)
        }
        for future in as_completed(futures):
            result = future.result()
            if "error" in result:
                errors += 1
            else:
                completed += 1
                total_completion_tokens += result.get("completion_tokens", 0)

            total = completed + errors
            if total % 100 == 0:
                elapsed = time.time() - start_time
                print(f"  Progress: {total}/{len(prompts)} "
                      f"({completed} ok, {errors} errors) "
                      f"[{elapsed:.1f}s, {total/elapsed:.1f} req/s]")

    elapsed = time.time() - start_time
    print(f"\nDone! {completed}/{len(prompts)} completed, {errors} errors")
    print(f"Total completion tokens: {total_completion_tokens}")
    print(f"Time: {elapsed:.1f}s ({completed/elapsed:.1f} req/s)")
    print(f"Embeddings saved by scheduler to --trail-embedding-save-dir")


if __name__ == "__main__":
    main()
