#!/usr/bin/env python3
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
"""
OpenVLA Consistency and Performance Benchmark

Compares SGLang inference vs PyTorch/Transformers reference implementation:
1. Output consistency (action token accuracy)
2. Latency (time to first token, total generation time)
3. Throughput (tokens/second, requests/second)

Usage:
    python test/manual/test_openvla_consistency.py

Requirements:
    - GPU with sufficient VRAM (~16GB for fp16)
    - pip install timm transformers pillow requests
"""

import argparse
import gc
import statistics
import time
from dataclasses import dataclass
from io import BytesIO
from typing import List, Optional, Tuple

import requests
import torch
from PIL import Image

# Test images
TEST_IMAGES = [
    "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/man_ironing_on_back_of_suv.png",
    "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/sgl_logo.png",
]

# Test prompts (OpenVLA format)
TEST_PROMPTS = [
    "In: What action should the robot take to pick up the red block?\nOut:",
    "In: What action should the robot take to move left?\nOut:",
    "In: What action should the robot take to open the drawer?\nOut:",
]

MODEL_PATH = "openvla/openvla-7b"


@dataclass
class BenchmarkResult:
    """Results from a single inference run."""

    action_tokens: List[int]
    latency_ms: float
    time_to_first_token_ms: Optional[float] = None


@dataclass
class BenchmarkSummary:
    """Aggregated benchmark results."""

    name: str
    num_runs: int
    mean_latency_ms: float
    std_latency_ms: float
    min_latency_ms: float
    max_latency_ms: float
    throughput_rps: float  # requests per second
    tokens_per_second: float


def load_test_image(url: str) -> Image.Image:
    """Load image from URL."""
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def clear_gpu_memory():
    """Clear GPU memory between benchmarks."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


class TransformersRunner:
    """Reference implementation using HuggingFace Transformers."""

    def __init__(self, model_path: str, device: str = "cuda:0"):
        from transformers import AutoModelForVision2Seq, AutoProcessor

        print(f"[Transformers] Loading model from {model_path}...")
        self.device = device
        self.processor = AutoProcessor.from_pretrained(
            model_path, trust_remote_code=True
        )

        # Load without specifying attention implementation to avoid compatibility issues
        # OpenVLA's custom model may not support sdpa/flash_attention attributes
        try:
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                attn_implementation="eager",  # Use eager to avoid sdpa check
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(device)
        except Exception as e:
            print(f"[Transformers] Loading with eager failed: {e}")
            print("[Transformers] Trying without attn_implementation...")
            self.model = AutoModelForVision2Seq.from_pretrained(
                model_path,
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            ).to(device)

        self.model.eval()
        print("[Transformers] Model loaded successfully")

    def predict(self, image: Image.Image, prompt: str) -> BenchmarkResult:
        """Run inference and return results with timing."""
        start_time = time.perf_counter()

        inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)

        with torch.no_grad():
            # Generate 7 action tokens
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=7,
                do_sample=False,
                pad_token_id=self.processor.tokenizer.pad_token_id,
            )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Extract generated tokens (remove input tokens)
        input_len = inputs["input_ids"].shape[1]
        action_tokens = outputs[0, input_len:].cpu().tolist()

        return BenchmarkResult(
            action_tokens=action_tokens,
            latency_ms=latency_ms,
        )

    def cleanup(self):
        """Release model resources."""
        del self.model
        del self.processor
        clear_gpu_memory()


class SGLangRunner:
    """SGLang implementation using the Engine API."""

    def __init__(self, model_path: str):
        from sglang import Engine

        print(f"[SGLang] Loading model from {model_path}...")
        self.engine = Engine(
            model_path=model_path,
            trust_remote_code=True,
            enable_multimodal=True,
            mem_fraction_static=0.80,
            disable_cuda_graph=True,  # For consistency testing
        )
        print("[SGLang] Model loaded successfully")

    def predict(self, image: Image.Image, prompt: str) -> BenchmarkResult:
        """Run inference and return results with timing."""
        start_time = time.perf_counter()

        output = self.engine.generate(
            prompt=prompt,
            image_data=[image],
            sampling_params={
                "temperature": 0,
                "max_new_tokens": 7,
            },
        )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # Extract action tokens from output
        # SGLang returns output_ids at the top level
        # For OpenVLA, action tokens are in range [31744, 31999] (vocab_size - 256 to vocab_size - 1)
        output_ids = output.get("output_ids", [])

        return BenchmarkResult(
            action_tokens=output_ids[:7] if output_ids else [],
            latency_ms=latency_ms,
        )

    def cleanup(self):
        """Release model resources."""
        self.engine.shutdown()
        clear_gpu_memory()


class SGLangServerRunner:
    """SGLang implementation using the OpenAI-compatible server API."""

    def __init__(self, model_path: str, base_url: str = "http://127.0.0.1:30000"):
        import openai

        from sglang.srt.utils import kill_process_tree
        from sglang.test.test_utils import (
            DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            popen_launch_server,
        )

        print(f"[SGLang Server] Launching server with {model_path}...")
        self.base_url = base_url
        self.api_key = "sk-benchmark"
        self.kill_process_tree = kill_process_tree

        self.process = popen_launch_server(
            model_path,
            base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=self.api_key,
            other_args=[
                "--trust-remote-code",
                "--enable-multimodal",
                "--mem-fraction-static=0.80",
            ],
        )

        self.client = openai.Client(
            api_key=self.api_key,
            base_url=f"{base_url}/v1",
        )
        print("[SGLang Server] Server launched successfully")

    def predict(self, image: Image.Image, prompt: str) -> BenchmarkResult:
        """Run inference via OpenAI API and return results with timing."""
        import base64
        from io import BytesIO

        # Convert image to base64
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        img_base64 = base64.b64encode(buffer.getvalue()).decode("utf-8")
        img_url = f"data:image/png;base64,{img_base64}"

        start_time = time.perf_counter()

        response = self.client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": img_url}},
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            temperature=0,
            max_tokens=7,
        )

        end_time = time.perf_counter()
        latency_ms = (end_time - start_time) * 1000

        # The response contains generated text (action tokens as text)
        output_text = response.choices[0].message.content

        return BenchmarkResult(
            action_tokens=[],  # Token IDs not directly available via API
            latency_ms=latency_ms,
        )

    def cleanup(self):
        """Stop the server."""
        if self.process:
            self.kill_process_tree(self.process.pid)
        clear_gpu_memory()


def run_benchmark(
    runner,
    images: List[Image.Image],
    prompts: List[str],
    num_warmup: int = 2,
    num_runs: int = 10,
) -> Tuple[List[BenchmarkResult], BenchmarkSummary]:
    """Run benchmark with warmup and multiple iterations."""
    results = []

    # Warmup runs
    print(f"  Running {num_warmup} warmup iterations...")
    for i in range(num_warmup):
        img = images[i % len(images)]
        prompt = prompts[i % len(prompts)]
        _ = runner.predict(img, prompt)

    # Benchmark runs
    print(f"  Running {num_runs} benchmark iterations...")
    for i in range(num_runs):
        img = images[i % len(images)]
        prompt = prompts[i % len(prompts)]
        result = runner.predict(img, prompt)
        results.append(result)
        print(f"    Run {i + 1}/{num_runs}: {result.latency_ms:.2f}ms")

    # Calculate summary statistics
    latencies = [r.latency_ms for r in results]
    total_tokens = sum(len(r.action_tokens) for r in results)
    total_time_s = sum(latencies) / 1000

    summary = BenchmarkSummary(
        name=runner.__class__.__name__,
        num_runs=num_runs,
        mean_latency_ms=statistics.mean(latencies),
        std_latency_ms=statistics.stdev(latencies) if len(latencies) > 1 else 0,
        min_latency_ms=min(latencies),
        max_latency_ms=max(latencies),
        throughput_rps=num_runs / total_time_s if total_time_s > 0 else 0,
        tokens_per_second=total_tokens / total_time_s if total_time_s > 0 else 0,
    )

    return results, summary


def decode_action_tokens(
    tokens: List[int], vocab_size: int = 32000
) -> Tuple[List[int], List[float]]:
    """Decode action tokens to bins and normalized actions.

    OpenVLA uses the formula: bin = vocab_size - token - 1

    Args:
        tokens: List of action token IDs.
        vocab_size: Base vocabulary size (32000 for OpenVLA).

    Returns:
        Tuple of (bin_indices, normalized_actions).
    """
    bins = [max(0, min(255, vocab_size - t - 1)) for t in tokens]
    actions = [(2.0 * b + 1.0) / 256.0 - 1.0 for b in bins]
    return bins, actions


def compare_outputs(
    ref_results: List[BenchmarkResult],
    test_results: List[BenchmarkResult],
    tolerance: int = 0,
) -> Tuple[int, int, float]:
    """Compare action tokens between reference and test results.

    Compares using bin indices (derived from tokens via HF formula).
    """
    total_tokens = 0
    matching_tokens = 0

    for ref, test in zip(ref_results, test_results):
        ref_bins, _ = decode_action_tokens(ref.action_tokens)
        test_bins, _ = decode_action_tokens(test.action_tokens)

        # Compare bin by bin
        for i in range(min(len(ref_bins), len(test_bins))):
            total_tokens += 1
            diff = abs(ref_bins[i] - test_bins[i])
            if diff <= tolerance:
                matching_tokens += 1

        # Account for length differences
        total_tokens += abs(len(ref_bins) - len(test_bins))

    accuracy = matching_tokens / total_tokens if total_tokens > 0 else 0
    return matching_tokens, total_tokens, accuracy


def print_summary(summary: BenchmarkSummary):
    """Print formatted benchmark summary."""
    print(f"\n{'=' * 60}")
    print(f"  {summary.name} Results")
    print(f"{'=' * 60}")
    print(f"  Runs:              {summary.num_runs}")
    print(f"  Mean Latency:      {summary.mean_latency_ms:.2f} ms")
    print(f"  Std Latency:       {summary.std_latency_ms:.2f} ms")
    print(f"  Min Latency:       {summary.min_latency_ms:.2f} ms")
    print(f"  Max Latency:       {summary.max_latency_ms:.2f} ms")
    print(f"  Throughput:        {summary.throughput_rps:.2f} req/s")
    print(f"  Tokens/sec:        {summary.tokens_per_second:.2f}")
    print(f"{'=' * 60}")


def main():
    parser = argparse.ArgumentParser(description="OpenVLA Consistency Benchmark")
    parser.add_argument(
        "--model", type=str, default=MODEL_PATH, help="Model path or HF model ID"
    )
    parser.add_argument(
        "--num-warmup", type=int, default=2, help="Number of warmup iterations"
    )
    parser.add_argument(
        "--num-runs", type=int, default=10, help="Number of benchmark iterations"
    )
    parser.add_argument(
        "--skip-transformers",
        action="store_true",
        help="Skip transformers reference benchmark",
    )
    parser.add_argument(
        "--skip-sglang", action="store_true", help="Skip SGLang benchmark"
    )
    parser.add_argument(
        "--use-server",
        action="store_true",
        help="Use SGLang server mode instead of Engine",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("  OpenVLA Consistency and Performance Benchmark")
    print("=" * 60)
    print(f"Model: {args.model}")
    print(f"Warmup runs: {args.num_warmup}")
    print(f"Benchmark runs: {args.num_runs}")
    print()

    # Load test images
    print("Loading test images...")
    images = [load_test_image(url) for url in TEST_IMAGES]
    print(f"Loaded {len(images)} images")

    ref_results = None
    ref_summary = None
    sglang_results = None
    sglang_summary = None

    # Run Transformers benchmark (reference)
    if not args.skip_transformers:
        print("\n" + "=" * 60)
        print("  Running Transformers (Reference) Benchmark")
        print("=" * 60)
        try:
            runner = TransformersRunner(args.model)
            ref_results, ref_summary = run_benchmark(
                runner, images, TEST_PROMPTS, args.num_warmup, args.num_runs
            )
            print_summary(ref_summary)
            runner.cleanup()
        except Exception as e:
            print(f"[ERROR] Transformers benchmark failed: {e}")
            import traceback

            traceback.print_exc()

    # Run SGLang benchmark
    if not args.skip_sglang:
        print("\n" + "=" * 60)
        print("  Running SGLang Benchmark")
        print("=" * 60)
        try:
            if args.use_server:
                runner = SGLangServerRunner(args.model)
            else:
                runner = SGLangRunner(args.model)

            sglang_results, sglang_summary = run_benchmark(
                runner, images, TEST_PROMPTS, args.num_warmup, args.num_runs
            )
            print_summary(sglang_summary)
            runner.cleanup()
        except Exception as e:
            print(f"[ERROR] SGLang benchmark failed: {e}")
            import traceback

            traceback.print_exc()

    # Compare results
    if ref_results and sglang_results:
        print("\n" + "=" * 60)
        print("  Output Comparison")
        print("=" * 60)

        matching, total, accuracy = compare_outputs(ref_results, sglang_results)
        print(f"  Matching tokens:   {matching}/{total}")
        print(f"  Token accuracy:    {accuracy * 100:.2f}%")

        # With tolerance of 1 (adjacent bins)
        matching_t1, total_t1, accuracy_t1 = compare_outputs(
            ref_results, sglang_results, tolerance=1
        )
        print(f"  Accuracy (±1 bin): {accuracy_t1 * 100:.2f}%")

    # Performance comparison
    if ref_summary and sglang_summary:
        print("\n" + "=" * 60)
        print("  Performance Comparison")
        print("=" * 60)

        speedup = ref_summary.mean_latency_ms / sglang_summary.mean_latency_ms
        print(f"  Transformers mean:  {ref_summary.mean_latency_ms:.2f} ms")
        print(f"  SGLang mean:        {sglang_summary.mean_latency_ms:.2f} ms")
        print(f"  Speedup:            {speedup:.2f}x")
        print()
        print(f"  Transformers RPS:   {ref_summary.throughput_rps:.2f}")
        print(f"  SGLang RPS:         {sglang_summary.throughput_rps:.2f}")

    print("\n" + "=" * 60)
    print("  Benchmark Complete")
    print("=" * 60)


if __name__ == "__main__":
    main()
