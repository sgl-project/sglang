#!/usr/bin/env python3
"""
Benchmark script for FLUX.2-Klein latency testing.

This script implements Phase 1.1: Latency Benchmark from A02 test design.
It measures end-to-end image generation latency for FLUX.2-Klein model
using both SGLang and Diffusers backends.

Usage:
    # Test SGLang backend
    python bench_flux_klein.py --backend sglang --port 30000

    # Test Diffusers backend
    python bench_flux_klein.py --backend diffusers --port 30000

    # Test both backends
    python bench_flux_klein.py --backend both --port 30000

    # Custom configuration
    python bench_flux_klein.py \
        --backend sglang \
        --batch-sizes 1 2 4 \
        --resolutions 512x512 768x768 \
        --num-inference-steps 20 30 \
        --num-runs 5 \
        --output-dir results
"""

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI


class FluxKleinBenchmark:
    """Benchmark class for FLUX.2-Klein latency testing."""

    def __init__(
        self,
        model_path: str = "black-forest-labs/FLUX.2-klein-4B",
        base_url: str = "http://localhost:30000/v1",
        api_key: str = "sk-proj-1234567890",
    ):
        """
        Initialize the benchmark.

        Args:
            model_path: HuggingFace model ID or path
            base_url: Base URL for the API server
            api_key: API key for authentication
        """
        self.model_path = model_path
        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.prompt = "A beautiful landscape with mountains and lakes"

    def generate_image(
        self,
        width: int,
        height: int,
        num_inference_steps: int,
        n: int = 1,
    ) -> Tuple[float, bool, Optional[str]]:
        """
        Generate an image and measure latency.

        Args:
            width: Image width
            height: Image height
            num_inference_steps: Number of inference steps
            n: Number of images to generate (batch size)

        Returns:
            Tuple of (latency_seconds, success, error_message)
        """
        start_time = time.perf_counter()

        try:
            response = self.client.images.generate(
                model=self.model_path,
                prompt=self.prompt,
                size=f"{width}x{height}",
                n=n,
                num_inference_steps=num_inference_steps,
                response_format="b64_json",
            )

            end_time = time.perf_counter()
            latency = end_time - start_time

            # Verify response
            if response and response.data and len(response.data) > 0:
                return latency, True, None
            else:
                return latency, False, "Empty response"

        except Exception as e:
            end_time = time.perf_counter()
            latency = end_time - start_time
            return latency, False, str(e)

    def run_latency_test(
        self,
        batch_sizes: List[int],
        resolutions: List[Tuple[int, int]],
        num_inference_steps_list: List[int],
        num_runs: int = 10,
    ) -> Dict:
        """
        Run latency benchmark tests.

        Args:
            batch_sizes: List of batch sizes to test
            resolutions: List of (width, height) tuples
            num_inference_steps_list: List of inference step counts
            num_runs: Number of runs per configuration

        Returns:
            Dictionary containing all test results
        """
        results = {
            "model_path": self.model_path,
            "prompt": self.prompt,
            "test_config": {
                "batch_sizes": batch_sizes,
                "resolutions": [f"{w}x{h}" for w, h in resolutions],
                "num_inference_steps": num_inference_steps_list,
                "num_runs": num_runs,
            },
            "results": [],
        }

        total_tests = (
            len(batch_sizes)
            * len(resolutions)
            * len(num_inference_steps_list)
            * num_runs
        )
        test_count = 0

        print(f"\n{'='*60}")
        print(f"Starting Latency Benchmark for {self.model_path}")
        print(f"{'='*60}")
        print(f"Total test configurations: {total_tests}")
        print(f"Prompt: {self.prompt}")
        print(f"{'='*60}\n")

        for batch_size in batch_sizes:
            for width, height in resolutions:
                for num_steps in num_inference_steps_list:
                    config_key = f"batch_{batch_size}_res_{width}x{height}_steps_{num_steps}"

                    print(
                        f"Testing: batch_size={batch_size}, "
                        f"resolution={width}x{height}, "
                        f"steps={num_steps}"
                    )

                    latencies = []
                    successes = []
                    errors = []

                    for run_idx in range(num_runs):
                        test_count += 1
                        latency, success, error = self.generate_image(
                            width=width,
                            height=height,
                            num_inference_steps=num_steps,
                            n=batch_size,
                        )

                        latencies.append(latency)
                        successes.append(success)
                        if error:
                            errors.append(error)

                        print(
                            f"  Run {run_idx+1}/{num_runs}: "
                            f"latency={latency:.3f}s, "
                            f"success={success}"
                        )

                    # Calculate statistics
                    if latencies:
                        latencies_array = np.array(latencies)
                        success_rate = sum(successes) / len(successes)

                        config_result = {
                            "config": {
                                "batch_size": batch_size,
                                "width": width,
                                "height": height,
                                "resolution": f"{width}x{height}",
                                "num_inference_steps": num_steps,
                            },
                            "latency_stats": {
                                "mean": float(np.mean(latencies_array)),
                                "median": float(np.median(latencies_array)),
                                "std": float(np.std(latencies_array)),
                                "min": float(np.min(latencies_array)),
                                "max": float(np.max(latencies_array)),
                                "p50": float(np.percentile(latencies_array, 50)),
                                "p90": float(np.percentile(latencies_array, 90)),
                                "p99": float(np.percentile(latencies_array, 99)),
                            },
                            "success_rate": success_rate,
                            "total_runs": num_runs,
                            "successful_runs": sum(successes),
                            "failed_runs": num_runs - sum(successes),
                            "errors": list(set(errors)) if errors else [],
                            "raw_latencies": latencies,
                        }

                        results["results"].append(config_result)

                        print(
                            f"  Summary: mean={config_result['latency_stats']['mean']:.3f}s, "
                            f"median={config_result['latency_stats']['median']:.3f}s, "
                            f"success_rate={success_rate:.2%}"
                        )
                        print()

        return results

    def save_results(self, results: Dict, output_path: Path):
        """Save results to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to: {output_path}")


def parse_resolutions(res_str: str) -> List[Tuple[int, int]]:
    """Parse resolution string like '512x512,768x768' into list of tuples."""
    resolutions = []
    for res in res_str.split(","):
        res = res.strip()
        if "x" in res:
            w, h = res.split("x")
            resolutions.append((int(w), int(h)))
        else:
            raise ValueError(f"Invalid resolution format: {res}")
    return resolutions


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark FLUX.2-Klein latency for Phase 1.1 testing"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="black-forest-labs/FLUX.2-klein-4B",
        help="HuggingFace model ID or path",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["sglang", "diffusers", "both"],
        default="sglang",
        help="Backend to test",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default="http://localhost:30000/v1",
        help="Base URL for the API server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=30000,
        help="Port number (overrides base-url port if specified)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="sk-proj-1234567890",
        help="API key for authentication",
    )
    parser.add_argument(
        "--batch-sizes",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Batch sizes to test",
    )
    parser.add_argument(
        "--resolutions",
        type=str,
        default="512x512,768x768,1024x1024",
        help="Resolutions to test (format: WxH,WxH,...)",
    )
    parser.add_argument(
        "--num-inference-steps",
        type=int,
        nargs="+",
        default=[20, 30, 50],
        help="Number of inference steps to test",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=10,
        help="Number of runs per configuration",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="benchmark/results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="A beautiful landscape with mountains and lakes",
        help="Prompt to use for testing",
    )

    args = parser.parse_args()

    # Parse resolutions
    resolutions = parse_resolutions(args.resolutions)

    # Update base_url with port if specified
    if args.port:
        base_url = f"http://localhost:{args.port}/v1"
    else:
        base_url = args.base_url

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    backends_to_test = []
    if args.backend in ["sglang", "both"]:
        backends_to_test.append("sglang")
    if args.backend in ["diffusers", "both"]:
        backends_to_test.append("diffusers")

    all_results = {}

    for backend in backends_to_test:
        print(f"\n{'='*60}")
        print(f"Testing {backend.upper()} backend")
        print(f"{'='*60}")

        # Create benchmark instance
        benchmark = FluxKleinBenchmark(
            model_path=args.model_path,
            base_url=base_url,
            api_key=args.api_key,
        )
        benchmark.prompt = args.prompt

        # Run tests
        results = benchmark.run_latency_test(
            batch_sizes=args.batch_sizes,
            resolutions=resolutions,
            num_inference_steps_list=args.num_inference_steps,
            num_runs=args.num_runs,
        )

        # Add backend info
        results["backend"] = backend
        all_results[backend] = results

        # Save results
        output_file = output_dir / f"{backend}_latency_results.json"
        benchmark.save_results(results, output_file)

    # Save combined results if testing both backends
    if len(backends_to_test) > 1:
        combined_output = output_dir / "combined_latency_results.json"
        with open(combined_output, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\nCombined results saved to: {combined_output}")

    print(f"\n{'='*60}")
    print("Benchmark completed!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
