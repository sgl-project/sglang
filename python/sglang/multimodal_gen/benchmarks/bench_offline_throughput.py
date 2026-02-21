"""
Benchmark offline throughput for multimodal generation models (Image/Video Generation).

This script benchmarks generation throughput without running a server, using low-level APIs.
It provides detailed metrics on throughput, latency, and resource utilization.

# Usage Examples

## Text-to-Video with VBench dataset
python -m sglang.multimodal_gen.benchmarks.bench_offline_throughput \\
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \\
    --dataset vbench \\
    --num-prompts 20 \\
    --batch-size 1 \\
    --width 512 --height 512 --num-frames 16

## Random dataset for stress testing
python -m sglang.multimodal_gen.benchmarks.bench_offline_throughput \\
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \\
    --dataset random \\
    --num-prompts 100 \\
    --batch-size 1 \\
    --num-inference-steps 20 \\
    --output-file results.json
"""

import argparse
import dataclasses
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

from tqdm import tqdm

from sglang.multimodal_gen.benchmarks.datasets import RandomDataset, VBenchDataset
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    configure_logger,
    init_logger,
)
from sglang.multimodal_gen.test.test_utils import print_divider, print_value_formatted

logger = init_logger(__name__)


# ============================================================================
# Core Data Structures & Configuration
# ============================================================================


@dataclass
class BatchOutput:
    """Container for batch generation results."""

    latency: float = 0.0
    latency_per_sample: float = 0.0
    num_samples: int = 0
    total_frames_or_pixels: int = 0
    peak_memory_mb: float = 0.0
    success: bool = False
    error: str = ""


@dataclasses.dataclass
class BenchArgs:
    """Benchmark configuration for multimodal generation."""

    # Diffusion Model Configuration
    num_inference_steps: int = 20
    guidance_scale: float = 7.5
    seed: int = 42
    disable_safety_checker: bool = False

    # Output Configuration
    width: int = 32
    height: int = 32
    num_frames: int = 1
    fps: int = 24

    # Dataset & Benchmark
    dataset: str = "random"
    dataset_path: str = ""
    task_name: str = "unknown"
    num_prompts: int = 10
    batch_size: int = 1

    # Benchmark Execution
    skip_warmup: bool = False
    profile: bool = False
    num_runs: int = 1
    output_file: str = ""
    disable_tqdm: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser):
        """Add benchmark-specific CLI arguments."""
        # Diffusion Model Configuration
        parser.add_argument(
            "--num-inference-steps",
            type=int,
            default=20,
            help="Number of denoising steps",
        )
        parser.add_argument(
            "--guidance-scale",
            type=float,
            default=7.5,
            help="Classifier-free guidance scale",
        )
        parser.add_argument("--seed", type=int, default=42, help="Random seed")
        parser.add_argument(
            "--disable-safety-checker",
            action="store_true",
            help="Disable NSFW detection",
        )

        # Output Configuration
        parser.add_argument("--width", type=int, default=32, help="Image/video width")
        parser.add_argument("--height", type=int, default=32, help="Image/video height")
        parser.add_argument(
            "--num-frames", type=int, default=1, help="Number of frames for video"
        )
        parser.add_argument("--fps", type=int, default=24, help="FPS for video")

        # Dataset & Benchmark
        parser.add_argument(
            "--dataset",
            type=str,
            default="random",
            choices=["vbench", "random"],
            help="Dataset to use",
        )
        parser.add_argument(
            "--dataset-path",
            type=str,
            default="",
            help="Path to dataset (prompts file or image directory)",
        )
        parser.add_argument(
            "--num-prompts",
            type=int,
            default=10,
            help="Total number of prompts to benchmark",
        )
        parser.add_argument(
            "--batch-size", type=int, default=1, help="Batch size per generation call"
        )

        # Benchmark Execution
        parser.add_argument(
            "--skip-warmup", action="store_true", help="Skip warmup batch"
        )
        parser.add_argument(
            "--profile",
            action="store_true",
            help="Enable torch profiler (use env var SGLANG_TORCH_PROFILER_DIR)",
        )
        parser.add_argument(
            "--num-runs", type=int, default=1, help="Number of benchmark runs"
        )
        parser.add_argument(
            "--output-file",
            type=str,
            default="",
            help="Output JSON file for results (append mode)",
        )
        parser.add_argument(
            "--disable-tqdm",
            action="store_true",
            help="Disable progress bar",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        """Create BenchArgs from parsed CLI arguments."""
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


# ============================================================================
# Engine/Runtime Integration
# ============================================================================


def initialize_engine(server_args: ServerArgs) -> Any:
    """Initialize diffusion pipeline engine."""
    logger.info("Initializing engine...")

    # Initialize DiffGenerator in local mode
    engine = DiffGenerator.from_server_args(server_args, local_mode=True)

    logger.info(f"Engine initialized successfully")
    return engine


def generate_batch(
    engine: Any,
    bench_args: BenchArgs,
    prompts: List[str],
    user_sampling_params: Dict[str, Any],
) -> BatchOutput:
    """Generate batch of images/videos synchronously."""
    output = BatchOutput()
    start_time = time.perf_counter()

    try:
        # Generate using DiffGenerator
        for i, prompt in enumerate(prompts):
            sampling_params_kwargs = dict(user_sampling_params)
            sampling_params_kwargs["prompt"] = prompt
            result = engine.generate(sampling_params_kwargs=sampling_params_kwargs)

            # Extract metrics from result
            if result is not None:
                if isinstance(result, list):
                    output.total_frames_or_pixels += len(result)
                else:
                    output.total_frames_or_pixels += 1

        output.latency = time.perf_counter() - start_time
        output.latency_per_sample = output.latency / len(prompts)
        output.num_samples = len(prompts)
        output.success = True

        logger.debug(
            f"Batch generated: {len(prompts)} samples in {output.latency:.2f}s"
        )

    except Exception as e:
        output.latency = time.perf_counter() - start_time
        output.success = False
        output.error = str(e)
        logger.error(f"Batch generation failed: {e}")

    return output


# ============================================================================
# Metrics Calculation
# ============================================================================


def calculate_metrics(
    outputs: List[BatchOutput],
    total_duration: float,
    task_type: str,
    resolution: Tuple[int, int, int],
    num_requests: int,
) -> Dict[str, Any]:
    """Calculate generation-specific throughput metrics."""
    successful = [o for o in outputs if o.success]
    num_success = len(successful)
    total_frames = sum(o.total_frames_or_pixels for o in successful)
    peak_memory = max((o.peak_memory_mb for o in outputs), default=0)

    # Resolution-based calculations
    width, height, frames = resolution
    pixels_per_sample = width * height * frames
    total_pixels = num_success * pixels_per_sample

    metrics = {
        "num_requests": num_requests,
        "successful_requests": num_success,
        "failed_requests": num_requests - num_success,
        "total_duration_seconds": total_duration,
        "total_frames_generated": total_frames,
        "total_pixels_generated": total_pixels,
        "images_per_second": num_success / total_duration if total_duration > 0 else 0,
        "frames_per_second": total_frames / total_duration if total_duration > 0 else 0,
        "megapixels_per_second": (
            total_pixels / (total_duration * 1e6) if total_duration > 0 else 0
        ),
        "requests_per_second": (
            num_success / total_duration if total_duration > 0 else 0
        ),
        "latency_per_request_seconds": (
            total_duration / num_success if num_success > 0 else 0
        ),
        "peak_memory_mb": peak_memory,
    }

    return metrics


# ============================================================================
# Main Benchmark Loop
# ============================================================================


def throughput_test(
    server_args: ServerArgs,
    bench_args: BenchArgs,
) -> Dict[str, Any]:
    """Main throughput benchmark function."""
    configure_logger(server_args=server_args)
    logger.info("Starting offline throughput benchmark...")

    # Initialize engine
    engine = initialize_engine(server_args)

    # Load dataset
    logger.info(f"Loading {bench_args.dataset} dataset...")
    if bench_args.dataset == "vbench":
        bench_args.task_name = engine.server_args.pipeline_config.task_type
        dataset = VBenchDataset(bench_args)
    elif bench_args.dataset == "random":
        dataset = RandomDataset(bench_args)
    else:
        raise ValueError(f"Unknown dataset: {bench_args.dataset}")

    # Prepare sampling parameters
    sampling_params = {
        "guidance_scale": bench_args.guidance_scale,
        "num_inference_steps": bench_args.num_inference_steps,
        "height": bench_args.height,
        "width": bench_args.width,
        "num_frames": bench_args.num_frames,
        "seed": bench_args.seed,
    }
    if bench_args.disable_safety_checker:
        sampling_params["safety_checker"] = None

    # Warmup (optional)
    if not bench_args.skip_warmup:
        logger.info("Running warmup batch...")
        warmup_indices = list(range(min(bench_args.batch_size, len(dataset))))
        warmup_batch = dataset.get_batch(warmup_indices)
        warmup_prompts = [item["prompt"] for item in warmup_batch]
        generate_batch(engine, bench_args, warmup_prompts, sampling_params)

    # Main benchmark loop
    logger.info(f"Running benchmark with {bench_args.num_prompts} prompts...")
    outputs: List[BatchOutput] = []
    total_indices = list(range(min(bench_args.num_prompts, len(dataset))))

    start_time = time.perf_counter()

    # Process in batches
    num_batches = (
        len(total_indices) + bench_args.batch_size - 1
    ) // bench_args.batch_size
    pbar = tqdm(
        total=num_batches,
        disable=bench_args.disable_tqdm,
        desc="Benchmark",
    )

    for batch_start in range(0, len(total_indices), bench_args.batch_size):
        batch_end = min(batch_start + bench_args.batch_size, len(total_indices))
        batch_indices = total_indices[batch_start:batch_end]

        batch_items = dataset.get_batch(batch_indices)
        batch_prompts = [item["prompt"] for item in batch_items]

        # Generate batch
        batch_output = generate_batch(
            engine, bench_args, batch_prompts, sampling_params
        )
        outputs.append(batch_output)

        pbar.update(1)

    pbar.close()
    total_duration = time.perf_counter() - start_time

    # Calculate metrics
    resolution = (bench_args.width, bench_args.height, bench_args.num_frames)
    metrics = calculate_metrics(
        outputs,
        total_duration,
        task_type="unknown",
        resolution=resolution,
        num_requests=len(total_indices),
    )

    # Display results
    display_results(
        metrics,
        bench_args,
        model_path=server_args.model_path,
    )

    # Save results
    if bench_args.output_file:
        save_results(metrics, bench_args, server_args)

    return metrics


# ============================================================================
# Output & Reporting
# ============================================================================


def display_results(
    metrics: Dict[str, Any],
    bench_args: BenchArgs,
    model_path: str,
):
    """Display benchmark results in console."""
    print(
        "\n{s:{c}^{n}}".format(s=" Offline Throughput Benchmark Result ", n=110, c="=")
    )
    print_value_formatted("Model:", model_path)
    print_value_formatted("Dataset:", bench_args.dataset)
    print_value_formatted(
        "Resolution:",
        f"{bench_args.width}x{bench_args.height}x{bench_args.num_frames}",
    )
    print_value_formatted("Num Inference Steps:", bench_args.num_inference_steps)
    print_divider(75)
    print_value_formatted("Total Requests:", metrics["num_requests"])
    print_value_formatted("Successful Requests:", metrics["successful_requests"])
    print_value_formatted("Failed Requests:", metrics["failed_requests"])
    print_value_formatted(
        "Total Duration (seconds):", metrics["total_duration_seconds"]
    )
    print_divider(75)
    print_value_formatted("Frames Generated:", metrics["total_frames_generated"])
    print_value_formatted(
        "Megapixels Generated:", metrics["total_pixels_generated"] / 1e6
    )
    print_divider(75)
    print_value_formatted(
        "Frame Throughput (frames/sec):", metrics["frames_per_second"]
    )
    print_value_formatted("MP Throughput (MP/sec):", metrics["megapixels_per_second"])
    print_value_formatted("Requests Per Second:", metrics["requests_per_second"])
    print_value_formatted(
        "Latency Per Request (sec):", metrics["latency_per_request_seconds"]
    )
    print_value_formatted("Peak Memory (MB):", metrics["peak_memory_mb"])
    print_divider(110, "=")


def save_results(
    metrics: Dict[str, Any],
    bench_args: BenchArgs,
    server_args: ServerArgs,
):
    """Save benchmark results to JSON file."""
    result = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_path": server_args.model_path,
            "task_type": "unknown",
            "backend": "engine",
        },
        "configuration": {
            "num_inference_steps": bench_args.num_inference_steps,
            "guidance_scale": bench_args.guidance_scale,
            "seed": bench_args.seed,
            "batch_size": bench_args.batch_size,
            "num_prompts": bench_args.num_prompts,
            "resolution": f"{bench_args.width}x{bench_args.height}x{bench_args.num_frames}",
            "dataset": bench_args.dataset,
        },
        "results": metrics,
    }

    with open(bench_args.output_file, "a") as f:
        f.write(json.dumps(result) + "\n")

    logger.info(f"Results saved to {bench_args.output_file}")


# ============================================================================
# CLI & Entry Point
# ============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Offline throughput benchmark for multimodal generation models"
    )

    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)

    args = parser.parse_args()

    # Create ServerArgs and BenchArgs
    server_args = ServerArgs.from_cli_args(args)
    bench_args = BenchArgs.from_cli_args(args)

    # Set global server args
    set_global_server_args(server_args)

    # Run benchmark
    result = throughput_test(server_args, bench_args)

    return result


if __name__ == "__main__":
    main()
