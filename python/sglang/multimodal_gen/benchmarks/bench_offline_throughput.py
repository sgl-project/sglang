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

## Reproducible JSONL request manifest with durable output evidence
python -m sglang.multimodal_gen.benchmarks.bench_offline_throughput \\
    --model-path Wan-AI/Wan2.1-T2V-1.3B-Diffusers \\
    --request-manifest workload.jsonl \\
    --save-output-dir artifacts \\
    --output-file results.jsonl
"""

import argparse
import dataclasses
import json
import os
import re
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

from sglang.multimodal_gen.benchmarks.datasets import RandomDataset, VBenchDataset
from sglang.multimodal_gen.benchmarks.request_manifest import (
    LoadedRequestManifest,
    file_sha256,
    load_request_manifest,
)
from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import DiffGenerator
from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    configure_logger,
    init_logger,
)
from sglang.multimodal_gen.test.test_utils import print_divider, print_value_formatted

logger = init_logger(__name__)


@dataclass
class BatchOutput:
    """Container for batch generation results."""

    latency: float = 0.0
    latency_per_sample: float = 0.0
    num_samples: int = 0
    total_frames: int = 0
    peak_memory_mb: float = 0.0
    success: bool = False
    error: str = ""
    requests: List["RequestOutput"] = dataclasses.field(default_factory=list)


@dataclass
class RequestOutput:
    """Evidence recorded for one benchmark request."""

    request_id: str
    prompt: str
    sampling_params: Dict[str, Any]
    success: bool
    latency_seconds: float
    error: str = ""
    output_file_paths: List[str] = dataclasses.field(default_factory=list)
    output_sha256: List[str] = dataclasses.field(default_factory=list)


@dataclass
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
    num_outputs_per_prompt: int = 1
    batch_size: int = 1
    random_request_config: str = None
    random_request_seed: int = 42
    request_manifest: str = ""

    # Benchmark Execution
    skip_warmup: bool = False
    output_file: str = ""
    save_output_dir: str = ""
    disable_tqdm: bool = False

    # Profiling
    profile: bool = False
    num_profiled_timesteps: int = 5
    profile_all_stages: bool = False

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
            "--task-name",
            type=str,
            default="unknown",
            help="Task name for benchmark identification",
        )
        parser.add_argument(
            "--num-prompts",
            type=int,
            default=10,
            help="Total number of prompts to benchmark",
        )
        parser.add_argument(
            "--num-outputs-per-prompt",
            type=int,
            default=1,
            help="Number of generated outputs requested per prompt",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=1,
            help="Batch size per generation call (currently only bs=1 is supported)",
        )

        parser.add_argument(
            "--random-request-config",
            type=str,
            default=None,
            help=(
                "JSON string defining random request profiles. "
                "Each profile may contain: width, height, num_inference_steps, etc. "
                "The 'weight' field controls sampling probability (relative weight)."
            ),
        )
        parser.add_argument(
            "--random-request-seed",
            type=int,
            default=42,
            help="Random seed for sampling request profiles (default: 42).",
        )
        parser.add_argument(
            "--request-manifest",
            type=str,
            default="",
            help=(
                "JSONL request manifest. When set, every non-empty line is run once "
                "and --dataset/--dataset-path/--num-prompts are ignored."
            ),
        )

        # Benchmark Execution
        parser.add_argument(
            "--skip-warmup", action="store_true", help="Skip warmup batch"
        )
        parser.add_argument(
            "--output-file",
            type=str,
            default="",
            help="Output JSON file for results (append mode)",
        )
        parser.add_argument(
            "--save-output-dir",
            type=str,
            default="",
            help=(
                "Persist generated media in this directory and record a SHA256 "
                "digest for each output."
            ),
        )
        parser.add_argument(
            "--disable-tqdm",
            action="store_true",
            help="Disable progress bar",
        )
        parser.add_argument(
            "--profile",
            action="store_true",
            help=(
                "Enable PyTorch profiler for diffusion generation. "
                "Set SGLANG_DIFFUSION_TORCH_PROFILER_DIR to control trace output directory."
            ),
        )
        parser.add_argument(
            "--num-profiled-timesteps",
            type=int,
            default=5,
            help=(
                "Number of denoising timesteps to profile after warmup. "
                "Use -1 to profile all denoising timesteps."
            ),
        )
        parser.add_argument(
            "--profile-all-stages",
            action="store_true",
            help="Profile all diffusion pipeline stages instead of only denoising steps.",
        )

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace):
        """Create BenchArgs from parsed CLI arguments."""
        attrs = [attr.name for attr in dataclasses.fields(cls)]
        return cls(**{attr: getattr(args, attr) for attr in attrs})


def initialize_engine(server_args: ServerArgs) -> DiffGenerator:
    """Initialize diffusion pipeline engine."""
    logger.info("Initializing engine...")
    engine = DiffGenerator.from_server_args(server_args, local_mode=True)
    logger.info("Engine initialized successfully")
    return engine


def generate_batch(
    engine: DiffGenerator,
    bench_args: BenchArgs,
    prompts: List[str],
    user_sampling_params: List[Dict[str, Any]],
    request_ids: Optional[List[str]] = None,
) -> BatchOutput:
    """Generate batch of images/videos synchronously."""
    assert len(user_sampling_params) == len(prompts), (
        f"user_sampling_params length ({len(user_sampling_params)}) must match "
        f"prompts length ({len(prompts)})"
    )
    if request_ids is None:
        request_ids = [f"request-{idx:05d}" for idx in range(len(prompts))]
    assert len(request_ids) == len(prompts), (
        f"request_ids length ({len(request_ids)}) must match "
        f"prompts length ({len(prompts)})"
    )

    output = BatchOutput()
    start_time = time.perf_counter()

    torch.get_device_module().reset_peak_memory_stats()

    for prompt, params, request_id in zip(prompts, user_sampling_params, request_ids):
        request_start_time = time.perf_counter()
        try:
            sampling_params_kwargs = dict(params)
            sampling_params_kwargs["prompt"] = prompt
            sampling_params_kwargs["request_id"] = request_id
            result = engine.generate(sampling_params_kwargs=sampling_params_kwargs)

            results = result if isinstance(result, list) else [result]
            results = [item for item in results if item is not None]
            if not results:
                raise RuntimeError("Engine returned no generation result")

            output_paths = [
                str(item.output_file_path)
                for item in results
                if item.output_file_path is not None
            ]
            output_hashes = []
            for output_path in output_paths:
                if not os.path.isfile(output_path):
                    raise RuntimeError(
                        f"Generated output does not exist: {output_path}"
                    )
                output_hashes.append(file_sha256(output_path))

            output.total_frames += int(params.get("num_frames", 1)) * len(results)
            output.num_samples += 1
            output.requests.append(
                RequestOutput(
                    request_id=request_id,
                    prompt=prompt,
                    sampling_params=dict(params),
                    success=True,
                    latency_seconds=time.perf_counter() - request_start_time,
                    output_file_paths=output_paths,
                    output_sha256=output_hashes,
                )
            )
        except Exception as e:
            logger.error(f"Generation failed for prompt '{prompt[:50]}...': {e}")
            output.error = str(e)
            output.requests.append(
                RequestOutput(
                    request_id=request_id,
                    prompt=prompt,
                    sampling_params=dict(params),
                    success=False,
                    latency_seconds=time.perf_counter() - request_start_time,
                    error=str(e),
                )
            )

    output.latency = time.perf_counter() - start_time
    output.latency_per_sample = output.latency / len(prompts) if prompts else 0.0
    output.success = output.num_samples > 0
    output.peak_memory_mb = torch.get_device_module().max_memory_allocated() / (
        1024 * 1024
    )

    logger.debug(
        f"Batch generated: {output.num_samples}/{len(prompts)} samples in {output.latency:.2f}s"
    )

    return output


def calculate_metrics(
    outputs: List[BatchOutput],
    total_duration: float,
    resolution: Tuple[int, int, int],
    num_requests: int,
    all_sampling_params: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Calculate generation-specific throughput metrics."""
    num_success = sum(o.num_samples for o in outputs)
    total_frames = sum(o.total_frames for o in outputs)
    peak_memory = max((o.peak_memory_mb for o in outputs), default=0)
    request_outputs = [request for output in outputs for request in output.requests]
    successful_request_outputs = [
        request for request in request_outputs if request.success
    ]
    request_latencies = [
        request.latency_seconds for request in successful_request_outputs
    ]

    width, height, frames = resolution
    if all_sampling_params:
        total_pixels = sum(
            request.sampling_params.get("width", width)
            * request.sampling_params.get("height", height)
            * request.sampling_params.get("num_frames", frames)
            for request in successful_request_outputs
        )
    else:
        total_pixels = num_success * width * height * frames

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
        "request_latency_mean_seconds": (
            statistics.fmean(request_latencies) if request_latencies else 0
        ),
        "request_latency_median_seconds": (
            statistics.median(request_latencies) if request_latencies else 0
        ),
        "request_latency_min_seconds": min(request_latencies, default=0),
        "request_latency_max_seconds": max(request_latencies, default=0),
        "request_results": [dataclasses.asdict(request) for request in request_outputs],
        "peak_memory_mb": peak_memory,
    }

    return metrics


def throughput_test(
    server_args: ServerArgs,
    bench_args: BenchArgs,
) -> Dict[str, Any]:
    """Main throughput benchmark function."""
    configure_logger(server_args=server_args)
    logger.info("Starting offline throughput benchmark...")

    engine = initialize_engine(server_args)
    bench_args.task_name = str(engine.server_args.pipeline_config.task_type)

    if bench_args.request_manifest and bench_args.random_request_config:
        raise ValueError(
            "--request-manifest and --random-request-config are mutually exclusive"
        )
    if bench_args.random_request_config and bench_args.dataset != "random":
        raise ValueError(
            "--random-request-config can only be used with --dataset random"
        )

    if bench_args.num_outputs_per_prompt != 1:
        raise ValueError(
            "bench_offline_throughput currently supports only --num-outputs-per-prompt 1"
        )

    manifest: Optional[LoadedRequestManifest] = None
    if bench_args.request_manifest:
        logger.info(f"Loading request manifest {bench_args.request_manifest}...")
        manifest = load_request_manifest(bench_args.request_manifest)
        manifest_requests = manifest.requests
        total_count = len(manifest_requests)
        all_prompts = [request.prompt for request in manifest_requests]
        all_request_ids = [request.request_id for request in manifest_requests]
        all_sampling_params = [
            dict(request.sampling_params) for request in manifest_requests
        ]
    elif bench_args.dataset == "vbench":
        logger.info(f"Loading {bench_args.dataset} dataset...")
        dataset = VBenchDataset(bench_args)
        total_count = min(bench_args.num_prompts, len(dataset))
        dataset_requests = [dataset[i] for i in range(total_count)]
    elif bench_args.dataset == "random":
        logger.info(f"Loading {bench_args.dataset} dataset...")
        dataset = RandomDataset(bench_args)
        total_count = min(bench_args.num_prompts, len(dataset))
        dataset_requests = [dataset[i] for i in range(total_count)]
    else:
        raise ValueError(f"Unknown dataset: {bench_args.dataset}")

    _sampling_params = {
        "guidance_scale": bench_args.guidance_scale,
        "num_inference_steps": bench_args.num_inference_steps,
        "height": bench_args.height,
        "width": bench_args.width,
        "num_frames": bench_args.num_frames,
        "fps": bench_args.fps,
        "num_outputs_per_prompt": bench_args.num_outputs_per_prompt,
        "seed": bench_args.seed,
        "profile": bench_args.profile,
        "num_profiled_timesteps": bench_args.num_profiled_timesteps,
        "profile_all_stages": bench_args.profile_all_stages,
    }
    if bench_args.disable_safety_checker:
        _sampling_params["safety_checker"] = None

    if manifest is None:
        all_prompts = [request.prompt for request in dataset_requests]
        all_request_ids = [request.request_id for request in dataset_requests]
        all_sampling_params = []
        for i, request in enumerate(dataset_requests):
            params = dict(_sampling_params)
            if bench_args.random_request_config:
                params.update(dataset.get_sampling_params(i))
            if request.image_paths:
                params["image_path"] = request.image_paths
            all_sampling_params.append(params)
    else:
        for params in all_sampling_params:
            defaults = dict(_sampling_params)
            defaults.update(params)
            params.clear()
            params.update(defaults)

    if bench_args.save_output_dir:
        output_dir = Path(bench_args.save_output_dir).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        for index, (request_id, params) in enumerate(
            zip(all_request_ids, all_sampling_params)
        ):
            safe_request_id = re.sub(r"[^A-Za-z0-9._-]+", "_", request_id).strip("._")
            safe_request_id = safe_request_id or "request"
            params.update(
                {
                    "output_path": str(output_dir),
                    "output_file_name": f"{index:05d}-{safe_request_id}",
                    "return_file_paths_only": True,
                    "save_output": True,
                }
            )

    if not bench_args.skip_warmup:
        logger.info("Running warmup batch...")
        warmup_count = min(bench_args.batch_size, total_count)
        warmup_prompts = all_prompts[:warmup_count]
        warmup_sampling_params = [
            {
                **p,
                "profile": False,
                "save_output": False,
                "return_file_paths_only": False,
                "output_path": None,
                "output_file_name": None,
            }
            for p in all_sampling_params[:warmup_count]
        ]
        generate_batch(
            engine,
            bench_args,
            warmup_prompts,
            warmup_sampling_params,
            request_ids=[
                f"warmup-{request_id}" for request_id in all_request_ids[:warmup_count]
            ],
        )

    logger.info(f"Running benchmark with {total_count} prompts...")
    outputs: List[BatchOutput] = []

    start_time = time.perf_counter()

    num_batches = (total_count + bench_args.batch_size - 1) // bench_args.batch_size
    pbar = tqdm(
        total=num_batches,
        disable=bench_args.disable_tqdm,
        desc="Benchmark",
    )

    for batch_start in range(0, total_count, bench_args.batch_size):
        batch_end = min(batch_start + bench_args.batch_size, total_count)
        batch_prompts = all_prompts[batch_start:batch_end]
        batch_sampling_params = all_sampling_params[batch_start:batch_end]
        batch_request_ids = all_request_ids[batch_start:batch_end]

        batch_output = generate_batch(
            engine,
            bench_args,
            batch_prompts,
            batch_sampling_params,
            request_ids=batch_request_ids,
        )
        outputs.append(batch_output)

        pbar.update(1)

    pbar.close()
    total_duration = time.perf_counter() - start_time

    resolution = (bench_args.width, bench_args.height, bench_args.num_frames)
    metrics = calculate_metrics(
        outputs,
        total_duration,
        resolution=resolution,
        num_requests=total_count,
        all_sampling_params=all_sampling_params,
    )

    display_results(
        metrics,
        bench_args,
        model_path=server_args.model_path,
    )

    if bench_args.output_file:
        save_results(metrics, bench_args, server_args, manifest=manifest)

    return metrics


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
    print_value_formatted(
        "Request Latency Mean (sec):", metrics["request_latency_mean_seconds"]
    )
    print_value_formatted(
        "Request Latency Median (sec):", metrics["request_latency_median_seconds"]
    )
    print_value_formatted("Peak Memory (MB):", metrics["peak_memory_mb"])
    print_divider(110, "=")


def save_results(
    metrics: Dict[str, Any],
    bench_args: BenchArgs,
    server_args: ServerArgs,
    manifest: Optional[LoadedRequestManifest] = None,
):
    """Save benchmark results to JSON file."""
    result = {
        "metadata": {
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "model_path": server_args.model_path,
            "task_type": bench_args.task_name,
            "backend": "engine",
            "request_manifest": manifest.path if manifest else None,
            "request_manifest_sha256": manifest.sha256 if manifest else None,
        },
        "configuration": {
            "num_inference_steps": bench_args.num_inference_steps,
            "guidance_scale": bench_args.guidance_scale,
            "seed": bench_args.seed,
            "batch_size": bench_args.batch_size,
            "num_prompts": metrics["num_requests"],
            "resolution": f"{bench_args.width}x{bench_args.height}x{bench_args.num_frames}",
            "dataset": "request_manifest" if manifest else bench_args.dataset,
            "save_output_dir": (
                str(Path(bench_args.save_output_dir).expanduser().resolve())
                if bench_args.save_output_dir
                else None
            ),
        },
        "results": metrics,
    }

    with open(bench_args.output_file, "a") as f:
        f.write(json.dumps(result) + "\n")

    logger.info(f"Results saved to {bench_args.output_file}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Offline throughput benchmark for multimodal generation models"
    )

    ServerArgs.add_cli_args(parser)
    BenchArgs.add_cli_args(parser)

    args, unknown_args = parser.parse_known_args()

    server_args = ServerArgs.from_cli_args(args, unknown_args)
    bench_args = BenchArgs.from_cli_args(args)

    set_global_server_args(server_args)

    result = throughput_test(server_args, bench_args)

    return result


if __name__ == "__main__":
    main()
