"""
Benchmark online serving for diffusion models (Image/Video Generation).


Usage:
    # launch a server and benchmark on it

    # T2V or T2I or any other multimodal generation model
    sglang serve Wan-AI/Wan2.2-T2V-A14B-Diffusers --num-gpus 1 --port 1231

    # benchmark it and make sure the port is the same as the server's port
    python3 -m sglang.multimodal_gen.benchmarks.bench_serving --dataset vbench --num-prompts 20 --port 1231
"""

import argparse
import asyncio
import json
import os
import time
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm

from sglang.multimodal_gen.benchmarks.datasets import (
    RandomDataset,
    RequestFuncInput,
    RequestFuncOutput,
    VBenchDataset,
)
from sglang.multimodal_gen.runtime.utils.logging_utils import (
    configure_logger,
    init_logger,
)
from sglang.multimodal_gen.test.test_utils import print_divider, print_value_formatted

logger = init_logger(__name__)


async def async_request_image_sglang(
    input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    output = RequestFuncOutput()
    output.start_time = time.perf_counter()

    # Check if we need to use multipart (for image edits with input images)
    if input.image_paths and len(input.image_paths) > 0:
        # Use multipart/form-data for image edits
        data = aiohttp.FormData()
        data.add_field("model", input.model)
        data.add_field("prompt", input.prompt)
        data.add_field("response_format", "b64_json")

        if input.width and input.height:
            data.add_field("size", f"{input.width}x{input.height}")

        # Merge extra parameters
        for key, value in input.extra_body.items():
            data.add_field(key, str(value))

        # Add image file(s)
        for idx, img_path in enumerate(input.image_paths):
            if os.path.exists(img_path):
                data.add_field(
                    "image",
                    open(img_path, "rb"),
                    filename=os.path.basename(img_path),
                    content_type="application/octet-stream",
                )
            else:
                output.error = f"Image file not found: {img_path}"
                output.success = False
                if pbar:
                    pbar.update(1)
                return output

        try:
            async with session.post(input.api_url, data=data) as response:
                if response.status == 200:
                    resp_json = await response.json()
                    output.response_body = resp_json
                    output.success = True
                    if "peak_memory_mb" in resp_json:
                        output.peak_memory_mb = resp_json["peak_memory_mb"]
                else:
                    output.error = f"HTTP {response.status}: {await response.text()}"
                    output.success = False
        except Exception as e:
            output.error = str(e)
            output.success = False
    else:
        # Use JSON for text-to-image generation
        payload = {
            "model": input.model,
            "prompt": input.prompt,
            "n": 1,
            "response_format": "b64_json",
        }

        if input.width and input.height:
            payload["size"] = f"{input.width}x{input.height}"

        # Merge extra parameters
        payload.update(input.extra_body)

        try:
            async with session.post(input.api_url, json=payload) as response:
                if response.status == 200:
                    resp_json = await response.json()
                    output.response_body = resp_json
                    output.success = True
                    if "peak_memory_mb" in resp_json:
                        output.peak_memory_mb = resp_json["peak_memory_mb"]
                else:
                    output.error = f"HTTP {response.status}: {await response.text()}"
                    output.success = False
        except Exception as e:
            output.error = str(e)
            output.success = False

    output.latency = time.perf_counter() - output.start_time

    if pbar:
        pbar.update(1)
    return output


async def async_request_video_sglang(
    input: RequestFuncInput,
    session: aiohttp.ClientSession,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    output = RequestFuncOutput()
    output.start_time = time.perf_counter()

    # 1. Submit Job
    job_id = None
    # Check if we need to upload images (Multipart) or just send JSON
    if input.image_paths and len(input.image_paths) > 0:
        # Use multipart/form-data
        data = aiohttp.FormData()
        data.add_field("model", input.model)
        data.add_field("prompt", input.prompt)

        if input.width and input.height:
            data.add_field("size", f"{input.width}x{input.height}")

        # Add extra body fields to form data if possible, or assume simple key-values
        # Note: Nested dicts in extra_body might need JSON serialization if API expects it stringified
        if input.extra_body:
            data.add_field("extra_body", json.dumps(input.extra_body))

        # Explicitly add fps/num_frames if they are not in extra_body (bench_serving logic overrides)
        if input.num_frames:
            data.add_field("num_frames", str(input.num_frames))
        if input.fps:
            data.add_field("fps", str(input.fps))

        # Add image file
        # Currently only support single image upload as 'input_reference' per API spec
        img_path = input.image_paths[0]
        if os.path.exists(img_path):
            data.add_field(
                "input_reference",
                open(img_path, "rb"),
                filename=os.path.basename(img_path),
                content_type="application/octet-stream",
            )
        else:
            output.error = f"Image file not found: {img_path}"
            output.success = False
            if pbar:
                pbar.update(1)
            return output

        try:
            async with session.post(input.api_url, data=data) as response:
                if response.status == 200:
                    resp_json = await response.json()
                    job_id = resp_json.get("id")
                else:
                    output.error = (
                        f"Submit failed HTTP {response.status}: {await response.text()}"
                    )
                    output.success = False
                    if pbar:
                        pbar.update(1)
                    return output
        except Exception as e:
            output.error = f"Submit exception: {str(e)}"
            output.success = False
            if pbar:
                pbar.update(1)
            return output

    else:
        # Use JSON
        payload: Dict[str, Any] = {
            "model": input.model,
            "prompt": input.prompt,
        }
        if input.width and input.height:
            payload["size"] = f"{input.width}x{input.height}"
        if input.num_frames:
            payload["num_frames"] = input.num_frames
        if input.fps:
            payload["fps"] = input.fps

        payload.update(input.extra_body)

        try:
            async with session.post(input.api_url, json=payload) as response:
                if response.status == 200:
                    resp_json = await response.json()
                    job_id = resp_json.get("id")
                else:
                    output.error = (
                        f"Submit failed HTTP {response.status}: {await response.text()}"
                    )
                    output.success = False
                    if pbar:
                        pbar.update(1)
                    return output
        except Exception as e:
            output.error = f"Submit exception: {str(e)}"
            output.success = False
            if pbar:
                pbar.update(1)
            return output

    if not job_id:
        output.error = "No job_id returned"
        output.success = False
        if pbar:
            pbar.update(1)
        return output

    # 2. Poll for completion
    # Assuming the API returns a 'status' field.
    # We construct the check URL. Assuming api_url is like .../v1/videos
    # The check url should be .../v1/videos/{id}
    check_url = f"{input.api_url}/{job_id}"

    while True:
        try:
            async with session.get(check_url) as response:
                if response.status == 200:
                    status_data = await response.json()
                    status = status_data.get("status")
                    if status == "completed":
                        output.success = True
                        output.response_body = status_data
                        if "peak_memory_mb" in status_data:
                            output.peak_memory_mb = status_data["peak_memory_mb"]
                        break
                    elif status == "failed":
                        output.success = False
                        output.error = f"Job failed: {status_data.get('error')}"
                        break
                    else:
                        # queued or processing
                        await asyncio.sleep(1.0)
                else:
                    output.success = False
                    output.error = (
                        f"Poll failed HTTP {response.status}: {await response.text()}"
                    )
                    break
        except Exception as e:
            output.success = False
            output.error = f"Poll exception: {str(e)}"
            break

    output.latency = time.perf_counter() - output.start_time

    if pbar:
        pbar.update(1)
    return output


def calculate_metrics(outputs: List[RequestFuncOutput], total_duration: float):
    success_outputs = [o for o in outputs if o.success]
    error_outputs = [o for o in outputs if not o.success]

    num_success = len(success_outputs)
    latencies = [o.latency for o in success_outputs]
    peak_memories = [o.peak_memory_mb for o in success_outputs if o.peak_memory_mb > 0]

    metrics = {
        "duration": total_duration,
        "completed_requests": num_success,
        "failed_requests": len(error_outputs),
        "throughput_qps": num_success / total_duration if total_duration > 0 else 0,
        "latency_mean": np.mean(latencies) if latencies else 0,
        "latency_median": np.median(latencies) if latencies else 0,
        "latency_p99": np.percentile(latencies, 99) if latencies else 0,
        "latency_p50": np.percentile(latencies, 50) if latencies else 0,
        "peak_memory_mb_max": max(peak_memories) if peak_memories else 0,
        "peak_memory_mb_mean": np.mean(peak_memories) if peak_memories else 0,
        "peak_memory_mb_median": np.median(peak_memories) if peak_memories else 0,
    }

    return metrics


def wait_for_service(base_url: str, timeout: int = 1200) -> None:
    logger.info(f"Waiting for service at {base_url}...")
    start_time = time.time()
    while True:
        try:
            # Try /health endpoint first
            resp = requests.get(f"{base_url}/health", timeout=1)
            if resp.status_code == 200:
                logger.info("Service is ready.")
                break
        except requests.exceptions.RequestException:
            pass

        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Service at {base_url} did not start within {timeout} seconds."
            )

        time.sleep(1)


async def benchmark(args):
    from huggingface_hub import model_info

    # Construct base_url if not provided
    if args.base_url is None:
        args.base_url = f"http://{args.host}:{args.port}"

    # Wait for service
    wait_for_service(args.base_url)

    # Fetch model info
    try:
        resp = requests.get(f"{args.base_url}/v1/model_info", timeout=5)
        if resp.status_code == 200:
            info = resp.json()
            if "model_path" in info and info["model_path"]:
                args.model = info["model_path"]
                logger.info(f"Updated model name from server: {args.model}")
    except Exception as e:
        logger.info(f"Failed to fetch model info: {e}. Using default: {args.model}")

    valid_tasks = (
        "text-to-video",
        "image-to-video",
        "video-to-video",
        "text-to-image",
        "image-to-image",
    )

    # Resolve task_name with priority: args.task > local config > HF pipeline_tag
    if args.task:
        task_name = args.task
        logger.info(f"Using task from --task: {task_name}")
    elif os.path.exists(args.model):
        config_path = os.path.join(args.model, "config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config = json.load(f)
            task_name = config.get("pipeline_tag", "text-to-image")
            logger.info(f"Inferred task from local config.json: {task_name}")
        else:
            task_name = "text-to-image"
            logger.info(f"No config.json found, defaulting task to: {task_name}")
    else:
        task_name = model_info(args.model).pipeline_tag
        logger.info(f"Inferred task from HuggingFace pipeline_tag: {task_name}")

    if task_name not in valid_tasks:
        raise ValueError(
            f"Task '{task_name}' is not a valid multimodal generation task. "
            f"Use --task to specify one of: {', '.join(valid_tasks)}"
        )

    if task_name in ("text-to-video", "image-to-video", "video-to-video"):
        api_url = f"{args.base_url}/v1/videos"
        request_func = async_request_video_sglang
    else:  # text-to-image or image-to-image
        api_url = (
            f"{args.base_url}/v1/images/edits"
            if task_name == "image-to-image"
            else f"{args.base_url}/v1/images/generations"
        )
        request_func = async_request_image_sglang

    setattr(args, "task_name", task_name)

    if args.dataset == "vbench":
        dataset = VBenchDataset(args, api_url, args.model)
    elif args.dataset == "random":
        dataset = RandomDataset(args, api_url, args.model)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    logger.info(f"Loading requests...")
    requests_list = dataset.get_requests()
    logger.info(f"Prepared {len(requests_list)} requests from {args.dataset} dataset.")

    # Limit concurrency
    if args.max_concurrency is not None:
        semaphore = asyncio.Semaphore(args.max_concurrency)
    else:
        semaphore = None

    async def limited_request_func(req, session, pbar):
        if semaphore:
            async with semaphore:
                return await request_func(req, session, pbar)
        else:
            return await request_func(req, session, pbar)

    # Run benchmark
    pbar = tqdm(total=len(requests_list), disable=args.disable_tqdm)

    async with aiohttp.ClientSession() as session:
        start_time = time.perf_counter()
        tasks = []
        for req in requests_list:
            if args.request_rate != float("inf"):
                # Poisson process: inter-arrival times follow exponential distribution
                interval = np.random.exponential(1.0 / args.request_rate)
                await asyncio.sleep(interval)

            task = asyncio.create_task(limited_request_func(req, session, pbar))
            tasks.append(task)

        outputs = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start_time

    pbar.close()

    # Calculate metrics
    metrics = calculate_metrics(outputs, total_duration)

    print("\n{s:{c}^{n}}".format(s=" Serving Benchmark Result ", n=60, c="="))

    # Section 1: Configuration
    print_value_formatted("Task:", task_name)
    print_value_formatted("Model:", args.model)
    print_value_formatted("Dataset:", args.dataset)

    # Section 2: Execution & Traffic
    print_divider(50)
    print_value_formatted("Benchmark duration (s):", metrics["duration"])
    print_value_formatted("Request rate:", str(args.request_rate))
    print_value_formatted(
        "Max request concurrency:",
        str(args.max_concurrency) if args.max_concurrency else "not set",
    )
    print_value_formatted(
        "Successful requests:",
        f"{metrics['completed_requests']}/{len(requests_list)}",
    )

    # Section 3: Performance Metrics
    print_divider(50)

    print_value_formatted("Request throughput (req/s):", metrics["throughput_qps"])

    print_value_formatted("Latency Mean (s):", metrics["latency_mean"])
    print_value_formatted("Latency Median (s):", metrics["latency_median"])
    print_value_formatted("Latency P99 (s):", metrics["latency_p99"])

    if metrics["peak_memory_mb_max"] > 0:
        print_divider(50)
        print_value_formatted("Peak Memory Max (MB):", metrics["peak_memory_mb_max"])
        print_value_formatted("Peak Memory Mean (MB):", metrics["peak_memory_mb_mean"])
        print_value_formatted(
            "Peak Memory Median (MB):", metrics["peak_memory_mb_median"]
        )

    print_divider(60)

    if args.output_file:
        with open(args.output_file, "w") as f:
            json.dump(metrics, f, indent=2)
        print(f"Metrics saved to {args.output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark serving for diffusion models."
    )
    parser.add_argument(
        "--backend",
        type=str,
        default=None,
        help="DEPRECATED: --task is deprecated and will be ignored. The task will be inferred from --model.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Base URL of the server (e.g., http://localhost:30000). Overrides host/port.",
    )
    parser.add_argument("--host", type=str, default="localhost", help="Server host.")
    parser.add_argument("--port", type=int, default=30000, help="Server port.")
    parser.add_argument("--model", type=str, default="default", help="Model name.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="vbench",
        choices=["vbench", "random"],
        help="Dataset to use.",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=[
            "text-to-video",
            "image-to-video",
            "text-to-image",
            "image-to-image",
            "video-to-video",
        ],
        default=None,
        help="The task will be inferred from huggingface pipeline_tag. When huggingface pipeline_tag is not provided, --task will be used.",
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=None,
        help="Path to local dataset file (optional).",
    )
    parser.add_argument(
        "--num-prompts", type=int, default=10, help="Number of prompts to benchmark."
    )
    parser.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent requests, default to `1`. This can be used "
        "to help simulate an environment where a higher level component "
        "is enforcing a maximum number of concurrent requests. While the "
        "--request-rate argument controls the rate at which requests are "
        "initiated, this argument will control how many are actually allowed "
        "to execute at a time. This means that when used in combination, the "
        "actual request rate may be lower than specified with --request-rate, "
        "if the server is not processing requests fast enough to keep up.",
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second. If this is inf, then all the requests are sent at time 0. "
        "Otherwise, we use Poisson process to synthesize the request arrival times. Default is inf.",
    )
    parser.add_argument("--width", type=int, default=None, help="Image/Video width.")
    parser.add_argument("--height", type=int, default=None, help="Image/Video height.")
    parser.add_argument(
        "--num-frames", type=int, default=None, help="Number of frames (for video)."
    )
    parser.add_argument("--fps", type=int, default=None, help="FPS (for video).")
    parser.add_argument(
        "--output-file", type=str, default=None, help="Output JSON file for metrics."
    )
    parser.add_argument(
        "--disable-tqdm", action="store_true", help="Disable progress bar."
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Log level.",
    )

    args = parser.parse_args()

    configure_logger(args)

    asyncio.run(benchmark(args))
