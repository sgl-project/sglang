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
import glob
import json
import os
import re
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm

from sglang.multimodal_gen.runtime.utils.logging_utils import (
    configure_logger,
    init_logger,
)

logger = init_logger(__name__)


def is_dir_not_empty(path):
    return os.path.isdir(path) and bool(os.listdir(path))


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    model: str
    width: Optional[int] = None
    height: Optional[int] = None
    num_frames: Optional[int] = None
    fps: Optional[int] = None
    extra_body: Dict[str, Any] = field(default_factory=dict)
    image_paths: Optional[List[str]] = None
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class RequestFuncOutput:
    success: bool = False
    latency: float = 0.0
    error: str = ""
    start_time: float = 0.0
    response_body: Dict[str, Any] = field(default_factory=dict)
    peak_memory_mb: float = 0.0


class BaseDataset(ABC):
    def __init__(self, args, api_url: str, model: str):
        self.args = args
        self.api_url = api_url
        self.model = model

    @abstractmethod
    def __len__(self) -> int:
        pass

    @abstractmethod
    def __getitem__(self, idx: int) -> RequestFuncInput:
        pass

    @abstractmethod
    def get_requests(self) -> List[RequestFuncInput]:
        pass


class VBenchDataset(BaseDataset):
    """
    Dataset loader for VBench prompts.
    Supports t2v, i2v.
    """

    T2V_PROMPT_URL = "https://raw.githubusercontent.com/Vchitect/VBench/master/prompts/prompts_per_dimension/subject_consistency.txt"
    I2V_DOWNLOAD_SCRIPT_URL = "https://raw.githubusercontent.com/Vchitect/VBench/master/vbench2_beta_i2v/download_data.sh"

    def __init__(self, args, api_url: str, model: str):
        super().__init__(args, api_url, model)
        self.cache_dir = os.path.join(os.path.expanduser("~"), ".cache", "sglang")
        self.items = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        if self.args.task_name in ("text-to-video", "text-to-image", "video-to-video"):
            return self._load_t2v_prompts()
        elif self.args.task_name in ("image-to-video", "image-to-image"):
            return self._load_i2v_data()
        else:
            raise ValueError(
                f"Illegal task name is found in VBenchDataset {args.task_name}"
            )

    def _download_file(self, url: str, dest_path: str) -> None:
        """Download a file from URL to destination path."""
        os.makedirs(os.path.dirname(dest_path), exist_ok=True)
        resp = requests.get(url)
        resp.raise_for_status()
        with open(dest_path, "w") as f:
            f.write(resp.text)

    def _load_t2v_prompts(self) -> List[Dict[str, Any]]:
        path = self.args.dataset_path

        if not path:
            path = os.path.join(self.cache_dir, "vbench_subject_consistency.txt")
            if not os.path.exists(path):
                logger.info(f"Downloading VBench T2V prompts to {path}...")
                try:
                    self._download_file(self.T2V_PROMPT_URL, path)
                except Exception as e:
                    logger.info(f"Failed to download VBench prompts: {e}")
                    return [{"prompt": "A cat sitting on a bench"}] * 50

        prompts = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append({"prompt": line})

        return self._resize_data(prompts)

    def _auto_download_i2v_dataset(self) -> str:
        """Auto-download VBench I2V dataset and return the dataset directory."""
        vbench_i2v_dir = os.path.join(self.cache_dir, "vbench_i2v", "vbench2_beta_i2v")
        info_json_path = os.path.join(vbench_i2v_dir, "data", "i2v-bench-info.json")
        crop_dir = os.path.join(vbench_i2v_dir, "data", "crop")
        origin_dir = os.path.join(vbench_i2v_dir, "data", "origin")

        if (
            os.path.exists(info_json_path)
            and is_dir_not_empty(crop_dir)
            and is_dir_not_empty(origin_dir)
        ):
            return vbench_i2v_dir

        logger.info(f"Downloading VBench I2V dataset to {vbench_i2v_dir}...")
        try:
            cache_root = os.path.join(self.cache_dir, "vbench_i2v")
            script_path = os.path.join(cache_root, "download_data.sh")

            self._download_file(self.I2V_DOWNLOAD_SCRIPT_URL, script_path)
            os.chmod(script_path, 0o755)

            logger.info("Executing download_data.sh (this may take a while)...")
            import subprocess

            result = subprocess.run(
                ["bash", script_path],
                cwd=cache_root,
                capture_output=True,
                text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"Download script failed: {result.stderr}")
            missing_packages = re.findall(r"(\S+): command not found", result.stderr)
            if missing_packages:
                missing_packages = list(set(missing_packages))
                package_list = ", ".join(f"'{cmd}'" for cmd in missing_packages)
                raise RuntimeError(
                    f"Download script failed because the following commands are not installed: {package_list}.\n"
                    "Please install them (e.g., on Ubuntu: `sudo apt install ...`) and try again."
                )
            logger.info(
                f"Successfully downloaded VBench I2V dataset to {vbench_i2v_dir}"
            )
        except Exception as e:
            logger.info(f"Failed to download VBench I2V dataset: {e}")
            logger.info("Please manually download following instructions at:")
            logger.info(
                "https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v#22-download"
            )
            return None

        return vbench_i2v_dir if os.path.exists(info_json_path) else None

    def _load_from_i2v_json(self, json_path: str) -> List[Dict[str, Any]]:
        """Load I2V data from i2v-bench-info.json format."""
        with open(json_path, "r") as f:
            items = json.load(f)

        base_dir = os.path.dirname(
            os.path.dirname(json_path)
        )  # Go up to vbench2_beta_i2v
        origin_dir = os.path.join(base_dir, "data", "origin")

        data = []
        for item in items:
            img_path = os.path.join(origin_dir, item.get("file_name", ""))
            if os.path.exists(img_path):
                data.append({"prompt": item.get("caption", ""), "image_path": img_path})
            else:
                logger.warning(f"Image not found: {img_path}")

        logger.info(f"Loaded {len(data)} I2V samples from VBench I2V dataset")
        return data

    def _scan_directory_for_images(self, path: str) -> List[Dict[str, Any]]:
        """Scan directory for image files."""
        exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
        files = []

        for ext in exts:
            files.extend(glob.glob(os.path.join(path, ext)))
            files.extend(glob.glob(os.path.join(path, ext.upper())))

            # Also check in data/origin subdirectory
            origin_dir = os.path.join(path, "data", "origin")
            if os.path.exists(origin_dir):
                files.extend(glob.glob(os.path.join(origin_dir, ext)))
                files.extend(glob.glob(os.path.join(origin_dir, ext.upper())))

        return [
            {"prompt": os.path.splitext(os.path.basename(f))[0], "image_path": f}
            for f in files
        ]

    def _create_dummy_data(self) -> List[Dict[str, Any]]:
        """Create dummy data with a placeholder image in cache directory."""
        logger.info("No I2V data found. Using dummy placeholders.")

        dummy_image = os.path.join(self.cache_dir, "dummy_image.jpg")
        if not os.path.exists(dummy_image):
            try:
                from PIL import Image

                os.makedirs(self.cache_dir, exist_ok=True)
                img = Image.new("RGB", (100, 100), color="red")
                img.save(dummy_image)
                logger.info(f"Created dummy image at {dummy_image}")
            except ImportError:
                logger.info("PIL not installed, cannot create dummy image.")
                return []

        return [{"prompt": "A moving cat", "image_path": dummy_image}] * 10

    def _load_i2v_data(self) -> List[Dict[str, Any]]:
        """Load I2V data from VBench I2V dataset or user-provided path."""
        path = self.args.dataset_path
        # Auto-download if no path provided
        if not path:
            path = self._auto_download_i2v_dataset()
            if not path:
                return self._resize_data(self._create_dummy_data())

        # Try to load from i2v-bench-info.json
        info_json_candidates = [
            os.path.join(path, "data", "i2v-bench-info.json"),
            path if path.endswith(".json") else None,
        ]

        for json_path in info_json_candidates:
            if json_path and os.path.exists(json_path):
                try:
                    return self._resize_data(self._load_from_i2v_json(json_path))
                except Exception as e:
                    logger.info(f"Failed to load {json_path}: {e}")

        # Fallback: scan directory for images
        if os.path.isdir(path):
            data = self._scan_directory_for_images(path)
            if data:
                return self._resize_data(data)

        # Last resort: dummy data
        return self._resize_data(self._create_dummy_data())

    def _resize_data(self, data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Resize data to match num_prompts."""
        if not self.args.num_prompts:
            return data

        if len(data) < self.args.num_prompts:
            factor = (self.args.num_prompts // len(data)) + 1
            data = data * factor

        return data[: self.args.num_prompts]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> RequestFuncInput:
        item = self.items[idx]
        image_paths = [item["image_path"]] if "image_path" in item else None

        return RequestFuncInput(
            prompt=item.get("prompt", ""),
            api_url=self.api_url,
            model=self.model,
            width=self.args.width,
            height=self.args.height,
            num_frames=self.args.num_frames,
            fps=self.args.fps,
            image_paths=image_paths,
        )

    def get_requests(self) -> List[RequestFuncInput]:
        return [self[i] for i in range(len(self))]


class RandomDataset(BaseDataset):
    def __init__(self, args, api_url: str, model: str):
        self.args = args
        self.api_url = api_url
        self.model = model
        self.num_prompts = args.num_prompts or 100

    def __len__(self) -> int:
        return self.num_prompts

    def __getitem__(self, idx: int) -> RequestFuncInput:
        return RequestFuncInput(
            prompt=f"Random prompt {idx} for benchmarking diffusion models",
            api_url=self.api_url,
            model=self.model,
            width=self.args.width,
            height=self.args.height,
            num_frames=self.args.num_frames,
            fps=self.args.fps,
        )

    def get_requests(self) -> List[RequestFuncInput]:
        return [self[i] for i in range(len(self))]


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
        payload = {
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

    task_name = model_info(args.model).pipeline_tag

    if args.task != task_name:
        logger.warning(
            f"Task from args {args.task} is different from huggingface pipeline_tag {task_name}, args.task will be ignored!"
        )

    if task_name in ("text-to-video", "image-to-video", "video-to-video"):
        api_url = f"{args.base_url}/v1/videos"
        request_func = async_request_video_sglang
    elif task_name in ("text-to-image", "image-to-image"):
        if task_name == "image-to-image":
            api_url = f"{args.base_url}/v1/images/edits"
        else:
            api_url = f"{args.base_url}/v1/images/generations"
        request_func = async_request_image_sglang
    else:
        raise ValueError(
            f"The task name {task_name} of model {args.model} is not a valid task name for multimodal generation. Please check the model path."
        )

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
    print("{:<40} {:<15}".format("Task:", task_name))
    print("{:<40} {:<15}".format("Model:", args.model))
    print("{:<40} {:<15}".format("Dataset:", args.dataset))

    # Section 2: Execution & Traffic
    print(f"{'-' * 50}")
    print("{:<40} {:<15.2f}".format("Benchmark duration (s):", metrics["duration"]))
    print("{:<40} {:<15}".format("Request rate:", str(args.request_rate)))
    print(
        "{:<40} {:<15}".format(
            "Max request concurrency:",
            str(args.max_concurrency) if args.max_concurrency else "not set",
        )
    )
    print(
        "{:<40} {}/{:<15}".format(
            "Successful requests:", metrics["completed_requests"], len(requests_list)
        )
    )

    # Section 3: Performance Metrics
    print(f"{'-' * 50}")

    print(
        "{:<40} {:<15.2f}".format(
            "Request throughput (req/s):", metrics["throughput_qps"]
        )
    )
    print("{:<40} {:<15.4f}".format("Latency Mean (s):", metrics["latency_mean"]))
    print("{:<40} {:<15.4f}".format("Latency Median (s):", metrics["latency_median"]))
    print("{:<40} {:<15.4f}".format("Latency P99 (s):", metrics["latency_p99"]))

    if metrics["peak_memory_mb_max"] > 0:
        print(f"{'-' * 50}")
        print(
            "{:<40} {:<15.2f}".format(
                "Peak Memory Max (MB):", metrics["peak_memory_mb_max"]
            )
        )
        print(
            "{:<40} {:<15.2f}".format(
                "Peak Memory Mean (MB):", metrics["peak_memory_mb_mean"]
            )
        )
        print(
            "{:<40} {:<15.2f}".format(
                "Peak Memory Median (MB):", metrics["peak_memory_mb_median"]
            )
        )

    print("=" * 60)

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
