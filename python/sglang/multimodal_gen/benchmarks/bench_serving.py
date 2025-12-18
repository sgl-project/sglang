"""
Benchmark online serving for diffusion models (Image/Video Generation).


Usage:

    t2v:
    python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
         --backend sglang-image --dataset vbench --task t2v --num-prompts 20

    i2v:
    python3 -m sglang.multimodal_gen.benchmarks.bench_serving \
         --backend sglang-image --dataset vbench --task i2v --num-prompts 20




"""

import argparse
import asyncio
import json
import os
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import aiohttp
import numpy as np
import requests
from tqdm.asyncio import tqdm


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

    # Placeholder for I2V, usually requires local file mapping

    def __init__(self, args, api_url: str, model: str):
        super().__init__(args, api_url, model)
        self.items = self._load_data()

    def _load_data(self) -> List[Dict[str, Any]]:
        if self.args.task == "t2v":
            return self._load_t2v_prompts()
        elif self.args.task == "i2v":
            return self._load_i2v_data()
        elif self.args.task in ["ti2v", "ti2i"]:
            return self._load_i2v_data()  # Reuse logic for now
        else:
            # Default to T2V if task not specified or unknown
            return self._load_t2v_prompts()

    def _load_t2v_prompts(self) -> List[Dict[str, Any]]:
        path = self.args.dataset_path

        # If no path provided, try to use default VBench prompt file
        if not path:
            path = os.path.join(
                os.path.expanduser("~"),
                ".cache",
                "sglang",
                "vbench_subject_consistency.txt",
            )
            if not os.path.exists(path):
                print(f"Downloading VBench prompts to {path}...")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                try:
                    response = requests.get(self.T2V_PROMPT_URL)
                    response.raise_for_status()
                    with open(path, "w") as f:
                        f.write(response.text)
                except Exception as e:
                    print(f"Failed to download VBench prompts: {e}")
                    # Fallback to dummy prompts if download fails
                    return [{"prompt": "A cat sitting on a bench"}] * 50

        prompts = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    prompts.append({"prompt": line})

        return self._resize_data(prompts)

    def _load_i2v_data(self) -> List[Dict[str, Any]]:
        """
        Load I2V data from VBench I2V dataset.
        Expects dataset_path to be a directory containing the i2v-bench-info.json file, or a JSON file with image paths.
        If not provided, auto-downloads the VBench I2V dataset.
        """
        path = self.args.dataset_path

        # Auto-download VBench I2V dataset if path is not provided and task is i2v
        if not path and self.args.task == "i2v":
            cache_dir = os.path.join(
                os.path.expanduser("~"), ".cache", "sglang", "vbench_i2v"
            )
            vbench_i2v_dir = os.path.join(cache_dir, "vbench2_beta_i2v")
            info_json_path = os.path.join(vbench_i2v_dir, "data", "i2v-bench-info.json")

            if not os.path.exists(info_json_path):
                print(f"Downloading VBench I2V dataset to {cache_dir}...")
                try:
                    os.makedirs(cache_dir, exist_ok=True)

                    # Download the download_data.sh script from GitHub raw URL
                    script_url = "https://raw.githubusercontent.com/Vchitect/VBench/master/vbench2_beta_i2v/download_data.sh"
                    script_path = os.path.join(cache_dir, "download_data.sh")

                    print(f"Downloading download_data.sh from {script_url}...")
                    resp = requests.get(script_url)
                    resp.raise_for_status()

                    with open(script_path, "w") as f:
                        f.write(resp.text)

                    # Make the script executable and run it
                    import subprocess

                    os.chmod(script_path, 0o755)

                    print("Executing download_data.sh to fetch VBench I2V dataset...")
                    print("This may take a while as it downloads image data...")

                    # Run the script in the cache directory
                    result = subprocess.run(
                        ["bash", script_path],
                        cwd=cache_dir,
                        capture_output=True,
                        text=True,
                    )

                    if result.returncode != 0:
                        print(f"Download script failed: {result.stderr}")
                        raise RuntimeError(f"Failed to download VBench I2V dataset")

                    print(
                        f"Successfully downloaded VBench I2V dataset to {vbench_i2v_dir}"
                    )

                except Exception as e:
                    print(f"Failed to download VBench I2V dataset: {e}")
                    print("Please manually download following instructions at:")
                    print(
                        "https://github.com/Vchitect/VBench/tree/master/vbench2_beta_i2v#22-download"
                    )

            if os.path.exists(info_json_path):
                path = vbench_i2v_dir

        data = []

        # Load from i2v-bench-info.json if it exists
        if path:
            info_json = None
            if os.path.isdir(path):
                # Check for i2v-bench-info.json in the directory
                possible_json = os.path.join(path, "data", "i2v-bench-info.json")
                if os.path.exists(possible_json):
                    info_json = possible_json
            elif os.path.isfile(path) and path.endswith(".json"):
                info_json = path

            if info_json:
                try:
                    with open(info_json, "r") as f:
                        items = json.load(f)

                    # VBench I2V format: each item has file_name, caption, etc.
                    base_dir = (
                        os.path.dirname(info_json)
                        if os.path.isfile(info_json)
                        else path
                    )
                    origin_dir = (
                        os.path.join(base_dir, "origin")
                        if os.path.isdir(path)
                        else os.path.join(os.path.dirname(base_dir), "origin")
                    )

                    for item in items:
                        file_name = item.get("file_name", "")
                        caption = item.get("caption", "")
                        img_path = os.path.join(origin_dir, file_name)

                        if os.path.exists(img_path):
                            data.append({"prompt": caption, "image_path": img_path})
                        else:
                            print(f"Warning: Image not found: {img_path}")

                    print(f"Loaded {len(data)} I2V samples from VBench I2V dataset")
                    return self._resize_data(data)
                except Exception as e:
                    print(f"Failed to load i2v-bench-info.json: {e}")

        # Fallback: scan directory for images
        if path and os.path.isdir(path):
            import glob

            exts = ["*.jpg", "*.jpeg", "*.png", "*.webp"]
            files = []
            for ext in exts:
                files.extend(glob.glob(os.path.join(path, ext)))
                files.extend(glob.glob(os.path.join(path, ext.upper())))
                # Also check in origin subdirectory
                origin_dir = os.path.join(path, "data", "origin")
                if os.path.exists(origin_dir):
                    files.extend(glob.glob(os.path.join(origin_dir, ext)))
                    files.extend(glob.glob(os.path.join(origin_dir, ext.upper())))

            for f in files:
                # Use filename as prompt or empty
                prompt = os.path.splitext(os.path.basename(f))[0]
                data.append({"prompt": prompt, "image_path": f})

        if not data:
            print(
                "No I2V data found or provided. Using dummy placeholders (requires manual image path setup)."
            )
            # Dummy data - this will likely fail if the image file doesn't exist
            # User must provide a valid image path for I2V to work
            dummy_image = "dummy_image.jpg"
            if not os.path.exists(dummy_image):
                # Create a blank dummy image for testing
                try:
                    from PIL import Image

                    img = Image.new("RGB", (100, 100), color="red")
                    img.save(dummy_image)
                    print(f"Created dummy image at {dummy_image}")
                except ImportError:
                    print("PIL not installed, cannot create dummy image.")

            data = [{"prompt": "A moving cat", "image_path": dummy_image}] * 10

        return self._resize_data(data)

    def _resize_data(self, data):
        if self.args.num_prompts:
            if len(data) < self.args.num_prompts:
                factor = (self.args.num_prompts // len(data)) + 1
                data = data * factor
            data = data[: self.args.num_prompts]
        return data

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

    metrics = {
        "duration": total_duration,
        "completed_requests": num_success,
        "failed_requests": len(error_outputs),
        "throughput_qps": num_success / total_duration if total_duration > 0 else 0,
        "latency_mean": np.mean(latencies) if latencies else 0,
        "latency_median": np.median(latencies) if latencies else 0,
        "latency_p99": np.percentile(latencies, 99) if latencies else 0,
        "latency_p50": np.percentile(latencies, 50) if latencies else 0,
    }

    return metrics


def wait_for_service(base_url: str, timeout: int = 120) -> None:
    print(f"Waiting for service at {base_url}...")
    start_time = time.time()
    while True:
        try:
            # Try /health endpoint first
            resp = requests.get(f"{base_url}/health", timeout=1)
            if resp.status_code == 200:
                print("Service is ready.")
                break
        except requests.exceptions.RequestException:
            pass

        if time.time() - start_time > timeout:
            raise TimeoutError(
                f"Service at {base_url} did not start within {timeout} seconds."
            )

        time.sleep(1)


async def benchmark(args):
    # Construct base_url if not provided
    if args.base_url is None:
        args.base_url = f"http://{args.host}:{args.port}"

    # Wait for service
    wait_for_service(args.base_url)

    # Setup dataset
    if args.backend == "sglang-image":
        if args.task == "i2v":
            api_url = f"{args.base_url}/v1/images/edits"
        else:
            api_url = f"{args.base_url}/v1/images/generations"
        request_func = async_request_image_sglang
    elif args.backend == "sglang-video":
        api_url = f"{args.base_url}/v1/videos"
        request_func = async_request_video_sglang
    else:
        raise ValueError(f"Unknown backend: {args.backend}")

    if args.dataset == "vbench":
        dataset = VBenchDataset(args, api_url, args.model)
    elif args.dataset == "random":
        dataset = RandomDataset(args, api_url, args.model)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    requests_list = dataset.get_requests()
    print(f"Prepared {len(requests_list)} requests from {args.dataset} dataset.")

    # Limit concurrency
    semaphore = asyncio.Semaphore(args.max_concurrency)

    async def limited_request_func(req, session, pbar):
        async with semaphore:
            return await request_func(req, session, pbar)

    # Run benchmark
    pbar = tqdm(total=len(requests_list), disable=args.disable_tqdm)

    async with aiohttp.ClientSession() as session:
        start_time = time.perf_counter()
        tasks = []
        for req in requests_list:
            if args.request_rate != float("inf"):
                # Simple rate limiting
                interval = 1.0 / args.request_rate
                await asyncio.sleep(interval)

            task = asyncio.create_task(limited_request_func(req, session, pbar))
            tasks.append(task)

        outputs = await asyncio.gather(*tasks)
        total_duration = time.perf_counter() - start_time

    pbar.close()

    # Calculate metrics
    metrics = calculate_metrics(outputs, total_duration)

    print("\n" + "=" * 40)
    print("Benchmark Results")
    print("=" * 40)
    print(f"Backend: {args.backend}")
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Total Duration: {metrics['duration']:.2f} s")
    print(f"Throughput: {metrics['throughput_qps']:.2f} req/s")
    print(f"Success Rate: {metrics['completed_requests']}/{len(requests_list)}")
    print(f"Latency Mean: {metrics['latency_mean']:.4f} s")
    print(f"Latency Median: {metrics['latency_median']:.4f} s")
    print(f"Latency P99: {metrics['latency_p99']:.4f} s")
    print("=" * 40)

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
        required=True,
        choices=["sglang-image", "sglang-video"],
        help="Backend type.",
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
        default="t2v",
        choices=["t2v", "i2v", "ti2v", "ti2i"],
        help="Task type.",
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
        "--max-concurrency", type=int, default=10, help="Maximum concurrent requests."
    )
    parser.add_argument(
        "--request-rate", type=float, default=float("inf"), help="Request rate (req/s)."
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

    args = parser.parse_args()

    asyncio.run(benchmark(args))
