"""
Benchmark online serving with video inputs.

Usage:
# Default: 10s video, 1 request, connects to localhost:30000
python3 -m sglang.bench_serving_video --backend sglang --model <model-id>

# Custom video parameters
python3 -m sglang.bench_serving_video --backend sglang --model <model-id> \
    --video-seconds 20 --video-height 720 --video-width 1280 --video-fps 30

# Stress test (repeat the same video 10 times - Cache Hit)
python3 -m sglang.bench_serving_video --backend sglang --model <model-id> --num-prompts 10

# Stress test (use 10 DIFFERENT videos - Cache Miss)
python3 -m sglang.bench_serving_video --backend sglang --model <model-id> --num-prompts 10 --unique-video
"""

import argparse
import asyncio
import json
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass
from typing import Any, AsyncGenerator, Dict, List, Optional

import aiohttp
import numpy as np
from tqdm.asyncio import tqdm

# Check dependencies
try:
    import cv2
except ImportError:
    print("Error: 'opencv-python' is required for video generation.")
    print("Please install it via: pip install opencv-python")
    print(
        "If you are running in a Linux environment and encounter 'cv2.error: OpenCV(4.x.x) ...' related to missing shared libraries, you may need to run:"
    )
    print("    sudo apt-get update")
    print("    sudo apt-get install -y libgl1 libglib2.0-0")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("Error: 'numpy' is required for video generation.")
    print("Please install it via: pip install numpy")


@dataclass
class RequestFuncInput:
    prompt: str
    api_url: str
    model: str
    video_url: str
    output_len: int
    extra_request_body: Dict[str, Any]


@dataclass
class RequestFuncOutput:
    generated_text: str = ""
    success: bool = False
    latency: float = 0.0
    ttft: float = 0.0  # Time to first token
    output_len: int = 0
    error: str = ""

    @staticmethod
    def init_new(request_func_input: RequestFuncInput):
        return RequestFuncOutput()


@dataclass
class BenchmarkMetrics:
    completed: int
    total_input_video_seconds: float
    total_output_tokens: int
    request_throughput: float
    video_throughput_seconds: float  # Video seconds processed per second
    output_throughput: float
    mean_ttft_ms: float
    median_ttft_ms: float
    p99_ttft_ms: float
    mean_e2e_latency_ms: float
    median_e2e_latency_ms: float
    p99_e2e_latency_ms: float
    concurrency: float


def generate_synthetic_video(
    filename: str,
    duration_sec: float,
    width: int,
    height: int,
    fps: int,
    seed_index: int = 0,
):
    """
    Generates a synthetic video using OpenCV.

    Features for meaningful complexity:
    1. Background noise (prevents trivial spatial compression).
    2. Bouncing ball (provides motion vectors).
    3. Frame counter text (provides temporal uniqueness).
    """
    # Set random seed to ensure uniqueness if seed_index varies
    np.random.seed(seed_index)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(filename, fourcc, fps, (width, height))

    total_frames = int(duration_sec * fps)

    # Ball state
    cx, cy = width // 2, height // 2
    vx, vy = 5, 5
    radius = min(width, height) // 10

    # Pre-generate some noise to blend to save time, but shift it to simulate texture
    noise_base = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)

    for i in range(total_frames):
        # 1. Background: Blended noise (simple texture)
        # We roll the noise to create a "static" effect without regenerating full frame noise
        noise_base = np.roll(noise_base, shift=1, axis=0)
        frame = noise_base // 4  # Darken noise

        # 2. Bouncing Ball
        cx += vx
        cy += vy

        if cx - radius < 0 or cx + radius > width:
            vx = -vx
            cx += vx
        if cy - radius < 0 or cy + radius > height:
            vy = -vy
            cy += vy

        cv2.circle(frame, (cx, cy), radius, (0, 0, 255), -1)

        # 3. Text overlay (Frame count + Timestamp + ID)
        time_str = f"ID: {seed_index} | Time: {i/fps:.2f}s | Frame: {i}"
        cv2.putText(
            frame,
            time_str,
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        out.write(frame)

    out.release()


def _create_bench_client_session():
    timeout = aiohttp.ClientTimeout(total=6 * 60 * 60)
    return aiohttp.ClientSession(timeout=timeout)


async def async_request_openai_chat_completions(
    request_func_input: RequestFuncInput,
    pbar: Optional[tqdm] = None,
) -> RequestFuncOutput:
    api_url = request_func_input.api_url

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": request_func_input.prompt},
                {
                    "type": "video_url",
                    "video_url": {
                        "url": request_func_input.video_url,
                    },
                },
            ],
        }
    ]

    async with _create_bench_client_session() as session:
        payload = {
            "model": request_func_input.model,
            "messages": messages,
            "max_completion_tokens": request_func_input.output_len,
            "stream": False,  # We benchmark E2E latency mostly
            **request_func_input.extra_request_body,
        }

        output = RequestFuncOutput.init_new(request_func_input)
        st = time.perf_counter()

        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status == 200:
                    resp_json = await response.json()
                    output.generated_text = resp_json["choices"][0]["message"][
                        "content"
                    ]
                    output.output_len = resp_json.get("usage", {}).get(
                        "completion_tokens", 0
                    )

                    # Calculate TTFT/Latency
                    # For non-streaming, TTFT ~= Latency
                    output.latency = time.perf_counter() - st
                    output.ttft = output.latency
                    output.success = True
                else:
                    output.error = response.reason or str(response.status)
                    output.success = False
        except Exception as e:
            output.success = False
            output.error = str(e)

    if pbar:
        pbar.update(1)
    return output


def calculate_metrics(
    outputs: List[RequestFuncOutput],
    dur_s: float,
    total_input_video_seconds: float,
) -> BenchmarkMetrics:
    completed = 0
    output_tokens = 0
    ttfts = []
    latencies = []

    for o in outputs:
        if o.success:
            completed += 1
            output_tokens += o.output_len
            ttfts.append(o.ttft)
            latencies.append(o.latency)

    if completed == 0:
        return None

    return BenchmarkMetrics(
        completed=completed,
        total_input_video_seconds=total_input_video_seconds,
        total_output_tokens=output_tokens,
        request_throughput=completed / dur_s,
        video_throughput_seconds=total_input_video_seconds / dur_s,
        output_throughput=output_tokens / dur_s,
        mean_ttft_ms=np.mean(ttfts) * 1000,
        median_ttft_ms=np.median(ttfts) * 1000,
        p99_ttft_ms=np.percentile(ttfts, 99) * 1000,
        mean_e2e_latency_ms=np.mean(latencies) * 1000,
        median_e2e_latency_ms=np.median(latencies) * 1000,
        p99_e2e_latency_ms=np.percentile(latencies, 99) * 1000,
        concurrency=np.sum(latencies) / dur_s,
    )


async def benchmark(args):
    print(f"Benchmarking video serving with args: {args}")

    # 1. Generate Video(s)
    temp_dir = tempfile.mkdtemp()
    video_urls = []

    try:
        # Determine how many unique videos to generate
        num_videos_to_gen = args.num_prompts if args.unique_video else 1

        print(
            f"Generating {num_videos_to_gen} video file(s)... "
            f"({'Unique content' if args.unique_video else 'Reused content'})"
        )

        for i in range(num_videos_to_gen):
            filename = f"bench_video_{i}.mp4"
            video_path = os.path.join(temp_dir, filename)

            generate_synthetic_video(
                filename=video_path,
                duration_sec=args.video_seconds,
                width=args.video_width,
                height=args.video_height,
                fps=args.video_fps,
                seed_index=i,
            )

            abs_path = f"file://{os.path.abspath(video_path)}"
            video_urls.append(abs_path)

            if i == 0:
                # Print size of the first video as a reference
                try:
                    video_size_bytes = os.path.getsize(video_path)
                    video_size_mb = video_size_bytes / (1024 * 1024)
                    print(
                        f"Video size (sample): {video_size_bytes} bytes ({video_size_mb:.2f} MB)"
                    )
                except OSError:
                    pass

        # If not unique, repeat the first video URL for all requests
        if not args.unique_video:
            video_urls = [video_urls[0]] * args.num_prompts

        # 2. Prepare Requests
        api_url = f"http://{args.host}:{args.port}/v1/chat/completions"
        extra_body = (
            json.loads(args.extra_request_body) if args.extra_request_body else {}
        )

        prompts = []
        for i in range(args.num_prompts):
            inp = RequestFuncInput(
                prompt=f"Describe what is happening in this video (ID {i}) in detail.",
                api_url=api_url,
                model=args.model,
                video_url=video_urls[i],
                output_len=args.output_len,
                extra_request_body=extra_body,
            )
            prompts.append(inp)

        pbar = tqdm(total=args.num_prompts) if not args.disable_tqdm else None

        # Pre-warm (always use the first video)
        print("Warming up...")
        warmup_input = RequestFuncInput(
            prompt="Describe this video briefly.",
            api_url=api_url,
            model=args.model,
            video_url=video_urls[0],
            output_len=16,
            extra_request_body=extra_body,
        )
        await async_request_openai_chat_completions(warmup_input)
        print("Warmup done.")

        start_time = time.perf_counter()

        tasks = []
        async for inp in get_request(prompts, args.request_rate):
            tasks.append(
                asyncio.create_task(async_request_openai_chat_completions(inp, pbar))
            )

        outputs = await asyncio.gather(*tasks)
        end_time = time.perf_counter()

        if pbar:
            pbar.close()

        # 3. Calculate Metrics
        duration = end_time - start_time
        total_vid_sec = args.num_prompts * args.video_seconds

        metrics = calculate_metrics(outputs, duration, total_vid_sec)

        # 4. Report
        if metrics:
            print("\n" + "=" * 50)
            print(f" Video Serving Benchmark Result ({args.backend})")
            print("=" * 50)
            print(
                f"Mode:                 {'Unique Videos (Cache Miss)' if args.unique_video else 'Single Video (Cache Hit)'}"
            )
            print(f"Successful requests:      {metrics.completed}")
            print(f"Benchmark duration:       {duration:.2f} s")
            print(
                f"Total video input:        {metrics.total_input_video_seconds:.2f} s"
            )
            print(f"Total output tokens:      {metrics.total_output_tokens}")
            print("-" * 50)
            print(f"Request Throughput:       {metrics.request_throughput:.2f} req/s")
            print(
                f"Video Throughput:         {metrics.video_throughput_seconds:.2f} vid_sec/s"
            )
            print(f"Output Token Throughput:  {metrics.output_throughput:.2f} tok/s")
            print("-" * 50)
            print(f"Mean E2E Latency:         {metrics.mean_e2e_latency_ms:.2f} ms")
            print(f"Median E2E Latency:       {metrics.median_e2e_latency_ms:.2f} ms")
            print(f"P99 E2E Latency:          {metrics.p99_e2e_latency_ms:.2f} ms")
            print("=" * 50)

            # Dump to file
            result = {
                "backend": args.backend,
                "video_config": {
                    "seconds": args.video_seconds,
                    "resolution": f"{args.video_width}x{args.video_height}",
                    "fps": args.video_fps,
                    "unique_video": args.unique_video,
                },
                "num_prompts": args.num_prompts,
                "throughput_vid_sec": metrics.video_throughput_seconds,
                "output_throughput": metrics.output_throughput,
                "mean_latency_ms": metrics.mean_e2e_latency_ms,
            }

            if args.output_file:
                with open(args.output_file, "a") as f:
                    f.write(json.dumps(result) + "\n")
            return result
        else:
            print("All requests failed.")
            return None

    finally:
        # Cleanup
        pass
        shutil.rmtree(temp_dir, ignore_errors=True)


async def get_request(
    prompts: List[RequestFuncInput], request_rate: float
) -> AsyncGenerator[RequestFuncInput, None]:
    prompts_iter = iter(prompts)
    for request in prompts_iter:
        yield request

        if request_rate == float("inf"):
            continue

        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark video serving.")
    parser.add_argument("--backend", type=str, default="sglang", help="Backend name")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Server host")
    parser.add_argument("--port", type=int, default=30000, help="Server port")
    parser.add_argument(
        "--model", type=str, default="Qwen/Qwen2.5-VL-3B-Instruct", help="Model name"
    )
    parser.add_argument(
        "--num-prompts", type=int, default=1, help="Number of requests to send"
    )
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Number of requests per second.",
    )
    parser.add_argument("--output-len", type=int, default=128, help="Max output tokens")

    # Video Config
    parser.add_argument(
        "--video-seconds", type=float, default=10.0, help="Duration of generated video"
    )
    parser.add_argument("--video-width", type=int, default=1280, help="Video width")
    parser.add_argument("--video-height", type=int, default=720, help="Video height")
    parser.add_argument("--video-fps", type=int, default=30, help="Video FPS")
    parser.add_argument(
        "--unique-video",
        action="store_true",
        help="If set, generates a unique video for every prompt (disables caching).",
    )

    # Advanced
    parser.add_argument(
        "--extra-request-body",
        type=str,
        default=None,
        help="JSON string for extra body params",
    )
    parser.add_argument(
        "--output-file", type=str, default=None, help="File to save results"
    )
    parser.add_argument(
        "--disable-tqdm", action="store_true", help="Disable progress bar"
    )

    args = parser.parse_args()
    asyncio.run(benchmark(args))
