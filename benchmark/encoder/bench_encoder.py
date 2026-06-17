"""
Encoder Benchmark Script (QPS Mode)

Benchmark encoder performance in EPD separation mode.
Supports mooncake (encode-only) and zmq_to_tokenizer (encode + TCP send) backends.
Supports image, audio, and video modalities.

Usage:
    # Image benchmark @ 20 QPS (default)
    python bench_encoder.py --encoder-url http://localhost:30000 --qps 20 --duration 60

    # Audio benchmark @ 10 QPS
    python bench_encoder.py --encoder-url http://localhost:30000 --qps 10 --duration 60 \
        --modality audio --audio-duration 2.0

    # Video benchmark @ 5 QPS (URL required)
    python bench_encoder.py --encoder-url http://localhost:30000 --qps 5 --duration 60 \
        --modality video --video-url https://example.com/video.mp4

    # ZMQ with mock receiver @ 10 QPS
    python bench_encoder.py --encoder-url http://localhost:30000 --backend zmq \
        --receiver-url tcp://127.0.0.1:12345 --qps 10 --duration 60

    # Using image/audio/video URL instead of random generation
    python bench_encoder.py --encoder-url http://localhost:30000 --qps 20 \
        --image-url https://example.com/image.jpg
    python bench_encoder.py --encoder-url http://localhost:30000 --qps 10 \
        --modality audio --audio-url https://example.com/audio.wav
    python bench_encoder.py --encoder-url http://localhost:30000 --qps 5 \
        --modality video --video-url https://example.com/video.mp4
"""

import argparse
import asyncio
import base64
import io
import time
import uuid
import wave
from dataclasses import dataclass, field
from typing import List, Optional, Set

import aiohttp
import numpy as np
from PIL import Image

# Try to import rich for better output, fall back to basic print
try:
    from rich.console import Console
    from rich.table import Table

    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False


@dataclass
class RequestResult:
    """Result of a single request."""

    req_id: str
    success: bool
    latency_ms: float
    embedding_len: Optional[int] = None
    embedding_dim: Optional[int] = None
    error: Optional[str] = None
    send_time: float = 0.0
    complete_time: float = 0.0


@dataclass
class BenchmarkStats:
    """Aggregated benchmark statistics."""

    results: List[RequestResult] = field(default_factory=list)
    start_time: float = 0.0
    end_time: float = 0.0
    send_end_time: float = 0.0

    @property
    def total_requests(self) -> int:
        return len(self.results)

    @property
    def successful_requests(self) -> int:
        return sum(1 for r in self.results if r.success)

    @property
    def failed_requests(self) -> int:
        return sum(1 for r in self.results if not r.success)

    @property
    def latencies(self) -> np.ndarray:
        return np.array(
            [r.latency_ms for r in self.results if r.success], dtype=np.float64
        )

    @property
    def actual_qps(self) -> float:
        window_end = self.send_end_time if self.send_end_time > 0 else self.end_time
        duration = window_end - self.start_time
        if duration <= 0:
            return 0.0
        return self.successful_requests / duration


@dataclass
class ImageConfig:
    size: int = 448
    num_per_request: int = 1
    num_unique: int = 32
    urls: Optional[List[str]] = None


@dataclass
class AudioConfig:
    duration: float = 1.0
    sample_rate: int = 24000
    num_per_request: int = 1
    num_unique: int = 32
    urls: Optional[List[str]] = None


@dataclass
class VideoConfig:
    urls: Optional[List[str]] = None


@dataclass
class BenchmarkConfig:
    """Per-run benchmark configuration. Modality-specific subsections are only
    read when `modality` matches — e.g. `audio` fields are ignored when
    modality != "audio"."""

    encoder_url: str
    qps: float
    duration: float
    warmup: float
    modality: str
    backend: str
    receiver_url: Optional[str] = None
    image: ImageConfig = field(default_factory=ImageConfig)
    audio: AudioConfig = field(default_factory=AudioConfig)
    video: VideoConfig = field(default_factory=VideoConfig)


def generate_random_image(size: int) -> str:
    """Generate a random RGB image and return as base64 string."""
    arr = np.random.randint(0, 256, size=(size, size, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buffer = io.BytesIO()
    img.save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode("utf-8")


def generate_random_audio(duration: float = 1.0, sample_rate: int = 24000) -> str:
    """Generate a random WAV audio and return as base64 data URI."""
    num_samples = int(duration * sample_rate)
    audio_data = np.random.randint(
        -32768, 32768, size=num_samples, dtype=np.int16
    ).tobytes()

    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav_file:
        wav_file.setnchannels(1)  # mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_data)
    return (
        f"data:audio/wav;base64,{base64.b64encode(buffer.getvalue()).decode('utf-8')}"
    )


class MMItemProvider:
    """Supplies the ``mm_items`` list for each request.

    URL inputs are reused verbatim every request (the URLs are explicit). For
    random inputs a pool of unique items is pre-generated once and rotated
    per request, so consecutive requests send different payloads instead of one
    identical payload — which would be unrepresentative and would hit the
    encoder's multimodal cache (``--enable-mm-global-cache``) after the first
    request. Pre-generating the pool keeps the per-request hot path free of
    CPU-heavy media generation.
    """

    def __init__(self, pool: List[str], num_per_request: int, rotate: bool):
        self._pool = pool
        self._num = max(1, num_per_request)
        self._rotate = rotate and len(pool) > 0
        self._cursor = 0

    def next(self) -> List[str]:
        if not self._rotate:
            return self._pool
        n = len(self._pool)
        items = [self._pool[(self._cursor + i) % n] for i in range(self._num)]
        self._cursor = (self._cursor + self._num) % n
        return items


def _parse_receiver_url(receiver_url: str) -> tuple[str, int]:
    """Parse tcp://host:port into (host, port). Accepts bare host:port too."""
    url = receiver_url.replace("tcp://", "")
    if ":" not in url:
        raise ValueError(
            f"Invalid receiver URL format: {receiver_url}. Expected tcp://host:port"
        )
    host, port_str = url.rsplit(":", 1)
    return host, int(port_str)


_DEFAULT_INPUT_TEXT = {
    "image": "Describe this image.",
    "audio": "Transcribe this audio.",
    "video": "Describe this video.",
}


def build_request_payload(
    req_id: str,
    mm_items: List[str],
    backend: str,
    modality: str = "image",
    prefill_host: Optional[str] = None,
    embedding_port: Optional[int] = None,
) -> dict:
    """Build the request payload for /encode endpoint."""
    payload = {
        "mm_items": mm_items,
        "req_id": req_id,
        "num_parts": 1,
        "part_idx": 0,
        "modality": modality,
        "input_text": _DEFAULT_INPUT_TEXT.get(modality, ""),
    }

    if backend == "zmq" and prefill_host is not None:
        payload["prefill_host"] = prefill_host
        payload["embedding_port"] = embedding_port
    else:
        # Mooncake mode: no send needed, just encode
        payload["prefill_host"] = None
        payload["embedding_port"] = None

    return payload


async def send_request(
    session: aiohttp.ClientSession,
    encoder_url: str,
    payload: dict,
    timeout: float = 300.0,
) -> RequestResult:
    """Send a single encode request."""
    req_id = payload["req_id"]
    send_time = time.perf_counter()

    try:
        async with session.post(
            f"{encoder_url}/encode",
            json=payload,
            timeout=aiohttp.ClientTimeout(total=timeout),
        ) as response:
            complete_time = time.perf_counter()
            latency_ms = (complete_time - send_time) * 1000

            if response.status == 200:
                result = await response.json()
                return RequestResult(
                    req_id=req_id,
                    success=True,
                    latency_ms=latency_ms,
                    embedding_len=result.get("embedding_len") if result else None,
                    embedding_dim=result.get("embedding_dim") if result else None,
                    send_time=send_time,
                    complete_time=complete_time,
                )
            else:
                error_text = await response.text()
                return RequestResult(
                    req_id=req_id,
                    success=False,
                    latency_ms=latency_ms,
                    error=f"HTTP {response.status}: {error_text[:200]}",
                    send_time=send_time,
                    complete_time=complete_time,
                )
    except asyncio.TimeoutError:
        return RequestResult(
            req_id=req_id,
            success=False,
            latency_ms=(time.perf_counter() - send_time) * 1000,
            error="Request timeout",
            send_time=send_time,
            complete_time=time.perf_counter(),
        )
    except Exception as e:
        return RequestResult(
            req_id=req_id,
            success=False,
            latency_ms=(time.perf_counter() - send_time) * 1000,
            error=str(e),
            send_time=send_time,
            complete_time=time.perf_counter(),
        )


async def run_benchmark(config: BenchmarkConfig) -> BenchmarkStats:
    """Run the benchmark with QPS-based load."""
    stats = BenchmarkStats()
    interval = 1.0 / config.qps  # Time between requests
    task_refs: Set[asyncio.Task] = set()  # GC anchor; tasks removed in callback
    collected_results: List[RequestResult] = []
    completed_count = 0

    # Parse receiver URL once; surfaces malformed values at startup.
    prefill_host: Optional[str] = None
    embedding_port: Optional[int] = None
    if config.backend == "zmq" and config.receiver_url:
        prefill_host, embedding_port = _parse_receiver_url(config.receiver_url)

    def _on_task_done(task: asyncio.Task) -> None:
        nonlocal completed_count
        completed_count += 1
        task_refs.discard(task)
        try:
            collected_results.append(task.result())
        except Exception as e:
            collected_results.append(
                RequestResult(
                    req_id="unknown",
                    success=False,
                    latency_ms=0,
                    error=str(e),
                )
            )

    # Prepare the mm_items provider based on modality.
    if config.modality == "video":
        video = config.video
        if not video.urls:
            raise ValueError("--video-url is required for video modality")
        print(f"Using {len(video.urls)} video URL(s)...")
        provider = MMItemProvider(video.urls, num_per_request=1, rotate=False)
        print("Video preparation complete.")
    elif config.modality == "audio":
        audio = config.audio
        if audio.urls:
            print(f"Using {len(audio.urls)} audio URL(s)...")
            provider = MMItemProvider(audio.urls, num_per_request=1, rotate=False)
        else:
            pool_size = max(audio.num_unique, audio.num_per_request)
            print(
                f"Generating {pool_size} unique random audio(s) "
                f"({audio.duration}s @ {audio.sample_rate}Hz), "
                f"{audio.num_per_request}/request..."
            )
            pool = [
                generate_random_audio(audio.duration, audio.sample_rate)
                for _ in range(pool_size)
            ]
            provider = MMItemProvider(
                pool, num_per_request=audio.num_per_request, rotate=True
            )
        print("Audio preparation complete.")
    else:
        # image modality (default)
        image = config.image
        if image.urls:
            print(f"Using {len(image.urls)} image URL(s)...")
            provider = MMItemProvider(image.urls, num_per_request=1, rotate=False)
        else:
            pool_size = max(image.num_unique, image.num_per_request)
            print(
                f"Generating {pool_size} unique random image(s) of size "
                f"{image.size}x{image.size}, {image.num_per_request}/request..."
            )
            pool = [
                f"data:image/jpeg;base64,{generate_random_image(image.size)}"
                for _ in range(pool_size)
            ]
            provider = MMItemProvider(
                pool, num_per_request=image.num_per_request, rotate=True
            )
        print("Image preparation complete.")

    connector = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup phase
        if config.warmup > 0:
            print(f"\nWarmup phase ({config.warmup}s)...")
            warmup_start = time.perf_counter()
            warmup_count = 0
            while time.perf_counter() - warmup_start < config.warmup:
                req_id = f"warmup_{warmup_count}"
                payload = build_request_payload(
                    req_id,
                    provider.next(),
                    config.backend,
                    config.modality,
                    prefill_host,
                    embedding_port,
                )
                task = asyncio.create_task(
                    send_request(session, config.encoder_url, payload)
                )
                task_refs.add(task)
                task.add_done_callback(_on_task_done)
                warmup_count += 1
                await asyncio.sleep(interval)

            # Wait for all warmup tasks to finish, then reset for main phase.
            while completed_count < warmup_count:
                await asyncio.sleep(0.01)
            task_refs.clear()
            collected_results.clear()
            completed_count = 0
            print(f"Warmup complete ({warmup_count} requests)")

        # Main benchmark phase
        print(f"\nBenchmark phase ({config.duration}s @ {config.qps} QPS)...")
        stats.start_time = time.perf_counter()
        request_count = 0
        next_send_time = stats.start_time
        sending_complete = False
        total_to_send = int(config.duration * config.qps)
        progress_interval = 0.25
        last_progress_time = 0.0

        while True:
            now = time.perf_counter()

            # Send requests to maintain target QPS (only during sending phase)
            if not sending_complete:
                while (
                    next_send_time <= now
                    and time.perf_counter() - stats.start_time < config.duration
                ):
                    req_id = f"bench_{request_count}_{uuid.uuid4().hex[:8]}"
                    payload = build_request_payload(
                        req_id,
                        provider.next(),
                        config.backend,
                        config.modality,
                        prefill_host,
                        embedding_port,
                    )
                    task = asyncio.create_task(
                        send_request(session, config.encoder_url, payload)
                    )
                    task_refs.add(task)
                    task.add_done_callback(_on_task_done)
                    request_count += 1
                    next_send_time += interval

                # Check if sending phase is complete
                if time.perf_counter() - stats.start_time >= config.duration:
                    sending_complete = True
                    stats.send_end_time = time.perf_counter()

            # Count completed requests
            completed = completed_count
            in_flight = request_count - completed
            all_done = completed >= request_count and sending_complete

            # Throttle progress redraws (~4 Hz); always redraw on final tick.
            if all_done or now - last_progress_time >= progress_interval:
                denom = request_count if sending_complete else total_to_send
                progress_pct = (completed / denom * 100) if denom > 0 else 0
                bar_width = 30
                filled = int(bar_width * progress_pct / 100)
                bar = "█" * filled + "░" * (bar_width - filled)
                phase = "waiting" if sending_complete else "sending"
                print(
                    f"\r  [{bar}] {progress_pct:5.1f}% | sent={request_count} "
                    f"recv={completed} in-flight={in_flight} ({phase})",
                    end="",
                    flush=True,
                )
                last_progress_time = now

            if all_done:
                break

            now = time.perf_counter()
            next_wakeup = last_progress_time + progress_interval
            if not sending_complete:
                next_wakeup = min(next_wakeup, next_send_time)
            sleep_for = min(max(next_wakeup - now, 0.0), progress_interval)
            await asyncio.sleep(sleep_for)

        # Clear progress line
        print()

        stats.end_time = time.perf_counter()
        stats.results = collected_results

    return stats


def _format_input_config(args: argparse.Namespace) -> dict:
    """Return modality-specific config rows as {key: value} dict."""
    config = {}
    if args.modality == "video":
        config["Video Source"] = f"{len(args.video_url)} URL(s)"
    elif args.modality == "audio":
        if args.audio_url:
            config["Audio Source"] = f"{len(args.audio_url)} URL(s)"
        else:
            config["Audio Duration"] = f"{args.audio_duration}s"
            config["Sample Rate"] = f"{args.audio_sample_rate}Hz"
            config["Audios/Request"] = str(args.num_audios)
    else:
        if args.image_url:
            config["Image Source"] = f"{len(args.image_url)} URL(s)"
        else:
            config["Image Size"] = f"{args.image_size}x{args.image_size}"
            config["Images/Request"] = str(args.num_images)
    return config


def print_results(stats: BenchmarkStats, args: argparse.Namespace):
    """Print benchmark results."""
    input_config = _format_input_config(args)

    latencies = stats.latencies
    if len(latencies) > 0:
        avg = float(latencies.mean())
        lat_min = float(latencies.min())
        lat_max = float(latencies.max())
        p50, p90, p95, p99 = np.percentile(latencies, [50, 90, 95, 99])
    else:
        avg = lat_min = lat_max = p50 = p90 = p95 = p99 = 0.0

    if RICH_AVAILABLE:
        console = Console()

        console.print("\n[bold cyan]Encoder Benchmark Results (QPS Mode)[/bold cyan]")
        console.print("=" * 50)

        # Config table
        config_table = Table(show_header=False, box=None)
        config_table.add_column("Key", style="dim")
        config_table.add_column("Value")
        config_table.add_row("Modality", args.modality)
        config_table.add_row("Backend", args.backend)
        config_table.add_row("Target QPS", str(args.qps))
        config_table.add_row("Duration", f"{args.duration}s")
        for key, value in input_config.items():
            config_table.add_row(key, value)
        console.print(config_table)

        console.print()

        # Results table
        results_table = Table(title="Results", show_header=True)
        results_table.add_column("Metric", style="cyan")
        results_table.add_column("Value", justify="right")

        results_table.add_row("Total Requests", str(stats.total_requests))
        results_table.add_row(
            "Successful", f"[green]{stats.successful_requests}[/green]"
        )
        results_table.add_row(
            "Failed",
            f"[red]{stats.failed_requests}[/red]" if stats.failed_requests > 0 else "0",
        )
        results_table.add_row("Actual QPS", f"{stats.actual_qps:.2f}")

        console.print(results_table)

        # Latency table
        latency_table = Table(title="Latency (ms)", show_header=True)
        latency_table.add_column("Metric", style="cyan")
        latency_table.add_column("Value", justify="right")

        latency_table.add_row("avg", f"{avg:.2f}")
        latency_table.add_row("min", f"{lat_min:.2f}")
        latency_table.add_row("max", f"{lat_max:.2f}")
        latency_table.add_row("P50", f"{p50:.2f}")
        latency_table.add_row("P90", f"{p90:.2f}")
        latency_table.add_row("P95", f"{p95:.2f}")
        latency_table.add_row("P99", f"{p99:.2f}")

        console.print(latency_table)

        # Embedding info
        success_results = [r for r in stats.results if r.success and r.embedding_len]
        if success_results:
            r = success_results[0]
            console.print(
                f"\n[dim]Embedding: {r.embedding_len} tokens × {r.embedding_dim} dim[/dim]"
            )

    else:
        # Fallback to basic print
        print("\n" + "=" * 50)
        print("Encoder Benchmark Results (QPS Mode)")
        print("=" * 50)

        print(f"\nModality: {args.modality}")
        print(f"Backend: {args.backend}")
        print(f"Target QPS: {args.qps} | Duration: {args.duration}s")
        for key, value in input_config.items():
            print(f"{key}: {value}")

        print(
            f"\nRequests: {stats.total_requests} | Success: {stats.successful_requests} | Failed: {stats.failed_requests}"
        )
        print(f"Actual QPS: {stats.actual_qps:.2f}")

        print(f"\nLatency (ms):")
        print(f"  avg: {avg:.2f} | min: {lat_min:.2f} | max: {lat_max:.2f}")
        print(f"  P50: {p50:.2f} | P90: {p90:.2f} | P95: {p95:.2f} | P99: {p99:.2f}")

        success_results = [r for r in stats.results if r.success and r.embedding_len]
        if success_results:
            r = success_results[0]
            print(f"\nEmbedding: {r.embedding_len} tokens x {r.embedding_dim} dim")


def main():
    parser = argparse.ArgumentParser(
        description="Encoder Benchmark Script (QPS Mode)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--encoder-url",
        type=str,
        required=True,
        help="Encoder server URL (e.g., http://localhost:30000)",
    )
    parser.add_argument(
        "--qps",
        type=float,
        required=True,
        help="Target queries per second",
    )

    # Optional arguments
    parser.add_argument(
        "--modality",
        type=str,
        choices=["image", "audio", "video"],
        default="image",
        help="Input modality to benchmark",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=60.0,
        help="Benchmark duration in seconds",
    )
    parser.add_argument(
        "--warmup",
        type=float,
        default=5.0,
        help="Warmup duration in seconds",
    )
    parser.add_argument(
        "--backend",
        type=str,
        choices=["mooncake", "zmq"],
        default="mooncake",
        help="Encoder transfer backend",
    )

    # ZMQ mode arguments
    parser.add_argument(
        "--receiver-url",
        type=str,
        default=None,
        help="Mock receiver URL for ZMQ mode (e.g., tcp://127.0.0.1:12345). Required if backend=zmq.",
    )

    # Image-specific arguments
    image_group = parser.add_argument_group("Image options (--modality image)")
    image_group.add_argument(
        "--image-size",
        type=int,
        default=448,
        help="Random image size (width=height)",
    )
    image_group.add_argument(
        "--num-images",
        type=int,
        default=1,
        help="Number of images per request",
    )
    image_group.add_argument(
        "--image-url",
        type=str,
        action="append",
        default=None,
        help="Image URL to use (can be specified multiple times). If provided, skips random image generation.",
    )

    # Audio-specific arguments
    audio_group = parser.add_argument_group("Audio options (--modality audio)")
    audio_group.add_argument(
        "--audio-duration",
        type=float,
        default=1.0,
        help="Duration of random audio in seconds",
    )
    audio_group.add_argument(
        "--audio-sample-rate",
        type=int,
        default=24000,
        help="Sample rate of random audio in Hz",
    )
    audio_group.add_argument(
        "--num-audios",
        type=int,
        default=1,
        help="Number of audio clips per request",
    )
    audio_group.add_argument(
        "--audio-url",
        type=str,
        action="append",
        default=None,
        help="Audio URL to use (can be specified multiple times). If provided, skips random audio generation.",
    )

    # Video-specific arguments
    video_group = parser.add_argument_group("Video options (--modality video)")
    video_group.add_argument(
        "--video-url",
        type=str,
        action="append",
        default=None,
        help="Video URL to use (can be specified multiple times). Required for video modality.",
    )

    args = parser.parse_args()

    # Validate ZMQ mode arguments
    if args.backend == "zmq" and args.receiver_url is None:
        parser.error("--receiver-url is required when --backend=zmq")
    if args.modality == "video" and not args.video_url:
        parser.error("--video-url is required when --modality=video")

    print("=" * 50)
    print("Encoder Benchmark (QPS Mode)")
    print("=" * 50)
    print(f"Encoder URL: {args.encoder_url}")
    print(f"Modality: {args.modality}")
    print(f"Backend: {args.backend}")
    print(f"Target QPS: {args.qps}")
    print(f"Duration: {args.duration}s (warmup: {args.warmup}s)")

    if args.backend == "zmq":
        print(f"Receiver: {args.receiver_url}")

    if args.modality == "video":
        print(f"Video URLs: {len(args.video_url)} URL(s)")
    elif args.modality == "audio":
        if args.audio_url:
            print(f"Audio URLs: {len(args.audio_url)} URL(s)")
        else:
            print(
                f"Audio: {args.audio_duration}s @ {args.audio_sample_rate}Hz x {args.num_audios}/req (random)"
            )
    else:
        if args.image_url:
            print(f"Image URLs: {len(args.image_url)} URL(s)")
        else:
            print(
                f"Image: {args.image_size}x{args.image_size} x {args.num_images}/req (random)"
            )

    # Run benchmark
    config = BenchmarkConfig(
        encoder_url=args.encoder_url,
        qps=args.qps,
        duration=args.duration,
        warmup=args.warmup,
        modality=args.modality,
        backend=args.backend,
        receiver_url=args.receiver_url,
        image=ImageConfig(
            size=args.image_size,
            num_per_request=args.num_images,
            urls=args.image_url,
        ),
        audio=AudioConfig(
            duration=args.audio_duration,
            sample_rate=args.audio_sample_rate,
            num_per_request=args.num_audios,
            urls=args.audio_url,
        ),
        video=VideoConfig(urls=args.video_url),
    )
    stats = asyncio.run(run_benchmark(config))

    # Print results
    print_results(stats, args)


if __name__ == "__main__":
    main()
