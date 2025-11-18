# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import base64
import dataclasses
import json
import os
import shlex
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path
from typing import Optional, Sequence

from PIL import Image

from sglang.multimodal_gen.configs.sample.base import DataType
from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.performance_logger import (
    get_diffusion_perf_log_dir,
)

logger = init_logger(__name__)


def run_command(command) -> Optional[float]:
    """Runs a command and returns the execution time and status."""
    print(f"Running command: {shlex.join(command)}")

    duration = None
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    ) as process:
        for line in process.stdout:
            sys.stdout.write(line)
            if "Pixel data generated" in line:
                words = line.split(" ")
                duration = float(words[-2])

    if process.returncode == 0:
        return duration
    else:
        print(f"Command failed with exit code {process.returncode}")
        return None


def probe_port(host="127.0.0.1", port=30010, timeout=2.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(timeout)
        try:
            s.connect((host, port))
            return True
        except OSError:
            return False


def is_in_ci() -> bool:
    return get_bool_env_var("SGLANG_IS_IN_CI")


def get_dynamic_server_port() -> int:
    cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    if not cuda_devices:
        cuda_devices = "0"
    try:
        first_device_id = int(cuda_devices.split(",")[0].strip()[0])
    except (ValueError, IndexError):
        first_device_id = 0

    if is_in_ci():
        base_port = 10000 + first_device_id * 2000
    else:
        base_port = 20000 + first_device_id * 1000

    return base_port + 1000


def is_mp4(data):
    idx = data.find(b"ftyp")
    return 0 <= idx <= 32


def is_jpeg(data: bytes) -> bool:
    # JPEG files start with: FF D8 FF
    return data.startswith(b"\xff\xd8\xff")


def is_png(data):
    # PNG files start with: 89 50 4E 47 0D 0A 1A 0A
    return data.startswith(b"\x89PNG\r\n\x1a\n")


def wait_for_port(host="127.0.0.1", port=30010, deadline=300.0, interval=0.5):
    end = time.time() + deadline
    last_err = None
    while time.time() < end:
        if probe_port(host, port, timeout=interval):
            return True
        time.sleep(interval)
    raise TimeoutError(f"Port {host}:{port} not ready. Last error: {last_err}")


def check_image_size(ut, image, width, height):
    # check image size
    ut.assertEqual(image.size, (width, height))


def get_perf_log_dir() -> Path:
    """Gets the performance log directory from the centralized sglang utility."""
    log_dir_str = get_diffusion_perf_log_dir()
    if not log_dir_str:
        raise RuntimeError(
            "Performance logging is disabled (SGLANG_PERF_LOG_DIR is empty), "
            "but a test tried to access the log directory."
        )
    return Path(log_dir_str)


def _ensure_log_path(log_dir: Path) -> Path:
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / "performance.log"


def clear_perf_log(log_dir: Path) -> Path:
    """Delete the perf log file so tests can watch for fresh entries."""
    log_path = _ensure_log_path(log_dir)
    if log_path.exists():
        log_path.unlink()
    logger.info("[server-test] Monitoring perf log at %s", log_path.as_posix())
    return log_path


def prepare_perf_log() -> tuple[Path, Path]:
    """Convenience helper to resolve and clear the perf log in one call."""
    log_dir = get_perf_log_dir()
    log_path = clear_perf_log(log_dir)
    return log_dir, log_path


def read_perf_records(log_path: Path) -> list[dict]:
    if not log_path.exists():
        return []
    records: list[dict] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return records


def wait_for_perf_record(
    tag: str,
    prev_len: int,
    log_path: Path,
    timeout: float = 120.0,
) -> tuple[dict, int]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        records = read_perf_records(log_path)
        if len(records) > prev_len:
            for rec in records[prev_len:]:
                if rec.get("tag") == tag:
                    return rec, len(records)
        time.sleep(0.5)
    raise AssertionError(
        f"Timeout waiting for perf log entry '{tag}' (start_len={prev_len})"
    )


def wait_for_stage_metrics(
    request_id: str,
    prev_len: int,
    expected_count: int,
    log_path: Path,
    timeout: float = 120.0,
) -> tuple[dict[str, float], int]:
    deadline = time.time() + timeout
    metrics: dict[str, float] = {}
    while time.time() < deadline:
        records = read_perf_records(log_path)
        for rec in records[prev_len:]:
            if (
                rec.get("tag") == "pipeline_stage_metric"
                and rec.get("request_id") == request_id
            ):
                stage = rec.get("stage")
                duration = rec.get("duration_ms")
                if stage is not None and duration is not None:
                    metrics[str(stage)] = float(duration)
        if len(metrics) >= expected_count:
            return metrics, len(records)
        time.sleep(0.5)
    raise AssertionError(
        f"Timeout waiting for stage metrics for request {request_id} "
        f"(collected={len(metrics)} expected={expected_count})"
    )


def sample_step_indices(
    step_map: dict[int, float], fractions: Sequence[float]
) -> list[int]:
    if not step_map:
        return []
    max_idx = max(step_map.keys())
    indices = set()
    for fraction in fractions:
        idx = min(max_idx, max(0, int(round(fraction * max_idx))))
        if idx in step_map:
            indices.add(idx)
    return sorted(indices)


def validate_image(b64_json: str) -> None:
    """Decode and validate that image is PNG or JPEG."""
    image_bytes = base64.b64decode(b64_json)
    assert is_png(image_bytes) or is_jpeg(image_bytes), "Image must be PNG or JPEG"


def validate_video(b64_json: str) -> None:
    """Decode and validate that video is a valid format."""
    video_bytes = base64.b64decode(b64_json)
    is_mp4 = (
        video_bytes[:4] == b"\x00\x00\x00\x18" or video_bytes[:4] == b"\x00\x00\x00\x1c"
    )
    is_webm = video_bytes[:4] == b"\x1a\x45\xdf\xa3"
    assert is_mp4 or is_webm, "Video must be MP4 or WebM"


def validate_openai_video(video_bytes: bytes) -> None:
    """Validate that video is MP4 or WebM by magic bytes."""
    is_mp4 = (
        video_bytes.startswith(b"\x00\x00\x00\x18")
        or video_bytes.startswith(b"\x00\x00\x00\x1c")
        or video_bytes[4:8] == b"ftyp"
    )
    is_webm = video_bytes.startswith(b"\x1a\x45\xdf\xa3")
    assert is_mp4 or is_webm, "Video must be MP4 or WebM"


@dataclasses.dataclass
class TestResult:
    name: str
    key: str
    duration: Optional[float]
    succeed: bool

    @property
    def duration_str(self):
        return f"{self.duration:.4f}" if self.duration else "NA"


class TestCLIBase(unittest.TestCase):
    model_path: str = None
    extra_args = []
    data_type: DataType = None
    # tested on h100
    thresholds = {}

    width: int = 720
    height: int = 720
    output_path: str = "test_outputs"

    base_command = [
        "sglang",
        "generate",
        "--text-encoder-cpu-offload",
        "--pin-cpu-memory",
        "--prompt",
        "A curious raccoon",
        "--save-output",
        "--log-level=debug",
        f"--width={width}",
        f"--height={height}",
        f"--output-path={output_path}",
    ]

    results = []

    @classmethod
    def setUpClass(cls):
        cls.results = []

    def _run_command(self, name: str, model_path: str, test_key: str = "", args=[]):
        command = (
            self.base_command
            + [f"--model-path={model_path}"]
            + shlex.split(args or "")
            + ["--output-file-name", f"{name}"]
            + self.extra_args
        )
        duration = run_command(command)
        status = "Success" if duration else "Failed"
        succeed = duration is not None

        duration = float(duration) if succeed else None
        self.results.append(TestResult(name, test_key, duration, succeed))

        return name, duration, status


class TestGenerateBase(TestCLIBase):
    model_path: str = None
    extra_args = []
    data_type: DataType = None
    # tested on h100
    thresholds = {}

    width: int = 720
    height: int = 720
    output_path: str = "test_outputs"
    image_path: str | None = None
    prompt: str | None = "A curious raccoon"

    base_command = [
        "sglang",
        "generate",
        # "--text-encoder-cpu-offload",
        # "--pin-cpu-memory",
        f"--prompt",
        f"{prompt}",
        "--save-output",
        "--log-level=debug",
        f"--width={width}",
        f"--height={height}",
        f"--output-path={output_path}",
    ]

    results: list[TestResult] = []

    @classmethod
    def setUpClass(cls):
        cls.results = []

    @classmethod
    def tearDownClass(cls):
        # Print markdown table
        print("\n## Test Results\n")
        print("| Test Case                      | Duration | Status  |")
        print("|--------------------------------|----------|---------|")
        test_keys = ["test_single_gpu", "test_cfg_parallel", "test_usp", "test_mixed"]
        test_key_to_order = {
            test_key: order for order, test_key in enumerate(test_keys)
        }

        ordered_results: list[TestResult] = [None] * len(test_keys)
        for result in cls.results:
            order = test_key_to_order[result.key]
            ordered_results[order] = result

        for result in ordered_results:
            if not result:
                continue
            status = (
                "Succeed"
                if (
                    result.succeed
                    and float(result.duration) <= float(cls.thresholds[result.key])
                )
                else "Failed"
            )
            print(f"| {result.name:<30} | {result.duration_str:<8} | {status:<7} |")
        print()
        durations = [result.duration_str for result in cls.results]
        print(" | ".join([""] + durations + [""]))

    def _run_test(self, name: str, args, model_path: str, test_key: str):
        time_threshold = self.thresholds[test_key]
        name, duration, status = self._run_command(
            name, args=args, model_path=model_path, test_key=test_key
        )
        self.verify(status, name, duration, time_threshold)

    def verify(self, status, name, duration, time_threshold):
        print("-" * 80)
        print("\n" * 3)

        # test task status
        self.assertEqual(status, "Success", f"{name} command failed")
        self.assertIsNotNone(duration, f"Could not parse duration for {name}")
        self.assertLessEqual(
            duration,
            time_threshold,
            f"{name} failed with {duration:.4f}s > {time_threshold}s",
        )

        # test output file
        path = os.path.join(
            self.output_path, f"{name}.{self.data_type.get_default_extension()}"
        )
        self.assertTrue(os.path.exists(path), f"Output file not exist for {path}")
        if self.data_type == DataType.IMAGE:
            with Image.open(path) as image:
                check_image_size(self, image, self.width, self.height)
        logger.info(f"{name} passed in {duration:.4f}s (threshold: {time_threshold}s)")

    def model_name(self):
        return self.model_path.split("/")[-1]

    def test_single_gpu(self):
        """single gpu"""
        self._run_test(
            name=f"{self.model_name()}_single_gpu",
            args=None,
            model_path=self.model_path,
            test_key="test_single_gpu",
        )

    def test_cfg_parallel(self):
        """cfg parallel"""
        if self.data_type == DataType.IMAGE:
            return
        self._run_test(
            name=f"{self.model_name()}_cfg_parallel",
            args="--num-gpus 2 --enable-cfg-parallel",
            model_path=self.model_path,
            test_key="test_cfg_parallel",
        )

    def test_usp(self):
        """usp"""
        if self.data_type == DataType.IMAGE:
            return
        self._run_test(
            name=f"{self.model_name()}_usp",
            args="--num-gpus 4 --ulysses-degree=2 --ring-degree=2",
            model_path=self.model_path,
            test_key="test_usp",
        )

    def test_mixed(self):
        """mixed"""
        if self.data_type == DataType.IMAGE:
            return
        self._run_test(
            name=f"{self.model_name()}_mixed",
            args="--num-gpus 4 --ulysses-degree=2 --ring-degree=1 --enable-cfg-parallel",
            model_path=self.model_path,
            test_key="test_mixed",
        )
