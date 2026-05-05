# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import base64
import html
import io
import json
import math
import os
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urljoin

import cv2
import httpx
import numpy as np
import requests
from PIL import Image, ImageDraw, ImageFont

from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import (
    RequestPerfRecord,
    get_diffusion_perf_log_dir,
)

if TYPE_CHECKING:
    from sglang.multimodal_gen.test.server.testcase_configs import DiffusionTestCase

logger = init_logger(__name__)

SGL_TEST_FILES_CI_DATA_REVISION = "4d9eff3b05b0ffe1d3529e8bb148b63af11a4b92"
SGL_TEST_FILES_CONSISTENCY_GT_ROOT = (
    "https://raw.githubusercontent.com/"
    f"sgl-project/ci-data/{SGL_TEST_FILES_CI_DATA_REVISION}/"
    "diffusion-ci/consistency_gt"
)
SGL_TEST_FILES_OFFICIAL_CONSISTENCY_GT_BASE = (
    f"{SGL_TEST_FILES_CONSISTENCY_GT_ROOT}/official_generated"
)
SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE = (
    f"{SGL_TEST_FILES_CONSISTENCY_GT_ROOT}/sglang_generated"
)
SGL_TEST_FILES_CONSISTENCY_GT_BASE = SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE
SGL_TEST_FILES_CONSISTENCY_GT_BASES = (
    SGL_TEST_FILES_OFFICIAL_CONSISTENCY_GT_BASE,
    SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE,
)
CONSISTENCY_THRESHOLD_JSON_PATH = (
    Path(__file__).resolve().parent / "server" / "consistency_threshold.json"
)
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
DEFAULT_CLIP_THRESHOLD_IMAGE = 0.92
DEFAULT_CLIP_THRESHOLD_VIDEO = 0.90
DEFAULT_SSIM_THRESHOLD_IMAGE = 0.95
DEFAULT_PSNR_THRESHOLD_IMAGE = 28.0
DEFAULT_MEAN_ABS_DIFF_THRESHOLD_IMAGE = 8.0
DEFAULT_SSIM_THRESHOLD_VIDEO = 0.92
DEFAULT_PSNR_THRESHOLD_VIDEO = 24.0
DEFAULT_MEAN_ABS_DIFF_THRESHOLD_VIDEO = 10.0
_clip_model_cache: dict[str, Any] = {}
_consistency_gt_cache: dict[str, Any] = {}


def _load_clip_processor_with_roberta_processing_compat(
    clip_processor_cls, *args, **kwargs
):
    from tokenizers import processors

    roberta_processing = processors.RobertaProcessing

    def roberta_processing_compat(*processor_args, **processor_kwargs):
        if "sep" in processor_kwargs and "cls" in processor_kwargs:
            sep = processor_kwargs.pop("sep")
            cls_token = processor_kwargs.pop("cls")
            return roberta_processing(
                sep, cls_token, *processor_args, **processor_kwargs
            )
        return roberta_processing(*processor_args, **processor_kwargs)

    processors.RobertaProcessing = roberta_processing_compat
    try:
        return clip_processor_cls.from_pretrained(*args, **kwargs)
    finally:
        processors.RobertaProcessing = roberta_processing


# ---------------------------------------------------------------------------
# Common model IDs for diffusion tests
#
# Centralised here so every test file references the same constants instead
# of scattering hard-coded strings. When adding a new model that will be
# reused across tests, define it here.
# ---------------------------------------------------------------------------

DEFAULT_SMALL_MODEL_NAME_FOR_TEST = "Tongyi-MAI/Z-Image-Turbo"

# Qwen image generation models
DEFAULT_QWEN_IMAGE_MODEL_NAME_FOR_TEST = "Qwen/Qwen-Image"
DEFAULT_QWEN_IMAGE_2512_MODEL_NAME_FOR_TEST = "Qwen/Qwen-Image-2512"
DEFAULT_QWEN_IMAGE_EDIT_MODEL_NAME_FOR_TEST = "Qwen/Qwen-Image-Edit"
DEFAULT_QWEN_IMAGE_EDIT_2509_MODEL_NAME_FOR_TEST = "Qwen/Qwen-Image-Edit-2509"
DEFAULT_QWEN_IMAGE_EDIT_2511_MODEL_NAME_FOR_TEST = "Qwen/Qwen-Image-Edit-2511"
DEFAULT_QWEN_IMAGE_LAYERED_MODEL_NAME_FOR_TEST = "Qwen/Qwen-Image-Layered"

# JoyAI image editing models
DEFAULT_JOYAI_IMAGE_EDIT_MODEL_NAME_FOR_TEST = "jdopensource/JoyAI-Image-Edit-Diffusers"

# FLUX image generation models
DEFAULT_FLUX_1_DEV_MODEL_NAME_FOR_TEST = "black-forest-labs/FLUX.1-dev"
DEFAULT_FLUX_2_DEV_MODEL_NAME_FOR_TEST = "black-forest-labs/FLUX.2-dev"
DEFAULT_FLUX_2_KLEIN_4B_MODEL_NAME_FOR_TEST = "black-forest-labs/FLUX.2-klein-4B"
DEFAULT_FLUX_2_KLEIN_BASE_4B_MODEL_NAME_FOR_TEST = (
    "black-forest-labs/FLUX.2-klein-base-4B"
)

# Wan video generation models
DEFAULT_WAN_2_1_T2V_1_3B_MODEL_NAME_FOR_TEST = "Wan-AI/Wan2.1-T2V-1.3B-Diffusers"
DEFAULT_WAN_2_1_T2V_14B_MODEL_NAME_FOR_TEST = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
DEFAULT_WAN_2_1_I2V_14B_480P_MODEL_NAME_FOR_TEST = (
    "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers"
)
DEFAULT_WAN_2_1_I2V_14B_720P_MODEL_NAME_FOR_TEST = (
    "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
)
DEFAULT_WAN_2_2_TI2V_5B_MODEL_NAME_FOR_TEST = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
DEFAULT_WAN_2_2_T2V_A14B_MODEL_NAME_FOR_TEST = "Wan-AI/Wan2.2-T2V-A14B-Diffusers"
DEFAULT_WAN_2_2_I2V_A14B_MODEL_NAME_FOR_TEST = "Wan-AI/Wan2.2-I2V-A14B-Diffusers"

# MOVA video generation models
DEFAULT_MOVA_360P_MODEL_NAME_FOR_TEST = "OpenMOSS-Team/MOVA-360p"


def print_value_formatted(description: str, value: int | float | str):
    """Helper function to print a metric value formatted."""
    if isinstance(value, int):
        if value >= 1e6:
            value_str = f"{value / 1e6:<30.2f}M"
        elif value >= 1e3:
            value_str = f"{value / 1e3:<30.2f}K"
        else:
            value_str = f"{value:<30}"
    elif isinstance(value, float):
        value_str = f"{value:<30.2f}"
    else:
        value_str = f"{value:<30}"

    print(f"{description:<45} {value_str}")


def print_divider(length: int, char: str = "-"):
    """Helper function to print a divider line."""
    print(char * length)


def is_image_url(image_path: str | Path | None) -> bool:
    """Check if image_path is a URL."""
    if image_path is None:
        return False
    return isinstance(image_path, str) and (
        image_path.startswith("http://") or image_path.startswith("https://")
    )


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


def find_free_port(host: str = "127.0.0.1") -> int:
    """Bind to port 0 and let the OS assign an available port."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def wait_for_server_health(
    base_url: str,
    path: str = "/health",
    timeout: float = 180.0,
    interval: float = 1.0,
) -> None:
    """Poll ``GET <base_url><path>`` until it returns HTTP 200."""
    deadline = time.time() + timeout
    last_err: httpx.RequestError | None = None
    last_status: int | None = None
    while time.time() < deadline:
        try:
            r = httpx.get(urljoin(base_url, path), timeout=5.0)
            last_status = r.status_code
            if r.status_code == 200:
                return
        except httpx.RequestError as e:
            last_err = e
        time.sleep(interval)
    raise TimeoutError(
        f"Server at {urljoin(base_url, path)} not healthy after {timeout}s. "
        f"{last_status=} {last_err=}"
    )


def post_json(
    base_url: str,
    path: str,
    payload: dict,
    timeout: float = 300.0,
) -> httpx.Response:
    """POST JSON to ``<base_url><path>`` and return the response."""
    return httpx.post(urljoin(base_url, path), json=payload, timeout=timeout)


def run_command(command: list[str]) -> bool:
    """Run a CLI command and return whether it succeeded."""
    print(f"Running command: {' '.join(command)}", flush=True)
    with subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
    ) as process:
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
        process.wait()
        if process.returncode == 0:
            return True
        print(f"Command failed with exit code {process.returncode}", flush=True)
    return False


# ---------------------------------------------------------------------------
# GPU memory helpers (nvidia-smi)
# ---------------------------------------------------------------------------


def query_gpu_mem_used_mib(gpu_index: int = 0, required: bool = False) -> int | None:
    """Return GPU memory usage in MiB via ``nvidia-smi``, or *None* on failure.

    When *required* is ``True`` the function raises instead of returning ``None``.
    """
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                f"--id={gpu_index}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            text=True,
        ).strip()
        return int(out.splitlines()[0].strip())
    except Exception as e:
        logger.warning(f"nvidia-smi memory query failed: {type(e).__name__}: {e}")
        assert not required, (
            "nvidia-smi memory query is unavailable; "
            "cannot enforce GPU memory assertions."
        )
        return None


def require_gpu_mem_query(gpu_index: int = 0) -> int:
    """Same as :func:`query_gpu_mem_used_mib` but asserts availability.

    Raises ``AssertionError`` when ``nvidia-smi`` is unavailable instead of
    returning ``None``, so callers can rely on a valid ``int`` result.
    """
    mem = query_gpu_mem_used_mib(gpu_index, required=True)
    assert mem is not None
    return mem


def assert_gpu_mem_changed(
    label: str,
    before_mib: int,
    after_mib: int,
    min_delta_mib: int,
) -> None:
    """Assert that GPU memory changed by at least *min_delta_mib* MiB."""
    delta = abs(after_mib - before_mib)
    logger.debug(
        f"[MEM] {label}: before={before_mib} MiB  after={after_mib} MiB  |delta|={delta} MiB"
    )
    assert delta >= min_delta_mib, (
        f"GPU memory change too small for '{label}': "
        f"|after-before|={delta} MiB < {min_delta_mib} MiB "
        f"(before={before_mib} MiB, after={after_mib} MiB)"
    )


def is_mp4(data: bytes) -> bool:
    """Check if data represents a valid MP4 file by magic bytes."""
    if len(data) < 8:
        return False
    return data[4:8] == b"ftyp"


def is_jpeg(data: bytes) -> bool:
    # JPEG files start with: FF D8 FF
    return data.startswith(b"\xff\xd8\xff")


def is_png(data):
    # PNG files start with: 89 50 4E 47 0D 0A 1A 0A
    return data.startswith(b"\x89PNG\r\n\x1a\n")


def is_webp(data: bytes) -> bool:
    # WebP files start with: RIFF....WEBP
    return data[:4] == b"RIFF" and data[8:12] == b"WEBP"


def detect_image_format(data: bytes) -> str:
    """Detect image format from bytes (magic). Returns 'png'|'jpeg'|'webp'; default 'png'."""
    if len(data) < 12:
        return "png"
    if is_png(data):
        return "png"
    if is_jpeg(data):
        return "jpeg"
    if is_webp(data):
        return "webp"
    return "png"


def get_expected_image_format(
    output_format: str | None = None,
    background: str | None = None,
) -> str:
    """Infer expected image format based on request parameters.
    Args:
        output_format: The output_format parameter from the request (png/jpeg/webp/jpg)
        background: The background parameter from the request (transparent/opaque/auto)
    Returns:
        Expected file extension: "jpg", "png", or "webp"
    """
    fmt = (output_format or "").lower()
    if fmt in {"png", "webp", "jpeg", "jpg"}:
        return "jpg" if fmt == "jpeg" else fmt
    if (background or "auto").lower() == "transparent":
        return "png"
    return "jpg"  # Default


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


def read_perf_logs(log_path: Path) -> list[RequestPerfRecord]:
    if not log_path.exists():
        return []
    records: list[RequestPerfRecord] = []
    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                record_dict = json.loads(line)
                records.append(RequestPerfRecord(**record_dict))
            except json.JSONDecodeError:
                continue
    return records


def wait_for_req_perf_record(
    request_id: str,
    log_path: Path,
    timeout: float = 30.0,
) -> RequestPerfRecord | None:
    """
    the stage metrics of this request should be in the performance_log file with {request-id}
    """
    logger.info(f"Waiting for req perf record with request id: {request_id}")
    deadline = time.time() + timeout
    while time.time() < deadline:
        records = read_perf_logs(log_path)
        for record in records:
            if record.request_id == request_id:
                return record

        time.sleep(0.5)

    if os.environ.get("SGLANG_GEN_BASELINE", "0") == "1":
        return None

    logger.error(f"record: {records}")
    raise AssertionError(f"Timeout waiting for stage metrics for request {request_id} ")


def validate_image(b64_json: str) -> None:
    """Decode and validate that image is PNG or JPEG."""
    image_bytes = base64.b64decode(b64_json)
    assert is_png(image_bytes) or is_jpeg(image_bytes), "Image must be PNG or JPEG"


def validate_video(b64_json: str) -> None:
    """Decode and validate that video is a valid format."""
    video_bytes = base64.b64decode(b64_json)
    is_webm = video_bytes[:4] == b"\x1a\x45\xdf\xa3"
    assert is_mp4(video_bytes) or is_webm, "Video must be MP4 or WebM"


def validate_openai_video(video_bytes: bytes) -> None:
    """Validate that video is MP4 or WebM by magic bytes."""
    is_webm = video_bytes.startswith(b"\x1a\x45\xdf\xa3")
    assert is_mp4(video_bytes) or is_webm, "Video must be MP4 or WebM"


def validate_image_file(
    file_path: str,
    expected_filename: str,
    expected_width: int | None = None,
    expected_height: int | None = None,
    output_format: str | None = None,
    background: str | None = None,
) -> None:
    """Validate image output file: existence, extension, size, filename, format, dimensions."""
    # Infer expected format from request parameters
    expected_ext = get_expected_image_format(output_format, background)

    # 1. File existence
    assert os.path.exists(file_path), f"Image file does not exist: {file_path}"

    # 2. Extension check
    assert file_path.endswith(
        f".{expected_ext}"
    ), f"Expected .{expected_ext} extension, got: {file_path}"

    # 3. File size > 0
    file_size = os.path.getsize(file_path)
    assert file_size > 0, f"Image file is empty: {file_path}"

    # 4. Filename validation
    actual_filename = os.path.basename(file_path)
    assert (
        actual_filename == expected_filename
    ), f"Filename mismatch: expected '{expected_filename}', got '{actual_filename}'"

    # 5. Image format validation (magic bytes check based on expected format)
    with open(file_path, "rb") as f:
        header = f.read(12)  # Read enough bytes for webp detection
        if expected_ext == "png":
            assert is_png(header), f"File is not a valid PNG: {file_path}"
        elif expected_ext == "jpg":
            assert is_jpeg(header), f"File is not a valid JPEG: {file_path}"
        elif expected_ext == "webp":
            assert is_webp(header), f"File is not a valid WebP: {file_path}"

    # 6. Image dimension validation (reuse PIL)
    if expected_width is not None and expected_height is not None:
        with Image.open(file_path) as img:
            width, height = img.size
            assert (
                width == expected_width
            ), f"Width mismatch: expected {expected_width}, got {width}"
            assert (
                height == expected_height
            ), f"Height mismatch: expected {expected_height}, got {height}"


def _get_video_dimensions_from_metadata(
    cap: cv2.VideoCapture,
) -> tuple[int, int] | None:
    """Get video dimensions from metadata properties.

    Args:
        cap: OpenCV VideoCapture object

    Returns:
        Tuple of (width, height) if successful, None if metadata is invalid
    """
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)

    if width == 0 or height == 0:
        return None

    return int(width), int(height)


def _get_video_dimensions_from_frame(cap: cv2.VideoCapture) -> tuple[int, int]:
    """Get video dimensions by reading the first frame.

    Args:
        cap: OpenCV VideoCapture object

    Returns:
        Tuple of (width, height)

    """
    ret, frame = cap.read()
    if not ret or frame is None:
        raise ValueError("Unable to read video frame to get dimensions")

    # frame.shape is (height, width, channels)
    height, width = frame.shape[:2]
    return int(width), int(height)


def get_video_dimensions(file_path: str) -> tuple[int, int]:
    """Get video dimensions (width, height) from a video file.

    Tries to get dimensions from metadata first, falls back to reading first frame.

    Returns:
        Tuple of (width, height)

    """
    cap = cv2.VideoCapture(file_path)
    try:
        # Try to get dimensions from metadata first
        dimensions = _get_video_dimensions_from_metadata(cap)
        if dimensions is not None:
            return dimensions

        # Fall back to reading first frame
        return _get_video_dimensions_from_frame(cap)
    finally:
        cap.release()


def get_video_frame_count(file_path: str) -> int:
    """Return the number of frames in a video file using OpenCV."""
    cap = cv2.VideoCapture(file_path)
    try:
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if count > 0:
            return count
        # Fallback: count frames manually
        n = 0
        while cap.read()[0]:
            n += 1
        return n
    finally:
        cap.release()


def validate_video_file(
    file_path: str,
    expected_filename: str,
    expected_width: int | None = None,
    expected_height: int | None = None,
) -> None:
    """Validate video output file: existence, extension, size, filename, format, dimensions."""
    # 1. File existence
    assert os.path.exists(file_path), f"Video file does not exist: {file_path}"

    # 2. Extension check
    assert file_path.endswith(".mp4"), f"Expected .mp4 extension, got: {file_path}"

    # 3. File size > 0
    file_size = os.path.getsize(file_path)
    assert file_size > 0, f"Video file is empty: {file_path}"

    # 4. Filename validation
    actual_filename = os.path.basename(file_path)
    assert (
        actual_filename == expected_filename
    ), f"Filename mismatch: expected '{expected_filename}', got '{actual_filename}'"

    # 5. Video format validation (reuse is_mp4)
    with open(file_path, "rb") as f:
        header = f.read(32)
        assert is_mp4(header), f"File is not a valid MP4: {file_path}"

    # 6. Video dimension validation (using OpenCV)
    if expected_width is not None and expected_height is not None:
        actual_width, actual_height = get_video_dimensions(file_path)
        assert (
            actual_width == expected_width
        ), f"Video width mismatch: expected {expected_width}, got {actual_width}"
        assert (
            actual_height == expected_height
        ), f"Video height mismatch: expected {expected_height}, got {actual_height}"


def _load_threshold_json() -> dict[str, Any]:
    """Load consistency_threshold.json; returns {} if missing."""
    if not CONSISTENCY_THRESHOLD_JSON_PATH.exists():
        return {}
    with CONSISTENCY_THRESHOLD_JSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


@dataclass
class ConsistencyThresholds:
    clip_threshold: float
    ssim_threshold: float
    psnr_threshold: float
    mean_abs_diff_threshold: float


def get_consistency_thresholds(
    case_id: str,
    is_video: bool,
    metadata: dict[str, Any] | None = None,
) -> ConsistencyThresholds:
    """Get all consistency thresholds for a case."""
    if metadata is None:
        metadata = _load_threshold_json()

    case_meta = metadata.get("cases", {}).get(case_id, {})
    suffix = "video" if is_video else "image"

    defaults = {
        "clip_threshold": metadata.get(
            f"default_clip_threshold_{suffix}",
            DEFAULT_CLIP_THRESHOLD_VIDEO if is_video else DEFAULT_CLIP_THRESHOLD_IMAGE,
        ),
        "ssim_threshold": metadata.get(
            f"default_ssim_threshold_{suffix}",
            DEFAULT_SSIM_THRESHOLD_VIDEO if is_video else DEFAULT_SSIM_THRESHOLD_IMAGE,
        ),
        "psnr_threshold": metadata.get(
            f"default_psnr_threshold_{suffix}",
            DEFAULT_PSNR_THRESHOLD_VIDEO if is_video else DEFAULT_PSNR_THRESHOLD_IMAGE,
        ),
        "mean_abs_diff_threshold": metadata.get(
            f"default_mean_abs_diff_threshold_{suffix}",
            (
                DEFAULT_MEAN_ABS_DIFF_THRESHOLD_VIDEO
                if is_video
                else DEFAULT_MEAN_ABS_DIFF_THRESHOLD_IMAGE
            ),
        ),
    }

    return ConsistencyThresholds(
        clip_threshold=float(
            case_meta.get("clip_threshold", defaults["clip_threshold"])
        ),
        ssim_threshold=float(
            case_meta.get("ssim_threshold", defaults["ssim_threshold"])
        ),
        psnr_threshold=float(
            case_meta.get("psnr_threshold", defaults["psnr_threshold"])
        ),
        mean_abs_diff_threshold=float(
            case_meta.get(
                "mean_abs_diff_threshold", defaults["mean_abs_diff_threshold"]
            )
        ),
    )


def get_clip_threshold(
    case: "DiffusionTestCase",
    metadata: dict[str, Any] | None = None,
) -> float:
    """Get CLIP similarity threshold for a consistency test case."""
    return get_consistency_thresholds(
        case_id=case.id,
        is_video=case.server_args.modality == "video",
        metadata=metadata,
    ).clip_threshold


@dataclass
class FrameConsistencyMetrics:
    frame_index: int
    clip_similarity: float
    ssim: float
    psnr: float
    mean_abs_diff: float
    clip_passed: bool
    ssim_passed: bool
    psnr_passed: bool
    mean_abs_diff_passed: bool


@dataclass
class ConsistencyResult:
    """Result of a consistency comparison."""

    case_id: str
    passed: bool
    similarity_scores: list[float]
    min_similarity: float
    threshold: float
    min_ssim: float
    min_psnr: float
    max_mean_abs_diff: float
    thresholds: ConsistencyThresholds
    frame_metrics: list[FrameConsistencyMetrics]


@dataclass
class LoadedConsistencyGT:
    images: list[np.ndarray]
    embeddings: list[np.ndarray]


def get_clip_model() -> tuple[Any, Any]:
    """Get CLIP model and processor."""
    global _clip_model_cache

    if "model" not in _clip_model_cache:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError as exc:
            raise ImportError(
                "transformers and torch are required for CLIP consistency check."
            ) from exc

        logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME}")
        try:
            processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        except TypeError as e:
            if "RobertaProcessing" not in str(e):
                raise
            logger.warning(
                "Fast CLIP processor failed (%s), retrying with use_fast=False", e
            )
            processor = _load_clip_processor_with_roberta_processing_compat(
                CLIPProcessor,
                CLIP_MODEL_NAME,
                use_fast=False,
            )
        model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
        model.eval()

        _clip_model_cache["model"] = model
        _clip_model_cache["processor"] = processor
        _clip_model_cache["device"] = device
        logger.info(f"CLIP model loaded on {device}")

    return _clip_model_cache["model"], _clip_model_cache["processor"]


def compute_clip_embedding(image: np.ndarray) -> np.ndarray:
    """Compute a normalized CLIP image embedding."""
    try:
        import torch
    except ImportError as exc:
        raise ImportError("torch is required for CLIP consistency check.") from exc

    model, processor = get_clip_model()
    device = _clip_model_cache["device"]

    pil_image = Image.fromarray(image)
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        if hasattr(image_features, "image_embeds"):
            image_features = image_features.image_embeds
        elif hasattr(image_features, "pooler_output"):
            image_features = image_features.pooler_output
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy().flatten()


def compute_clip_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """Compute cosine similarity between two CLIP embeddings."""
    return float(np.dot(emb1, emb2))


def _ensure_rgb_uint8_image(image: np.ndarray) -> np.ndarray:
    """Normalize image input for pixel-wise consistency metrics."""
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"Expected RGB HWC image, got shape={image.shape}")
    if image.dtype == np.uint8:
        return image
    image = np.clip(image, 0, 255)
    return image.astype(np.uint8)


def compute_ssim(image: np.ndarray, gt_image: np.ndarray) -> float:
    """Compute SSIM between two RGB images."""
    from skimage.metrics import structural_similarity

    image = _ensure_rgb_uint8_image(image)
    gt_image = _ensure_rgb_uint8_image(gt_image)
    if image.shape != gt_image.shape:
        raise ValueError(
            f"Image shape mismatch for SSIM: output={image.shape}, gt={gt_image.shape}"
        )
    return float(structural_similarity(image, gt_image, channel_axis=2, data_range=255))


def compute_psnr(image: np.ndarray, gt_image: np.ndarray) -> float:
    """Compute PSNR between two RGB images."""
    from skimage.metrics import peak_signal_noise_ratio

    image = _ensure_rgb_uint8_image(image)
    gt_image = _ensure_rgb_uint8_image(gt_image)
    if image.shape != gt_image.shape:
        raise ValueError(
            f"Image shape mismatch for PSNR: output={image.shape}, gt={gt_image.shape}"
        )
    return float(peak_signal_noise_ratio(gt_image, image, data_range=255))


def compute_mean_abs_diff(image: np.ndarray, gt_image: np.ndarray) -> float:
    """Compute mean absolute pixel difference between two RGB images."""
    image = _ensure_rgb_uint8_image(image)
    gt_image = _ensure_rgb_uint8_image(gt_image)
    if image.shape != gt_image.shape:
        raise ValueError(
            f"Image shape mismatch for mean_abs_diff: output={image.shape}, gt={gt_image.shape}"
        )
    return float(np.abs(image.astype(np.float32) - gt_image.astype(np.float32)).mean())


def output_format_to_ext(output_format: str | None) -> str:
    """Map output_format to file extension. Used by GT naming and consistency check."""
    if not output_format:
        return "jpg"
    of = output_format.lower()
    if of == "jpeg":
        return "jpg"
    if of in ("png", "webp", "jpg"):
        return of
    return "png"


def _consistency_gt_filenames(
    case_id: str, num_gpus: int, is_video: bool, output_format: str | None = None
) -> list[str]:
    """Return the list of GT image filenames for a case. Reused by GT generation and consistency check."""
    n = num_gpus
    if is_video:
        return [
            f"{case_id}_{n}gpu_frame_0.png",
            f"{case_id}_{n}gpu_frame_mid.png",
            f"{case_id}_{n}gpu_frame_last.png",
        ]
    ext = output_format_to_ext(output_format)
    return [f"{case_id}_{n}gpu.{ext}"]


def get_consistency_gt_candidates(
    case_id: str, num_gpus: int, is_video: bool, output_format: str | None = None
) -> list[str]:
    """Return candidate GT filenames for local consistency data."""
    n = num_gpus
    if is_video:
        return [
            f"{case_id}_{n}gpu_frame_0.png",
            f"{case_id}_{n}gpu_frame_mid.png",
            f"{case_id}_{n}gpu_frame_last.png",
        ]
    base = f"{case_id}_{n}gpu"
    preferred = output_format_to_ext(output_format)
    exts = [preferred] + [e for e in ("png", "jpg", "webp") if e != preferred]
    return [f"{base}.{e}" for e in exts]


def get_consistency_gt_remote_files(
    case_id: str, num_gpus: int, is_video: bool, output_format: str | None = None
) -> list[tuple[str, str]]:
    """Return GT filenames with their remote raw URLs."""
    files = _find_remote_consistency_gt_files(
        case_id, num_gpus, is_video, output_format
    )
    if files:
        return files

    return _remote_consistency_gt_candidates(
        SGL_TEST_FILES_CONSISTENCY_GT_BASE, case_id, num_gpus, is_video, output_format
    )


def _remote_consistency_gt_candidates(
    base_url: str,
    case_id: str,
    num_gpus: int,
    is_video: bool,
    output_format: str | None = None,
) -> list[tuple[str, str]]:
    filenames = get_consistency_gt_candidates(
        case_id, num_gpus, is_video, output_format
    )
    return [(filename, f"{base_url}/{filename}") for filename in filenames]


def _remote_file_exists(url: str) -> bool:
    try:
        return requests.head(url, timeout=10, allow_redirects=True).status_code == 200
    except requests.RequestException:
        return False


def _find_remote_consistency_gt_files(
    case_id: str,
    num_gpus: int,
    is_video: bool,
    output_format: str | None = None,
) -> list[tuple[str, str]]:
    for base_url in SGL_TEST_FILES_CONSISTENCY_GT_BASES:
        candidates = _remote_consistency_gt_candidates(
            base_url, case_id, num_gpus, is_video, output_format
        )
        if is_video:
            if all(_remote_file_exists(url) for _, url in candidates):
                return candidates
        else:
            for filename, url in candidates:
                if _remote_file_exists(url):
                    return [(filename, url)]
    return []


def _get_consistency_gt_dir() -> Path | None:
    """Return the local GT directory when configured."""
    d = os.environ.get("SGLANG_CONSISTENCY_GT_DIR")
    if not d:
        return None
    return Path(d).resolve()


def _get_consistency_gt_cache_key(
    case_id: str,
    num_gpus: int,
    is_video: bool,
    output_format: str | None,
) -> str:
    gt_dir = _get_consistency_gt_dir()
    source = str(gt_dir) if gt_dir is not None else "remote"
    return f"{case_id}:{num_gpus}:{is_video}:{output_format or ''}:{source}"


def load_consistency_gt(
    case_id: str,
    num_gpus: int,
    is_video: bool = False,
    output_format: str | None = None,
) -> LoadedConsistencyGT:
    """Load GT images and CLIP embeddings for consistency checks."""
    cache_key = _get_consistency_gt_cache_key(
        case_id, num_gpus, is_video, output_format
    )
    cached = _consistency_gt_cache.get(cache_key)
    if cached is not None:
        return cached

    filenames = _consistency_gt_filenames(case_id, num_gpus, is_video, output_format)
    images: list[np.ndarray] = []

    gt_dir = _get_consistency_gt_dir()
    if gt_dir is not None:
        candidates = get_consistency_gt_candidates(
            case_id, num_gpus, is_video, output_format
        )
        if is_video:
            for fn in candidates:
                path = gt_dir / fn
                if not path.exists():
                    raise FileNotFoundError(f"GT image not found: {path}")
                arr = np.array(Image.open(path).convert("RGB"))
                images.append(arr)
        else:
            path = None
            for fn in candidates:
                candidate = gt_dir / fn
                if candidate.exists():
                    path = candidate
                    break
            if path is None:
                raise FileNotFoundError(
                    f"GT image not found in {gt_dir}. Tried: {', '.join(candidates)}"
                )
            images.append(np.array(Image.open(path).convert("RGB")))
        logger.info(f"Loaded {len(images)} GT images for {case_id} from {gt_dir}")
    else:
        remote_files = _find_remote_consistency_gt_files(
            case_id, num_gpus, is_video, output_format
        )
        if not remote_files:
            raise FileNotFoundError(
                f"GT image not found for {case_id}. Tried: {', '.join(filenames)}"
            )
        for _, url in remote_files:
            resp = requests.get(url, timeout=30)
            if resp.status_code != 200:
                raise FileNotFoundError(f"GT image not found: {url}")
            images.append(np.array(Image.open(io.BytesIO(resp.content)).convert("RGB")))
        source_dir = remote_files[0][1].rsplit("/", 1)[0]
        logger.info(f"Loaded {len(images)} GT images for {case_id} from {source_dir}")

    embeddings = [compute_clip_embedding(arr) for arr in images]
    loaded_gt = LoadedConsistencyGT(images=images, embeddings=embeddings)
    _consistency_gt_cache[cache_key] = loaded_gt
    return loaded_gt


def load_gt_embeddings(
    case_id: str,
    num_gpus: int,
    is_video: bool = False,
    output_format: str | None = None,
) -> list[np.ndarray]:
    """Load GT images and convert them into CLIP embeddings."""
    return load_consistency_gt(
        case_id=case_id,
        num_gpus=num_gpus,
        is_video=is_video,
        output_format=output_format,
    ).embeddings


def gt_exists(
    case_id: str,
    num_gpus: int,
    is_video: bool = False,
    output_format: str | None = None,
) -> bool:
    """Check whether GT image(s) exist."""
    gt_dir = _get_consistency_gt_dir()
    if gt_dir is not None:
        candidates = get_consistency_gt_candidates(
            case_id, num_gpus, is_video, output_format
        )
        if is_video:
            return all((gt_dir / c).exists() for c in candidates)
        return any((gt_dir / c).exists() for c in candidates)

    return bool(
        _find_remote_consistency_gt_files(case_id, num_gpus, is_video, output_format)
    )


def extract_key_frames_from_video(
    video_bytes: bytes,
    num_frames: int | None = None,
) -> list[np.ndarray]:
    """
    Extract key frames (first, middle, last) from video bytes.

    Args:
        video_bytes: Raw video bytes (MP4 format)
        num_frames: Total number of frames (if known), used for validation

    Returns:
        List of numpy arrays [first_frame, middle_frame, last_frame].
    """
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name

    try:
        cap = cv2.VideoCapture(tmp_path)
        if not cap.isOpened():
            raise ValueError("Failed to open video file")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames < 1:
            raise ValueError("Video has no frames")

        first_idx = 0
        mid_idx = total_frames // 2
        last_idx = total_frames - 1
        key_indices = [first_idx, mid_idx, last_idx]

        frames = []
        for idx in key_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame at index {idx}")
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

        cap.release()
        logger.info(
            f"Extracted {len(frames)} key frames from video "
            f"(total: {total_frames}, indices: {key_indices})"
        )
        return frames

    finally:
        os.unlink(tmp_path)


def image_bytes_to_numpy(image_bytes: bytes) -> np.ndarray:
    """Convert image bytes to numpy array."""
    img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    return np.array(img)


def compare_with_gt(
    output_frames: list[np.ndarray],
    gt_data: LoadedConsistencyGT,
    thresholds: ConsistencyThresholds,
    case_id: str,
) -> ConsistencyResult:
    """Compare output frames with GT using CLIP and pixel-level metrics."""
    if len(output_frames) != len(gt_data.embeddings):
        raise ValueError(
            f"Frame count mismatch: output={len(output_frames)}, gt={len(gt_data.embeddings)}"
        )

    similarity_scores = []
    frame_metrics: list[FrameConsistencyMetrics] = []

    for i, (out_frame, gt_frame, gt_emb) in enumerate(
        zip(output_frames, gt_data.images, gt_data.embeddings)
    ):
        out_frame = _ensure_rgb_uint8_image(out_frame)
        gt_frame = _ensure_rgb_uint8_image(gt_frame)
        if out_frame.shape != gt_frame.shape:
            raise ValueError(
                f"Frame shape mismatch for case {case_id}, frame {i}: "
                f"output={out_frame.shape}, gt={gt_frame.shape}"
            )
        out_emb = compute_clip_embedding(out_frame)
        clip_similarity = compute_clip_similarity(out_emb, gt_emb)
        ssim = compute_ssim(out_frame, gt_frame)
        psnr = compute_psnr(out_frame, gt_frame)
        mean_abs_diff = compute_mean_abs_diff(out_frame, gt_frame)
        similarity_scores.append(clip_similarity)
        frame_metrics.append(
            FrameConsistencyMetrics(
                frame_index=i,
                clip_similarity=clip_similarity,
                ssim=ssim,
                psnr=psnr,
                mean_abs_diff=mean_abs_diff,
                clip_passed=clip_similarity >= thresholds.clip_threshold,
                ssim_passed=ssim >= thresholds.ssim_threshold,
                psnr_passed=psnr >= thresholds.psnr_threshold,
                mean_abs_diff_passed=(
                    mean_abs_diff <= thresholds.mean_abs_diff_threshold
                ),
            )
        )

    min_similarity = min(similarity_scores)
    min_ssim = min(metric.ssim for metric in frame_metrics)
    min_psnr = min(metric.psnr for metric in frame_metrics)
    max_mean_abs_diff = max(metric.mean_abs_diff for metric in frame_metrics)
    passed = all(
        metric.clip_passed
        and metric.ssim_passed
        and metric.psnr_passed
        and metric.mean_abs_diff_passed
        for metric in frame_metrics
    )

    result = ConsistencyResult(
        case_id=case_id,
        passed=passed,
        similarity_scores=similarity_scores,
        min_similarity=min_similarity,
        threshold=thresholds.clip_threshold,
        min_ssim=min_ssim,
        min_psnr=min_psnr,
        max_mean_abs_diff=max_mean_abs_diff,
        thresholds=thresholds,
        frame_metrics=frame_metrics,
    )

    status = "PASSED" if passed else "FAILED"
    print(f"\n{'=' * 60}")
    print(f"[Consistency Check] {case_id}: {status}")
    print(
        "  Thresholds: "
        f"clip>={thresholds.clip_threshold}, "
        f"ssim>={thresholds.ssim_threshold}, "
        f"psnr>={thresholds.psnr_threshold}, "
        f"mean_abs_diff<={thresholds.mean_abs_diff_threshold}"
    )
    print(f"  Min similarity: {min_similarity:.4f}")
    print(f"  Min SSIM: {min_ssim:.4f}")
    print(f"  Min PSNR: {min_psnr:.4f}")
    print(f"  Max mean_abs_diff: {max_mean_abs_diff:.4f}")
    print("  Frame details:")
    for metric in frame_metrics:
        frame_status = (
            "PASS"
            if (
                metric.clip_passed
                and metric.ssim_passed
                and metric.psnr_passed
                and metric.mean_abs_diff_passed
            )
            else "FAIL"
        )
        print(
            f"    Frame {metric.frame_index}: "
            f"clip={metric.clip_similarity:.4f} "
            f"ssim={metric.ssim:.4f} "
            f"psnr={metric.psnr:.4f} "
            f"mean_abs_diff={metric.mean_abs_diff:.4f} "
            f"{frame_status}"
        )
    print(f"{'=' * 60}\n")

    return result


def _safe_artifact_name(name: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in name)


def _format_metric_value(value: float) -> str:
    if math.isinf(value):
        return "inf"
    if math.isnan(value):
        return "nan"
    return f"{value:.4f}"


def _json_metric_value(value: float) -> float | str:
    if math.isinf(value) or math.isnan(value):
        return _format_metric_value(value)
    return round(value, 6)


def _metric_items(metric: FrameConsistencyMetrics) -> list[tuple[str, float, bool]]:
    return [
        ("clip", metric.clip_similarity, metric.clip_passed),
        ("ssim", metric.ssim, metric.ssim_passed),
        ("psnr", metric.psnr, metric.psnr_passed),
        ("mean_abs_diff", metric.mean_abs_diff, metric.mean_abs_diff_passed),
    ]


def _text_width(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> int:
    box = draw.textbbox((0, 0), text, font=font)
    return box[2] - box[0]


def _resize_for_comparison(image: np.ndarray, max_size: tuple[int, int]) -> Image.Image:
    pil_image = Image.fromarray(_ensure_rgb_uint8_image(image)).copy()
    pil_image.thumbnail(max_size, Image.Resampling.LANCZOS)
    return pil_image


def _draw_metric_items(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    metric: FrameConsistencyMetrics,
    font: ImageFont.ImageFont,
) -> None:
    cursor = x
    for index, (name, value, passed) in enumerate(_metric_items(metric)):
        text = f"{name}={_format_metric_value(value)}"
        fill = (30, 110, 55) if passed else (185, 35, 35)
        draw.text((cursor, y), text, fill=fill, font=font)
        cursor += _text_width(draw, text, font)
        if index != 3:
            separator = " | "
            draw.text((cursor, y), separator, fill=(95, 95, 95), font=font)
            cursor += _text_width(draw, separator, font)


def _make_consistency_failure_image(
    case_id: str,
    num_gpus: int,
    output_frames: list[np.ndarray],
    gt_data: LoadedConsistencyGT,
    result: ConsistencyResult,
    is_video: bool,
) -> Image.Image:
    font = ImageFont.load_default()
    max_thumb_size = (520, 520) if len(output_frames) == 1 else (480, 320)
    gt_thumbs = [
        _resize_for_comparison(image, max_thumb_size) for image in gt_data.images
    ]
    output_thumbs = [
        _resize_for_comparison(image, max_thumb_size) for image in output_frames
    ]
    thumb_width = max_thumb_size[0]

    margin = 24
    column_gap = 24
    label_height = 42
    metric_height = 30
    row_gap = 18
    frame_rows = []
    for gt_image, output_image in zip(gt_thumbs, output_thumbs):
        image_height = max(gt_image.height, output_image.height)
        frame_rows.append((gt_image, output_image, image_height))

    header_lines = [
        f"Consistency failure: {case_id}",
        f"modality={'video' if is_video else 'image'} | gpus={num_gpus} | frames={len(output_frames)}",
        (
            "thresholds: "
            f"clip>={result.thresholds.clip_threshold} "
            f"ssim>={result.thresholds.ssim_threshold} "
            f"psnr>={result.thresholds.psnr_threshold} "
            f"mean_abs_diff<={result.thresholds.mean_abs_diff_threshold}"
        ),
        (
            "worst: "
            f"clip={_format_metric_value(result.min_similarity)} "
            f"ssim={_format_metric_value(result.min_ssim)} "
            f"psnr={_format_metric_value(result.min_psnr)} "
            f"mean_abs_diff={_format_metric_value(result.max_mean_abs_diff)}"
        ),
    ]
    header_height = 24 + len(header_lines) * 18 + 16
    width = max(960, margin * 2 + thumb_width * 2 + column_gap)
    height = (
        margin
        + header_height
        + sum(label_height + row[2] + metric_height for row in frame_rows)
        + row_gap * max(0, len(frame_rows) - 1)
        + margin
    )

    image = Image.new("RGB", (width, height), (245, 246, 248))
    draw = ImageDraw.Draw(image)

    y = margin
    for line in header_lines:
        draw.text((margin, y), line, fill=(25, 25, 25), font=font)
        y += 18
    y = margin + header_height

    left_x = margin
    right_x = margin + thumb_width + column_gap
    for idx, (gt_image, output_image, image_height) in enumerate(frame_rows):
        row_height = label_height + image_height + metric_height
        draw.rectangle(
            [margin - 8, y - 8, width - margin + 8, y + row_height + 8],
            fill=(255, 255, 255),
            outline=(222, 225, 230),
        )
        frame_label = "image" if len(frame_rows) == 1 else f"frame {idx}"
        draw.text((left_x, y), f"GT {frame_label}", fill=(35, 35, 35), font=font)
        draw.text(
            (right_x, y), f"CI generated {frame_label}", fill=(35, 35, 35), font=font
        )

        image_y = y + label_height
        image.paste(gt_image, (left_x + (thumb_width - gt_image.width) // 2, image_y))
        image.paste(
            output_image,
            (right_x + (thumb_width - output_image.width) // 2, image_y),
        )

        metric_y = image_y + image_height + 10
        _draw_metric_items(draw, left_x, metric_y, result.frame_metrics[idx], font)
        y += row_height + row_gap

    return image


def _consistency_failure_record(
    case_id: str,
    num_gpus: int,
    result: ConsistencyResult,
    is_video: bool,
    output_format: str | None,
    image_name: str,
    gt_remote_files: list[tuple[str, str]] | None,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "num_gpus": num_gpus,
        "is_video": is_video,
        "output_format": output_format,
        "comparison_png": image_name,
        "metrics": {
            "min_clip_similarity": _json_metric_value(result.min_similarity),
            "min_ssim": _json_metric_value(result.min_ssim),
            "min_psnr": _json_metric_value(result.min_psnr),
            "max_mean_abs_diff": _json_metric_value(result.max_mean_abs_diff),
        },
        "thresholds": {
            "clip_threshold": result.thresholds.clip_threshold,
            "ssim_threshold": result.thresholds.ssim_threshold,
            "psnr_threshold": result.thresholds.psnr_threshold,
            "mean_abs_diff_threshold": result.thresholds.mean_abs_diff_threshold,
        },
        "frames": [
            {
                "frame_index": metric.frame_index,
                "clip_similarity": _json_metric_value(metric.clip_similarity),
                "ssim": _json_metric_value(metric.ssim),
                "psnr": _json_metric_value(metric.psnr),
                "mean_abs_diff": _json_metric_value(metric.mean_abs_diff),
                "clip_passed": metric.clip_passed,
                "ssim_passed": metric.ssim_passed,
                "psnr_passed": metric.psnr_passed,
                "mean_abs_diff_passed": metric.mean_abs_diff_passed,
            }
            for metric in result.frame_metrics
        ],
        "gt_files": [
            {"filename": filename, "url": url}
            for filename, url in (gt_remote_files or [])
        ],
    }


def _write_consistency_failure_index(
    out_dir: Path,
    records: list[dict[str, Any]],
) -> None:
    sections = []
    for record in sorted(records, key=lambda r: (r["case_id"], r["num_gpus"])):
        case_id = html.escape(record["case_id"])
        png = html.escape(record["comparison_png"])
        metrics = record["metrics"]
        sections.append(
            "<section>"
            f"<h2>{case_id} ({record['num_gpus']} GPU)</h2>"
            "<p>"
            f"clip={metrics['min_clip_similarity']} | "
            f"ssim={metrics['min_ssim']} | "
            f"psnr={metrics['min_psnr']} | "
            f"mean_abs_diff={metrics['max_mean_abs_diff']}"
            "</p>"
            f'<img src="{png}" alt="{case_id} comparison">'
            "</section>"
        )

    doc = (
        '<!doctype html><html><head><meta charset="utf-8">'
        "<title>Diffusion consistency failures</title>"
        "<style>"
        "body{font-family:sans-serif;margin:24px;background:#f5f6f8;color:#202124}"
        "section{margin:0 0 28px;padding:16px;background:white;border:1px solid #ddd;border-radius:6px}"
        "h2{font-size:18px;margin:0 0 8px}"
        "p{margin:0 0 12px;color:#444}"
        "img{max-width:100%;height:auto;border:1px solid #ddd}"
        "</style></head><body>"
        "<h1>Diffusion consistency failures</h1>" + "".join(sections) + "</body></html>"
    )
    (out_dir / "index.html").write_text(doc, encoding="utf-8")


def save_consistency_failure_artifact(
    artifact_dir: str | Path | None,
    case_id: str,
    num_gpus: int,
    output_frames: list[np.ndarray],
    gt_data: LoadedConsistencyGT,
    result: ConsistencyResult,
    is_video: bool,
    output_format: str | None = None,
    gt_remote_files: list[tuple[str, str]] | None = None,
) -> Path | None:
    if not artifact_dir:
        return None

    out_dir = Path(artifact_dir) / "consistency_failures"
    out_dir.mkdir(parents=True, exist_ok=True)

    safe_case_id = _safe_artifact_name(case_id)
    image_name = f"{safe_case_id}.png"
    image_path = out_dir / image_name
    comparison = _make_consistency_failure_image(
        case_id=case_id,
        num_gpus=num_gpus,
        output_frames=output_frames,
        gt_data=gt_data,
        result=result,
        is_video=is_video,
    )
    comparison.save(image_path)

    record = _consistency_failure_record(
        case_id=case_id,
        num_gpus=num_gpus,
        result=result,
        is_video=is_video,
        output_format=output_format,
        image_name=image_name,
        gt_remote_files=gt_remote_files,
    )
    case_json_path = out_dir / f"{safe_case_id}.json"
    case_json_path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")

    summary_path = out_dir / "summary.json"
    records = []
    if summary_path.exists():
        records = json.loads(summary_path.read_text(encoding="utf-8"))
    records = [
        item
        for item in records
        if not (item.get("case_id") == case_id and item.get("num_gpus") == num_gpus)
    ]
    records.append(record)
    summary_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")
    _write_consistency_failure_index(out_dir, records)
    return image_path
