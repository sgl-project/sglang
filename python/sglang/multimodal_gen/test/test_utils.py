# Copied and adapted from: https://github.com/hao-ai-lab/FastVideo
import base64
import io
import json
import os
import socket
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from PIL import Image

from sglang.multimodal_gen.runtime.utils.common import get_bool_env_var
from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.runtime.utils.perf_logger import (
    RequestPerfRecord,
    get_diffusion_perf_log_dir,
)
from sglang.multimodal_gen.test.server.testcase_configs import DiffusionTestCase

logger = init_logger(__name__)

# --- Consistency testing: GT from sgl-test-files ---
SGL_TEST_FILES_CONSISTENCY_GT_BASE = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/main/diffusion-ci/consistency_gt"
CONSISTENCY_THRESHOLD_JSON_PATH = (
    Path(__file__).resolve().parent / "server" / "consistency_threshold.json"
)
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"
DEFAULT_CLIP_THRESHOLD_IMAGE = 0.92
DEFAULT_CLIP_THRESHOLD_VIDEO = 0.90
_clip_model_cache: dict[str, Any] = {}


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


# --- Consistency testing (GT from sgl-test-files, embeddings computed on the fly) ---


def _load_threshold_json() -> dict[str, Any]:
    """Load consistency_threshold.json; returns {} if missing."""
    if not CONSISTENCY_THRESHOLD_JSON_PATH.exists():
        return {}
    with CONSISTENCY_THRESHOLD_JSON_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_clip_threshold(
    case: DiffusionTestCase,
    metadata: dict[str, Any] | None = None,
) -> float:
    """
    Get CLIP similarity threshold for a consistency test case.
    Uses consistency_threshold.json: default_clip_threshold_image/video and
    optional per-case override in cases.<case_id>.clip_threshold.
    """
    if metadata is None:
        metadata = _load_threshold_json()
    case_meta = metadata.get("cases", {}).get(case.id, {})
    is_video = case.server_args.modality == "video"
    default = (
        metadata.get("default_clip_threshold_video", DEFAULT_CLIP_THRESHOLD_VIDEO)
        if is_video
        else metadata.get("default_clip_threshold_image", DEFAULT_CLIP_THRESHOLD_IMAGE)
    )
    return float(case_meta.get("clip_threshold", default))


@dataclass
class ConsistencyResult:
    """Result of a consistency comparison."""

    case_id: str
    passed: bool
    similarity_scores: list[float]
    min_similarity: float
    threshold: float
    frame_details: list[dict[str, Any]]


def get_clip_model() -> tuple[Any, Any]:
    """
    Get CLIP model and processor (lazy loading with singleton pattern).

    Returns:
        Tuple of (model, processor)

    Raises:
        ImportError: If transformers is not installed
    """
    global _clip_model_cache

    if "model" not in _clip_model_cache:
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor
        except ImportError:
            raise ImportError(
                "transformers and torch are required for CLIP consistency check. "
                "Install with: pip install transformers torch"
            )

        logger.info(f"Loading CLIP model: {CLIP_MODEL_NAME}")
        processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
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
    """
    Compute CLIP embedding for a single image.

    Args:
        image: numpy array (H, W, C) in RGB format

    Returns:
        768-dimensional numpy array (L2 normalized)
    """
    try:
        import torch
    except ImportError:
        raise ImportError(
            "torch is required for CLIP consistency check. "
            "Install with: pip install torch"
        )

    model, processor = get_clip_model()
    device = _clip_model_cache["device"]

    pil_image = Image.fromarray(image)
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy().flatten()


def compute_clip_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two CLIP embeddings.
    """
    similarity = np.dot(emb1, emb2)
    return float(similarity)


def _consistency_gt_filenames(case_id: str, is_video: bool) -> list[str]:
    """Return the list of GT image filenames for a case."""
    if is_video:
        return [
            f"{case_id}_frame_0.png",
            f"{case_id}_frame_mid.png",
            f"{case_id}_frame_last.png",
        ]
    return [f"{case_id}.png"]


def load_gt_embeddings(
    case_id: str, num_gpus: int, is_video: bool = False
) -> list[np.ndarray]:
    """
    Load ground truth by downloading PNG(s) from sgl-test-files and computing CLIP embeddings.

    Args:
        case_id: Test case identifier.
        num_gpus: Unused, kept for API compatibility.
        is_video: Whether this is a video case (default: False).

    Returns:
        List of numpy arrays (embeddings) for each key frame.
        Image: 1 embedding, Video: 3 embeddings (first, mid, last)

    Raises:
        FileNotFoundError: If GT images are not available at the expected URLs.
    """
    filenames = _consistency_gt_filenames(case_id, is_video)
    embeddings = []

    for fn in filenames:
        url = f"{SGL_TEST_FILES_CONSISTENCY_GT_BASE}/{fn}"
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            raise FileNotFoundError(f"GT image not found: {url}")

        arr = np.array(Image.open(io.BytesIO(resp.content)).convert("RGB"))
        emb = compute_clip_embedding(arr)
        embeddings.append(emb)

    logger.info(
        f"Loaded {len(embeddings)} GT embeddings for {case_id} from sgl-test-files"
    )
    return embeddings


def gt_exists(case_id: str, num_gpus: int, is_video: bool = False) -> bool:
    """Check if GT image(s) exist at sgl-test-files (by requesting the first required file)."""
    filenames = _consistency_gt_filenames(case_id, is_video)
    fn = filenames[0]
    url = f"{SGL_TEST_FILES_CONSISTENCY_GT_BASE}/{fn}"
    try:
        r = requests.head(url, timeout=10)
        return r.status_code == 200
    except Exception:
        return False


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
    gt_embeddings: list[np.ndarray],
    threshold: float,
    case_id: str,
) -> ConsistencyResult:
    """
    Compare output frames with ground truth embeddings using CLIP similarity.
    """
    if len(output_frames) != len(gt_embeddings):
        raise ValueError(
            f"Frame count mismatch: output={len(output_frames)}, gt={len(gt_embeddings)}"
        )

    similarity_scores = []
    frame_details = []

    for i, (out_frame, gt_emb) in enumerate(zip(output_frames, gt_embeddings)):
        out_emb = compute_clip_embedding(out_frame)
        similarity = compute_clip_similarity(out_emb, gt_emb)
        similarity_scores.append(similarity)
        frame_details.append(
            {
                "frame_index": i,
                "similarity": similarity,
                "passed": similarity >= threshold,
                "output_shape": out_frame.shape,
            }
        )

    min_similarity = min(similarity_scores)
    passed = all(s >= threshold for s in similarity_scores)

    result = ConsistencyResult(
        case_id=case_id,
        passed=passed,
        similarity_scores=similarity_scores,
        min_similarity=min_similarity,
        threshold=threshold,
        frame_details=frame_details,
    )

    status = "PASSED" if passed else "FAILED"
    print(f"\n{'='*60}")
    print(f"[CLIP Consistency] {case_id}: {status}")
    print(f"  Threshold: {threshold}")
    print(f"  Min similarity: {min_similarity:.4f}")
    print(f"  Frame details:")
    for detail in frame_details:
        frame_status = "✓" if detail["passed"] else "✗"
        print(
            f"    Frame {detail['frame_index']}: similarity={detail['similarity']:.4f} {frame_status}"
        )
    print(f"{'='*60}\n")

    if passed:
        logger.info(
            f"[Consistency] {case_id}: PASSED (min_similarity={min_similarity:.4f}, threshold={threshold})"
        )
    else:
        logger.error(
            f"[Consistency] {case_id}: FAILED (min_similarity={min_similarity:.4f}, threshold={threshold})"
        )
        for detail in frame_details:
            if not detail["passed"]:
                logger.error(
                    f"  Frame {detail['frame_index']}: similarity={detail['similarity']:.4f} < {threshold}"
                )

    return result
