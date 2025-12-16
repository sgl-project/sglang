"""
Utility functions for consistency testing of diffusion models.

This module provides functions for:
- Loading and comparing ground truth (GT) frames
- Computing SSIM similarity metrics
- Extracting key frames from videos
- Saving GT frames for consistency testing
"""

from __future__ import annotations

import io
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from sglang.multimodal_gen.runtime.utils.logging_utils import init_logger
from sglang.multimodal_gen.test.server.testcase_configs import DiffusionTestCase

logger = init_logger(__name__)

# Remote GT base URL (sgl-test-files repository)
GT_REMOTE_BASE_URL = "https://raw.githubusercontent.com/sgl-project/sgl-test-files/refs/heads/main/images/consistency_gt"

# Path to the GT metadata file (kept in sglang repo)
GT_METADATA_PATH = Path(__file__).parent.parent / "consistency_gt" / "gt_metadata.json"

# Default thresholds
DEFAULT_SSIM_THRESHOLD_IMAGE = 0.98
DEFAULT_SSIM_THRESHOLD_VIDEO = 0.95
DEFAULT_SEED = 1024
DEFAULT_GENERATOR_DEVICE = "cuda"

# Environment variable for staging directory
GT_STAGING_DIR_ENV = "SGLANG_GT_STAGING_DIR"


@dataclass
class ConsistencyConfig:
    """Configuration for a consistency test case."""

    case_id: str
    num_gpus: int
    seed: int
    generator_device: str
    ssim_threshold: float
    key_frame_indices: list[
        int | str
    ]  # int for specific index, "mid", "last" for dynamic
    resolution: str | None = None
    num_frames: int | None = None


@dataclass
class ConsistencyResult:
    """Result of a consistency comparison."""

    case_id: str
    passed: bool
    ssim_scores: list[float]
    min_ssim: float
    threshold: float
    frame_details: list[dict[str, Any]]


def load_gt_metadata() -> dict[str, Any]:
    """Load the ground truth metadata from gt_metadata.json."""
    if not GT_METADATA_PATH.exists():
        logger.warning(f"GT metadata not found at {GT_METADATA_PATH}")
        return {"cases": {}}

    with GT_METADATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_gt_metadata(metadata: dict[str, Any]) -> None:
    """Save the ground truth metadata to gt_metadata.json."""
    GT_METADATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with GT_METADATA_PATH.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)


def get_staging_dir() -> Path | None:
    """Get the staging directory from environment variable.

    Returns:
        Path to staging directory, or None if not set.
    """
    staging_dir = os.environ.get(GT_STAGING_DIR_ENV)
    if staging_dir:
        return Path(staging_dir)
    return None


def load_gt_frames(case_id: str, num_gpus: int) -> list[np.ndarray]:
    """
    Load ground truth frames for a specific case from remote repository.

    Downloads frames from sgl-test-files repository and caches locally.

    Returns:
        List of numpy arrays (RGB images) for each key frame.

    Raises:
        FileNotFoundError: If GT frames are not available remotely.
    """
    from sglang.utils import download_and_cache_file

    # Determine frame names from metadata
    metadata = load_gt_metadata()
    case_meta = metadata.get("cases", {}).get(case_id, {})

    if case_meta.get("num_frames", 1) == 3:
        # Video with 3 key frames
        frame_names = ["frame_0.png", "frame_mid.png", "frame_last.png"]
    else:
        # Single image
        frame_names = ["frame_0.png"]

    frames = []
    for frame_name in frame_names:
        # Construct URL and cache path
        url = f"{GT_REMOTE_BASE_URL}/{num_gpus}-gpu/{case_id}/{frame_name}"
        cache_filename = f"/tmp/sglang_gt_{num_gpus}gpu_{case_id}_{frame_name}"

        try:
            # Download and cache
            local_path = download_and_cache_file(url, cache_filename)
        except Exception as e:
            raise FileNotFoundError(
                f"GT frame not found at {url}. "
                f"Please upload GT files to sgl-test-files repository. Error: {e}"
            )

        img = Image.open(local_path).convert("RGB")
        frames.append(np.array(img))

    logger.info(f"Loaded {len(frames)} GT frames for {case_id} from remote repository")
    return frames


def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Compute Structural Similarity Index (SSIM) between two images.

    Args:
        img1: First image as numpy array (H, W, C) in RGB format
        img2: Second image as numpy array (H, W, C) in RGB format

    Returns:
        SSIM score between 0 and 1 (1 = identical)
    """
    try:
        from skimage.metrics import structural_similarity as ssim
    except ImportError:
        raise ImportError(
            "scikit-image is required for SSIM computation. "
            "Install it with: pip install scikit-image"
        )

    # Ensure images have the same shape
    if img1.shape != img2.shape:
        raise ValueError(f"Image shapes do not match: {img1.shape} vs {img2.shape}")

    # Compute SSIM for each channel and average
    # Use channel_axis parameter for multichannel images
    score = ssim(img1, img2, channel_axis=2, data_range=255)
    return float(score)


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
        List of numpy arrays [first_frame, middle_frame, last_frame]
    """
    try:
        import cv2
    except ImportError:
        raise ImportError(
            "opencv-python is required for video frame extraction. "
            "Install it with: pip install opencv-python"
        )

    # Write video to temporary file for OpenCV to read
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

        # Calculate key frame indices
        first_idx = 0
        mid_idx = total_frames // 2
        last_idx = total_frames - 1

        key_indices = [first_idx, mid_idx, last_idx]
        # Remove duplicates while preserving order (for very short videos)
        key_indices = list(dict.fromkeys(key_indices))

        frames = []
        for idx in key_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if not ret:
                raise ValueError(f"Failed to read frame at index {idx}")
            # Convert BGR to RGB
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


def compare_frames_with_gt(
    output_frames: list[np.ndarray],
    gt_frames: list[np.ndarray],
    threshold: float,
    case_id: str,
) -> ConsistencyResult:
    """
    Compare output frames with ground truth frames using SSIM.

    Args:
        output_frames: List of output frames as numpy arrays
        gt_frames: List of GT frames as numpy arrays
        threshold: SSIM threshold for passing
        case_id: Test case identifier

    Returns:
        ConsistencyResult with comparison details
    """
    if len(output_frames) != len(gt_frames):
        raise ValueError(
            f"Frame count mismatch: output={len(output_frames)}, gt={len(gt_frames)}"
        )

    ssim_scores = []
    frame_details = []

    for i, (out_frame, gt_frame) in enumerate(zip(output_frames, gt_frames)):
        # Resize if shapes don't match (shouldn't happen normally)
        if out_frame.shape != gt_frame.shape:
            logger.warning(
                f"Frame {i} shape mismatch: output={out_frame.shape}, gt={gt_frame.shape}. "
                "Resizing output to match GT."
            )
            out_pil = Image.fromarray(out_frame)
            out_pil = out_pil.resize((gt_frame.shape[1], gt_frame.shape[0]))
            out_frame = np.array(out_pil)

        ssim_score = compute_ssim(out_frame, gt_frame)
        ssim_scores.append(ssim_score)
        frame_details.append(
            {
                "frame_index": i,
                "ssim": ssim_score,
                "passed": ssim_score >= threshold,
                "output_shape": out_frame.shape,
                "gt_shape": gt_frame.shape,
            }
        )

    min_ssim = min(ssim_scores)
    passed = all(s >= threshold for s in ssim_scores)

    result = ConsistencyResult(
        case_id=case_id,
        passed=passed,
        ssim_scores=ssim_scores,
        min_ssim=min_ssim,
        threshold=threshold,
        frame_details=frame_details,
    )

    if passed:
        logger.info(
            f"[Consistency] {case_id}: PASSED (min_ssim={min_ssim:.4f}, threshold={threshold})"
        )
    else:
        logger.error(
            f"[Consistency] {case_id}: FAILED (min_ssim={min_ssim:.4f}, threshold={threshold})"
        )
        for detail in frame_details:
            if not detail["passed"]:
                logger.error(
                    f"  Frame {detail['frame_index']}: SSIM={detail['ssim']:.4f} < {threshold}"
                )

    return result


def get_consistency_config(
    case: DiffusionTestCase,
    metadata: dict[str, Any] | None = None,
) -> ConsistencyConfig:
    """
    Get consistency configuration for a test case.

    Uses metadata if available, otherwise uses defaults.
    """
    if metadata is None:
        metadata = load_gt_metadata()

    case_meta = metadata.get("cases", {}).get(case.id, {})

    # Determine if this is image or video
    is_video = case.server_args.modality == "video"

    # Get threshold (use case-specific if available, otherwise default)
    default_threshold = (
        metadata.get("default_ssim_threshold_video", DEFAULT_SSIM_THRESHOLD_VIDEO)
        if is_video
        else metadata.get("default_ssim_threshold_image", DEFAULT_SSIM_THRESHOLD_IMAGE)
    )
    threshold = case_meta.get("ssim_threshold", default_threshold)

    # Get key frame indices
    if is_video:
        key_frame_indices = case_meta.get("key_frame_indices", [0, "mid", "last"])
    else:
        key_frame_indices = [0]

    return ConsistencyConfig(
        case_id=case.id,
        num_gpus=case.server_args.num_gpus,
        seed=case_meta.get("seed", metadata.get("default_seed", DEFAULT_SEED)),
        generator_device=case_meta.get(
            "generator_device",
            metadata.get("default_generator_device", DEFAULT_GENERATOR_DEVICE),
        ),
        ssim_threshold=threshold,
        key_frame_indices=key_frame_indices,
        resolution=case.sampling_params.output_size,
        num_frames=case.sampling_params.num_frames,
    )


def save_frames_as_gt(
    frames: list[np.ndarray],
    case_id: str,
    num_gpus: int,
    is_video: bool = False,
    output_dir: Path | None = None,
) -> Path:
    """
    Save frames as ground truth PNG files to staging directory.

    Args:
        frames: List of frames as numpy arrays
        case_id: Test case identifier
        num_gpus: Number of GPUs used
        is_video: Whether this is a video (affects naming)
        output_dir: Output directory (required, GT files are no longer stored locally)

    Returns:
        Path to the GT directory
    """
    if output_dir is None:
        raise ValueError(
            "output_dir is required. GT files are now stored in sgl-test-files repository."
        )
    case_dir = output_dir / f"{num_gpus}-gpu" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)

    # Clear existing frames
    for old_file in case_dir.glob("frame_*.png"):
        old_file.unlink()

    # Save frames with appropriate naming
    if is_video and len(frames) == 3:
        names = ["frame_0.png", "frame_mid.png", "frame_last.png"]
    else:
        names = [f"frame_{i}.png" for i in range(len(frames))]

    for frame, name in zip(frames, names):
        img = Image.fromarray(frame)
        img.save(case_dir / name)
        logger.info(f"Saved GT frame: {case_dir / name}")

    return case_dir


def gt_exists(case_id: str, num_gpus: int) -> bool:
    """Check if GT exists (cached locally or available remotely)."""
    import requests

    # Check local cache first
    cache_filename = f"/tmp/sglang_gt_{num_gpus}gpu_{case_id}_frame_0.png"
    if os.path.exists(cache_filename):
        return True

    # Check remote
    url = f"{GT_REMOTE_BASE_URL}/{num_gpus}-gpu/{case_id}/frame_0.png"
    try:
        response = requests.head(url, timeout=10)
        return response.status_code == 200
    except Exception:
        return False


def save_gt_to_staging(
    frames: list[np.ndarray],
    case: DiffusionTestCase,
    is_video: bool,
) -> Path | None:
    """
    Save GT frames to staging directory for later upload to sgl-test-files.

    This function is called when GT doesn't exist during test execution.
    The staging directory can be uploaded as a CI artifact for developers
    to download and upload to the sgl-test-files repository.

    Note: Only frame files are saved, not metadata. Users should manually
    upload frame files to sgl-test-files repository.

    Args:
        frames: List of frames as numpy arrays
        case: Test case configuration
        is_video: Whether this is a video (affects frame naming)

    Returns:
        Path to saved GT directory, or None if staging is not enabled.
    """
    staging_dir = get_staging_dir()
    if staging_dir is None:
        return None

    num_gpus = case.server_args.num_gpus
    case_id = case.id

    # Save frames to staging directory (no metadata)
    gt_path = save_frames_as_gt(
        frames=frames,
        case_id=case_id,
        num_gpus=num_gpus,
        is_video=is_video,
        output_dir=staging_dir,
    )

    # Print detailed instructions for uploading to sgl-test-files
    logger.info(f"[Staging] Saved GT frames for {case_id} to {gt_path}")
    logger.info("=" * 60)
    logger.info("GT FILES READY FOR UPLOAD")
    logger.info("=" * 60)
    logger.info(f"Staging directory: {gt_path}")
    logger.info("")
    logger.info("To add these GT files to the repository:")
    logger.info("1. Clone sgl-test-files repo (if not already):")
    logger.info("   git clone https://github.com/sgl-project/sgl-test-files.git")
    logger.info("")
    logger.info("2. Copy the generated files:")
    logger.info(
        f"   cp -r {gt_path}/* sgl-test-files/images/consistency_gt/{num_gpus}-gpu/{case_id}/"
    )
    logger.info("")
    logger.info("3. Commit and push:")
    logger.info("   cd sgl-test-files")
    logger.info("   git add images/consistency_gt/")
    logger.info(f'   git commit -m "Add consistency GT for {case_id}"')
    logger.info("   git push")
    logger.info("")
    logger.info("4. After the PR is merged, re-run the tests.")
    logger.info("=" * 60)

    return gt_path
