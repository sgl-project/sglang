"""
Utility functions for consistency testing of diffusion models.

This module provides functions for:
- Loading and comparing ground truth (GT) CLIP embeddings
- Computing CLIP similarity metrics
- Extracting key frames from videos
- Saving GT embeddings for consistency testing
"""

from __future__ import annotations

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

# Path to the GT metadata file (kept in sglang repo)
GT_METADATA_PATH = Path(__file__).parent.parent / "consistency_gt" / "gt_metadata.json"

# Path to the embeddings directory (kept in sglang repo)
EMBEDDINGS_DIR = Path(__file__).parent.parent / "consistency_gt" / "embeddings"

# CLIP configuration
CLIP_MODEL_NAME = "openai/clip-vit-large-patch14"

# Default thresholds
DEFAULT_CLIP_THRESHOLD_IMAGE = 0.92
DEFAULT_CLIP_THRESHOLD_VIDEO = 0.90
DEFAULT_SEED = 1024
DEFAULT_GENERATOR_DEVICE = "cuda"

# Environment variable for staging directory
GT_STAGING_DIR_ENV = "SGLANG_GT_STAGING_DIR"

# Global cache for CLIP model (singleton pattern)
_clip_model_cache: dict[str, Any] = {}


@dataclass
class ConsistencyConfig:
    """Configuration for a consistency test case."""

    case_id: str
    num_gpus: int
    seed: int
    generator_device: str
    clip_threshold: float
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
    similarity_scores: list[float]
    min_similarity: float
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

        # Move to GPU if available
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

    # Convert numpy to PIL Image
    pil_image = Image.fromarray(image)

    # Process image
    inputs = processor(images=pil_image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Get embedding
    with torch.no_grad():
        image_features = model.get_image_features(**inputs)
        # L2 normalize
        image_features = image_features / image_features.norm(dim=-1, keepdim=True)

    return image_features.cpu().numpy().flatten()


def compute_clip_similarity(emb1: np.ndarray, emb2: np.ndarray) -> float:
    """
    Compute cosine similarity between two CLIP embeddings.

    Args:
        emb1: 768-dimensional numpy array
        emb2: 768-dimensional numpy array

    Returns:
        Cosine similarity score [-1, 1], typically [0.5, 1.0] for images
    """
    # Both embeddings should already be L2 normalized
    # Cosine similarity = dot product for normalized vectors
    similarity = np.dot(emb1, emb2)
    return float(similarity)


def load_gt_embeddings(
    case_id: str, num_gpus: int, is_video: bool = False
) -> list[np.ndarray]:
    """
    Load ground truth CLIP embeddings for a specific case.

    Args:
        case_id: Test case identifier.
        num_gpus: Number of GPUs used.
        is_video: Whether this is a video case (default: False).

    Returns:
        List of numpy arrays (embeddings) for each key frame.
        Image: 1 embedding, Video: 3 embeddings (first, mid, last)

    Raises:
        FileNotFoundError: If GT embeddings are not available.
    """
    embedding_path = EMBEDDINGS_DIR / f"{num_gpus}-gpu" / f"{case_id}.npy"

    if not embedding_path.exists():
        raise FileNotFoundError(
            f"GT embedding not found at {embedding_path}. "
            f"Please generate GT embeddings first."
        )

    embeddings = np.load(embedding_path)

    # Handle shape: (768,) for image, (3, 768) for video
    if embeddings.ndim == 1:
        embeddings = [embeddings]
    else:
        embeddings = list(embeddings)

    logger.info(
        f"Loaded {len(embeddings)} GT embeddings for {case_id} from {embedding_path}"
    )
    return embeddings


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
        Always returns exactly 3 frames to ensure consistency with GT naming.
        For very short videos, frames may be duplicated.
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

        # Always use 3 indices to ensure consistent frame count for GT naming
        # (frame_0.png, frame_mid.png, frame_last.png)
        # For very short videos, some indices may be duplicated
        key_indices = [first_idx, mid_idx, last_idx]

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
    import io

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

    Args:
        output_frames: List of output frames as numpy arrays
        gt_embeddings: List of GT CLIP embeddings
        threshold: Similarity threshold for passing
        case_id: Test case identifier

    Returns:
        ConsistencyResult with comparison details
    """
    if len(output_frames) != len(gt_embeddings):
        raise ValueError(
            f"Frame count mismatch: output={len(output_frames)}, gt={len(gt_embeddings)}"
        )

    similarity_scores = []
    frame_details = []

    for i, (out_frame, gt_emb) in enumerate(zip(output_frames, gt_embeddings)):
        # Compute CLIP embedding for output frame
        out_emb = compute_clip_embedding(out_frame)

        # Compute similarity
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

    # Always print detailed similarity info
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
        metadata.get("default_clip_threshold_video", DEFAULT_CLIP_THRESHOLD_VIDEO)
        if is_video
        else metadata.get("default_clip_threshold_image", DEFAULT_CLIP_THRESHOLD_IMAGE)
    )
    threshold = case_meta.get("clip_threshold", default_threshold)

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
        clip_threshold=threshold,
        key_frame_indices=key_frame_indices,
        resolution=case.sampling_params.output_size,
        num_frames=case.sampling_params.num_frames,
    )


def save_embeddings_as_gt(
    frames: list[np.ndarray],
    case_id: str,
    num_gpus: int,
    is_video: bool = False,
    output_dir: Path | None = None,
) -> Path:
    """
    Convert frames to CLIP embeddings and save as .npy file.

    Args:
        frames: List of frames as numpy arrays
        case_id: Test case identifier
        num_gpus: Number of GPUs used
        is_video: Whether this is a video (affects expected frame count)
        output_dir: Output directory (defaults to EMBEDDINGS_DIR)

    Returns:
        Path to the saved .npy file
    """
    if output_dir is None:
        output_dir = EMBEDDINGS_DIR

    case_dir = output_dir / f"{num_gpus}-gpu"
    case_dir.mkdir(parents=True, exist_ok=True)

    # Compute embeddings for all frames
    embeddings = []
    for frame in frames:
        emb = compute_clip_embedding(frame)
        embeddings.append(emb)

    # Stack embeddings: (768,) for single frame, (N, 768) for multiple
    if len(embeddings) == 1:
        embeddings_array = embeddings[0]
    else:
        embeddings_array = np.stack(embeddings)

    # Save as .npy
    output_path = case_dir / f"{case_id}.npy"
    np.save(output_path, embeddings_array)
    logger.info(f"Saved GT embeddings: {output_path} (shape: {embeddings_array.shape})")

    return output_path


def gt_exists(case_id: str, num_gpus: int) -> bool:
    """Check if GT embedding exists locally."""
    embedding_path = EMBEDDINGS_DIR / f"{num_gpus}-gpu" / f"{case_id}.npy"
    return embedding_path.exists()


def save_gt_to_staging(
    frames: list[np.ndarray],
    case: DiffusionTestCase,
    is_video: bool,
) -> Path | None:
    """
    Save GT embeddings to staging directory for later commit to sglang repo.

    This function is called when GT doesn't exist during test execution.
    The staging directory can be committed directly to the sglang repository.

    Args:
        frames: List of frames as numpy arrays
        case: Test case configuration
        is_video: Whether this is a video (affects frame naming)

    Returns:
        Path to saved GT embedding file, or None if staging is not enabled.
    """
    staging_dir = get_staging_dir()
    if staging_dir is None:
        return None

    num_gpus = case.server_args.num_gpus
    case_id = case.id

    gt_path = save_embeddings_as_gt(
        frames=frames,
        case_id=case_id,
        num_gpus=num_gpus,
        is_video=is_video,
        output_dir=staging_dir,
    )

    logger.info(f"[Staging] Saved GT embeddings for {case_id} to {gt_path}")
    return gt_path
