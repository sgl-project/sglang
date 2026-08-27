import math

import numpy as np

from sglang.multimodal_gen.test import test_utils
from sglang.multimodal_gen.test.test_utils import (
    ConsistencyThresholds,
    LoadedConsistencyGT,
    compare_with_gt,
    compute_mean_abs_diff,
    compute_psnr,
    compute_ssim,
    save_consistency_failure_artifact,
)


def _solid_image(value: int, size: int = 32) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


def test_consistency_gt_urls_are_pinned_to_ci_data_revision():
    revision_path = f"/ci-data/{test_utils.SGL_TEST_FILES_CI_DATA_REVISION}/"

    assert "/ci-data/main/" not in test_utils.SGL_TEST_FILES_CONSISTENCY_GT_ROOT
    assert revision_path in test_utils.SGL_TEST_FILES_OFFICIAL_CONSISTENCY_GT_BASE
    assert revision_path in test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE


def test_pixel_metrics_identical_image():
    image = _solid_image(128)

    ssim = compute_ssim(image, image)
    psnr = compute_psnr(image, image)
    mean_abs_diff = compute_mean_abs_diff(image, image)

    assert ssim == 1.0
    assert math.isinf(psnr)
    assert mean_abs_diff == 0.0


def test_pixel_metrics_detect_different_image():
    image = _solid_image(128)
    other = _solid_image(0)

    ssim = compute_ssim(image, other)
    psnr = compute_psnr(image, other)
    mean_abs_diff = compute_mean_abs_diff(image, other)

    assert ssim < 0.95
    assert psnr < 28.0
    assert mean_abs_diff > 8.0


def test_compare_with_gt_passes_for_identical_image(monkeypatch):
    gt_image = _solid_image(128)

    monkeypatch.setattr(
        test_utils,
        "compute_clip_embedding",
        lambda image: np.array([1.0, 0.0], dtype=np.float32),
    )

    result = compare_with_gt(
        output_frames=[gt_image.copy()],
        gt_data=LoadedConsistencyGT(
            images=[gt_image.copy()],
            embeddings=[np.array([1.0, 0.0], dtype=np.float32)],
        ),
        thresholds=ConsistencyThresholds(
            clip_threshold=0.92,
            ssim_threshold=0.95,
            psnr_threshold=28.0,
            mean_abs_diff_threshold=8.0,
        ),
        case_id="unit_image_pass",
    )

    assert result.passed is True
    assert result.min_similarity == 1.0
    assert result.min_ssim == 1.0
    assert math.isinf(result.min_psnr)
    assert result.max_mean_abs_diff == 0.0


def test_compare_with_gt_uses_worst_frame_for_video(monkeypatch):
    gt_frame_0 = _solid_image(128)
    gt_frame_1 = _solid_image(128)
    bad_frame = _solid_image(0)

    monkeypatch.setattr(
        test_utils,
        "compute_clip_embedding",
        lambda image: np.array([1.0, 0.0], dtype=np.float32),
    )

    result = compare_with_gt(
        output_frames=[gt_frame_0.copy(), bad_frame],
        gt_data=LoadedConsistencyGT(
            images=[gt_frame_0.copy(), gt_frame_1.copy()],
            embeddings=[
                np.array([1.0, 0.0], dtype=np.float32),
                np.array([1.0, 0.0], dtype=np.float32),
            ],
        ),
        thresholds=ConsistencyThresholds(
            clip_threshold=0.92,
            ssim_threshold=0.95,
            psnr_threshold=28.0,
            mean_abs_diff_threshold=8.0,
        ),
        case_id="unit_video_fail",
    )

    assert result.passed is False
    assert result.min_similarity == 1.0
    assert result.min_ssim < 0.95
    assert result.min_psnr < 28.0
    assert result.max_mean_abs_diff > 8.0
    assert any(
        not metric.ssim_passed
        or not metric.psnr_passed
        or not metric.mean_abs_diff_passed
        for metric in result.frame_metrics
    )


def test_save_consistency_failure_artifact(tmp_path, monkeypatch):
    gt_image = _solid_image(128)
    bad_image = _solid_image(0)

    monkeypatch.setattr(
        test_utils,
        "compute_clip_embedding",
        lambda image: np.array([1.0, 0.0], dtype=np.float32),
    )

    result = compare_with_gt(
        output_frames=[bad_image],
        gt_data=LoadedConsistencyGT(
            images=[gt_image],
            embeddings=[np.array([1.0, 0.0], dtype=np.float32)],
        ),
        thresholds=ConsistencyThresholds(
            clip_threshold=0.92,
            ssim_threshold=0.95,
            psnr_threshold=28.0,
            mean_abs_diff_threshold=8.0,
        ),
        case_id="unit_image_fail",
    )

    artifact_path = save_consistency_failure_artifact(
        artifact_dir=tmp_path,
        case_id="unit_image_fail",
        num_gpus=1,
        output_frames=[bad_image],
        gt_data=LoadedConsistencyGT(
            images=[gt_image],
            embeddings=[np.array([1.0, 0.0], dtype=np.float32)],
        ),
        result=result,
        is_video=False,
        output_format="png",
        gt_remote_files=[("unit_image_fail_1gpu.png", "https://example.com/gt.png")],
    )

    assert artifact_path is not None
    assert artifact_path.exists()
    assert artifact_path.suffix == ".png"
    assert (tmp_path / "consistency_failures" / "summary.json").exists()
    assert (tmp_path / "consistency_failures" / "index.html").exists()
