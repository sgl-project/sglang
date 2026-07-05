import math

import numpy as np
import pytest

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


def _set_official_gt_outputs(monkeypatch, outputs_by_case):
    monkeypatch.setattr(
        test_utils,
        "_official_consistency_gt_outputs_for_case",
        lambda case_id: frozenset(outputs_by_case.get(case_id, ())),
    )


@pytest.fixture(autouse=True)
def _disable_remote_official_gt_case_map(monkeypatch):
    _set_official_gt_outputs(monkeypatch, {})


def test_consistency_gt_urls_are_pinned_to_ci_data_revision():
    revision_path = f"/ci-data/{test_utils.SGL_TEST_FILES_CI_DATA_REVISION}/"

    assert "/ci-data/main/" not in test_utils.SGL_TEST_FILES_CONSISTENCY_GT_ROOT
    assert revision_path in test_utils.SGL_TEST_FILES_OFFICIAL_CONSISTENCY_GT_BASE
    assert revision_path in test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE


def test_remote_file_exists_returns_false_for_definitive_404(monkeypatch):
    class Response:
        status_code = 404

        def close(self):
            pass

    monkeypatch.setattr(test_utils.requests, "head", lambda *args, **kwargs: Response())

    assert test_utils._remote_file_exists("https://example.com/missing.png") is False


def test_remote_video_gt_candidates_survive_inconclusive_probe(monkeypatch):
    monkeypatch.setattr(test_utils, "_remote_file_exists", lambda url: None)

    files = test_utils._find_remote_consistency_gt_files(
        "unit_video",
        1,
        is_video=True,
    )

    assert [filename for filename, _ in files] == [
        "unit_video_1gpu_frame_0.png",
        "unit_video_1gpu_frame_mid.png",
        "unit_video_1gpu_frame_last.png",
    ]


def test_remote_image_gt_prefers_official_when_present(monkeypatch):
    monkeypatch.setenv(test_utils.CONSISTENCY_PLATFORM_ENV, "h100")
    official_prefix = test_utils.SGL_TEST_FILES_OFFICIAL_CONSISTENCY_GT_BASE + "/"
    expected_filename = f"unit_image_1gpu.{test_utils.output_format_to_ext(None)}"
    _set_official_gt_outputs(monkeypatch, {"unit_image": [expected_filename]})
    monkeypatch.setattr(
        test_utils,
        "_remote_file_exists",
        lambda url: url.startswith(official_prefix),
    )

    files = test_utils._find_remote_consistency_gt_files(
        "unit_image",
        1,
        is_video=False,
    )

    assert files == [
        (
            expected_filename,
            (
                f"{test_utils.SGL_TEST_FILES_OFFICIAL_CONSISTENCY_GT_BASE}"
                f"/{expected_filename}"
            ),
        )
    ]


def test_remote_image_gt_ignores_unmapped_official_file(monkeypatch):
    monkeypatch.setenv(test_utils.CONSISTENCY_PLATFORM_ENV, "h100")
    official_prefix = test_utils.SGL_TEST_FILES_OFFICIAL_CONSISTENCY_GT_BASE + "/"
    sglang_prefix = test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE + "/"
    expected_filename = f"unit_image_1gpu.{test_utils.output_format_to_ext(None)}"
    _set_official_gt_outputs(monkeypatch, {})
    monkeypatch.setattr(
        test_utils,
        "_remote_file_exists",
        lambda url: url.startswith(official_prefix) or url.startswith(sglang_prefix),
    )

    files = test_utils._find_remote_consistency_gt_files(
        "unit_image",
        1,
        is_video=False,
    )

    assert files == [
        (
            expected_filename,
            (
                f"{test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE}"
                f"/{expected_filename}"
            ),
        )
    ]


def test_remote_video_gt_ignores_unmapped_official_files(monkeypatch):
    monkeypatch.setenv(test_utils.CONSISTENCY_PLATFORM_ENV, "h100")
    official_prefix = test_utils.SGL_TEST_FILES_OFFICIAL_CONSISTENCY_GT_BASE + "/"
    sglang_prefix = test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE + "/"
    case_id = "unit_video"
    _set_official_gt_outputs(monkeypatch, {})
    monkeypatch.setattr(
        test_utils,
        "_remote_file_exists",
        lambda url: url.startswith(official_prefix) or url.startswith(sglang_prefix),
    )

    files = test_utils._find_remote_consistency_gt_files(case_id, 2, is_video=True)

    assert files == [
        (
            f"{case_id}_2gpu_frame_0.png",
            (
                f"{test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE}"
                f"/{case_id}_2gpu_frame_0.png"
            ),
        ),
        (
            f"{case_id}_2gpu_frame_mid.png",
            (
                f"{test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE}"
                f"/{case_id}_2gpu_frame_mid.png"
            ),
        ),
        (
            f"{case_id}_2gpu_frame_last.png",
            (
                f"{test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE}"
                f"/{case_id}_2gpu_frame_last.png"
            ),
        ),
    ]


def test_ltx_hq_remote_gt_uses_sglang_generated_when_official_declared(monkeypatch):
    monkeypatch.setenv(test_utils.CONSISTENCY_PLATFORM_ENV, "h100")
    case_id = "ltx_2_3_hq_pipeline"
    filenames = [
        f"{case_id}_1gpu_frame_0.png",
        f"{case_id}_1gpu_frame_mid.png",
        f"{case_id}_1gpu_frame_last.png",
    ]
    _set_official_gt_outputs(monkeypatch, {case_id: filenames})
    monkeypatch.setattr(test_utils, "_remote_file_exists", lambda url: True)

    files = test_utils._find_remote_consistency_gt_files(case_id, 1, is_video=True)

    assert files == [
        (filename, f"{test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE}/{filename}")
        for filename in filenames
    ]


def test_remote_image_gt_falls_back_to_sglang_when_official_missing(monkeypatch):
    monkeypatch.setenv(test_utils.CONSISTENCY_PLATFORM_ENV, "h100")
    sglang_prefix = test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE + "/"
    expected_filename = f"unit_image_1gpu.{test_utils.output_format_to_ext(None)}"
    monkeypatch.setattr(
        test_utils,
        "_remote_file_exists",
        lambda url: url.startswith(sglang_prefix),
    )

    files = test_utils._find_remote_consistency_gt_files(
        "unit_image",
        1,
        is_video=False,
    )

    assert files == [
        (
            expected_filename,
            (
                f"{test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE}"
                f"/{expected_filename}"
            ),
        )
    ]


def test_remote_image_gt_skips_official_for_quarantined_case(monkeypatch):
    monkeypatch.setenv(test_utils.CONSISTENCY_PLATFORM_ENV, "h100")
    official_prefix = test_utils.SGL_TEST_FILES_OFFICIAL_CONSISTENCY_GT_BASE + "/"
    sglang_prefix = test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE + "/"
    case_id = "qwen_image_edit_2509_ti2i"
    expected_filename = f"{case_id}_1gpu.{test_utils.output_format_to_ext(None)}"
    monkeypatch.setattr(
        test_utils,
        "_remote_file_exists",
        lambda url: url.startswith(official_prefix) or url.startswith(sglang_prefix),
    )

    files = test_utils._find_remote_consistency_gt_files(
        case_id,
        1,
        is_video=False,
    )

    assert files == [
        (
            expected_filename,
            (
                f"{test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE}"
                f"/{expected_filename}"
            ),
        )
    ]


def test_remote_platform_video_gt_prefers_platform_sglang_before_default_official(
    monkeypatch,
):
    monkeypatch.setenv(test_utils.CONSISTENCY_PLATFORM_ENV, "5090")
    sglang_platform_prefix = (
        f"{test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE}/5090/"
    )
    official_default_prefix = (
        f"{test_utils.SGL_TEST_FILES_OFFICIAL_CONSISTENCY_GT_BASE}/unit_video_1gpu_"
    )
    monkeypatch.setattr(
        test_utils,
        "_remote_file_exists",
        lambda url: url.startswith(sglang_platform_prefix)
        or url.startswith(official_default_prefix),
    )

    files = test_utils._find_remote_consistency_gt_files(
        "unit_video",
        1,
        is_video=True,
    )

    assert files == [
        (
            "5090/unit_video_1gpu_frame_0.png",
            (
                f"{test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE}"
                "/5090/unit_video_1gpu_frame_0.png"
            ),
        ),
        (
            "5090/unit_video_1gpu_frame_mid.png",
            (
                f"{test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE}"
                "/5090/unit_video_1gpu_frame_mid.png"
            ),
        ),
        (
            "5090/unit_video_1gpu_frame_last.png",
            (
                f"{test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE}"
                "/5090/unit_video_1gpu_frame_last.png"
            ),
        ),
    ]


def test_remote_video_gt_skips_official_for_quarantined_case(monkeypatch):
    monkeypatch.setenv(test_utils.CONSISTENCY_PLATFORM_ENV, "h100")
    official_prefix = test_utils.SGL_TEST_FILES_OFFICIAL_CONSISTENCY_GT_BASE + "/"
    sglang_prefix = test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE + "/"
    case_id = "ltx_2_two_stage_t2v"
    monkeypatch.setattr(
        test_utils,
        "_remote_file_exists",
        lambda url: url.startswith(official_prefix) or url.startswith(sglang_prefix),
    )

    files = test_utils._find_remote_consistency_gt_files(
        case_id,
        2,
        is_video=True,
    )

    assert files == [
        (
            f"{case_id}_2gpu_frame_0.png",
            (
                f"{test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE}"
                f"/{case_id}_2gpu_frame_0.png"
            ),
        ),
        (
            f"{case_id}_2gpu_frame_mid.png",
            (
                f"{test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE}"
                f"/{case_id}_2gpu_frame_mid.png"
            ),
        ),
        (
            f"{case_id}_2gpu_frame_last.png",
            (
                f"{test_utils.SGL_TEST_FILES_SGLANG_CONSISTENCY_GT_BASE}"
                f"/{case_id}_2gpu_frame_last.png"
            ),
        ),
    ]


def test_remote_npu_image_gt_prefers_official_ascend_when_present(monkeypatch):
    monkeypatch.setenv(test_utils.CONSISTENCY_PLATFORM_ENV, "h100")
    official_ascend_prefix = (
        test_utils.SGL_TEST_FILES_OFFICIAL_CONSISTENCY_GT_BASE_ASCEND + "/"
    )
    expected_filename = f"unit_npu_image_1gpu.{test_utils.output_format_to_ext(None)}"
    _set_official_gt_outputs(monkeypatch, {"unit_npu_image": [expected_filename]})
    monkeypatch.setattr(
        test_utils,
        "_remote_file_exists",
        lambda url: url.startswith(official_ascend_prefix),
    )

    files = test_utils._find_remote_consistency_gt_files(
        "unit_npu_image",
        1,
        is_video=False,
    )

    assert files == [
        (
            expected_filename,
            (
                f"{test_utils.SGL_TEST_FILES_OFFICIAL_CONSISTENCY_GT_BASE_ASCEND}"
                f"/{expected_filename}"
            ),
        )
    ]


def test_platform_gt_candidates_prefer_platform_then_default(monkeypatch):
    monkeypatch.setenv(test_utils.CONSISTENCY_PLATFORM_ENV, "5090")

    assert test_utils.get_consistency_gt_candidates(
        "unit_image",
        1,
        is_video=False,
        output_format="png",
    ) == [
        "5090/unit_image_1gpu.png",
        "5090/unit_image_1gpu.jpg",
        "5090/unit_image_1gpu.webp",
        "unit_image_1gpu.png",
        "unit_image_1gpu.jpg",
        "unit_image_1gpu.webp",
    ]


def test_threshold_metadata_merges_platform_override():
    metadata = test_utils._merge_threshold_metadata(
        {
            "cases": {
                "case_a": {
                    "clip_threshold": 0.9,
                    "ssim_threshold": 0.9,
                    "psnr_threshold": 20.0,
                    "mean_abs_diff_threshold": 10.0,
                }
            },
            "default_clip_threshold_image": 0.92,
        },
        {
            "cases": {
                "case_a": {
                    "clip_threshold": 0.8,
                    "ssim_threshold": 0.7,
                    "psnr_threshold": 12.0,
                    "mean_abs_diff_threshold": 20.0,
                }
            }
        },
    )

    assert metadata["default_clip_threshold_image"] == 0.92
    assert metadata["cases"]["case_a"]["psnr_threshold"] == 12.0


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
    assert (
        tmp_path / "consistency_failures" / "generated" / "unit_image_fail_1gpu.png"
    ).exists()
