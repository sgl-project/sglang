import io

import numpy as np
import pytest
from PIL import Image

from scripts.ci.utils.diffusion import publish_diffusion_gt as publish_gt


def _encode_png(rgb):
    buffer = io.BytesIO()
    Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8)).save(buffer, format="PNG")
    return buffer.getvalue()


def _structured_image():
    x = np.linspace(0, 255, 256, dtype=np.float32)
    y = np.linspace(0, 255, 256, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    image = np.stack([xx, yy, 255 - xx], axis=-1)
    image[48:160, 72:184, 0] = 235
    image[48:160, 72:184, 1] = 80
    image[48:160, 72:184, 2] = 55
    return _encode_png(image)


def _low_detail_noise():
    rng = np.random.default_rng(0)
    image = rng.normal(92, 5, size=(256, 256, 3))
    return _encode_png(image)


def _high_frequency_noise():
    rng = np.random.default_rng(1)
    image = rng.integers(0, 256, size=(256, 256, 3), dtype=np.uint8)
    return _encode_png(image)


def test_quality_gate_accepts_structured_image():
    metrics = publish_gt.compute_image_quality_metrics(_structured_image())

    assert publish_gt.get_quality_failure_reasons(metrics) == []


def test_quality_gate_flags_low_detail_noise():
    metrics = publish_gt.compute_image_quality_metrics(_low_detail_noise())

    assert "low-contrast low-detail output" in publish_gt.get_quality_failure_reasons(
        metrics
    )


def test_quality_gate_flags_high_frequency_noise():
    metrics = publish_gt.compute_image_quality_metrics(_high_frequency_noise())

    assert "high-frequency random noise" in publish_gt.get_quality_failure_reasons(
        metrics
    )


def test_old_new_metrics_flags_large_drift():
    metrics = publish_gt.compute_old_new_metrics(
        _structured_image(), _low_detail_noise()
    )

    assert metrics.ssim < publish_gt.OLD_NEW_MIN_SSIM
    assert metrics.mean_abs_diff > publish_gt.OLD_NEW_MAX_MEAN_ABS_DIFF


def test_gt_file_validation_rejects_suspicious_update(monkeypatch):
    monkeypatch.setattr(
        publish_gt,
        "get_remote_blob_content",
        lambda repo_owner, repo_name, blob_sha, token: _structured_image(),
    )

    with pytest.raises(SystemExit):
        files_to_upload = [
            (
                "diffusion-ci/consistency_gt/sglang_generated/example.png",
                _low_detail_noise(),
            )
        ]
        publish_gt.validate_gt_files(
            files_to_upload,
            files_to_upload,
            {
                "diffusion-ci/consistency_gt/sglang_generated/example.png": {
                    "sha": "old"
                }
            },
            "token",
        )


def test_gt_file_validation_rejects_suspicious_unchanged_output():
    files_to_upload = [
        (
            "diffusion-ci/consistency_gt/sglang_generated/example.png",
            _low_detail_noise(),
        )
    ]

    with pytest.raises(SystemExit):
        publish_gt.validate_gt_files(
            files_to_upload,
            [],
            {
                "diffusion-ci/consistency_gt/sglang_generated/example.png": {
                    "sha": "old"
                }
            },
            "token",
        )


def test_gt_file_validation_allows_replacing_suspicious_old_gt(monkeypatch):
    monkeypatch.setattr(
        publish_gt,
        "get_remote_blob_content",
        lambda repo_owner, repo_name, blob_sha, token: _low_detail_noise(),
    )

    files_to_upload = [
        (
            "diffusion-ci/consistency_gt/sglang_generated/example.png",
            _structured_image(),
        )
    ]

    publish_gt.validate_gt_files(
        files_to_upload,
        files_to_upload,
        {"diffusion-ci/consistency_gt/sglang_generated/example.png": {"sha": "old"}},
        "token",
    )
