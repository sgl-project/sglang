"""K3 GPU preprocess: shared batched pipeline hook contract (CPU parts)."""

import sys

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.srt.multimodal.processors.kimi_k3 import _fill_transparent_bg
from sglang.srt.multimodal.processors.kimi_k25 import _resize_bicubic_if_needed
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _natural_image(height: int, width: int) -> np.ndarray:
    """Deterministic natural-image-like content: gradients, hard edges,
    and high-frequency texture (the aliasing-sensitive case)."""
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = (
        127
        + 60 * np.sin(2 * np.pi * xx / (width / 7.3))
        + 50 * np.cos(2 * np.pi * yy / (height / 5.1))
    )
    edges = 255.0 * ((xx // 9 + yy // 7) % 2)
    tex = 30.0 * np.sin(xx * 12.9898 + yy * 78.233)
    img = np.clip(0.55 * base + 0.30 * edges + 0.15 * (127 + tex), 0, 255)
    return np.stack(
        [img, np.roll(img, 13, axis=0), np.roll(img, 29, axis=1)], axis=-1
    ).astype(np.uint8)


def test_resize_matches_pil_bicubic_golden():
    """The GPU resize must reproduce the checkpoint processor's
    PIL.Image.resize(..., BICUBIC) downscale: PIL antialiases (kernel support
    scales with the ratio) and returns uint8. Without antialias=True the
    difference on textured content reaches tens of pixel levels."""
    arr = _natural_image(1200, 1600)
    for target_w, target_h in ((800, 600), (1120, 840)):
        golden = np.asarray(
            Image.fromarray(arr).resize((target_w, target_h), Image.Resampling.BICUBIC)
        ).astype(np.float32)

        x = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        ours = (
            _resize_bicubic_if_needed(x, target_h, target_w)
            .squeeze(0)
            .permute(1, 2, 0)
            .numpy()
        )

        diff = np.abs(ours - golden)
        # Integer pixel domain: everything within 1 level, most pixels exact.
        assert diff.max() <= 1.0, f"max |diff|={diff.max()} at {target_w}x{target_h}"
        assert (diff == 0).mean() > 0.7, f"bitwise ratio={(diff == 0).mean():.3f}"


def test_fill_transparent_bg_matches_checkpoint_composite():
    """Composite + truncation must match the checkpoint's numpy reference:
    alpha * rgb + (1 - alpha) * chessboard, then astype(np.uint8)."""
    cfg = {
        "pattern": "chessboard",
        "chessboard_square_size": 8,
        "chessboard_square_on_top_left": True,
        "chessboard_white_value": 255,
        "chessboard_gray_value": 180,
    }
    rgba = _natural_image(32, 40)
    alpha = ((np.mgrid[0:32, 0:40][0] * 6) % 256).astype(np.uint8)
    img = np.concatenate([rgba, alpha[..., None]], axis=-1)

    # Checkpoint reference (media_utils.fill_transparent_bg_with).
    bg = np.ones((32, 40, 3), dtype=np.uint8) * 255
    for y in range(0, 32, 8):
        for x0 in range(0, 40, 8):
            if (y // 8 + x0 // 8) % 2 == 1:
                bg[y : y + 8, x0 : x0 + 8] = 180
    a3 = np.stack([alpha.astype(np.float32) / 255.0] * 3, axis=2)
    golden = (a3 * img[:, :, :3] + (1 - a3) * bg).astype(np.uint8)

    x = torch.from_numpy(img).float().permute(2, 0, 1).unsqueeze(0)
    ours = _fill_transparent_bg(x, cfg).squeeze(0).permute(1, 2, 0).numpy()
    assert np.array_equal(ours, golden.astype(np.float32))


def test_fill_transparent_bg_batch_matches_per_image():
    """Compositing a batch must be bitwise identical to per-image calls
    (the batched pipeline applies it to whole resize groups)."""
    torch.manual_seed(0)
    batch = torch.rand(3, 4, 8, 6) * 255.0
    cfg = {"pattern": "chessboard", "chessboard_square_size": 2}

    batched = _fill_transparent_bg(batch, cfg)
    per_image = torch.cat(
        [_fill_transparent_bg(batch[i : i + 1], cfg) for i in range(batch.shape[0])]
    )
    assert torch.equal(batched, per_image)


def test_fill_transparent_bg_rgb_passthrough_batch():
    batch = torch.rand(2, 3, 4, 4) * 255.0
    assert _fill_transparent_bg(batch, {"pattern": "white"}) is batch


def test_fill_transparent_bg_no_config_drops_alpha():
    batch = torch.rand(2, 4, 4, 4) * 255.0
    out = _fill_transparent_bg(batch, None)
    assert out.shape == (2, 3, 4, 4)
    assert torch.equal(out, batch[:, :3])


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
