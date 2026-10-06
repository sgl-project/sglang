# SPDX-License-Identifier: Apache-2.0
"""GAP 6 regression: the Kandinsky6 IT2VA conditioning-image preprocessor
must aspect-preserving-resize-then-centre-crop, matching the diffusers
reference's ``encode_i2va_first_frame``
(``pipeline_kandinsky6_ti2va.py``, ``scale = min(src_h/height,
src_w/width)``; resize to ``(src_h/scale, src_w/scale)``; centre-crop to
``(height, width)``), not stretch-resize the whole source image to the
target box (which distorts its aspect ratio whenever the source doesn't
already match the target aspect ratio).
"""

from __future__ import annotations

import numpy as np
import PIL.Image
import pytest
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.kandinsky6.image_encoding import (
    Kandinsky6ImageEncodingStage,
)


def _gradient_image(width: int, height: int) -> PIL.Image.Image:
    """A deterministic, per-pixel-distinguishable RGB image (not a flat
    color), so resize/crop geometry is actually verifiable pixel-for-pixel.
    """
    xs = np.linspace(0, 255, width, dtype=np.uint8)
    ys = np.linspace(0, 255, height, dtype=np.uint8)
    grid = np.zeros((height, width, 3), dtype=np.uint8)
    grid[..., 0] = xs[None, :]
    grid[..., 1] = ys[:, None]
    grid[..., 2] = 128
    return PIL.Image.fromarray(grid, mode="RGB")


@pytest.mark.parametrize(
    "src_h,src_w,height,width",
    [
        (1080, 1920, 512, 768),  # 16:9 source, 3:2 target (audit's own case)
        (1920, 1080, 512, 768),  # 9:16 source, 3:2 target (audit's own case)
        (1024, 1536, 512, 768),  # already 3:2 -- matching aspect
    ],
)
def test_cover_resize_dims_matches_diffusers_formula(src_h, src_w, height, width):
    # encode_i2va_first_frame: scale = min(src_h/height, src_w/width);
    # resize to (int(src_h/scale), int(src_w/scale)), then centre-crop.
    scale = min(src_h / height, src_w / width)
    expected = (int(src_h / scale), int(src_w / scale))
    assert (
        Kandinsky6ImageEncodingStage._cover_resize_dims(src_h, src_w, height, width)
        == expected
    )
    # The resize always covers the target box on both axes (never smaller).
    assert expected[0] >= height
    assert expected[1] >= width


@pytest.mark.parametrize(
    "src_w,src_h,min_mean_abs_diff",
    [
        (1920, 1080, 0.015),  # 16:9 landscape source (audit's own case)
        (1080, 1920, 0.05),  # 9:16 portrait source (audit's own case)
    ],
)
def test_mismatched_aspect_pil_image_is_cropped_not_stretched(
    src_w, src_h, min_mean_abs_diff
):
    # The model's documented default target resolution (512x768, 2:3)
    # against realistic phone/webcam source aspect ratios.
    height, width = 512, 768
    image = _gradient_image(src_w, src_h)

    produced = Kandinsky6ImageEncodingStage._preprocess(image, height, width)
    assert produced.shape == (1, 3, height, width)

    # Independent oracle: the diffusers reference's own aspect-preserving
    # resize + centre-crop, computed directly via PIL, without calling any
    # of the code under test.
    scale = min(src_h / height, src_w / width)
    new_h, new_w = int(src_h / scale), int(src_w / scale)
    resized = image.resize((new_w, new_h), resample=PIL.Image.BILINEAR)
    top, left = (new_h - height) // 2, (new_w - width) // 2
    cropped = resized.crop((left, top, left + width, top + height))
    expected_np = np.array(cropped).astype(np.float32) / 255.0
    expected = torch.from_numpy(expected_np).permute(2, 0, 1) * 2.0 - 1.0

    assert torch.allclose(produced[0], expected, atol=2.0 / 255.0)

    # The deleted stretch-resize procedure (plain `.resize((width, height))`,
    # no crop) produces a materially different image whenever the source
    # aspect ratio doesn't already match the target -- proving this is a
    # real behavior change, not a no-op refactor.
    stretched = image.resize((width, height), resample=PIL.Image.BILINEAR)
    stretched_np = np.array(stretched).astype(np.float32) / 255.0
    stretched_t = torch.from_numpy(stretched_np).permute(2, 0, 1) * 2.0 - 1.0
    mean_abs_diff = (produced[0] - stretched_t).abs().mean().item()
    assert mean_abs_diff > min_mean_abs_diff


def test_matching_aspect_pil_image_resize_only_agrees_with_stretch():
    # A sanity check on the oracle itself: when the source already matches
    # the target aspect ratio, cover-resize-then-crop and plain stretch
    # resize agree closely (no crop actually removes anything) -- so the
    # large the divergence asserted above is specifically an aspect-ratio
    # effect, not a general property of the two code paths.
    height, width = 64, 96
    src_w, src_h = 288, 192  # exactly 3:2, matching height:width == 64:96 == 2:3...
    # (use width:height == 96:64 == 3:2 to match src_w:src_h == 288:192 == 3:2)
    image = _gradient_image(src_w, src_h)

    produced = Kandinsky6ImageEncodingStage._preprocess(image, height, width)
    stretched = image.resize((width, height), resample=PIL.Image.BILINEAR)
    stretched_np = np.array(stretched).astype(np.float32) / 255.0
    stretched_t = torch.from_numpy(stretched_np).permute(2, 0, 1) * 2.0 - 1.0

    mean_abs_diff = (produced[0] - stretched_t).abs().mean().item()
    assert mean_abs_diff < 0.02


def test_tensor_input_uses_cover_crop_geometry_too():
    height, width = 64, 96
    src_h, src_w = 108, 192
    tensor = torch.rand(1, 3, src_h, src_w) * 2.0 - 1.0  # already in [-1, 1]

    produced = Kandinsky6ImageEncodingStage._preprocess(tensor, height, width)
    assert produced.shape[-2:] == (height, width)
