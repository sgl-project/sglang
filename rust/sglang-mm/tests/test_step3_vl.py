"""CPU parity for Step3/Step5 preprocessing, independent of Engine imports."""

import io
import json

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from sglang.srt.rust_extensions._multimodal import step3_vl

SPEC = dict(
    image_token_id=7,
    image_start_id=1,
    image_end_id=2,
    patch_start_id=3,
    patch_end_id=4,
    patch_newline_id=5,
    image_size=728,
    patch_size=504,
    image_feature_tokens=169,
    patch_feature_tokens=81,
    image_mean=[0.48145466, 0.4578275, 0.40821073],
    image_std=[0.26862954, 0.26130258, 0.27577711],
)


def encode(array, fmt="PNG"):
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format=fmt)
    return buf.getvalue()


def preprocess(data, mode=None):
    return step3_vl.preprocess_batch(
        [data], json.dumps(SPEC), None if mode is None else [mode]
    )[0]


def reference(image, size):
    tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255
    tensor = F.normalize(tensor, SPEC["image_mean"], SPEC["image_std"])
    return (
        F.resize(
            tensor,
            [size, size],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        .numpy()
        .ravel()
    )


@pytest.mark.parametrize("size", [(1, 1), (53, 37), (728, 728), (1008, 1008)])
def test_pixels_and_row_major_tiles(size):
    """Compare normalization/AA resize and tile order against torchvision."""
    w, h = size
    array = np.random.default_rng(42).integers(0, 256, (h, w, 3), dtype=np.uint8)
    image = Image.fromarray(array)
    pixels, patches, newlines = preprocess(encode(array), "pil")
    np.testing.assert_allclose(pixels, reference(image, 728), atol=5e-4, rtol=0)
    if w == 1008:
        expected = np.concatenate(
            [
                reference(image.crop((x, y, x + 504, y + 504)), 504)
                for y in (0, 504)
                for x in (0, 504)
            ]
        )
        np.testing.assert_allclose(patches, expected, atol=5e-4, rtol=0)
        assert newlines == [False, True, False, False]
    else:
        assert len(patches) == 0
        assert newlines == []


@pytest.mark.parametrize("suffix,mode", [(b"", "tensor"), (b"trailing-data", "pil")])
def test_jpeg_selection_matches_python_loader(suffix, mode):
    """Python's fast JPEG gate requires both SOI and terminal EOI markers."""
    array = np.random.default_rng(8).integers(0, 256, (400, 700, 3), dtype=np.uint8)
    data = encode(array, "JPEG") + suffix
    auto = preprocess(data)
    explicit = preprocess(data, mode)
    for actual, expected in zip(auto, explicit):
        np.testing.assert_array_equal(actual, expected)


def test_multiple_images_token_spans():
    ids, spans = step3_vl.expand_input_ids(
        [99, 7, 98, 7], [[False, True, False, False], []], json.dumps(SPEC)
    )
    image = [1] + [7] * 169 + [2]
    patch = [3] + [7] * 81 + [4]
    expected = [99] + patch * 2 + [5] + patch * 2 + image + [98] + image
    assert ids == expected
    for item in spans:
        for start, end in item:
            assert ids[start : end + 1] == [7] * (end - start + 1)
    assert [len(item) for item in spans] == [5, 1]


def test_invalid_batch_rejected():
    with pytest.raises(ValueError):
        preprocess(b"not an image")
    with pytest.raises(ValueError):
        step3_vl.preprocess_batch([], json.dumps(SPEC), ["pil"])
    with pytest.raises(ValueError):
        step3_vl.expand_input_ids([7], [], json.dumps(SPEC))


def test_pil_resize_preserves_horizontal_first_rounding():
    """The Pillow <= 12 reference uses horizontal-then-vertical uint8 rounding.

    Pillow 13 can reverse those passes for tall images. Spell out the
    reference passes so upgrading Pillow cannot silently change this oracle.
    """
    array = np.random.default_rng(42).integers(0, 256, (1600, 505, 3), dtype=np.uint8)
    image = Image.fromarray(array)
    resized = image.resize((504, 1600), Image.Resampling.BILINEAR).resize(
        (504, 1512), Image.Resampling.BILINEAR
    )
    _, patches, newlines = preprocess(encode(array), "pil")
    expected = np.concatenate(
        [reference(resized.crop((0, y, 504, y + 504)), 504) for y in (0, 504, 1008)]
    )
    np.testing.assert_allclose(patches, expected, atol=5e-4, rtol=0)
    assert newlines == [True, True, False]


def test_pillow13_vertical_first_rounding():
    """Pillow 13's reordered passes must remain an explicit opt-in."""
    array = np.random.default_rng(42).integers(0, 256, (1600, 505, 3), dtype=np.uint8)
    image = Image.fromarray(array)
    resized = image.resize((505, 1512), Image.Resampling.BILINEAR).resize(
        (504, 1512), Image.Resampling.BILINEAR
    )
    _, patches, newlines = preprocess(encode(array), "pil_v13")
    expected = np.concatenate(
        [reference(resized.crop((0, y, 504, y + 504)), 504) for y in (0, 504, 1008)]
    )
    np.testing.assert_allclose(patches, expected, atol=5e-4, rtol=0)
    assert newlines == [True, True, False]
    old = preprocess(encode(array), "pil")[1]
    assert np.max(np.abs(patches - old)) > 0.01
