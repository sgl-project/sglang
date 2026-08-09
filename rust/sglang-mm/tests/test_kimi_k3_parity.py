"""Bit-exact parity: Rust Kimi-K3 preprocessing vs the checkpoint's PIL path.

The reference below transcribes the checkpoint's ``media_utils.py`` (fill
stage ``"after_resize"``). Each stage — load/mode canonicalization, resize,
composite — and the end-to-end f32 tensors are compared with
``assert_array_equal`` (no tolerance).
"""

import io
import math

import numpy as np
import pytest
from PIL import Image

from sglang.srt.multimodal._core import kimi_k3 as _rs
from sglang.srt.multimodal.kimi_k3_rust import (
    bg_tuple,
    rust_preprocess_images,
    to_effective_array,
)

# The K3 checkpoint's preprocessor_config.json values.
PATCH_SIZE = 14
MERGE_KERNEL_SIZE = 2
IN_PATCH_LIMIT = 65536
PATCH_LIMIT_ON_ONE_SIDE = 512
MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
BG_CFG = {
    "pattern": "chessboard",
    "chessboard_square_size": 8,
    "chessboard_square_on_top_left": True,
    "chessboard_white_value": 255,
    "chessboard_gray_value": 180,
}


# --- PIL reference: transcribed from the checkpoint's media_utils.py ---------


def ref_chessboard(height, width, square, on_top_left, white, gray):
    bg = np.ones((height, width, 3), dtype=np.uint8) * white
    for y in range(0, height, square):
        for x in range(0, width, square):
            if (y // square + x // square) % 2 == (1 if on_top_left else 0):
                bg[y : y + square, x : x + square] = gray
    return bg


def ref_fill_transparent_bg(image: Image.Image, bg_cfg) -> Image.Image:
    if bg_cfg is None:
        return image.convert("RGB")
    if image.mode == "RGB":
        return image
    if not ("A" in image.getbands() or "transparency" in image.info):
        return image.convert("RGB")
    img = np.array(image.convert("RGBA"))
    height, width = img.shape[:2]
    pattern = bg_cfg["pattern"]
    if pattern == "white":
        bg = np.full((height, width, 3), 255, dtype=np.uint8)
    elif pattern == "black":
        bg = np.zeros((height, width, 3), dtype=np.uint8)
    elif pattern == "gray":
        bg = np.full((height, width, 3), 128, dtype=np.uint8)
    else:
        bg = ref_chessboard(
            height,
            width,
            bg_cfg["chessboard_square_size"],
            bg_cfg["chessboard_square_on_top_left"],
            bg_cfg["chessboard_white_value"],
            bg_cfg["chessboard_gray_value"],
        )
    alpha = img[:, :, 3].astype(np.float32) / 255.0
    alpha_3d = np.stack([alpha] * 3, axis=2)
    result = alpha_3d * img[:, :, :3] + (1 - alpha_3d) * bg
    return Image.fromarray(result.astype(np.uint8))


def ref_navit_resize(width, height):
    s1 = math.sqrt(
        IN_PATCH_LIMIT
        / (max(1.0, width // PATCH_SIZE) * max(1.0, height // PATCH_SIZE))
    )
    s2 = PATCH_LIMIT_ON_ONE_SIDE * PATCH_SIZE / width
    s3 = PATCH_LIMIT_ON_ONE_SIDE * PATCH_SIZE / height
    scale = min(1.0, s1, s2, s3)
    new_w = min(max(1, int(width * scale)), PATCH_LIMIT_ON_ONE_SIDE * PATCH_SIZE)
    new_h = min(max(1, int(height * scale)), PATCH_LIMIT_ON_ONE_SIDE * PATCH_SIZE)
    factor = MERGE_KERNEL_SIZE * PATCH_SIZE
    return {
        "new_width": new_w,
        "new_height": new_h,
        "pad_width": (factor - new_w % factor) % factor,
        "pad_height": (factor - new_h % factor) % factor,
    }


def ref_preprocess(image: Image.Image, cfg, bg_cfg):
    """resize (original mode) -> composite -> pad -> normalize -> patchify."""
    resized = image.resize(
        (cfg["new_width"], cfg["new_height"]), resample=Image.Resampling.BICUBIC
    )
    arr = np.asarray(ref_fill_transparent_bg(resized, bg_cfg))
    arr = np.pad(
        arr,
        ((0, cfg["pad_height"]), (0, cfg["pad_width"]), (0, 0)),
        mode="constant",
        constant_values=0,
    )
    x = (arr / 255.0).astype(np.float32)
    x -= np.array(MEAN)
    x *= 1.0 / np.array(STD)
    h, w, c = x.shape
    patches = x.reshape(1, h // PATCH_SIZE, PATCH_SIZE, w // PATCH_SIZE, PATCH_SIZE, c)
    patches = patches.transpose(0, 1, 3, 5, 2, 4).reshape(-1, c, PATCH_SIZE, PATCH_SIZE)
    return patches, np.array([1, h // PATCH_SIZE, w // PATCH_SIZE], dtype=np.int64)


# --- fixtures ----------------------------------------------------------------


def _rng(seed):
    return np.random.RandomState(seed)


def gradient_rgba(w, h, seed=0):
    arr = _rng(seed).randint(0, 256, (h, w, 4), dtype=np.uint8)
    # Sweep the full alpha range so the composite exercises every blend value.
    arr[:, :, 3] = np.linspace(0, 255, h * w, dtype=np.uint8).reshape(h, w)
    return Image.fromarray(arr, "RGBA")


def noise_rgb(w, h, seed=1):
    return Image.fromarray(_rng(seed).randint(0, 256, (h, w, 3), dtype=np.uint8), "RGB")


def jpeg_roundtrip(image: Image.Image) -> Image.Image:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=85)
    return Image.open(io.BytesIO(buf.getvalue()))


IMAGES = {
    "rgb_noise": noise_rgb(640, 480),
    "rgb_jpeg": jpeg_roundtrip(noise_rgb(1023, 767, seed=2)),
    "rgb_tiny": noise_rgb(3, 5, seed=3),
    "rgb_huge_downscale": noise_rgb(9000, 4000, seed=4),  # trips the patch limits
    "rgba_gradient": gradient_rgba(400, 300),
    "rgba_odd": gradient_rgba(37, 53, seed=5),
    "la": Image.fromarray(
        _rng(6).randint(0, 256, (120, 80, 2), dtype=np.uint8), "LA"
    ),
    "l": Image.fromarray(_rng(7).randint(0, 256, (90, 70), dtype=np.uint8), "L"),
}


# --- stage parity ------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(IMAGES))
def test_load_matches_reference_mode(name):
    """`to_effective_array` must hand Rust exactly the pixels PIL would
    resize: RGB stays put, alpha modes canonicalize to RGBA, L to RGB."""
    image = IMAGES[name]
    arr = to_effective_array(image)
    assert arr is not None
    if image.mode in ("RGBA", "LA"):
        np.testing.assert_array_equal(arr, np.asarray(image.convert("RGBA")))
    elif image.mode == "L":
        np.testing.assert_array_equal(arr, np.asarray(image.convert("RGB")))
    else:
        np.testing.assert_array_equal(arr, np.asarray(image))


def test_load_rejects_out_of_scope_modes():
    """Palette images resize with NEAREST in PIL — no equivalent array exists,
    so they must report None (PIL fallback) instead of silently diverging."""
    palette = noise_rgb(32, 32, seed=8).convert("P")
    assert to_effective_array(palette) is None
    assert to_effective_array(noise_rgb(8, 8, seed=9).convert("CMYK")) is None
    assert to_effective_array(np.zeros((4, 4, 3), dtype=np.uint8)) is None


@pytest.mark.parametrize("name", sorted(IMAGES))
def test_resize_bit_exact(name):
    image = IMAGES[name]
    arr = to_effective_array(image)
    cfg = ref_navit_resize(*image.size)
    tw, th = cfg["new_width"], cfg["new_height"]

    got = _rs.resize_bicubic(arr, tw, th).reshape(th, tw, -1)
    # The reference resizes in the image's own mode; the canonicalization is
    # per-channel so the converted-mode resize must equal it exactly.
    ref_mode = image.resize((tw, th), resample=Image.Resampling.BICUBIC)
    if arr.shape[2] == 4:
        ref = np.asarray(ref_mode.convert("RGBA"))
    else:
        ref = np.asarray(ref_mode.convert("RGB"))
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("pattern", ["chessboard", "white", "black", "gray", None])
def test_composite_bit_exact(pattern):
    bg_cfg = None if pattern is None else {**BG_CFG, "pattern": pattern}
    rgba = np.asarray(gradient_rgba(101, 67, seed=10))
    got = _rs.fill_transparent_bg(rgba, bg_tuple(bg_cfg)).reshape(67, 101, 3)
    ref = np.asarray(ref_fill_transparent_bg(Image.fromarray(rgba, "RGBA"), bg_cfg))
    np.testing.assert_array_equal(got, ref)


@pytest.mark.parametrize("name", sorted(IMAGES))
def test_end_to_end_bit_exact(name):
    """The full pipeline output — the tensors handed to the scheduler — must
    equal the PIL reference bit for bit, f32 with zero tolerance."""
    image = IMAGES[name]
    cfg = ref_navit_resize(*image.size)

    out = rust_preprocess_images(
        [image],
        [cfg],
        patch_size=PATCH_SIZE,
        image_mean=MEAN,
        image_std=STD,
        transparent_bg_config=BG_CFG,
    )
    assert out is not None
    pixel_values, grid_thws = out

    ref_patches, ref_grid = ref_preprocess(image, cfg, BG_CFG)
    np.testing.assert_array_equal(grid_thws[0].numpy(), ref_grid)
    assert pixel_values.numpy().dtype == np.float32
    np.testing.assert_array_equal(pixel_values.numpy(), ref_patches)


def test_multi_image_request_concatenates_like_reference():
    images = [IMAGES["rgb_noise"], IMAGES["rgba_gradient"], IMAGES["l"]]
    cfgs = [ref_navit_resize(*im.size) for im in images]
    out = rust_preprocess_images(
        images,
        cfgs,
        patch_size=PATCH_SIZE,
        image_mean=MEAN,
        image_std=STD,
        transparent_bg_config=BG_CFG,
    )
    assert out is not None
    pixel_values, grid_thws = out
    refs = [ref_preprocess(im, cfg, BG_CFG) for im, cfg in zip(images, cfgs)]
    np.testing.assert_array_equal(
        pixel_values.numpy(), np.concatenate([r[0] for r in refs])
    )
    np.testing.assert_array_equal(grid_thws.numpy(), np.stack([r[1] for r in refs]))


def test_fallback_poisons_whole_request():
    """One out-of-scope image sends the whole request to the PIL path — mixed
    Rust/PIL requests would make per-image provenance untraceable."""
    images = [IMAGES["rgb_noise"], noise_rgb(16, 16, seed=11).convert("P")]
    cfgs = [ref_navit_resize(*im.size) for im in images]
    assert (
        rust_preprocess_images(
            images,
            cfgs,
            patch_size=PATCH_SIZE,
            image_mean=MEAN,
            image_std=STD,
            transparent_bg_config=BG_CFG,
        )
        is None
    )
