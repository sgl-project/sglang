"""Python/Rust parity for the InternVL image tile pipeline."""

import io
import json

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.srt.rust_extensions._multimodal import internvl as _rs_internvl

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def _spec_json(image_size: int, max_num: int = 12) -> str:
    spec = {
        "image_token_id": 0,
        "img_context_token_id": 151857,
        "img_start_token_id": 151859,
        "img_end_token_id": 151858,
        "image_size": image_size,
        "num_image_token": 64,
        "max_num": max_num,
        "use_thumbnail": True,
        "image_mean": MEAN,
        "image_std": STD,
    }
    return json.dumps(spec)


def _reference_dynamic_preprocess(
    arr: np.ndarray, image_size: int, max_num: int
) -> np.ndarray:
    """Python `InternVLProcessor.dynamic_preprocess` on a CPU float tensor."""
    tensor = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(MEAN, dtype=torch.float32).view(-1, 1, 1)
    std = torch.tensor(STD, dtype=torch.float32).view(-1, 1, 1)
    tensor = (tensor - mean) / std

    _, height, width = tensor.shape
    aspect_ratio = width / height
    ratios = {
        (i, j)
        for n in range(1, max_num + 1)
        for i in range(1, n + 1)
        for j in range(1, n + 1)
        if i * j <= max_num
    }
    ratios = sorted(ratios, key=lambda item: item[0] * item[1])

    best_ratio_diff = float("inf")
    best_ratio = (1, 1)
    for cols, rows in ratios:
        diff = abs(aspect_ratio - cols / rows)
        blocks = cols * rows
        if diff < best_ratio_diff:
            best_ratio_diff = diff
            best_ratio = (cols, rows)
        elif diff == best_ratio_diff and blocks > best_ratio[0] * best_ratio[1]:
            best_ratio = (cols, rows)

    target_w = image_size * best_ratio[0]
    target_h = image_size * best_ratio[1]
    blocks = best_ratio[0] * best_ratio[1]
    resized = torch.nn.functional.interpolate(
        tensor.unsqueeze(0),
        size=(target_h, target_w),
        mode="bicubic",
        align_corners=False,
    ).squeeze(0)

    tiles = []
    for tile_index in range(blocks):
        x = (tile_index % best_ratio[0]) * image_size
        y = (tile_index // best_ratio[0]) * image_size
        tiles.append(resized[:, y : y + image_size, x : x + image_size])
    tiles.append(
        torch.nn.functional.interpolate(
            tensor.unsqueeze(0),
            size=(image_size, image_size),
            mode="bicubic",
            align_corners=False,
        ).squeeze(0)
    )
    return torch.stack(tiles).numpy()


def _rs_preprocess(arr: np.ndarray, image_size: int) -> np.ndarray:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    tiles, tile_count, size = _rs_internvl.preprocess(
        buf.getvalue(), _spec_json(image_size)
    )
    assert size == image_size
    return np.asarray(tiles).reshape(tile_count, 3, image_size, image_size)


CASES = [
    (21, 33, 0),
    (33, 21, 1),
    (32, 32, 2),
    (40, 100, 3),
    (100, 40, 4),
    (17, 80, 5),
    (80, 17, 6),
    (48, 48, 7),
]


@pytest.mark.parametrize("image_size", [16, 32])
@pytest.mark.parametrize(
    "height,width,seed",
    CASES,
    ids=[f"{h}x{w}-seed{s}" for h, w, s in CASES],
)
def test_internvl_tiles_match_python(height, width, seed, image_size):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)
    got = _rs_preprocess(arr, image_size)
    want = _reference_dynamic_preprocess(arr, image_size, max_num=12)
    assert got.shape == want.shape
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-5)
