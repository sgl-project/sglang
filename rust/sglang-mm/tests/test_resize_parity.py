import math
import time
from typing import Optional

import numpy as np
import pytest
from PIL import Image

import sglang.srt.multimodal._core.inkling


def py_scaled_dims(
    width: int,
    height: int,
    frac: Optional[float],
    cap: Optional[int],
):
    if frac is None:
        return width, height
    long_edge = max(width, height)
    if long_edge == 0:
        return width, height
    target = float(long_edge) * frac
    if cap is not None:
        target = min(target, float(max(cap, long_edge)))
    ratio = target / float(long_edge)
    if ratio == 1.0:
        return width, height

    def scale(value):
        return max(1, math.floor(float(value) * ratio + 0.5))

    return scale(width), scale(height)


def pil_resize(arr: np.ndarray, tw: int, th: int) -> np.ndarray:
    return np.array(
        Image.fromarray(arr).resize((tw, th), resample=Image.Resampling.LANCZOS),
        dtype=np.uint8,
    )


def rs_resize(arr: np.ndarray, tw: int, th: int) -> np.ndarray:
    return sglang.srt.multimodal._core.inkling.resize_rgb(arr, tw, th).reshape(
        th, tw, 3
    )


CASES = [
    (1080, 1920, 1152, 2048),
    (896, 896, 1792, 1792),
    (360, 640, 720, 1280),
    (37, 53, 74, 106),
    (100, 100, 173, 173),
    (1, 1, 2, 2),
    (256, 256, 100, 100),
    (720, 1280, 720, 1280),
    (3, 500, 6, 1000),
]


@pytest.mark.parametrize(
    "h,w,th,tw", CASES, ids=[f"{h}x{w}->{th}x{tw}" for h, w, th, tw in CASES]
)
def test_resize_bit_exact(h, w, th, tw):
    rng = np.random.default_rng(h * 10000 + w)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    np.testing.assert_array_equal(rs_resize(arr, tw, th), pil_resize(arr, tw, th))


def test_scaled_dims_sweep():
    rng = np.random.default_rng(0)
    sizes = [(int(a), int(b)) for a, b in rng.integers(1, 5000, (500, 2))]
    sizes += [(2048, 1024), (2049, 100), (1024, 2048), (1, 1), (4096, 4096)]
    for frac, cap in [(2.0, 2048), (1.5, 2048), (3.0, None), (None, None), (2.0, 1)]:
        for w, h in sizes:
            assert sglang.srt.multimodal._core.inkling.scaled_dims(
                w, h, frac, cap
            ) == py_scaled_dims(w, h, frac, cap), (
                w,
                h,
                frac,
                cap,
            )


def test_decode_patchify_rescaled_matches_pil_pipeline():
    import io

    import torch

    rng = np.random.default_rng(7)
    arr = rng.integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    h, w, bits = sglang.srt.multimodal._core.inkling.decode_patchify(
        buf.getvalue(), 40, 2.0, 2048
    )
    assert (w, h) == py_scaled_dims(1920, 1080, 2.0, 2048)
    ref_arr = pil_resize(arr, w, h)
    ref_bits = sglang.srt.multimodal._core.inkling.patchify_rgb(ref_arr, 40)
    np.testing.assert_array_equal(bits, ref_bits)
    assert torch.from_numpy(bits).view(torch.bfloat16).shape[0] > 0


def test_resize_bench():
    arr = np.random.default_rng(1).integers(0, 256, (1080, 1920, 3), dtype=np.uint8)
    tw, th = py_scaled_dims(1920, 1080, 2.0, 2048)
    pil_resize(arr, tw, th)
    rs_resize(arr, tw, th)
    t0 = time.perf_counter()
    for _ in range(10):
        pil_resize(arr, tw, th)
    t_pil = (time.perf_counter() - t0) / 10 * 1e3
    t0 = time.perf_counter()
    for _ in range(10):
        rs_resize(arr, tw, th)
    t_rs = (time.perf_counter() - t0) / 10 * 1e3
    print(
        f"\nresize 1920x1080->{tw}x{th}: PIL {t_pil:.1f}ms rust {t_rs:.1f}ms ({t_pil/t_rs:.1f}x)"
    )
