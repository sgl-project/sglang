import math
import time
from typing import Optional

import numpy as np
import pytest
from PIL import Image

from sglang.srt.rust_extensions._multimodal import common as _rs_common
from sglang.srt.rust_extensions._multimodal import inkling as _rs_inkling


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


def pil_resize(arr: np.ndarray, tw: int, th: int, filter=Image.Resampling.LANCZOS):
    return np.array(Image.fromarray(arr).resize((tw, th), resample=filter), np.uint8)


def tv_resize(arr: np.ndarray, tw: int, th: int) -> np.ndarray:
    """torchvision's uint8 antialias bicubic — ATen's fixed-point kernel."""
    import torch
    from torchvision.transforms.v2 import functional as F

    tensor = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    out = F.resize(
        tensor, [th, tw], interpolation=F.InterpolationMode.BICUBIC, antialias=True
    )
    return out[0].permute(1, 2, 0).numpy()


# Every resampler the Rust resize claims, and its reference: `aten_u8` for a
# default server, `pil_bicubic` for --disable-fast-image-processor, `pil_lanczos`
# for inkling.
REFERENCES = {
    "pil_lanczos": lambda a, tw, th: pil_resize(a, tw, th, Image.Resampling.LANCZOS),
    "pil_bicubic": lambda a, tw, th: pil_resize(a, tw, th, Image.Resampling.BICUBIC),
    "aten_u8": tv_resize,
}


def rs_resize(arr, tw: int, th: int, resample: str = "pil_lanczos") -> np.ndarray:
    return _rs_common.resize_rgb(arr, tw, th, resample).reshape(th, tw, 3)


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


@pytest.mark.parametrize("resample", sorted(REFERENCES))
@pytest.mark.parametrize(
    "h,w,th,tw", CASES, ids=[f"{h}x{w}->{th}x{tw}" for h, w, th, tw in CASES]
)
def test_resize_bit_exact(h, w, th, tw, resample):
    rng = np.random.default_rng(h * 10000 + w)
    arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
    np.testing.assert_array_equal(
        rs_resize(arr, tw, th, resample), REFERENCES[resample](arr, tw, th)
    )


@pytest.mark.parametrize("resample", sorted(REFERENCES))
def test_resize_bit_exact_random_sweep(resample):
    """`aten_u8`'s weight precision varies with the scale factor, so the fixed
    cases above are not enough coverage on their own."""
    rng = np.random.default_rng(7)
    for h, w, th, tw in rng.integers(1, 200, (40, 4)):
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        np.testing.assert_array_equal(
            rs_resize(arr, tw, th, resample),
            REFERENCES[resample](arr, tw, th),
            err_msg=f"{h}x{w}->{th}x{tw} under {resample}",
        )


def test_unknown_resample_rejected():
    arr = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="unknown resample"):
        _rs_common.resize_rgb(arr, 2, 2, "nearest")


def test_scaled_dims_sweep():
    rng = np.random.default_rng(0)
    sizes = [(int(a), int(b)) for a, b in rng.integers(1, 5000, (500, 2))]
    sizes += [(2048, 1024), (2049, 100), (1024, 2048), (1, 1), (4096, 4096)]
    for frac, cap in [(2.0, 2048), (1.5, 2048), (3.0, None), (None, None), (2.0, 1)]:
        for w, h in sizes:
            assert _rs_common.scaled_dims(w, h, frac, cap) == py_scaled_dims(
                w, h, frac, cap
            ), (
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
    h, w, bits = _rs_inkling.decode_patchify(buf.getvalue(), 40, 2.0, 2048)
    assert (w, h) == py_scaled_dims(1920, 1080, 2.0, 2048)
    ref_arr = pil_resize(arr, w, h)
    ref_bits = _rs_inkling.patchify_rgb(ref_arr, 40)
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
