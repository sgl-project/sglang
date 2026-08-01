import io
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from bench_parity import make_photo_like

import sglang.srt.multimodal.inkling.image_processing as ip


def encode(arr, fmt):
    buf = io.BytesIO()
    Image.fromarray(arr).save(
        buf, format=fmt, **({"quality": 90} if fmt == "JPEG" else {})
    )
    return buf.getvalue()


def run(images, use_rs: bool, rescale: bool):
    ip._rs_module = None
    os.environ["SGLANG_RS_MM_PREPROCESS"] = "1" if use_rs else "0"
    kwargs = (
        {}
        if rescale
        else {"rescale_image_frac": None, "rescale_image_max_upscaled_long_edge": None}
    )
    proc = ip.InklingImageProcessor(patch_size=40, **kwargs)
    out = proc.preprocess(images)
    assert (ip._rs_module is not False) == use_rs, "rust module gating mismatch"
    return out


def compare(tag, images, expect_exact, rescale=False):
    ref = run(images, use_rs=False, rescale=rescale)
    got = run(images, use_rs=True, rescale=rescale)
    assert ref["num_patches"] == got["num_patches"], tag
    assert ref["num_tokens"] == got["num_tokens"], tag
    a, b = ref["vision_patches_bthwc"], got["vision_patches_bthwc"]
    assert a.shape == b.shape and a.dtype == b.dtype, f"{tag}: {a.shape} vs {b.shape}"
    exact = torch.equal(
        a.contiguous().view(torch.uint16), b.contiguous().view(torch.uint16)
    )
    if expect_exact:
        assert exact, f"{tag}: expected bit-exact"
        print(f"  {tag}: bit-exact=True shape={tuple(a.shape)}")
    else:
        d = (a.float() - b.float()).abs()
        print(
            f"  {tag}: bit-exact={exact} max_abs={d.max():.6f} shape={tuple(a.shape)}"
        )
        assert d.max() < 0.25, f"{tag}: JPEG decoder diff too large"


arr1 = make_photo_like(1080, 1920, seed=1)
arr2 = make_photo_like(720, 1280, seed=2)
arr3 = make_photo_like(480, 640, seed=3)

print("=== integration: InklingImageProcessor env-gated rust path ===")
compare("single PNG", [encode(arr1, "PNG")], expect_exact=True)
compare("single JPEG", [encode(arr1, "JPEG")], expect_exact=False)
compare(
    "5x PNG batch",
    [encode(a, "PNG") for a in [arr1, arr2, arr3, arr1, arr2]],
    expect_exact=True,
)
compare(
    "mixed JPEG/PNG batch",
    [encode(arr1, "JPEG"), encode(arr2, "PNG")],
    expect_exact=False,
)
compare("PIL input (PNG roundtrip)", [Image.fromarray(arr3)], expect_exact=True)
compare("single PNG rescaled", [encode(arr1, "PNG")], expect_exact=True, rescale=True)
compare(
    "single JPEG rescaled", [encode(arr1, "JPEG")], expect_exact=False, rescale=True
)
compare(
    "3x mixed rescaled",
    [encode(arr1, "JPEG"), encode(arr2, "PNG"), encode(arr3, "PNG")],
    expect_exact=False,
    rescale=True,
)
print("INTEGRATION_OK")
