import io
import time

import numpy as np
import torch
from PIL import Image

import sglang.srt.multimodal._core.inkling
from sglang.srt.multimodal.inkling.image_processing import (
    IMAGE_MEAN,
    IMAGE_STD,
    PAD_NORM,
    _encode_image_bytes,
    _fill_patches_numba,
)

PS = 40


def ref_patchify(arr: np.ndarray) -> torch.Tensor:
    h, w, _ = arr.shape
    nph = (h + PS - 1) // PS
    npw = w // PS + 1
    patches = np.empty((nph * npw, PS, PS, 3), dtype=np.float32)
    _fill_patches_numba(arr, PS, patches, IMAGE_MEAN, IMAGE_STD, PAD_NORM)
    return torch.from_numpy(patches).to(torch.bfloat16)


def rs_patchify(arr: np.ndarray) -> torch.Tensor:
    h, w, _ = arr.shape
    nph = (h + PS - 1) // PS
    npw = w // PS + 1
    bits = sglang.srt.multimodal._core.inkling.patchify_rgb(arr, PS)
    return torch.from_numpy(bits).view(torch.bfloat16).reshape(nph * npw, PS, PS, 3)


def rs_decode_patchify(data: bytes) -> torch.Tensor:
    h, w, bits = sglang.srt.multimodal._core.inkling.decode_patchify(data, PS)
    nph = (h + PS - 1) // PS
    npw = w // PS + 1
    return torch.from_numpy(bits).view(torch.bfloat16).reshape(nph * npw, PS, PS, 3)


def make_photo_like(h: int, w: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:h, 0:w]
    base = np.stack(
        [
            127 + 100 * np.sin(yy / 97.0) * np.cos(xx / 131.0),
            127 + 100 * np.cos(yy / 61.0) * np.sin(xx / 89.0),
            127 + 100 * np.sin((xx + yy) / 149.0),
        ],
        axis=-1,
    )
    noise = rng.normal(0, 12, (h // 8 + 1, w // 8 + 1, 3))
    noise = np.kron(noise, np.ones((8, 8, 1)))[:h, :w]
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def encode(arr: np.ndarray, fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(
        buf, format=fmt, **({"quality": 90} if fmt == "JPEG" else {})
    )
    return buf.getvalue()


def parity_a():
    print("=== Parity A: patchify from decoded array (expect bit-exact) ===")
    rng = np.random.default_rng(42)
    for h, w in [(1080, 1920), (1920, 1080), (40, 40), (37, 53), (720, 1280), (1, 1)]:
        arr = rng.integers(0, 256, (h, w, 3), dtype=np.uint8)
        ref, got = ref_patchify(arr), rs_patchify(arr)
        exact = torch.equal(ref.view(torch.uint16), got.view(torch.uint16))
        print(f"  {h}x{w}: shape {tuple(got.shape)} bit-exact={exact}")
        assert exact, f"parity A failed at {h}x{w}"


def parity_b():
    print("=== Parity B: full decode path ===")
    arr = make_photo_like(1080, 1920)
    for fmt in ["PNG", "JPEG"]:
        data = encode(arr, fmt)
        ref = _encode_image_bytes(
            data,
            patch_size=PS,
            rescale_image_frac=None,
            rescale_image_max_upscaled_long_edge=None,
        )
        got = rs_decode_patchify(data)
        got2 = got.view(got.shape[0], 1, PS, PS, 3).expand(-1, 2, -1, -1, -1)
        if torch.equal(
            ref.contiguous().view(torch.uint16), got2.contiguous().view(torch.uint16)
        ):
            print(f"  {fmt}: bit-exact=True ({len(data)/1e6:.2f}MB)")
        else:
            d = (ref.float() - got2.float()).abs()
            print(
                f"  {fmt}: bit-exact=False  max_abs={d.max():.6f} mean_abs={d.mean():.8f} "
                f"(decoder difference; normalized-feature units)"
            )


def bench():
    print("=== Benchmark (1080p, patch_size=40) ===")
    arr = make_photo_like(1080, 1920)
    jpeg = encode(arr, "JPEG")
    n = 30

    _encode_image_bytes(
        jpeg,
        patch_size=PS,
        rescale_image_frac=None,
        rescale_image_max_upscaled_long_edge=None,
    )
    rs_decode_patchify(jpeg)
    sglang.srt.multimodal._core.inkling.decode_patchify_batch([jpeg] * 5, PS)

    def run(label, fn, iters=n, images_per_call=1):
        t0, c0 = time.perf_counter(), time.process_time()
        for _ in range(iters):
            fn()
        wall = (time.perf_counter() - t0) / iters / images_per_call * 1e3
        cpu = (time.process_time() - c0) / iters / images_per_call * 1e3
        print(f"  {label:42} wall {wall:8.2f} ms/img   cpu {cpu:8.2f} ms/img")
        return wall, cpu

    w_py, c_py = run(
        "python (PIL + numba + bf16 cast)",
        lambda: _encode_image_bytes(
            jpeg,
            patch_size=PS,
            rescale_image_frac=None,
            rescale_image_max_upscaled_long_edge=None,
        ),
    )
    w_rs, c_rs = run("rust  decode_patchify", lambda: rs_decode_patchify(jpeg))
    w_rb, c_rb = run(
        "rust  decode_patchify_batch (5 imgs/call)",
        lambda: sglang.srt.multimodal._core.inkling.decode_patchify_batch(
            [jpeg] * 5, PS
        ),
        iters=max(n // 5, 5),
        images_per_call=5,
    )

    run("python numba patchify only", lambda: ref_patchify(arr))
    run("rust  patchify_rgb only", lambda: rs_patchify(arr))

    print(
        f"\n  speedup vs python: single {w_py / w_rs:.1f}x wall / {c_py / c_rs:.1f}x cpu, "
        f"batch {w_py / w_rb:.1f}x wall / {c_py / c_rb:.1f}x cpu"
    )


if __name__ == "__main__":
    torch.set_num_threads(8)
    parity_a()
    parity_b()
    bench()
    print("\nOK")
