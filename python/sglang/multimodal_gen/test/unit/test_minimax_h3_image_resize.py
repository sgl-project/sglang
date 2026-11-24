# SPDX-License-Identifier: Apache-2.0
"""MiniMax H3 image output resize: the device path must equal Pillow byte for byte.

The algorithm (integer tap matrices + float64 matmuls) also runs on the CPU, so
most checks are CPU-runnable; the same checks run on a CUDA/ROCm device when one
is present.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3 import (
    image_canvas,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.minimax_h3.image_canvas import (
    minimax_h3_resize_image_output,
)

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])

# (input h, input w, output h, output w): the production tier (2.02x down), an
# integral and non-integral down-scale, an upscale, one identity axis, odd sizes.
GEOMETRIES = [
    (1536, 2720, 768, 1344),
    (256, 448, 128, 224),
    (320, 512, 200, 300),
    (96, 160, 150, 240),
    (128, 192, 128, 97),
    (97, 131, 61, 53),
]


def _noise(h: int, w: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _natural_like(h: int, w: int) -> np.ndarray:
    """Smooth gradients and sinusoids plus hard edges, saturated and black areas."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    r = 127.5 + 127.5 * np.sin(xx / 23.0) * np.cos(yy / 41.0)
    g = 255.0 * xx / max(w - 1, 1)
    b = 255.0 * yy / max(h - 1, 1)
    img = np.stack([r, g, b], axis=-1)
    img[h // 4 : h // 2, w // 4 : w // 2] = 255
    img[h // 2 : 3 * h // 4, w // 8 : w // 4] = 0
    img[:, w // 2] = [255, 0, 0]
    img += np.random.default_rng(1).normal(0.0, 2.0, img.shape)
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def _skip_big_on_cpu(device, geometry):
    # The production-size canvas needs ~40 GFLOP of float64 matmul: too slow on CPU.
    if device == "cpu" and geometry[0] * geometry[1] > 600_000:
        pytest.skip("production-size geometry is only run on an accelerator")


def _pillow(array: np.ndarray, width: int, height: int) -> np.ndarray:
    return np.asarray(
        Image.fromarray(array).resize((width, height), Image.Resampling.LANCZOS)
    )


@pytest.fixture(autouse=True)
def _clear_verdicts():
    image_canvas._GPU_RESIZE_VERDICT.clear()
    yield
    image_canvas._GPU_RESIZE_VERDICT.clear()


@pytest.mark.parametrize(
    "in_size,out_size", [(2720, 1344), (97, 53), (64, 64), (50, 120)]
)
def test_tap_matrix_rows_are_normalized_fixed_point(in_size, out_size):
    mat = image_canvas._pil_lanczos_coeffs(in_size, out_size)
    assert mat.shape == (out_size, in_size)
    # Rows sum to 1.0 in 22-bit fixed point up to the per-tap rounding error.
    row_sums = mat.sum(axis=1)
    assert np.abs(row_sums - (1 << 22)).max() <= 16


@pytest.mark.parametrize("geometry", GEOMETRIES[1:])
def test_tap_matrices_reproduce_pillow_with_integer_math(geometry):
    # CPU-only check of the coefficient construction against Pillow: apply the
    # integer taps with int64 numpy exactly as Pillow's 8 bpc code does.
    in_h, in_w, out_h, out_w = geometry
    array = _natural_like(in_h, in_w)
    kx = image_canvas._pil_lanczos_coeffs(in_w, out_w)
    ky = image_canvas._pil_lanczos_coeffs(in_h, out_h)

    def clip8(acc):
        return np.clip((acc + (1 << 21)) >> 22, 0, 255)

    tmp = clip8(np.einsum("hwc,ow->hoc", array.astype(np.int64), kx))
    out = clip8(np.einsum("hoc,yh->yoc", tmp, ky)).astype(np.uint8)
    np.testing.assert_array_equal(out, _pillow(array, out_w, out_h))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("geometry", GEOMETRIES)
@pytest.mark.parametrize("kind", ["noise", "natural"])
def test_matmul_resize_equals_pillow(device, geometry, kind):
    _skip_big_on_cpu(device, geometry)
    in_h, in_w, out_h, out_w = geometry
    array = _noise(in_h, in_w, seed=7) if kind == "noise" else _natural_like(in_h, in_w)
    q = torch.from_numpy(array).to(device).permute(2, 0, 1).contiguous()
    got = image_canvas._gpu_lanczos_resize_chw(q, out_w, out_h)
    assert got.dtype == torch.uint8 and got.shape == (3, out_h, out_w)
    np.testing.assert_array_equal(
        got.permute(1, 2, 0).cpu().numpy(), _pillow(array, out_w, out_h)
    )


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("geometry", GEOMETRIES[:3])
def test_self_check_accepts_exact_resize_and_caches_verdict(device, geometry):
    _skip_big_on_cpu(device, geometry)
    in_h, in_w, out_h, out_w = geometry
    dev = torch.device(device)
    assert image_canvas._gpu_resize_is_exact(in_h, in_w, out_h, out_w, dev)
    assert (in_h, in_w, out_h, out_w, dev.type) in image_canvas._GPU_RESIZE_VERDICT


def test_self_check_rejects_a_resize_that_differs_from_pillow(monkeypatch):
    real = image_canvas._gpu_lanczos_resize_chw

    def off_by_one(q, width, height):
        out = real(q, width, height)
        out[0, 0, 0] ^= 1
        return out

    monkeypatch.setattr(image_canvas, "_gpu_lanczos_resize_chw", off_by_one)
    assert not image_canvas._gpu_resize_is_exact(64, 96, 32, 48, torch.device("cpu"))


def test_self_check_rejects_a_resize_that_raises(monkeypatch):
    def boom(q, width, height):
        raise RuntimeError("no float64 matmul on this device")

    monkeypatch.setattr(image_canvas, "_gpu_lanczos_resize_chw", boom)
    assert not image_canvas._gpu_resize_is_exact(64, 96, 32, 48, torch.device("cpu"))


@pytest.mark.parametrize("threads", ["1", "4"])
@pytest.mark.parametrize("geometry", GEOMETRIES)
def test_pillow_fallback_is_one_pillow_call(monkeypatch, threads, geometry):
    monkeypatch.setenv("SGLANG_DIFFUSION_MINIMAX_H3_RESIZE_THREADS", threads)
    in_h, in_w, out_h, out_w = geometry
    array = _natural_like(in_h, in_w)
    np.testing.assert_array_equal(
        image_canvas._pil_lanczos_resize(array, out_w, out_h),
        _pillow(array, out_w, out_h),
    )


def _reference_output(pixels: torch.Tensor, width: int, height: int) -> torch.Tensor:
    """The pre-optimization implementation: host quantize + one Pillow call."""
    resized = []
    for image in pixels.cpu():
        array = (image * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 0).numpy()
        output = Image.fromarray(array).resize(
            (width, height), Image.Resampling.LANCZOS
        )
        resized.append(torch.from_numpy(np.array(output, copy=True)).permute(2, 0, 1))
    return torch.stack(resized).float().div_(255)


def _decoded_like(device, batch=2, h=256, w=448) -> torch.Tensor:
    gen = torch.Generator().manual_seed(3)
    smooth = torch.from_numpy(_natural_like(h, w)).permute(2, 0, 1).float() / 255
    noise = torch.rand(3, h, w, generator=gen)
    pixels = torch.stack([smooth, noise][:batch])
    # The decoder output is not confined to [0, 1]: exercise the clamp.
    pixels[0, :, :4, :4] = 1.3
    pixels[0, :, 4:8, :4] = -0.2
    return pixels.to(device)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("gpu_resize", ["1", "0"])
@pytest.mark.parametrize("out_hw", [(128, 224), (100, 150), (300, 500)])
def test_resize_output_equals_reference(monkeypatch, device, gpu_resize, out_hw):
    monkeypatch.setenv("SGLANG_DIFFUSION_MINIMAX_H3_RESIZE_GPU", gpu_resize)
    pixels = _decoded_like(device)
    out_h, out_w = out_hw
    got = minimax_h3_resize_image_output(pixels, out_w, out_h)
    assert got.device.type == "cpu" and got.dtype == torch.float32
    assert torch.equal(got, _reference_output(pixels, out_w, out_h))


def test_resize_output_falls_back_when_check_fails(monkeypatch):
    monkeypatch.setattr(image_canvas, "_gpu_resize_is_exact", lambda *a: False)
    pixels = _decoded_like("cuda" if torch.cuda.is_available() else "cpu")
    assert torch.equal(
        minimax_h3_resize_image_output(pixels, 224, 128),
        _reference_output(pixels, 224, 128),
    )


def test_resize_output_rejects_wrong_rank():
    with pytest.raises(ValueError, match="decoded"):
        minimax_h3_resize_image_output(torch.zeros(3, 8, 8), 4, 4)
