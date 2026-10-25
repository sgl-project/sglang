"""K3 GPU preprocess: shared batched pipeline hook contract (CPU parts)."""

import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from sglang.srt.multimodal.processors import kimi_k3, kimi_k25
from sglang.srt.multimodal.processors.kimi_k3 import _fill_transparent_bg
from sglang.srt.multimodal.processors.kimi_k25 import _resize_bicubic_if_needed
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


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


def test_large_resize_runs_on_cpu_and_matches(monkeypatch):
    """Sources above the GPU pixel cap resize on the CPU with the same kernel
    and come back on the caller's device."""
    x = torch.from_numpy(_natural_image(600, 800)).permute(2, 0, 1).unsqueeze(0)
    expected = _resize_bicubic_if_needed(x, 300, 400)

    monkeypatch.setattr(kimi_k25, "_MAX_GPU_RESIZE_PIXELS", 0)
    moved = []
    real_cpu = torch.Tensor.cpu
    monkeypatch.setattr(
        torch.Tensor, "cpu", lambda t: moved.append(t.shape) or real_cpu(t)
    )
    ours = _resize_bicubic_if_needed(x, 300, 400)

    assert moved == [x.shape]
    assert ours.device == x.device
    torch.testing.assert_close(ours, expected)


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


def _cpu_chw(image) -> torch.Tensor:
    return torch.from_numpy(np.asarray(image)).permute(2, 0, 1).contiguous()


def _norm_tensors():
    return torch.full((1, 3, 1, 1), 1 / 127.5), torch.full((1, 3, 1, 1), -1.0)


def test_gpu_preprocess_chunks_match_single_batch(monkeypatch):
    """Byte-bounded sub-batches produce the same patches as one batch."""
    images = [
        Image.fromarray(np.roll(_natural_image(112, 168), 5 * i, axis=1))
        for i in range(5)
    ]
    configs = [dict(new_height=56, new_width=84, pad_height=0, pad_width=0)] * 5
    scale, bias = _norm_tensors()
    args = (images, configs, scale, bias, 14)
    whole, whole_grids = kimi_k25._gpu_preprocess_images(*args, to_chw=_cpu_chw)

    monkeypatch.setattr(kimi_k25, "_GPU_PREPROCESS_CHUNK_BYTES", 1)
    assert len(kimi_k25._byte_bounded_chunks(list(range(5)), 56 * 84)) == 5
    chunked, chunked_grids = kimi_k25._gpu_preprocess_images(*args, to_chw=_cpu_chw)

    torch.testing.assert_close(chunked, whole)
    assert torch.equal(chunked_grids, whole_grids)


def test_prepare_image_features_falls_back_to_cpu_on_gpu_oom(monkeypatch):
    """A tokenizer-side GPU OOM is served by the CPU path, not surfaced."""
    images = [Image.fromarray(_natural_image(56, 84))] * 2
    scale, bias = _norm_tensors()
    cpu_calls = []

    def cpu_call(text, imgs):
        cpu_calls.append(len(imgs))
        configs = [kimi_k25.navit_resize_config(84, 56, 14, 2, 16384, 512, None)] * len(
            imgs
        )
        pixel_values, grids = kimi_k25._gpu_preprocess_images(
            imgs, configs, scale, bias, 14, to_chw=_cpu_chw
        )
        return {"pixel_values": pixel_values, "image_grid_thw": grids}

    def gpu_oom(*args, **kwargs):
        raise torch.OutOfMemoryError("CUDA out of memory")

    fake = SimpleNamespace(
        _patch_size=14,
        _merge_kernel_size=2,
        _in_patch_limit=16384,
        _patch_limit_on_one_side=512,
        _fixed_output_tokens=None,
        _transparent_bg_config=None,
        _image_token="<image>",
        _get_gpu_norm_tensors=lambda: (scale, bias),
        _cpu_call=cpu_call,
    )
    monkeypatch.setattr(kimi_k3, "_gpu_preprocess_images", gpu_oom)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    features, _, _, grids = kimi_k3.KimiK3GPUProcessorWrapper.prepare_image_features(
        fake, images
    )

    assert cpu_calls == [2]
    assert len(features) == len(grids) == 2


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
