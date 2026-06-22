import imageio.v2 as imageio
import numpy as np
import pytest

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams_hdmap_decode import (
    _read_frames_numpy,
)


def _write_synthetic_mp4(path, num_frames, h, w):
    """Write a synthetic RGB mp4 (ffmpeg via imageio-ffmpeg)."""
    rng = np.random.RandomState(0)
    with imageio.get_writer(path, codec="libx264", fps=10, quality=5) as wr:
        for _ in range(num_frames):
            wr.append_data(rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8))


def test_read_frames_numpy_reads_all(tmp_path):
    p = str(tmp_path / "v.mp4")
    _write_synthetic_mp4(p, num_frames=7, h=16, w=32)
    frames = _read_frames_numpy(p)
    assert len(frames) == 7
    assert frames[0].shape == (16, 32, 3)
    assert frames[0].dtype == np.uint8


def test_read_frames_numpy_early_stop(tmp_path):
    p = str(tmp_path / "v.mp4")
    _write_synthetic_mp4(p, num_frames=40, h=16, w=32)
    frames = _read_frames_numpy(p, max_frames=9)   # B: stop at total_pixel
    assert len(frames) == 9
    assert frames[0].shape == (16, 32, 3)


import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams_hdmap_decode import (
    decode_hdmap_baseline,
)


def test_baseline_shape_range_dtype(tmp_path):
    p = str(tmp_path / "v.mp4")
    _write_synthetic_mp4(p, num_frames=12, h=16, w=32)
    clip = decode_hdmap_baseline(p, total_pixel=9, h=32, w=48,
                                 device=torch.device("cpu"), dtype=torch.float32)
    assert clip.shape == (1, 3, 9, 32, 48)
    assert clip.dtype == torch.float32
    assert -1.0 - 1e-4 <= float(clip.min()) and float(clip.max()) <= 1.0 + 1e-4


def test_baseline_clamps_short_clip(tmp_path):
    p = str(tmp_path / "v.mp4")
    _write_synthetic_mp4(p, num_frames=4, h=16, w=32)
    clip = decode_hdmap_baseline(p, total_pixel=9, h=16, w=32,
                                 device=torch.device("cpu"), dtype=torch.float32)
    assert clip.shape == (1, 3, 9, 16, 32)


from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams_hdmap_decode import (
    decode_hdmap_numpy,
)


def test_numpy_variant_matches_baseline_when_no_resize(tmp_path):
    """A == baseline bit-identical when target res == native res (no resize)."""
    p = str(tmp_path / "v.mp4")
    _write_synthetic_mp4(p, num_frames=9, h=16, w=32)
    kw = dict(total_pixel=9, h=16, w=32, device=torch.device("cpu"), dtype=torch.float32)
    base = decode_hdmap_baseline(p, **kw)
    a = decode_hdmap_numpy(p, **kw)
    assert a.shape == base.shape == (1, 3, 9, 16, 32)
    assert torch.allclose(a, base, atol=1e-6)


def test_numpy_variant_with_resize_is_close(tmp_path):
    """A differs from baseline only by cv2-vs-PIL lanczos when res differs."""
    p = str(tmp_path / "v.mp4")
    _write_synthetic_mp4(p, num_frames=9, h=16, w=32)
    kw = dict(total_pixel=9, h=32, w=48, device=torch.device("cpu"), dtype=torch.float32)
    base = decode_hdmap_baseline(p, **kw)
    a = decode_hdmap_numpy(p, **kw)
    assert a.shape == base.shape
    # resize backend drift only -- bounded, not catastrophic. cv2 LANCZOS4 vs
    # PIL LANCZOS differ at extreme pixel values (random-noise test frame);
    # real HD-map rasters (sparse lines/boxes on black) drift far less.
    assert float((a - base).abs().max()) < 0.3
    assert float((a - base).abs().mean()) < 0.05


from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams_hdmap_decode import (
    decode_hdmap_limited,
    decode_hdmap_numpy_limited,
)


def test_limited_equals_baseline_first_total_pixel(tmp_path):
    """B decodes fewer frames but its output == baseline bit-identical."""
    p = str(tmp_path / "v.mp4")
    _write_synthetic_mp4(p, num_frames=60, h=16, w=32)   # long clip, only need 9
    kw = dict(total_pixel=9, h=16, w=32, device=torch.device("cpu"), dtype=torch.float32)
    base = decode_hdmap_baseline(p, **kw)
    b = decode_hdmap_limited(p, **kw)
    assert b.shape == base.shape
    assert torch.allclose(b, base, atol=1e-6)


def test_numpy_limited_equals_numpy_all(tmp_path):
    """AB == A bit-identical (same first total_pixel frames)."""
    p = str(tmp_path / "v.mp4")
    _write_synthetic_mp4(p, num_frames=60, h=32, w=48)
    kw = dict(total_pixel=9, h=32, w=48, device=torch.device("cpu"), dtype=torch.float32)
    a = decode_hdmap_numpy(p, **kw)
    ab = decode_hdmap_numpy_limited(p, **kw)
    assert ab.shape == a.shape
    assert torch.allclose(ab, a, atol=1e-6)
