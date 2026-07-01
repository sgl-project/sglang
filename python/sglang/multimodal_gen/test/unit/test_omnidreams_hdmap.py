# SPDX-License-Identifier: Apache-2.0
"""CPU unit tests for OmniDreams HD-map handling (no server, no GPU).

Two related concerns are pinned here:

1. **Decode** (``omnidreams_hdmap_decode``): the A/B/AB fast-path decoders
   (``decode_hdmap_baseline`` / ``decode_hdmap_numpy`` /
   ``decode_hdmap_limited`` / ``decode_hdmap_numpy_limited``) and
   ``_read_frames_numpy`` — frame counts, shapes, range/dtype, and the
   bit-identical equivalence invariants between the variants.
2. **Online validation** (``video_api._validate_http_hdmap_path``):
   ``hdmap_path`` is fed to ``load_video`` / ``load_image`` which open local
   files directly, so a raw filesystem path from an untrusted HTTP body is an
   arbitrary-file-read vector. The guard enforces that, over the HTTP API,
   ``hdmap_path`` may only be an ``http(s)://`` or ``data:`` URL. CLI callers
   build sampling params directly and bypass this guard.
"""

import imageio.v2 as imageio
import numpy as np
import pytest
import torch
from fastapi import HTTPException

from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
    _validate_http_hdmap_path,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams_hdmap_decode import (
    _read_frames_numpy,
    decode_hdmap_baseline,
    decode_hdmap_limited,
    decode_hdmap_numpy,
    decode_hdmap_numpy_limited,
)


def _write_synthetic_mp4(path, num_frames, h, w):
    """Write a synthetic RGB mp4 (ffmpeg via imageio-ffmpeg)."""
    rng = np.random.RandomState(0)
    with imageio.get_writer(path, codec="libx264", fps=10, quality=5) as wr:
        for _ in range(num_frames):
            wr.append_data(rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8))


# --------------------------------------------------------------------------- #
# Decode: _read_frames_numpy
# --------------------------------------------------------------------------- #
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
    frames = _read_frames_numpy(p, max_frames=9)  # B: stop at total_pixel
    assert len(frames) == 9
    assert frames[0].shape == (16, 32, 3)


# --------------------------------------------------------------------------- #
# Decode: decode_hdmap_baseline
# --------------------------------------------------------------------------- #
def test_baseline_shape_range_dtype(tmp_path):
    p = str(tmp_path / "v.mp4")
    _write_synthetic_mp4(p, num_frames=12, h=16, w=32)
    clip = decode_hdmap_baseline(
        p, total_pixel=9, h=32, w=48, device=torch.device("cpu"), dtype=torch.float32
    )
    assert clip.shape == (1, 3, 9, 32, 48)
    assert clip.dtype == torch.float32
    assert -1.0 - 1e-4 <= float(clip.min()) and float(clip.max()) <= 1.0 + 1e-4


def test_baseline_clamps_short_clip(tmp_path):
    p = str(tmp_path / "v.mp4")
    _write_synthetic_mp4(p, num_frames=4, h=16, w=32)
    clip = decode_hdmap_baseline(
        p, total_pixel=9, h=16, w=32, device=torch.device("cpu"), dtype=torch.float32
    )
    assert clip.shape == (1, 3, 9, 16, 32)


# --------------------------------------------------------------------------- #
# Decode: decode_hdmap_numpy (A) vs baseline
# --------------------------------------------------------------------------- #
def test_numpy_variant_matches_baseline_when_no_resize(tmp_path):
    """A == baseline bit-identical when target res == native res (no resize)."""
    p = str(tmp_path / "v.mp4")
    _write_synthetic_mp4(p, num_frames=9, h=16, w=32)
    kw = dict(
        total_pixel=9, h=16, w=32, device=torch.device("cpu"), dtype=torch.float32
    )
    base = decode_hdmap_baseline(p, **kw)
    a = decode_hdmap_numpy(p, **kw)
    assert a.shape == base.shape == (1, 3, 9, 16, 32)
    assert torch.allclose(a, base, atol=1e-6)


def test_numpy_variant_with_resize_is_close(tmp_path):
    """A differs from baseline only by cv2-vs-PIL lanczos when res differs."""
    p = str(tmp_path / "v.mp4")
    _write_synthetic_mp4(p, num_frames=9, h=16, w=32)
    kw = dict(
        total_pixel=9, h=32, w=48, device=torch.device("cpu"), dtype=torch.float32
    )
    base = decode_hdmap_baseline(p, **kw)
    a = decode_hdmap_numpy(p, **kw)
    assert a.shape == base.shape
    # resize backend drift only -- bounded, not catastrophic. cv2 LANCZOS4 vs
    # PIL LANCZOS differ at extreme pixel values (random-noise test frame);
    # real HD-map rasters (sparse lines/boxes on black) drift far less.
    assert float((a - base).abs().max()) < 0.3
    assert float((a - base).abs().mean()) < 0.05


# --------------------------------------------------------------------------- #
# Decode: limited variants (B / AB)
# --------------------------------------------------------------------------- #
def test_limited_equals_baseline_first_total_pixel(tmp_path):
    """B decodes fewer frames but its output == baseline bit-identical."""
    p = str(tmp_path / "v.mp4")
    _write_synthetic_mp4(p, num_frames=60, h=16, w=32)  # long clip, only need 9
    kw = dict(
        total_pixel=9, h=16, w=32, device=torch.device("cpu"), dtype=torch.float32
    )
    base = decode_hdmap_baseline(p, **kw)
    b = decode_hdmap_limited(p, **kw)
    assert b.shape == base.shape
    assert torch.allclose(b, base, atol=1e-6)


def test_numpy_limited_equals_numpy_all(tmp_path):
    """AB == A bit-identical (same first total_pixel frames)."""
    p = str(tmp_path / "v.mp4")
    _write_synthetic_mp4(p, num_frames=60, h=32, w=48)
    kw = dict(
        total_pixel=9, h=32, w=48, device=torch.device("cpu"), dtype=torch.float32
    )
    a = decode_hdmap_numpy(p, **kw)
    ab = decode_hdmap_numpy_limited(p, **kw)
    assert ab.shape == a.shape
    assert torch.allclose(ab, a, atol=1e-6)


# --------------------------------------------------------------------------- #
# Online validation: _validate_http_hdmap_path
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "value",
    [
        None,
        "http://host/clip_hdmap.mp4",
        "https://host/clip_hdmap.mp4",
        "data:video/mp4;base64,AAAA",
        ["http://host/a.mp4", "https://host/b.mp4"],
        # Non-string entries are tolerated (skipped) rather than rejected.
        [123, "https://host/b.mp4"],
        # Scheme matching is case-insensitive and ignores surrounding space.
        "  HTTPS://host/clip.mp4  ",
    ],
)
def test_allows_urls_and_none(value):
    # Must not raise.
    _validate_http_hdmap_path(value)


@pytest.mark.parametrize(
    "value",
    [
        "/root/blockdata/omni-dreams/clip_hdmap.mp4",
        "relative/clip_hdmap.mp4",
        "file:///root/blockdata/clip_hdmap.mp4",
        "ftp://host/clip_hdmap.mp4",
        # A single bad entry in an otherwise-valid list still fails.
        ["https://host/ok.mp4", "/root/blockdata/local.mp4"],
    ],
)
def test_rejects_local_and_non_http_schemes(value):
    with pytest.raises(HTTPException) as exc_info:
        _validate_http_hdmap_path(value)
    assert exc_info.value.status_code == 400
    assert "hdmap_path" in str(exc_info.value.detail)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
