"""Candidate HD-map video-decode helpers (A+B optimization spikes).

NOT wired into the production stage and does NOT touch the shared
``vision_utils.load_video`` (used by Wan/FLUX/etc.). Each function reproduces
the current ``_preprocess_hdmap_clip`` + ``_encode_hdmap`` output contract
(``[1, 3, total_pixel, H, W]`` in ``[-1, 1]``) so they are drop-in candidates
once the benchmark picks a winner.

Variants (orthogonal):
  * baseline      = PIL roundtrip, decode all frames  (current behavior)
  * A (numpy)     = cv2.resize + numpy->torch direct,  decode all frames
  * B (limited)   = PIL roundtrip,                     decode only total_pixel
  * AB             = numpy->torch direct,             decode only total_pixel

Numerics (see vision_utils.py:39-90): pil_to_numpy = arr/255, normalize = 2x-1,
so baseline per-pixel = frame/255*2-1 == (frame-127.5)/127.5 == the numpy path.
=> B == baseline bit-identical; A == baseline bit-identical when no resize;
   A vs baseline differ only by cv2 vs PIL lanczos when a resize is needed.
"""

import cv2
import imageio.v2 as imageio
import numpy as np
import PIL.Image
import torch

from sglang.multimodal_gen.runtime.models.vision_utils import (
    normalize,
    numpy_to_pt,
    pil_to_numpy,
    resize,
)


def _read_frames_numpy(path, max_frames=None):
    """Decode an mp4 to a list of ``[H, W, 3]`` uint8 numpy frames.

    Same ffmpeg backend as ``vision_utils.load_video`` (``imageio.get_reader``),
    but yields numpy directly -- no ``PIL.Image.fromarray`` roundtrip. When
    ``max_frames`` is set (the B optimization), iteration stops after that many
    frames so ffmpeg never decodes the tail that would be truncated anyway.
    """
    frames = []
    with imageio.get_reader(path) as reader:
        for frame in reader:
            frames.append(np.ascontiguousarray(frame)[..., :3])
            if max_frames is not None and len(frames) >= max_frames:
                break
    return frames


def _clamp_to_total(frames, total_pixel):
    """Truncate or repeat-last-frame so the clip is exactly ``total_pixel`` long.

    Mirrors ``OmniDreamsBeforeDenoisingStage._encode_hdmap`` (lines ~526-537).
    """
    n = len(frames)
    if n == 0:
        raise ValueError("no frames decoded")
    if n < total_pixel:
        frames = list(frames) + [frames[-1] for _ in range(total_pixel - n)]
    else:
        frames = list(frames[:total_pixel])
    return frames


def _preprocess_pil(frames_np, h, w, device, dtype):
    """Baseline per-frame path: PIL.fromarray -> resize(lanczos) -> pil_to_numpy
    -> numpy_to_pt -> normalize. Byte-accurate replica of the current
    ``OmniDreamsBeforeDenoisingStage._preprocess_pixels`` for numpy/PIL frames."""
    per = []
    for f in frames_np:
        img = PIL.Image.fromarray(f)
        img = resize(img, h, w)  # PIL LANCZOS
        arr = pil_to_numpy(img)  # [1, h, w, 3] float32 in [0,1]
        x = numpy_to_pt(arr)  # [1, 3, h, w] in [0,1]
        x = normalize(x)  # -> [-1,1]
        per.append(x)  # [1,3,h,w]
    clip = torch.cat(per, dim=0)  # [T, 3, h, w]
    return clip.unsqueeze(0).to(device=device, dtype=dtype)  # [1, T, 3, h, w]


def decode_hdmap_baseline(path, total_pixel, h, w, device, dtype):
    """Current production behavior: PIL roundtrip, decode ALL frames, truncate."""
    frames = _read_frames_numpy(path)  # all frames
    frames = _clamp_to_total(frames, total_pixel)
    clip = _preprocess_pil(frames, h, w, device, dtype)  # [1, T, 3, h, w]
    return clip.permute(0, 2, 1, 3, 4)  # [1, 3, T, h, w]


def _preprocess_numpy(frames_np, h, w, device, dtype):
    """A path: cv2.resize (LANCZOS4) on numpy, bulk numpy->torch, normalize.
    No PIL anywhere. Math matches baseline (arr/255*2-1) so output is bit-identical
    to baseline when the resize is a no-op (target == native res)."""
    resized = [
        cv2.resize(f, (w, h), interpolation=cv2.INTER_LANCZOS4) for f in frames_np
    ]
    arr = np.stack(resized, axis=0).astype(np.float32) / 255.0  # [T,h,w,3] in [0,1]
    x = torch.from_numpy(arr).permute(0, 3, 1, 2)  # [T,3,h,w] in [0,1]
    x = 2.0 * x - 1.0  # -> [-1,1]
    return x.unsqueeze(0).to(device=device, dtype=dtype)  # [1,T,3,h,w]


def decode_hdmap_numpy(path, total_pixel, h, w, device, dtype):
    """A: numpy->torch direct (cv2.resize), decode ALL frames, truncate."""
    frames = _read_frames_numpy(path)
    frames = _clamp_to_total(frames, total_pixel)
    clip = _preprocess_numpy(frames, h, w, device, dtype)  # [1,T,3,h,w]
    return clip.permute(0, 2, 1, 3, 4)  # [1,3,T,h,w]


def decode_hdmap_limited(path, total_pixel, h, w, device, dtype):
    """B: PIL roundtrip, decode ONLY total_pixel frames (early-stop ffmpeg)."""
    frames = _read_frames_numpy(path, max_frames=total_pixel)
    frames = _clamp_to_total(frames, total_pixel)
    clip = _preprocess_pil(frames, h, w, device, dtype)
    return clip.permute(0, 2, 1, 3, 4)


def decode_hdmap_numpy_limited(path, total_pixel, h, w, device, dtype):
    """AB: numpy->torch direct AND decode only total_pixel frames."""
    frames = _read_frames_numpy(path, max_frames=total_pixel)
    frames = _clamp_to_total(frames, total_pixel)
    clip = _preprocess_numpy(frames, h, w, device, dtype)
    return clip.permute(0, 2, 1, 3, 4)


def decode_hdmap_ab(path, total_pixel, h, w, device, dtype):
    """Production wire-in alias for ``decode_hdmap_numpy_limited`` (the A+B
    winner from the real-sample benchmark: 5-9x, near-zero drift).

    Returns ``[1, 3, total_pixel, h, w]`` in ``[-1, 1]`` -- the same contract as
    ``OmniDreamsBeforeDenoisingStage._preprocess_hdmap_clip`` -- so it is a
    drop-in for the per-frame video path, replacing the old
    ``load_video``-decodes-all-frames + PIL preprocess path.

    Native-res (no resize) output is bit-identical to the old path; resize output
    differs only by cv2-vs-PIL lanczos (drift ~0.07 on real hdmap rasters).
    """
    return decode_hdmap_numpy_limited(path, total_pixel, h, w, device, dtype)
