"""Production HD-map video decode for the per-frame (video) path.

Decodes an on-disk HD-map mp4 to ``[1, 3, total_pixel, H, W]`` in ``[-1, 1]``
matching the ``OmniDreamsBeforeDenoisingStage._preprocess_hdmap_clip`` contract,
so it is a drop-in for the per-frame video path (replacing the old
``load_video``-decodes-all-frames + PIL preprocess).

A+B optimization (the winner of the real-sample benchmark, 5-9x faster than
``load_video``): decode only ``total_pixel`` frames (early-stop ffmpeg) +
cv2.resize + numpy->torch direct (no PIL roundtrip). Near-zero drift vs the
PIL path: native-res (no resize) output is bit-identical; resize output differs
only by cv2-vs-PIL lanczos (~0.07 on real hdmap rasters).
"""

import cv2
import imageio.v2 as imageio
import numpy as np
import torch


def _read_frames_numpy(path, max_frames=None):
    """Decode an mp4 to a list of ``[H, W, 3]`` uint8 numpy frames.

    Same ffmpeg backend as ``vision_utils.load_video`` (``imageio.get_reader``),
    but yields numpy directly -- no ``PIL.Image.fromarray`` roundtrip. When
    ``max_frames`` is set, iteration stops after that many frames so ffmpeg never
    decodes the tail that would be truncated anyway.
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

    Mirrors ``OmniDreamsBeforeDenoisingStage._encode_hdmap``.
    """
    n = len(frames)
    if n == 0:
        raise ValueError("no frames decoded")
    if n < total_pixel:
        frames = list(frames) + [frames[-1] for _ in range(total_pixel - n)]
    else:
        frames = list(frames[:total_pixel])
    return frames


def _preprocess_numpy(frames_np, h, w, device, dtype):
    """cv2.resize (LANCZOS4) on numpy, bulk numpy->torch, normalize.

    No PIL anywhere. Math matches the PIL baseline (arr/255*2-1) so output is
    bit-identical when the resize is a no-op (target == native res).
    """
    resized = [
        cv2.resize(f, (w, h), interpolation=cv2.INTER_LANCZOS4) for f in frames_np
    ]
    arr = np.stack(resized, axis=0).astype(np.float32) / 255.0  # [T,h,w,3] in [0,1]
    x = torch.from_numpy(arr).permute(0, 3, 1, 2)  # [T,3,h,w] in [0,1]
    x = 2.0 * x - 1.0  # -> [-1,1]
    return x.unsqueeze(0).to(device=device, dtype=dtype)  # [1,T,3,h,w]


def decode_hdmap_ab(path, total_pixel, h, w, device, dtype):
    """Production HD-map decode: A+B (numpy->torch direct + early-stop ffmpeg).

    Returns ``[1, 3, total_pixel, h, w]`` in ``[-1, 1]`` -- the same contract as
    ``OmniDreamsBeforeDenoisingStage._preprocess_hdmap_clip``.
    """
    frames = _read_frames_numpy(path, max_frames=total_pixel)
    frames = _clamp_to_total(frames, total_pixel)
    clip = _preprocess_numpy(frames, h, w, device, dtype)  # [1,T,3,h,w]
    return clip.permute(0, 2, 1, 3, 4)  # [1,3,T,h,w]
