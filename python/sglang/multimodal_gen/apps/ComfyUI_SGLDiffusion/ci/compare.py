"""Compare an SGLD output against the native ComfyUI baseline.

Integrated mode runs a different attention/kernel stack than native ComfyUI,
so outputs are close but never bit-identical. Thresholds are therefore
tolerances, not equality, and are configurable per call or via the CLI of
run_workflows.py. Defaults are starting points that have not been calibrated
against real weights.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

DEFAULT_MIN_PSNR_DB = 25.0
DEFAULT_MIN_CORR = 0.90


@dataclass
class Comparison:
    psnr_db: float
    corr: float
    passed: bool
    reason: str = ""


def psnr(a, b, max_val=255.0):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    mse = float(np.mean((a - b) ** 2))
    if mse == 0.0:
        return float("inf")
    return 10.0 * np.log10(max_val**2 / mse)


def correlation(a, b):
    """Pearson correlation of the flattened arrays; 1.0 for two constants."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    da, db = a - a.mean(), b - b.mean()
    denom = np.sqrt((da**2).sum() * (db**2).sum())
    if denom == 0.0:
        return 1.0 if np.array_equal(a, b) else 0.0
    return float((da * db).sum() / denom)


def compare_arrays(
    test, ref, min_psnr_db=DEFAULT_MIN_PSNR_DB, min_corr=DEFAULT_MIN_CORR
) -> Comparison:
    if np.shape(test) != np.shape(ref):
        return Comparison(0.0, 0.0, False, f"shape {np.shape(test)} != {np.shape(ref)}")
    p, c = psnr(test, ref), correlation(test, ref)
    reasons = []
    if p < min_psnr_db:
        reasons.append(f"psnr {p:.2f} < {min_psnr_db}")
    if c < min_corr:
        reasons.append(f"corr {c:.3f} < {min_corr}")
    return Comparison(p, c, not reasons, "; ".join(reasons))


def load_image(path):
    from PIL import Image

    return np.asarray(Image.open(path).convert("RGB"))


def load_video_frames(path, which=("first", "last")):
    """Decode the first and/or last frame of a video as RGB uint8 arrays."""
    import av

    frames = []
    with av.open(str(path)) as container:
        for f in container.decode(video=0):
            frames.append(f.to_ndarray(format="rgb24"))
    if not frames:
        raise ValueError(f"no video frames in {path}")
    picks = {"first": frames[0], "last": frames[-1]}
    return [picks[w] for w in which]


def compare_files(test_path, ref_path, **thresholds) -> Comparison:
    """Compare two images or two videos (first and last frame, worst case)."""
    video_ext = (".mp4", ".webm", ".mkv", ".mov")
    if str(test_path).lower().endswith(video_ext):
        pairs = zip(load_video_frames(test_path), load_video_frames(ref_path))
    else:
        pairs = [(load_image(test_path), load_image(ref_path))]
    results = [compare_arrays(t, r, **thresholds) for t, r in pairs]
    worst = min(results, key=lambda r: r.psnr_db)
    return Comparison(
        worst.psnr_db,
        min(r.corr for r in results),
        all(r.passed for r in results),
        "; ".join(r.reason for r in results if r.reason),
    )
