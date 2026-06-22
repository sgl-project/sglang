# test/spikes/omnidreams_hdmap_decode_benchmark.py
"""CPU benchmark: A (drop PIL) / B (decode only total_pixel) / AB vs baseline.

Run (CPU):
    TORCHDYNAMO_DISABLE=1 /Users/cerdore/.python/sglang/bin/python \
        python/sglang/multimodal_gen/test/spikes/omnidreams_hdmap_decode_benchmark.py

Goal: decide whether to wire A/B into the production hdmap decode path.
NOT NVDEC -- pure CPU decode/preprocess cost.

Design:
  * Matrix over (frames_in_mp4, total_pixel, native_res, target_res) so A's PIL-churn
    cost and B's tail-decode cost are each isolated.
  * Repeats + median (ffmpeg warmup, GC noise).
  * Drift = max|variant - baseline| (resize backend only) per cell.

libx264 rounds the mp4 width up to a multiple of 16, so every native width below
is already mod-16 (else the "no-resize" cells would silently resize).
"""

from __future__ import annotations

import gc
import statistics
import tempfile
import time
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch

from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.omnidreams_hdmap_decode import (
    decode_hdmap_baseline,
    decode_hdmap_limited,
    decode_hdmap_numpy,
    decode_hdmap_numpy_limited,
)

VARIANTS = {
    "baseline": decode_hdmap_baseline,
    "A_numpy": decode_hdmap_numpy,
    "B_limited": decode_hdmap_limited,
    "AB_numpy_limited": decode_hdmap_numpy_limited,
}


def _write_synthetic_mp4(path, num_frames, h, w, seed=0):
    rng = np.random.RandomState(seed)
    with imageio.get_writer(str(path), codec="libx264", fps=30, quality=5) as wr:
        for _ in range(num_frames):
            wr.append_data(rng.randint(0, 256, size=(h, w, 3), dtype=np.uint8))
    return str(path)


def _time(fn, *args, repeats=5, warmup=1):
    for _ in range(warmup):
        fn(*args)
    gc.collect()
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        ts.append(time.perf_counter() - t0)
    gc.collect()
    return statistics.median(ts) * 1000.0  # ms


def main():
    device, dtype = torch.device("cpu"), torch.float32
    # (label, frames_in_mp4, total_pixel, native (H,W), target (H,W))
    cases = [
        ("short no-resize",        9,  9,  (16, 32),  (16, 32)),
        ("short resize",           9,  9,  (16, 32),  (32, 48)),
        ("long(60) need-9 no-rs",  60, 9,  (16, 32),  (16, 32)),
        ("long(60) need-9 rs",     60, 9,  (16, 32),  (32, 48)),
        ("long(200) need-49 rs",   200, 49, (32, 48),  (64, 96)),
    ]

    header = f"{'case':<24} {'variant':<16} {'median_ms':>10} {'vs_base':>8} {'drift_maxabs':>13}"
    print(header)
    print("-" * len(header))
    with tempfile.TemporaryDirectory() as td:
        for label, nf, tp, (nh, nw), (th, tw) in cases:
            p = _write_synthetic_mp4(Path(td) / f"{label}.mp4", nf, nh, nw)
            base_ms = None
            base_clip = None
            for name, fn in VARIANTS.items():
                ms = _time(fn, p, tp, th, tw, device, dtype)
                clip = fn(p, tp, th, tw, device, dtype)
                if name == "baseline":
                    base_ms, base_clip = ms, clip
                    drift = "-"
                else:
                    drift = f"{float((clip - base_clip).abs().max()):.4f}"
                speedup = f"{base_ms / ms:.2f}x" if base_ms else "-"
                print(f"{label:<24} {name:<16} {ms:>10.2f} {speedup:>8} {drift:>13}")
            print()

    print("Done. Drift = max|variant - baseline| (resize-backend-only; B/AB are ~0).")
    print("Decision rule: wire variant in if speedup is material AND drift is acceptable.")


if __name__ == "__main__":
    main()
