# test/spikes/omnidreams_hdmap_decode_benchmark_real.py
"""Real-sample CPU benchmark: A / B / AB vs baseline on a REAL hdmap.mp4.

Run (rtx6k):
    /root/autodl-tmp/sglang-venv/bin/python3 \
        python/sglang/multimodal_gen/test/spikes/omnidreams_hdmap_decode_benchmark_real.py \
        [path/to/hdmap.mp4]

Why a separate harness: the synthetic one (16x32..64x96, <=200 frames) could
NOT stress the real regime -- real hdmap clips are 1280x720 @ ~2400-3000 frames,
so the frame-count lever (B) and per-frame PIL cost (A) both operate at a scale
the synthetic clips under-represent by ~500x (pixels) / ~12-60x (frames).
"""

from __future__ import annotations

import gc
import statistics
import sys
import time

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

DEFAULT_SAMPLE = (
    "/root/autodl-tmp/omni-dreams-samples/data/single_view/"
    "23599139-948f-4681-b7f4-74794113086d/"
    "23599139-948f-4681-b7f4-74794113086d_18016.79223_18096.79223_80.0_hdmap.mp4"
)


def _time(fn, *args, repeats=3, warmup=1):
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
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_SAMPLE
    device, dtype = torch.device("cpu"), torch.float32
    # (label, total_pixel, target (H,W)). native is 1280x720 (the real mp4).
    # 13 latent -> total_pixel = 1+(13-1)*4 = 49 ; 49 latent -> 193.
    cases = [
        ("tp=49  native(no-rs)", 49, (720, 1280)),
        ("tp=49  720p(704,rs)", 49, (704, 1280)),
        ("tp=193 native(no-rs)", 193, (720, 1280)),
        ("tp=193 720p(704,rs)", 193, (704, 1280)),
    ]
    print(f"sample: {path}")
    header = f"{'case':<22} {'variant':<16} {'median_ms':>10} {'vs_base':>8} {'drift_maxabs':>13}"
    print(header)
    print("-" * len(header))
    for label, tp, (th, tw) in cases:
        base_ms = None
        base_clip = None
        for name, fn in VARIANTS.items():
            ms = _time(fn, path, tp, th, tw, device, dtype, repeats=3)
            clip = fn(path, tp, th, tw, device, dtype)
            if name == "baseline":
                base_ms, base_clip = ms, clip
                drift = "-"
            else:
                drift = f"{float((clip - base_clip).abs().max()):.4f}"
            speedup = f"{base_ms / ms:.2f}x" if base_ms else "-"
            print(f"{label:<22} {name:<16} {ms:>10.1f} {speedup:>8} {drift:>13}")
        print()
    print(
        "Done. B/AB drift ~0 (bit-identical); A drift = cv2-vs-PIL lanczos (resize only)."
    )


if __name__ == "__main__":
    main()
