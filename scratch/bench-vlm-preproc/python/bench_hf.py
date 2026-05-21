"""Benchmark HF Qwen2VLImageProcessor on the same fixtures the Rust bench uses.

Two paths reported:
  1. PIL-load + HF processor   (closest to what sglang's QwenVLImageProcessor does
                                when nvJPEG path is not taken — i.e. CPU path)
  2. HF processor on a preloaded PIL image (just the processor cost, no decode)
"""

from __future__ import annotations

import io
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

FIXTURES = Path(__file__).resolve().parent.parent / "fixtures"


def build_processor() -> Qwen2VLImageProcessor:
    return Qwen2VLImageProcessor(
        do_resize=True,
        do_rescale=True,
        do_normalize=True,
        patch_size=14,
        temporal_patch_size=2,
        merge_size=2,
        min_pixels=56 * 56 * 4,
        max_pixels=28 * 28 * 1280,
        image_mean=[0.48145466, 0.4578275, 0.40821073],
        image_std=[0.26862954, 0.26130258, 0.27577711],
    )


def per_image_bench(iters: int = 20):
    proc = build_processor()
    fixtures = sorted(
        [p for p in FIXTURES.iterdir() if p.suffix in (".jpg", ".jpeg", ".png")]
    )

    print(f"# HF Qwen2VLImageProcessor bench")
    print(f"# fixtures: {len(fixtures)} from {FIXTURES}")
    print(f"# torch threads: {torch.get_num_threads()}")
    print()
    print(f"## Per-image timing ({iters} iters each)")
    header = (
        f"{'fixture':<28} {'size_KB':>8} "
        f"{'decode_us':>10} {'process_us':>11} {'total_us':>10}   target          patches"
    )
    print(header)

    summary = []
    for path in fixtures:
        raw = path.read_bytes()
        size_kb = len(raw) // 1024
        # Warmup
        img = Image.open(io.BytesIO(raw)).convert("RGB")
        _ = proc(images=img, return_tensors="pt")

        decode_total = 0.0
        process_total = 0.0
        total_total = 0.0
        last_grid = None
        last_np = None
        for _ in range(iters):
            t0 = time.perf_counter()
            img = Image.open(io.BytesIO(raw)).convert("RGB")
            # Force PIL to actually decode rather than lazy-load:
            img.load()
            t1 = time.perf_counter()
            out = proc(images=img, return_tensors="pt")
            t2 = time.perf_counter()
            decode_total += t1 - t0
            process_total += t2 - t1
            total_total += t2 - t0
            last_grid = out["image_grid_thw"][0].tolist()
            last_np = out["pixel_values"].shape[0]

        decode_us = decode_total / iters * 1e6
        process_us = process_total / iters * 1e6
        total_us = total_total / iters * 1e6
        target = f"{last_grid[1]*14}x{last_grid[2]*14}"

        print(
            f"{path.name:<28} {size_kb:>8} "
            f"{decode_us:>10.1f} {process_us:>11.1f} {total_us:>10.1f}   {target:<14}  {last_np:>6}"
        )
        summary.append((path.name, total_us, last_np))

    return summary


def batch_bench(iters: int = 5):
    """Run all fixtures in one HF call (HF supports list of images)."""
    proc = build_processor()
    fixtures = sorted(
        [p for p in FIXTURES.iterdir() if p.suffix in (".jpg", ".jpeg", ".png")]
    )
    # Pre-decode to PIL so this measures only HF processor time
    raws = [p.read_bytes() for p in fixtures]

    print()
    print(f"## Batch timing ({iters} iters)")
    # warmup
    pil_imgs = [Image.open(io.BytesIO(r)).convert("RGB") for r in raws]
    for im in pil_imgs:
        im.load()
    _ = proc(images=pil_imgs, return_tensors="pt")

    total = 0.0
    for _ in range(iters):
        t0 = time.perf_counter()
        pil_imgs = [Image.open(io.BytesIO(r)).convert("RGB") for r in raws]
        for im in pil_imgs:
            im.load()
        _ = proc(images=pil_imgs, return_tensors="pt")
        total += time.perf_counter() - t0
    avg_ms = total / iters * 1000
    print(
        f"batch of {len(fixtures)} images (decode + process): "
        f"avg wall = {avg_ms:.2f} ms ({avg_ms/len(fixtures):.2f} ms/image)"
    )


if __name__ == "__main__":
    per_image_bench()
    batch_bench()
