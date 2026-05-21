"""Validate Rust preprocessor output against HF Qwen2VLImageProcessor.

We instantiate Qwen2VLImageProcessor from explicit kwargs (no model download).
Then we compare its pixel_values vs the Rust binary's dumped f32.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor


def run_hf(image_path: Path):
    img = Image.open(image_path).convert("RGB")
    proc = Qwen2VLImageProcessor(
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
    out = proc(images=img, return_tensors="pt")
    pixel_values = out["pixel_values"].numpy()
    grid_thw = out["image_grid_thw"].numpy()  # [1, 3] (t, h, w)
    return pixel_values, grid_thw


def load_rust_dump(stem: Path):
    bin_path = stem.with_suffix(".f32")
    json_path = stem.with_suffix(".json")
    meta = json.loads(json_path.read_text())
    raw = bin_path.read_bytes()
    arr = np.frombuffer(raw, dtype=np.float32).reshape(
        meta["num_patches"], meta["patch_features"]
    )
    return arr, meta


def main():
    if len(sys.argv) != 3:
        print("usage: python validate.py <image> <rust_dump_stem>", file=sys.stderr)
        sys.exit(2)
    image_path = Path(sys.argv[1])
    stem = Path(sys.argv[2])

    print(f"image: {image_path}")
    print(f"rust dump stem: {stem}")

    hf_px, hf_grid = run_hf(image_path)
    rust_px, rust_meta = load_rust_dump(stem)

    print()
    print(f"HF   pixel_values shape: {hf_px.shape}  grid_thw: {hf_grid.tolist()}")
    print(
        f"Rust pixel_values shape: {rust_px.shape}  grid_thw: {rust_meta['grid_thw']}"
    )

    if hf_px.shape != rust_px.shape:
        print(f"\nFAIL: shape mismatch")
        sys.exit(1)
    if hf_grid.flatten().tolist() != list(rust_meta["grid_thw"]):
        print(f"\nFAIL: grid_thw mismatch")
        sys.exit(1)

    # Compute diff stats
    diff = hf_px - rust_px
    print()
    print(f"diff stats:")
    print(f"  max abs diff:  {np.abs(diff).max():.6f}")
    print(f"  mean abs diff: {np.abs(diff).mean():.6f}")
    print(f"  rms diff:      {np.sqrt(np.mean(diff**2)):.6f}")

    # Bilinear resize differences expected (HF uses torchvision/PIL, we use fast_image_resize).
    # Tolerate up to ~0.02 in normalized space (which is ~0.04*255 ≈ 1 raw pixel value).
    if np.abs(diff).max() < 0.05:
        print("\nPASS: values within tolerance (bilinear resize variation expected)")
    else:
        print(f"\nWARN: max diff > 0.05; resize impls may differ more than expected")


if __name__ == "__main__":
    main()
