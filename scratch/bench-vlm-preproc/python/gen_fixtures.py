"""Generate deterministic test images for the VLM preprocessor bench.

We want a mix of sizes and content complexity. JPEG quality 90, PNG default.
Content is procedural so the JPEG entropy is realistic (not flat color).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from PIL import Image

OUT_DIR = Path(__file__).resolve().parent.parent / "fixtures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIZES = [
    ("small", 512, 512),
    ("medium", 1024, 768),
    ("large", 2048, 1536),
    ("xl", 4096, 3072),
]


def gradient_with_noise(h: int, w: int, seed: int) -> np.ndarray:
    """A gradient + sinusoidal pattern + low-amplitude noise.
    Looks roughly like a photo (decent JPEG entropy)."""
    rng = np.random.default_rng(seed)
    y = np.linspace(0, 1, h, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, w, dtype=np.float32)[None, :]

    r = (y * 0.6 + 0.4 * np.sin(20 * x + 1.1)) * 255
    g = (x * 0.6 + 0.4 * np.sin(15 * y + 0.3)) * 255
    b = ((x + y) * 0.3 + 0.4 * np.sin(8 * (x + y) + 2.1)) * 255

    img = np.stack([r, g, b], axis=-1)
    img += rng.normal(0, 8, img.shape).astype(np.float32)
    return np.clip(img, 0, 255).astype(np.uint8)


def main():
    for i, (tag, w, h) in enumerate(SIZES):
        arr = gradient_with_noise(h, w, seed=42 + i)
        pil = Image.fromarray(arr, mode="RGB")
        jpg = OUT_DIR / f"{tag}_{w}x{h}.jpg"
        png = OUT_DIR / f"{tag}_{w}x{h}.png"
        pil.save(jpg, format="JPEG", quality=90)
        pil.save(png, format="PNG", optimize=False)
        print(
            f"  wrote {jpg.name} ({jpg.stat().st_size/1024:.0f} KB) / "
            f"{png.name} ({png.stat().st_size/1024:.0f} KB)"
        )

    # XL JPEG variant WITH restart markers, used to exercise the parallel-strip
    # decode path. Otherwise our fixtures contain zero RST markers and the
    # parallel path falls back to single-thread (as it does in production for
    # most natural-image JPEGs).
    tag, w, h = "xl", 4096, 3072
    arr = gradient_with_noise(h, w, seed=42 + 3)  # match xl seed
    pil = Image.fromarray(arr, mode="RGB")
    out = OUT_DIR / f"{tag}_{w}x{h}_rst16.jpg"
    pil.save(out, format="JPEG", quality=90, restart_marker_blocks=16)
    print(
        f"  wrote {out.name} ({out.stat().st_size/1024:.0f} KB)  "
        f"[restart markers every 16 MCU blocks]"
    )

    print(f"\nFixtures in: {OUT_DIR}")


if __name__ == "__main__":
    main()
