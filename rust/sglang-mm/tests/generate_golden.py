import io
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.dirname(__file__))
from bench_parity import PS, make_photo_like, ref_patchify

OUT = (
    sys.argv[1]
    if len(sys.argv) > 1
    else os.path.join(os.path.dirname(__file__), "..", "tests", "golden")
)

CASES = [
    ("480x640", 480, 640, 10),
    ("200x320", 200, 320, 11),
    ("37x53", 37, 53, 12),
    ("40x40", 40, 40, 13),
]

os.makedirs(OUT, exist_ok=True)
for name, h, w, seed in CASES:
    arr = make_photo_like(h, w, seed=seed)
    bits = ref_patchify(arr).view(torch.uint16).numpy()
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    path = os.path.join(OUT, f"golden_{name}.npz")
    np.savez_compressed(
        path,
        arr=arr,
        bits=bits,
        png=np.frombuffer(buf.getvalue(), dtype=np.uint8),
        patch_size=np.int64(PS),
    )
    print(f"  {path}: input {h}x{w}, bits {bits.shape}")
print("GOLDEN_OK")
