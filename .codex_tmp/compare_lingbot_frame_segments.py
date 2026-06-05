#!/usr/bin/env python3
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np


def digest(x: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(str(x.shape).encode())
    h.update(x.tobytes())
    return h.hexdigest()


def compare(name: str, a: np.ndarray, b: np.ndarray) -> dict[str, object]:
    result: dict[str, object] = {
        "name": name,
        "a_shape": list(a.shape),
        "b_shape": list(b.shape),
        "same_shape": a.shape == b.shape,
        "a_sha256": digest(a),
        "b_sha256": digest(b),
    }
    if a.shape == b.shape:
        diff = b.astype(np.int16) - a.astype(np.int16)
        abs_diff = np.abs(diff)
        result.update(
            {
                "exact_equal": bool(np.array_equal(a, b)),
                "max_abs_diff": int(abs_diff.max(initial=0)),
                "mean_abs_diff": float(abs_diff.mean()),
                "nonzero_elements": int(np.count_nonzero(abs_diff)),
                "num_elements": int(abs_diff.size),
            }
        )
    return result


baseline = np.load(sys.argv[1])["frames"]
patch = np.load(sys.argv[2])["frames"]

segments = []
segments.append(compare("prefix_min_len", baseline[: len(patch)], patch))
segments.append(compare("chunk0", baseline[:9], patch[:9]))
segments.append(compare("chunk1_first9", baseline[9:18], patch[9:18]))
segments.append(compare("baseline_chunk0_vs_patch_chunk1", baseline[:9], patch[9:18]))
segments.append(compare("baseline_chunk1_tail3", baseline[18:21], patch[18:21]))
segments.append(compare("baseline_chunk2_vs_patch_chunk2", baseline[21:33], patch[18:30]))
segments.append(compare("baseline_chunk3_vs_patch_chunk3", baseline[33:45], patch[30:42]))
print(json.dumps(segments, indent=2, sort_keys=True))
