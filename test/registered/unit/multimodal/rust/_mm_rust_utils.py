"""Shared fixtures for the Rust multimodal suites.

Imported via ``sys.path`` from the sibling suites (unittest runs these files by
path, so a package-relative import would break ``python <file>``); the module
name is deliberately specific so it cannot shadow another suite's helpers on the
process-global ``sys.path``.
"""

import io

import numpy as np
from PIL import Image

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import is_in_ci

register_cpu_ci(
    est_time=0, suite="base-a-test-cpu", disabled="Rust multimodal test helpers"
)


def load_core():
    """The Rust ``_multimodal`` extension, or ``None`` (→ skip) when not built
    locally. In CI a missing extension is a hard failure, never a silent
    skip — the CPU suite builds it from source."""
    try:
        from sglang.srt.rust_extensions import _multimodal

        return _multimodal
    except ImportError:
        if is_in_ci():
            raise
        return None


def make_image(width, height, seed=0):
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:height, 0:width]
    base = np.stack(
        (x * 255 / max(width - 1, 1), y * 255 / max(height - 1, 1), (x + y) % 256),
        axis=-1,
    )
    return Image.fromarray(
        np.clip(base + rng.integers(0, 24, base.shape), 0, 255).astype(np.uint8)
    )


def image_bytes(width, height, seed=0):
    buffer = io.BytesIO()
    make_image(width, height, seed).save(buffer, format="PNG")
    return buffer.getvalue()
