"""Kimi image resize: the CPU fallback for large sources matches the GPU resize."""

import sys

import numpy as np
import pytest
import torch

from sglang.srt.multimodal.processors.kimi_k25 import _resize_bicubic_if_needed
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=10, stage="base-b", runner_config="1-gpu-small")
register_amd_ci(est_time=10, suite="stage-b-test-1-gpu-small-amd")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


def _natural_image(height: int, width: int) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    base = (
        127
        + 60 * np.sin(2 * np.pi * xx / (width / 7.3))
        + 50 * np.cos(2 * np.pi * yy / (height / 5.1))
    )
    edges = 255.0 * ((xx // 9 + yy // 7) % 2)
    tex = 30.0 * np.sin(xx * 12.9898 + yy * 78.233)
    img = np.clip(0.55 * base + 0.30 * edges + 0.15 * (127 + tex), 0, 255)
    return np.stack(
        [img, np.roll(img, 13, axis=0), np.roll(img, 29, axis=1)], axis=-1
    ).astype(np.uint8)


@pytest.mark.parametrize(
    "source_hw, target_hw",
    [
        ((1200, 1600), (600, 800)),
        ((1200, 1600), (840, 1120)),
        ((4032, 3024), (2576, 1932)),
    ],
)
def test_cpu_fallback_matches_gpu_resize(source_hw, target_hw):
    """Same source on the GPU, resized there and through the CPU fallback:
    within one pixel level everywhere, and mostly bit-exact."""
    x = torch.from_numpy(_natural_image(*source_hw)).permute(2, 0, 1).unsqueeze(0)
    x = x.cuda()

    gpu = _resize_bicubic_if_needed(x, *target_hw)
    cpu = _resize_bicubic_if_needed(x, *target_hw, max_gpu_pixels=0)

    assert gpu.is_cuda and cpu.device == x.device
    diff = (gpu.float() - cpu.float()).abs()
    assert diff.max().item() <= 1.0
    assert (diff == 0).float().mean().item() > 0.7


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
