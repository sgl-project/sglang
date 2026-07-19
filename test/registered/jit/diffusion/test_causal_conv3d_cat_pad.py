import sys

import pytest
import torch

from sglang.jit_kernel.diffusion.causal_conv3d_cat_pad import (
    fused_causal_conv3d_cat_pad_cuda,
)
from sglang.jit_kernel.diffusion.triton.causal_conv3d_pad import (
    fused_causal_conv3d_cat_pad as fused_causal_conv3d_cat_pad_triton,
)
from sglang.kernels.jit import get_ci_test_range
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=45, stage="base-b-kernel-unit", runner_config="4-gpu-b200")
register_amd_ci(est_time=10, stage="jit-kernel-unit", runner_config="amd")

DEVICE = "cuda"
DTYPE = torch.bfloat16

COSMOS3_CASES = get_ci_test_range(
    [
        (1024, 1, 30, 52, 1),
        (1024, 1, 30, 52, 2),
        (1024, 2, 60, 104, 1),
        (1024, 2, 60, 104, 2),
        (512, 4, 120, 208, 1),
        (512, 4, 120, 208, 2),
        (256, 4, 240, 416, 1),
        (256, 4, 240, 416, 2),
    ],
    [(1024, 1, 30, 52, 1), (512, 4, 120, 208, 2)],
)


def _make_inputs(
    channels: int,
    t_size: int,
    h_size: int,
    w_size: int,
    cache_t: int,
) -> tuple[torch.Tensor, torch.Tensor, tuple[int, ...]]:
    generator = torch.Generator(device=DEVICE)
    generator.manual_seed(channels * 1009 + t_size * 251 + h_size + cache_t)
    x = torch.randn(
        (1, channels, t_size, h_size, w_size),
        device=DEVICE,
        dtype=DTYPE,
        generator=generator,
    )
    cache_x = torch.randn(
        (1, channels, cache_t, h_size, w_size),
        device=DEVICE,
        dtype=DTYPE,
        generator=generator,
    )
    padding = (1, 1, 1, 1, cache_t, 0)
    return x, cache_x, padding


@pytest.mark.parametrize("channels,t_size,h_size,w_size,cache_t", COSMOS3_CASES)
def test_causal_conv3d_cat_pad(
    channels: int,
    t_size: int,
    h_size: int,
    w_size: int,
    cache_t: int,
) -> None:
    x, cache_x, padding = _make_inputs(channels, t_size, h_size, w_size, cache_t)
    actual = fused_causal_conv3d_cat_pad_cuda(x, cache_x, padding)
    expected = fused_causal_conv3d_cat_pad_triton(x, cache_x, padding)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_causal_conv3d_cat_pad_torch_compile() -> None:
    x, cache_x, padding = _make_inputs(1024, 1, 30, 52, 1)

    @torch.compile(fullgraph=True)
    def fn(x: torch.Tensor, cache_x: torch.Tensor) -> torch.Tensor:
        return fused_causal_conv3d_cat_pad_cuda(x, cache_x, padding)

    actual = fn(x, cache_x)
    expected = fused_causal_conv3d_cat_pad_triton(x, cache_x, padding)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
