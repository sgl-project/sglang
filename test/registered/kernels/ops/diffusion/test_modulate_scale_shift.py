import pytest
import torch

from sglang.kernels.ops.diffusion.modulate_scale_shift import (
    can_use_modulate_scale_shift_cuda,
    modulate_scale_shift,
    modulate_scale_shift_cuda,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")

# FLUX.1 1024^2 adaLN shapes (D=3072) plus batched and odd-length coverage.
CASES = [(1, 4096, 3072), (1, 512, 3072), (1, 4608, 3072), (2, 1024, 3072), (1, 17, 64)]


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)


def _eager(x, scale, shift):
    return x * (1 + scale[:, None]) + shift[:, None]


@pytest.mark.parametrize("shape", CASES)
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_modulate_scale_shift_matches_eager(shape, dtype):
    x = torch.randn(shape, device="cuda", dtype=dtype)
    scale = torch.randn((shape[0], shape[-1]), device="cuda", dtype=dtype)
    shift = torch.randn_like(scale)
    out = modulate_scale_shift_cuda(x, scale, shift)
    assert torch.equal(out, _eager(x, scale, shift))  # bitwise contract


def test_modulate_scale_shift_adaln_chunk_views():
    x = torch.randn((1, 4096, 3072), device="cuda", dtype=torch.bfloat16)
    emb = torch.randn((1, 6 * 3072), device="cuda", dtype=torch.bfloat16)
    shift, scale = emb.chunk(6, dim=1)[:2]
    assert can_use_modulate_scale_shift_cuda(x, scale, shift)
    out = modulate_scale_shift_cuda(x, scale, shift)
    assert torch.equal(out, _eager(x, scale, shift))


def test_modulate_scale_shift_guards_reject_fp32():
    x = torch.randn((1, 64, 64), device="cuda", dtype=torch.float32)
    row = torch.randn((1, 64), device="cuda", dtype=torch.float32)
    assert not can_use_modulate_scale_shift_cuda(x, row, row)
    assert torch.equal(modulate_scale_shift(x, row, row), _eager(x, row, row))


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
