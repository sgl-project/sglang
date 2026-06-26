import sys

import pytest
import torch
import torch.nn.functional as F

from sglang.jit_kernel.diffusion.triton.ltx2_dual_modulate import (
    ltx2_rmsnorm_ca_dual_modulate_from_temb,
    ltx2_rmsnorm_dual_modulate,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=8, suite="base-b-kernel-unit-1-gpu-large")

DEVICE = "cuda"
EPS = 1e-6


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)


def _rms_norm_ref(x: torch.Tensor) -> torch.Tensor:
    return F.rms_norm(x, normalized_shape=(x.shape[-1],), eps=EPS)


def _assert_close(actual: torch.Tensor, expected: torch.Tensor) -> None:
    torch.testing.assert_close(actual, expected, atol=1e-2, rtol=1e-2)


@torch.no_grad()
@pytest.mark.parametrize(
    ("batch", "seq", "hidden", "param_seq"),
    [(1, 4, 4096, 1), (2, 3, 2048, 1), (2, 3, 2048, 3)],
)
def test_ltx2_rmsnorm_dual_modulate(
    batch: int,
    seq: int,
    hidden: int,
    param_seq: int,
) -> None:
    x = torch.randn(batch, seq, hidden, device=DEVICE, dtype=torch.bfloat16)
    scale0 = torch.randn(batch, param_seq, hidden, device=DEVICE, dtype=torch.bfloat16)
    shift0 = torch.randn(batch, param_seq, hidden, device=DEVICE, dtype=torch.bfloat16)
    scale1 = torch.randn(batch, param_seq, hidden, device=DEVICE, dtype=torch.bfloat16)
    shift1 = torch.randn(batch, param_seq, hidden, device=DEVICE, dtype=torch.bfloat16)

    actual0, actual1 = ltx2_rmsnorm_dual_modulate(
        x, scale0, shift0, scale1, shift1, EPS
    )
    normed = _rms_norm_ref(x)
    expected0 = normed * (1 + scale0.expand_as(x)) + shift0.expand_as(x)
    expected1 = normed * (1 + scale1.expand_as(x)) + shift1.expand_as(x)

    _assert_close(actual0, expected0)
    _assert_close(actual1, expected1)


@torch.no_grad()
@pytest.mark.parametrize("batch,seq,hidden", [(1, 4, 4096), (2, 3, 2048)])
@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
def test_ltx2_rmsnorm_ca_dual_modulate_from_temb(
    batch: int,
    seq: int,
    hidden: int,
    table_dtype: torch.dtype,
) -> None:
    x = torch.randn(batch, seq, hidden, device=DEVICE, dtype=torch.bfloat16)
    temb_scale_shift = torch.randn(
        batch, seq, 4 * hidden, device=DEVICE, dtype=torch.bfloat16
    )
    scale_shift_table = torch.randn(
        4, hidden, device=DEVICE, dtype=table_dtype
    ).contiguous()

    actual0, actual1 = ltx2_rmsnorm_ca_dual_modulate_from_temb(
        x, temb_scale_shift, scale_shift_table, EPS
    )
    scale0, shift0, scale1, shift1 = (
        scale_shift_table.to(dtype=x.dtype).view(1, 1, 4, hidden)
        + temb_scale_shift.reshape(batch, seq, 4, hidden)
    ).unbind(dim=2)
    normed = _rms_norm_ref(x)
    expected0 = normed * (1 + scale0) + shift0
    expected1 = normed * (1 + scale1) + shift1

    _assert_close(actual0, expected0)
    _assert_close(actual1, expected1)


@torch.no_grad()
def test_ltx2_rmsnorm_ca_dual_modulate_rejects_shape() -> None:
    x = torch.randn(1, 4, 4096, device=DEVICE, dtype=torch.bfloat16)
    temb_scale_shift = torch.randn(1, 4, 4 * 4096, device=DEVICE, dtype=torch.bfloat16)
    scale_shift_table = torch.randn(3, 4096, device=DEVICE, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="scale_shift_table"):
        ltx2_rmsnorm_ca_dual_modulate_from_temb(
            x, temb_scale_shift, scale_shift_table, EPS
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
