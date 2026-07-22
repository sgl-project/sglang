"""The vectorized row-scale kernel must be BIT-identical to the triton
apply_log_scaling_tau kernel it replaces (same fp32 multiply + bf16 round),
including on the row-strided qkvr-slice layouts."""

import pytest
import torch

from sglang.kernels.ops.attention.log_scaling_tau import (
    _apply_log_scaling_tau_kernel,
)
from sglang.kernels.ops.model.inkling.inkling_row_scale import row_scale_bf16
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _triton_ref(x2d, tau):
    import triton

    rows, inner = x2d.shape
    out = torch.empty(rows, inner, dtype=x2d.dtype, device=x2d.device)
    total = rows * inner
    _apply_log_scaling_tau_kernel[(triton.cdiv(total, 1024),)](
        x2d, tau, out, x2d.stride(0), inner, total, BLOCK=1024
    )
    return out


@pytest.mark.parametrize("rows", [1, 3, 16, 200, 4096])
@pytest.mark.parametrize("inner", [8, 256, 2048, 16384])
@pytest.mark.parametrize("strided", [False, True])
def test_row_scale_bitexact(rows, inner, strided):
    torch.manual_seed(rows + inner)
    if strided:
        packed = torch.randn(rows, inner + 40, device="cuda", dtype=torch.bfloat16)
        x = packed[:, 8 : 8 + inner]
    else:
        x = torch.randn(rows, inner, device="cuda", dtype=torch.bfloat16)
    tau = 1.0 + 0.1 * torch.rand(rows, device="cuda", dtype=torch.float32)

    out = row_scale_bf16(x, tau)
    ref = _triton_ref(x, tau)
    assert torch.equal(out, ref)


@pytest.mark.parametrize("rows", [1, 3, 200, 4096])
@pytest.mark.parametrize("inner", [8, 256, 16384])
@pytest.mark.parametrize("strided", [False, True])
def test_row_compact_bitexact(rows, inner, strided):
    """The tau-less compaction flavor (kHasTau=false) must reproduce
    .contiguous() exactly on the same strided layouts row_scale handles --
    no other test exercises run_compact."""
    from sglang.kernels.ops.model.inkling.inkling_row_scale import row_compact_bf16

    torch.manual_seed(rows + inner)
    if strided:
        packed = torch.randn(rows, inner + 40, device="cuda", dtype=torch.bfloat16)
        x = packed[:, 8 : 8 + inner]
    else:
        x = torch.randn(rows, inner, device="cuda", dtype=torch.bfloat16)
    out = row_compact_bf16(x)
    assert out.is_contiguous()
    assert torch.equal(out, x.contiguous())


def test_dispatch_through_apply_log_scaling_tau():
    from sglang.kernels.ops.attention.log_scaling_tau import apply_log_scaling_tau

    torch.manual_seed(0)
    for shape, view in (((7, 16, 16), (-1, 1, 1)), ((7, 2048), (-1, 1))):
        x = torch.randn(*shape, device="cuda", dtype=torch.bfloat16)
        tau = 1.0 + 0.1 * torch.rand(7, device="cuda", dtype=torch.float32)
        out = apply_log_scaling_tau(x, tau.view(*view))
        ref = _triton_ref(x.view(7, -1), tau).view(x.shape)
        assert torch.equal(out, ref)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
