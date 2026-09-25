"""The vectorized row-scale kernel must be BIT-identical to the triton
apply_log_scaling_tau kernel it replaces (same fp32 multiply + bf16 round),
including on the row-strided qkvr-slice layouts."""

import pytest
import torch

from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


@pytest.mark.parametrize("rows", [1, 3, 200, 4096])
@pytest.mark.parametrize("inner", [8, 256, 16384])
@pytest.mark.parametrize("strided", [False, True])
def test_row_compact_bitexact(rows, inner, strided):
    """The tau-less compaction flavor (kHasTau=false) must reproduce
    .contiguous() exactly on the same strided layouts row_scale handles --
    no other test exercises run_compact."""
    from sglang.kernels.ops.memory.row_compact import row_compact_bf16

    torch.manual_seed(rows + inner)
    if strided:
        packed = torch.randn(rows, inner + 40, device="cuda", dtype=torch.bfloat16)
        x = packed[:, 8 : 8 + inner]
    else:
        x = torch.randn(rows, inner, device="cuda", dtype=torch.bfloat16)
    out = row_compact_bf16(x)
    assert out.is_contiguous()
    assert torch.equal(out, x.contiguous())


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__]))
