import sys

import pytest
import torch

from sglang.jit_kernel.diffusion.triton.ltx2_ada_values import ltx2_ada_values9
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci

register_cuda_ci(est_time=8, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_amd_ci(est_time=8, suite="nightly-amd-kernel-1-gpu", nightly=True)

DEVICE = "cuda"


@pytest.fixture(autouse=True)
def cuda_setup():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    torch.cuda.manual_seed(0)


def _reference(
    scale_shift_table: torch.Tensor,
    timestep: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    batch, seq, _ = timestep.shape
    hidden = scale_shift_table.shape[1]
    return (
        scale_shift_table.to(device=timestep.device, dtype=timestep.dtype)
        .view(1, 1, 9, hidden)
        .add(timestep.reshape(batch, seq, 9, hidden))
        .unbind(dim=2)
    )


@torch.no_grad()
@pytest.mark.parametrize("batch,seq,hidden", [(1, 1, 4096), (2, 3, 2048)])
@pytest.mark.parametrize("table_dtype", [torch.bfloat16, torch.float32])
def test_ltx2_ada_values9(
    batch: int,
    seq: int,
    hidden: int,
    table_dtype: torch.dtype,
) -> None:
    scale_shift_table = torch.randn(
        9, hidden, device=DEVICE, dtype=table_dtype
    ).contiguous()
    timestep = torch.randn(
        batch, seq, 9 * hidden, device=DEVICE, dtype=torch.bfloat16
    ).contiguous()

    actual = ltx2_ada_values9(scale_shift_table, timestep)
    expected = _reference(scale_shift_table, timestep)

    assert len(actual) == 9
    for actual_value, expected_value in zip(actual, expected):
        torch.testing.assert_close(actual_value, expected_value, atol=0, rtol=0)


@torch.no_grad()
def test_ltx2_ada_values9_rejects_unsupported_shape() -> None:
    scale_shift_table = torch.randn(8, 4096, device=DEVICE, dtype=torch.bfloat16)
    timestep = torch.randn(1, 1, 9 * 4096, device=DEVICE, dtype=torch.bfloat16)

    with pytest.raises(ValueError, match="scale_shift_table"):
        ltx2_ada_values9(scale_shift_table, timestep)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-s"]))
