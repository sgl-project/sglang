import sys

import pytest
import torch

from sglang.kernels.ops.layernorm.mhc import mhc_stream_mean
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="mhc_stream_mean requires CUDA",
)


def _assert_bitwise(got: torch.Tensor, ref: torch.Tensor):
    assert got.shape == ref.shape and got.dtype == ref.dtype
    assert torch.equal(got.view(torch.uint8), ref.view(torch.uint8))


@pytest.mark.parametrize("m", [0, 7, 4097])
@pytest.mark.parametrize("h", [1024, 4096])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_matches_torch_mean_bitwise(m: int, h: int, dtype: torch.dtype):
    # The kernel path relies on torch's CUDA mean adding the 4 streams in order in
    # fp32; a change in torch's reduction order turns this red.
    torch.manual_seed(0)
    x = torch.randn(m, 4, h, device="cuda", dtype=dtype) * 1e3
    _assert_bitwise(mhc_stream_mean(x), x.mean(dim=1))


def test_extreme_values_bitwise():
    torch.manual_seed(0)
    x = torch.randn(64, 4, 4096, device="cuda", dtype=torch.bfloat16)
    x[0, :, :4] = torch.tensor([float("inf"), float("-inf"), float("nan"), -0.0])
    x[1] = 3e38
    x[2] = 2.0**-60
    _assert_bitwise(mhc_stream_mean(x), x.mean(dim=1))


def test_strided_rows_and_out_slot():
    torch.manual_seed(0)
    base = torch.randn(256, 4, 4096 + 128, device="cuda", dtype=torch.bfloat16)
    x = base[..., :4096]
    packed = torch.empty(256, 3 * 4096, device="cuda", dtype=torch.bfloat16)
    slot = packed[:, 4096 : 2 * 4096]
    got = mhc_stream_mean(x, out=slot)
    assert got.data_ptr() == slot.data_ptr()
    _assert_bitwise(slot.clone(), x.mean(dim=1))


def test_out_with_other_dtype_takes_the_fallback():
    torch.manual_seed(0)
    x = torch.randn(32, 4, 4096, device="cuda", dtype=torch.bfloat16)
    out = torch.empty(32, 4096, device="cuda", dtype=torch.float32)
    mhc_stream_mean(x, out=out)
    _assert_bitwise(out, x.mean(dim=1).float())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
