import pytest
import torch

from sglang.kernels.ops.layernorm.mhc import hc_combine
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="hc_combine requires CUDA",
)


def _reference(
    x_flat: torch.Tensor, pre: torch.Tensor, hc: int, out_dtype: torch.dtype
) -> torch.Tensor:
    m, h = x_flat.shape[0], x_flat.shape[1] // hc
    return (pre.unsqueeze(-1) * x_flat.view(m, hc, h)).sum(dim=1).to(out_dtype)


@pytest.mark.parametrize("m", [1, 7, 64, 1024, 2048])
@pytest.mark.parametrize("hc,h", [(4, 128), (4, 512), (8, 256)])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_hc_combine_matches_reference(m: int, hc: int, h: int, dtype: torch.dtype):
    torch.manual_seed(0)
    x_flat = torch.randn(m, hc * h, device="cuda", dtype=dtype)
    pre = torch.randn(m, hc, device="cuda", dtype=dtype).contiguous()

    got = hc_combine(x_flat, pre, hc, dtype)
    ref = _reference(x_flat, pre, hc, dtype)

    assert got.shape == (m, h)
    assert got.dtype == dtype
    # hc_combine accumulates in fp32 while the reference accumulates in `dtype`,
    # so compare against the fp32 result the kernel is approximating.
    ref_fp32 = _reference(x_flat.float(), pre.float(), hc, torch.float32)
    scale = ref_fp32.abs().max().clamp(min=1e-6)
    assert (got.float() - ref_fp32).abs().max() / scale < 5e-2
    assert (ref.float() - got.float()).abs().max() / scale < 5e-2


def test_hc_combine_strided_pre():
    torch.manual_seed(0)
    m, hc, h = 32, 4, 128
    x_flat = torch.randn(m, hc * h, device="cuda", dtype=torch.bfloat16)
    # A non-contiguous view of pre, to check the kernel honours pre's strides.
    pre = torch.randn(m, hc, 2, device="cuda", dtype=torch.bfloat16)[..., 0]
    assert not pre.is_contiguous()
    got = hc_combine(x_flat, pre, hc, torch.bfloat16)
    ref = _reference(x_flat, pre, hc, torch.bfloat16)
    scale = ref.float().abs().max().clamp(min=1e-6)
    assert (got.float() - ref.float()).abs().max() / scale < 5e-2
