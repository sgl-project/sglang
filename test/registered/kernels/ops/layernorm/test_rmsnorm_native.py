import pytest
import torch

from sglang.kernels.ops.layernorm.rmsnorm_native import rmsnorm_native
from sglang.srt.layers.layernorm import RMSNorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-small")

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.version.cuda is None,
    reason="rmsnorm_native requires CUDA",
)


def _native(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    norm = RMSNorm(
        x.shape[-1], eps=eps, weight_dtype=torch.float32, force_native=True
    ).cuda()
    norm.weight.data.copy_(weight)
    return norm.forward_native(x)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
@pytest.mark.parametrize("rows, dim", [(1, 1024), (37, 1024), (1521, 1024), (5, 1000)])
@pytest.mark.parametrize("scale", [1.0, 1e-20, 1e4])
def test_rmsnorm_native_matches_forward_native_bitwise(dtype, rows, dim, scale):
    torch.manual_seed(0)
    x = (torch.randn(rows, dim, device="cuda") * scale).to(dtype)
    weight = torch.randn(dim, device="cuda", dtype=torch.float32)
    assert torch.equal(rmsnorm_native(x, weight, 1e-6), _native(x, weight, 1e-6))


def test_rmsnorm_native_non_contiguous_and_empty():
    torch.manual_seed(0)
    weight = torch.randn(1024, device="cuda", dtype=torch.float32)
    x = torch.randn(64, 2048, device="cuda", dtype=torch.bfloat16)[:, ::2]
    assert torch.equal(rmsnorm_native(x, weight, 1e-6), _native(x, weight, 1e-6))
    empty = torch.empty(0, 1024, device="cuda", dtype=torch.bfloat16)
    assert rmsnorm_native(empty, weight, 1e-6).shape == (0, 1024)


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_rmsnorm_native_folds_in_the_residual_add(dtype):
    torch.manual_seed(0)
    weight = torch.randn(1024, device="cuda", dtype=torch.float32)
    x = torch.randn(300, 1024, device="cuda", dtype=dtype)
    delta = torch.randn_like(x)
    out, total = rmsnorm_native(delta, weight, 1e-6, x)
    assert torch.equal(total, x + delta)
    assert torch.equal(out, _native(x + delta, weight, 1e-6))


def test_dsv41_vit_norm_uses_fused_path_with_native_bits():
    from sglang.srt.models.deepseek_v41_vit import _rms_norm

    torch.manual_seed(0)
    norm = _rms_norm(1024).cuda()
    norm.weight.data.normal_(1.0, 0.2)
    x = torch.randn(3, 517, 1024, device="cuda", dtype=torch.bfloat16)
    assert torch.equal(norm(x), norm.forward_native(x))
