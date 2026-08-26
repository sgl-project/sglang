import pytest
import torch

from sglang.kernels.ops.layernorm.grouped_gemma_rmsnorm import grouped_gemma_rmsnorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _reference_grouped_gemma_rmsnorm(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    eps: float,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Eager reference, mirrors GroupedGemmaRMSNorm.forward in hyperconnection.py.

    With compute_dtype=float64 this serves as the near-exact reference for
    precision assertions.
    """
    x_float = x.to(compute_dtype)
    hidden = x_float.shape[-1]
    x_grouped = x_float.reshape(*x_float.shape[:-1], hidden // group_size, group_size)
    variance = x_grouped.pow(2).mean(dim=-1, keepdim=True)
    x_norm = (x_grouped * torch.rsqrt(variance + eps)).flatten(-2)
    return x_norm * (1.0 + weight.to(compute_dtype))


# Tolerances are at the output-dtype quantization floor, measured against the
# fp64 reference on 4xB300 (sm103): bf16 max rel err 3.9e-3 (1 ulp), fp16
# 4.9e-4 (0.5 ulp). The kernel computes in fp32 like the eager reference.
_TOLERANCES = {
    torch.bfloat16: dict(rtol=5e-3, atol=5e-3),
    torch.float16: dict(rtol=1e-3, atol=1e-3),
}


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize(
    "num_tokens,hidden_size,group_size",
    [
        (1, 10240, 2560),  # production shape (HC 4 x 2560)
        (7, 10240, 2560),
        (128, 10240, 2560),
        (33, 1024, 512),
        (5, 512, 512),  # single group == plain gemma rmsnorm
        (1024, 2048, 1024),
    ],
)
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_grouped_gemma_rmsnorm_correctness(
    dtype, num_tokens, hidden_size, group_size, eps
):
    torch.manual_seed(0)
    x = torch.randn(num_tokens, hidden_size, dtype=dtype, device="cuda")
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda") * 0.2

    out = grouped_gemma_rmsnorm(x, weight, group_size, eps)
    expected = _reference_grouped_gemma_rmsnorm(
        x, weight, group_size, eps, compute_dtype=torch.float64
    ).to(dtype)

    torch.testing.assert_close(out, expected, **_TOLERANCES[dtype])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_grouped_gemma_rmsnorm_3d_input(dtype):
    torch.manual_seed(0)
    x = torch.randn(4, 8, 10240, dtype=dtype, device="cuda")
    weight = torch.randn(10240, dtype=dtype, device="cuda") * 0.2

    out = grouped_gemma_rmsnorm(x, weight, 2560, 1e-6)
    expected = _reference_grouped_gemma_rmsnorm(
        x, weight, 2560, 1e-6, compute_dtype=torch.float64
    ).to(dtype)

    assert out.shape == x.shape
    torch.testing.assert_close(out, expected, **_TOLERANCES[dtype])


def test_grouped_gemma_rmsnorm_out_param():
    x = torch.randn(64, 10240, dtype=torch.bfloat16, device="cuda")
    weight = torch.randn(10240, dtype=torch.bfloat16, device="cuda") * 0.2
    out = torch.empty_like(x)

    result = grouped_gemma_rmsnorm(x, weight, 2560, 1e-6, out=out)
    expected = _reference_grouped_gemma_rmsnorm(
        x, weight, 2560, 1e-6, compute_dtype=torch.float64
    ).to(x.dtype)

    assert result.data_ptr() == out.data_ptr()
    torch.testing.assert_close(result, expected, **_TOLERANCES[x.dtype])


def test_grouped_gemma_rmsnorm_unsupported_dtype():
    x = torch.randn(4, 10240, dtype=torch.float32, device="cuda")
    weight = torch.zeros(10240, dtype=torch.float32, device="cuda")
    with pytest.raises(RuntimeError, match="dtype"):
        grouped_gemma_rmsnorm(x, weight, 2560, 1e-6)


def test_grouped_gemma_rmsnorm_bad_group_size():
    x = torch.randn(4, 10240, dtype=torch.bfloat16, device="cuda")
    weight = torch.zeros(10240, dtype=torch.bfloat16, device="cuda")
    with pytest.raises(RuntimeError, match="group_size"):
        grouped_gemma_rmsnorm(x, weight, 1000, 1e-6)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
