import pytest
import torch

from sglang.srt.models.qwen4_exp import Qwen4ExpPLEGroupedNorm
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="1-gpu-large")
register_cuda_ci(est_time=30, stage="base-b-kernel-unit", runner_config="4-gpu-b200")


def _reference_ple_norm(
    x: torch.Tensor,
    weight: torch.Tensor,
    group_size: int,
    eps: float,
    compute_dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Eager reference, mirrors Qwen4ExpPLEGroupedNorm.forward's fallback path.

    With compute_dtype=float64 this serves as the near-exact reference for
    precision assertions.
    """
    x_float = x.to(compute_dtype)
    group_shape = x_float.shape[:-1] + (-1, group_size)
    variance = x_float.reshape(group_shape).pow(2).mean(dim=-1, keepdim=True)
    variance = variance.expand(group_shape).reshape_as(x_float)
    x_norm = x_float * torch.rsqrt(variance + eps)
    return x_norm * (weight.to(compute_dtype) + 1.0)


# Tolerances are at the output-dtype quantization floor. The fused path is the
# grouped_gemma_rmsnorm JIT kernel, which computes in fp32 like the eager
# fallback. Measured on B300 (sm103) vs the fp64 reference at the largest
# shape (8192 x 10240): the fp32 *eager* chain itself already sits 1 bf16 ulp
# (rel 7.8e-3) off the fp64 reference on ~147/83.8M elements (the jit kernel:
# ~143), so 5e-3 (half a bf16 ulp headroom) is unattainable at this shape and
# bf16 is set to 2 ulp. fp16 passes at 0.5 ulp.
_TOLERANCES = {
    torch.bfloat16: dict(rtol=8e-3, atol=8e-3),
    torch.float16: dict(rtol=1e-3, atol=1e-3),
}

# Real Qwen4-Exp PLE shape: HC 4 x hidden 2560, grouped per hidden_size.
_PLE_HIDDEN_SIZE = 10240
_PLE_GROUP_SIZE = 2560


def _make_norm(hidden_size, group_size, eps, dtype, seed=0):
    torch.manual_seed(seed)
    norm = Qwen4ExpPLEGroupedNorm(hidden_size, eps=eps, group_size=group_size)
    norm = norm.to(device="cuda", dtype=dtype)
    with torch.no_grad():
        norm.weight.copy_(torch.randn(hidden_size, device="cuda", dtype=dtype) * 0.2)
    return norm


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("num_tokens", [1, 7, 128, 8192])
@pytest.mark.parametrize("eps", [1e-6, 1e-5])
def test_qwen4_ple_norm_correctness(dtype, num_tokens, eps):
    torch.manual_seed(0)
    norm = _make_norm(_PLE_HIDDEN_SIZE, _PLE_GROUP_SIZE, eps, dtype)
    x = torch.randn(num_tokens, _PLE_HIDDEN_SIZE, dtype=dtype, device="cuda")

    out = norm(x)
    expected = _reference_ple_norm(
        x, norm.weight, _PLE_GROUP_SIZE, eps, compute_dtype=torch.float64
    ).to(dtype)

    assert norm._jit_group_size == _PLE_GROUP_SIZE  # fused path exercised
    torch.testing.assert_close(out, expected, **_TOLERANCES[dtype])


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
def test_qwen4_ple_norm_3d_input(dtype):
    # _apply_ple_norm flattens (token, hc, hidden) -> (token, hc * hidden), but
    # the module itself must handle arbitrary leading dims identically.
    torch.manual_seed(0)
    norm = _make_norm(_PLE_HIDDEN_SIZE, _PLE_GROUP_SIZE, 1e-6, dtype)
    x = torch.randn(4, 8, _PLE_HIDDEN_SIZE, dtype=dtype, device="cuda")

    out = norm(x)
    expected = _reference_ple_norm(
        x, norm.weight, _PLE_GROUP_SIZE, 1e-6, compute_dtype=torch.float64
    ).to(dtype)

    assert out.shape == x.shape
    torch.testing.assert_close(out, expected, **_TOLERANCES[dtype])


def test_qwen4_ple_norm_eager_fallback():
    # group_size not a multiple of 512 -> JIT kernel not eligible -> eager path.
    torch.manual_seed(0)
    dtype = torch.bfloat16
    norm = _make_norm(1000, 500, 1e-6, dtype)
    assert norm._jit_group_size is None
    x = torch.randn(33, 1000, dtype=dtype, device="cuda")

    out = norm(x)
    expected = _reference_ple_norm(
        x, norm.weight, 500, 1e-6, compute_dtype=torch.float64
    ).to(dtype)

    torch.testing.assert_close(out, expected, **_TOLERANCES[dtype])


def test_qwen4_ple_norm_non_cuda_fallback():
    torch.manual_seed(0)
    norm = Qwen4ExpPLEGroupedNorm(_PLE_HIDDEN_SIZE, eps=1e-6, group_size=_PLE_GROUP_SIZE)
    x = torch.randn(4, _PLE_HIDDEN_SIZE, dtype=torch.bfloat16)

    out = norm(x)
    expected = _reference_ple_norm(
        x, norm.weight, _PLE_GROUP_SIZE, 1e-6, compute_dtype=torch.float64
    ).to(torch.bfloat16)

    torch.testing.assert_close(out, expected, **_TOLERANCES[torch.bfloat16])


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v", "-s"]))
