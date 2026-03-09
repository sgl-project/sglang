# Adapted from sgl-kernel/tests/test_norm.py

import pytest
import torch

# JIT rmsnorm: fp16/bf16 only; hidden_size must be a multiple of 256, > 256, and <=8192
RMSNORM_HIDDEN_SIZES = [512, 1024, 3072, 3584, 4096, 8192]

# JIT fused_add_rmsnorm: fp16/bf16 only; hidden_size % 8 == 0, <=8192
FUSED_ADD_RMSNORM_HIDDEN_SIZES = [1024, 3072, 3584, 4096, 8192]

BS_LIST = [1, 19, 99, 989]


def _jit_rmsnorm(input, weight, output, eps):
    from sglang.jit_kernel.norm import rmsnorm

    rmsnorm(input, weight, output=output, eps=eps)


def _fi_rmsnorm(input, weight, out, eps):
    from flashinfer.norm import rmsnorm

    rmsnorm(input, weight, out=out, eps=eps)


def _jit_fused_add_rmsnorm(input, residual, weight, eps):
    from sglang.jit_kernel.norm import fused_add_rmsnorm

    fused_add_rmsnorm(input, residual, weight, eps)


def _fi_fused_add_rmsnorm(input, residual, weight, eps):
    from flashinfer.norm import fused_add_rmsnorm

    fused_add_rmsnorm(input, residual, weight, eps=eps)


@pytest.mark.parametrize("batch_size", BS_LIST)
@pytest.mark.parametrize("hidden_size", RMSNORM_HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("specify_out", [True, False])
def test_rmsnorm_jit(batch_size, hidden_size, dtype, specify_out):
    eps = 1e-6
    x = torch.randn(batch_size, hidden_size, device="cuda", dtype=dtype)
    w = torch.randn(hidden_size, device="cuda", dtype=dtype)

    # flashinfer reference
    x_ref = x.clone()
    _fi_rmsnorm(x_ref, w, out=x_ref, eps=eps)

    if specify_out:
        y = torch.empty_like(x)
        _jit_rmsnorm(x, w, output=y, eps=eps)
    else:
        y = x.clone()
        _jit_rmsnorm(y, w, output=y, eps=eps)

    torch.testing.assert_close(y, x_ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("batch_size", BS_LIST)
@pytest.mark.parametrize("hidden_size", FUSED_ADD_RMSNORM_HIDDEN_SIZES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_fused_add_rmsnorm_jit(batch_size, hidden_size, dtype):
    eps = 1e-6
    x = torch.randn(batch_size, hidden_size, dtype=dtype, device="cuda")
    residual = torch.randn_like(x)
    weight = torch.randn(hidden_size, dtype=dtype, device="cuda")

    # flashinfer reference
    x_ref = x.clone()
    r_ref = residual.clone()
    _fi_fused_add_rmsnorm(x_ref, r_ref, weight, eps=eps)

    x_jit = x.clone()
    r_jit = residual.clone()
    _jit_fused_add_rmsnorm(x_jit, r_jit, weight, eps)

    torch.testing.assert_close(x_jit, x_ref, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(r_jit, r_ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    pytest.main([__file__])
