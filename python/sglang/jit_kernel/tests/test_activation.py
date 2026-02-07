import pytest
import torch
import torch.nn.functional as F
from sglang.jit_kernel.activation import (
    silu_and_mul,
    gelu_and_mul,
    gelu_tanh_and_mul,
    gelu_quick,
)

@pytest.mark.parametrize("dim", [128, 512, 1024])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_silu_and_mul(dim, batch_size, dtype):
    x = torch.randn(batch_size, 2 * dim, device="cuda", dtype=dtype)
    y_ref = F.silu(x[..., :dim]) * x[..., dim:]
    y = silu_and_mul(x)
    atol = 1e-3 if dtype == torch.float16 else 2e-2
    rtol = 1e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(y, y_ref, rtol=rtol, atol=atol)

@pytest.mark.parametrize("dim", [128, 512, 1024])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gelu_and_mul(dim, batch_size, dtype):
    x = torch.randn(batch_size, 2 * dim, device="cuda", dtype=dtype)
    y_ref = F.gelu(x[..., :dim], approximate="none") * x[..., dim:]
    y = gelu_and_mul(x)
    atol = 1e-3 if dtype == torch.float16 else 2e-2
    rtol = 1e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(y, y_ref, rtol=rtol, atol=atol)

@pytest.mark.parametrize("dim", [128, 512, 1024])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gelu_tanh_and_mul(dim, batch_size, dtype):
    x = torch.randn(batch_size, 2 * dim, device="cuda", dtype=dtype)
    y_ref = F.gelu(x[..., :dim], approximate="tanh") * x[..., dim:]
    y = gelu_tanh_and_mul(x)
    atol = 1e-3 if dtype == torch.float16 else 2e-2
    rtol = 1e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(y, y_ref, rtol=rtol, atol=atol)

@pytest.mark.parametrize("dim", [128, 512, 1024])
@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_gelu_quick(dim, batch_size, dtype):
    x = torch.randn(batch_size, dim, device="cuda", dtype=dtype)
    y_ref = x * torch.sigmoid(1.702 * x)
    y = gelu_quick(x)
    atol = 1e-3 if dtype == torch.float16 else 2e-2
    rtol = 1e-3 if dtype == torch.float16 else 2e-2
    torch.testing.assert_close(y, y_ref, rtol=rtol, atol=atol)

if __name__ == "__main__":
    pytest.main([__file__])
