# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/4e8eb1879f9c3ba6d75511e5893183bf8f289a62/tests/test_activation.py

import pytest
import sgl_kernel
import torch


@pytest.mark.parametrize("dim", [128, 256, 512, 2048, 4096, 11008, 16384])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [1, 2, 4, 8, 16, 32, 64, 128, 512])
def test_fused_silu_mul(dim, batch_size, seq_len):
    x = torch.randn(batch_size, seq_len, 2 * dim).to(0).to(torch.float16)
    y_ref = x[..., dim:] * torch.nn.functional.silu(x[..., :dim])
    y = sgl_kernel.silu_and_mul(x)
    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dim", [128, 256, 512, 2048, 4096, 11008, 16384])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [1, 2, 4, 8, 16, 32, 64, 128, 512])
def test_fused_gelu_tanh_mul(dim, batch_size, seq_len):
    x = torch.randn(batch_size, seq_len, 2 * dim).to(0).to(torch.float16)
    y_ref = x[..., dim:] * torch.nn.functional.gelu(x[..., :dim], approximate="tanh")
    y = sgl_kernel.gelu_tanh_and_mul(x)
    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("dim", [128, 256, 512, 2048, 4096, 11008, 16384])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8, 16])
@pytest.mark.parametrize("seq_len", [1, 2, 4, 8, 16, 32, 64, 128, 512])
def test_fused_gelu_mul(dim, batch_size, seq_len):
    x = torch.randn(batch_size, seq_len, 2 * dim).to(0).to(torch.float16)
    y_ref = x[..., dim:] * torch.nn.functional.gelu(x[..., :dim], approximate="none")
    y = sgl_kernel.gelu_and_mul(x)
    torch.testing.assert_close(y_ref, y, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__])
