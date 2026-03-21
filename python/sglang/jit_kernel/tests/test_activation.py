# Adapted from https://github.com/flashinfer-ai/flashinfer/blob/main/tests/test_activation.py

import flashinfer
import pytest
import sgl_kernel
import torch

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

_DTYPES = [torch.float16, torch.bfloat16]
_SHAPES = [
    (1, 1, 128),
    (2, 4, 256),
    (4, 8, 512),
    (8, 16, 2048),
    (1, 1, 4096),
    (16, 32, 11008),
]


@pytest.mark.parametrize("batch_size,seq_len,dim", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_silu_and_mul(dtype, batch_size, seq_len, dim):
    x = torch.randn(batch_size, seq_len, 2 * dim, dtype=dtype, device="cuda")
    ref = sgl_kernel.silu_and_mul(x)
    out = flashinfer.silu_and_mul(x)
    torch.testing.assert_close(ref, out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size,seq_len,dim", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_gelu_tanh_and_mul(dtype, batch_size, seq_len, dim):
    x = torch.randn(batch_size, seq_len, 2 * dim, dtype=dtype, device="cuda")
    ref = sgl_kernel.gelu_tanh_and_mul(x)
    out = flashinfer.gelu_tanh_and_mul(x)
    torch.testing.assert_close(ref, out, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize("batch_size,seq_len,dim", _SHAPES)
@pytest.mark.parametrize("dtype", _DTYPES)
def test_gelu_and_mul(dtype, batch_size, seq_len, dim):
    x = torch.randn(batch_size, seq_len, 2 * dim, dtype=dtype, device="cuda")
    ref = sgl_kernel.gelu_and_mul(x)
    out = flashinfer.gelu_and_mul(x)
    torch.testing.assert_close(ref, out, rtol=1e-3, atol=1e-3)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
