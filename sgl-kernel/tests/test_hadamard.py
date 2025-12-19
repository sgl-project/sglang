import math

import pytest
import torch
import torch.nn.functional as F
from sgl_kernel import hadamard_transform


def _fwht_pow2_in_float32(x: torch.Tensor) -> torch.Tensor:
    """Fast Walsh–Hadamard transform for last-dim power-of-2 length.

    Computes the unnormalized Hadamard transform in float32 for stability.
    """
    n = x.shape[-1]
    x = x.reshape(-1, n).to(torch.float32)
    h = 1
    # Iterative butterfly: O(n log n), no materialized Hadamard matrix.
    while h < n:
        x = x.view(-1, n // (2 * h), 2, h)
        a = x[:, :, 0, :]
        b = x[:, :, 1, :]
        y = torch.empty_like(x)
        y[:, :, 0, :] = a + b
        y[:, :, 1, :] = a - b
        x = y.view(-1, n)
        h *= 2
    return x


def hadamard_transform_ref(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    x: (..., dim)
    out: (..., dim)
    """
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim))
    dim_padded = 2**log_dim
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    out_f = _fwht_pow2_in_float32(x)
    out_f = out_f * scale
    out = out_f[..., :dim].to(dtype=x.dtype).reshape(*x_shape)
    return out


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "dim",
    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 137, 1024, 2048, 4096, 8192, 16384, 32768],
)
def test_fast_hadamard_transform(dim, dtype):
    device = "cuda"

    if dtype == torch.float32:
        rtol, atol = 3e-4, 3e-3
    elif dtype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    else:  # float16
        rtol, atol = 3e-3, 5e-3

    torch.random.manual_seed(0)
    batch_size = 15

    x = torch.randn(batch_size, dim, device=device, dtype=dtype)
    x_ref = x.detach().clone().to(torch.float32)
    x_pt = x.detach().clone()

    scale = 1 / math.sqrt(dim)

    out = hadamard_transform(x, scale=scale)
    out_ref = hadamard_transform_ref(x_ref, scale=scale)
    out_pt = hadamard_transform_ref(x_pt, scale=scale)

    torch.testing.assert_close(
        out_pt.float(),
        out_ref,
        rtol=rtol,
        atol=atol,
        msg="Reference implementations mismatch",
    )
    torch.testing.assert_close(
        out.float(),
        out_ref,
        rtol=rtol,
        atol=atol,
        msg="fast_hadamard_transform output mismatch",
    )


if __name__ == "__main__":
    pytest.main([__file__])
