import math

import pytest
import torch
import torch.nn.functional as F
from scipy.linalg import hadamard

from sglang.srt.layers.attention.dsa.dsa_indexer import rotate_activation
from sglang.srt.utils import get_device


def hadamard_transform_ref(x, scale=1.0):
    """Reference impl for the general (power-of-2) hadamard_transform.

    Pads dim to the next power of 2, multiplies by the full H matrix
    via F.linear, then truncates back to the original dim.
    """
    x_shape = x.shape
    dim = x.shape[-1]
    x = x.reshape(-1, dim)
    log_dim = math.ceil(math.log2(dim)) if dim > 0 else 0
    dim_padded = 2**log_dim if dim > 0 else 1
    if dim != dim_padded:
        x = F.pad(x, (0, dim_padded - dim))
    H = torch.tensor(hadamard(dim_padded, dtype=float), dtype=x.dtype, device=x.device)
    out = F.linear(x, H)
    out = out * scale
    return out[..., :dim].reshape(*x_shape)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "dim",
    [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768],
)
def test_hadamard_transform(dim, dtype):
    device = get_device()

    # Tolerances from sgl-kernel/tests/test_hadamard.py (old AOT test)
    if dtype == torch.float32:
        rtol, atol = 3e-4, 3e-3
    elif dtype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    else:  # float16
        rtol, atol = 3e-3, 5e-3

    torch.random.manual_seed(0)
    batch_size = 15

    x = torch.randn(batch_size, dim, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(dim)

    out = rotate_activation(x)
    # Compute reference in float32 from a detached copy to avoid precision loss
    out_ref = hadamard_transform_ref(x.detach().clone().float(), scale=scale)

    torch.testing.assert_close(out.float(), out_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_hadamard_transform_3d_input(dtype):
    device = get_device()

    if dtype == torch.bfloat16:
        rtol, atol = 1e-2, 5e-2
    else:
        rtol, atol = 3e-3, 5e-3

    torch.random.manual_seed(0)

    x = torch.randn(4, 8, 256, device=device, dtype=dtype)
    scale = 1.0 / math.sqrt(256)

    out = rotate_activation(x)
    assert out.shape == x.shape

    out_ref = hadamard_transform_ref(x.detach().clone().float(), scale=scale)
    torch.testing.assert_close(out.float(), out_ref, rtol=rtol, atol=atol)
