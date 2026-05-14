import itertools
import unittest

import torch
from utils import precision

from sglang.test.test_utils import CustomTestCase

torch.manual_seed(42)


def _torch_hadamard_transform(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Reference pure-torch FWHT implementation."""
    n = x.size(-1)
    leading = x.shape[:-1]
    out = x.reshape(-1, n).float().clone()
    h = 1
    while h < n:
        out = out.view(-1, n // (2 * h), 2, h)
        a = out[:, :, 0, :]
        b = out[:, :, 1, :]
        out = torch.stack((a + b, a - b), dim=2).view(-1, n)
        h *= 2
    return (out.view(*leading, n) * scale).to(x.dtype)


class TestHadamardTransform(CustomTestCase):
    # Last dim must be power of 2
    N = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    BATCH = [1, 3, 7, 16]
    DTYPE = [torch.float32, torch.bfloat16, torch.float16]

    def _test_hadamard(self, batch, n, dtype):
        x = torch.randn([batch, n], dtype=dtype)
        scale = n**-0.5

        out = torch.ops.sgl_kernel.fast_hadamard_transform_cpu(x, scale)
        ref = _torch_hadamard_transform(x, scale)

        atol = rtol = precision[dtype]
        torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_basic(self):
        """Test across various batch sizes, dimensions, and dtypes."""
        for params in itertools.product(self.BATCH, self.N, self.DTYPE):
            with self.subTest(batch=params[0], n=params[1], dtype=params[2]):
                self._test_hadamard(*params)

    def test_3d_input(self):
        """Test with 3-D input tensor (e.g. [B, S, D])."""
        for n in [32, 64, 128, 256]:
            for dtype in [torch.bfloat16, torch.float32]:
                x = torch.randn([2, 8, n], dtype=dtype)
                scale = n**-0.5
                out = torch.ops.sgl_kernel.fast_hadamard_transform_cpu(x, scale)
                ref = _torch_hadamard_transform(x, scale)
                atol = rtol = precision[dtype]
                torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_4d_input(self):
        """Test with 4-D input tensor (e.g. [B, H, S, D])."""
        for n in [32, 128]:
            x = torch.randn([2, 4, 8, n], dtype=torch.bfloat16)
            scale = n**-0.5
            out = torch.ops.sgl_kernel.fast_hadamard_transform_cpu(x, scale)
            ref = _torch_hadamard_transform(x, scale)
            atol = rtol = precision[torch.bfloat16]
            torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_scale_one(self):
        """Test with scale=1.0 (no scaling)."""
        x = torch.randn([4, 64], dtype=torch.float32)
        out = torch.ops.sgl_kernel.fast_hadamard_transform_cpu(x, 1.0)
        ref = _torch_hadamard_transform(x, 1.0)
        torch.testing.assert_close(ref, out, atol=1e-5, rtol=1e-5)

    def test_large_dim(self):
        """Test with large last dimension typical of LLM hidden sizes."""
        for n in [2048, 4096]:
            x = torch.randn([2, n], dtype=torch.bfloat16)
            scale = n**-0.5
            out = torch.ops.sgl_kernel.fast_hadamard_transform_cpu(x, scale)
            ref = _torch_hadamard_transform(x, scale)
            atol = rtol = precision[torch.bfloat16]
            torch.testing.assert_close(ref, out, atol=atol, rtol=rtol)

    def test_involution(self):
        """FWHT applied twice (with scale=1/n) should recover the original."""
        n = 128
        x = torch.randn([4, n], dtype=torch.float32)
        # H * H = n * I, so applying twice with scale 1/n gives identity.
        y = torch.ops.sgl_kernel.fast_hadamard_transform_cpu(x, 1.0)
        z = torch.ops.sgl_kernel.fast_hadamard_transform_cpu(y, 1.0 / n)
        torch.testing.assert_close(x, z, atol=1e-4, rtol=1e-4)


if __name__ == "__main__":
    unittest.main()
