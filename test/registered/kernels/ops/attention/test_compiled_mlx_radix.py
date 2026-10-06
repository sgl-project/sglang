"""The pending-token tail must match committed KV without modifying the pool."""

import importlib.util
import unittest

import torch

from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.test_utils import CustomTestCase

register_mps_ci(est_time=30, suite="stage-a-unit-test-mps")


@unittest.skipUnless(
    torch.backends.mps.is_available() and importlib.util.find_spec("mlx") is not None,
    "Requires Torch MPS and MLX",
)
class TestCompiledRadix(CustomTestCase):
    def test_direct_mlx_inventory_and_shape_contract(self):
        import mlx.core as mx

        from sglang.kernels.ops.attention.compiled_mlx_radix import radix_decode
        from sglang.kernels.selector import select_kernel
        from sglang.kernels.spec import KernelBackend, PlatformInfo

        spec = select_kernel("attention.compiled_mlx_radix_decode")
        self.assertEqual(spec.backend, KernelBackend.MLX)
        self.assertTrue(spec.is_available(PlatformInfo(device_type="mps")))
        self.assertFalse(spec.is_available(PlatformInfo(device_type="cuda")))
        q = mx.zeros((1, 4, 64))
        k = mx.zeros((1, 2, 64))
        pool = mx.zeros((16, 2, 64))
        table = mx.zeros((2, 8), dtype=mx.int32)
        rows, lengths = mx.array([1]), mx.array([2])
        with self.assertRaisesRegex(ValueError, "Pending K/V"):
            radix_decode(q, k, k, pool, pool, table, rows, lengths, 0.125, tails=(q, q))
        with self.assertRaisesRegex(ValueError, "dtype"):
            radix_decode(
                q, k, k, pool, pool, table.astype(mx.float32), rows, lengths, 0.125
            )

    def test_read_only_tail_matches_committed_reference(self):
        from sglang.kernels.ops.attention.compiled_mlx_radix import radix_decode
        from sglang.kernels.ops.attention.mlx_radix_export import (
            radix_decode as reference,
        )
        from sglang.srt.utils.tensor_bridge import mlx_call_multi

        torch.manual_seed(17)
        for dtype, dim in (
            (torch.float32, 64),
            (torch.bfloat16, 128),
            (torch.float16, 256),
        ):
            with self.subTest(dtype=dtype), torch.no_grad():
                q = torch.randn(3, 4, dim, device="mps", dtype=dtype)
                k = torch.randn(3, 2, dim, device="mps", dtype=dtype)
                v, tk, tv = (torch.randn_like(k) for _ in range(3))
                kp = torch.randn(256, 2, dim, device="mps", dtype=dtype)
                vp = torch.randn_like(kp)
                table = torch.randperm(256, device="mps").reshape(4, 64).int()
                requests = torch.tensor([2, 0, 3], device="mps")
                lengths = torch.tensor([2, 17, 63], device="mps")
                args = (q, k, v, kp, vp, table, requests, lengths)
                scale = dim**-0.5
                before = kp.clone(), vp.clone()
                plain = mlx_call_multi(
                    lambda *x, scale=scale: (radix_decode(*x, scale),), *args
                )[0]
                tolerance = 1e-5 if dtype == torch.float32 else 0.006
                torch.testing.assert_close(
                    plain, reference(*args, scale), atol=tolerance, rtol=tolerance
                )
                actual = mlx_call_multi(
                    lambda *x, scale=scale: (
                        radix_decode(*x[:-2], scale, tails=x[-2:]),
                    ),
                    *args,
                    tk,
                    tv,
                )[0]
                committed_k, committed_v = kp.clone(), vp.clone()
                for index in range(3):
                    slot = table[requests[index], lengths[index] - 2]
                    committed_k[slot], committed_v[slot] = tk[index], tv[index]
                expected = reference(
                    q, k, v, committed_k, committed_v, table, requests, lengths, scale
                )
                torch.testing.assert_close(
                    actual, expected, atol=tolerance, rtol=tolerance
                )
                torch.testing.assert_close(kp, before[0], atol=0, rtol=0)
                torch.testing.assert_close(vp, before[1], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
