"""Paged reads and split reductions must preserve the uncommitted-token boundary."""

import importlib.util
import tempfile
import unittest
from itertools import product
from pathlib import Path
from unittest.mock import patch

import torch

from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.test_utils import CustomTestCase

register_mps_ci(est_time=45, suite="stage-a-unit-test-mps")


@unittest.skipUnless(
    torch.backends.mps.is_available() and importlib.util.find_spec("mlx") is not None,
    "Requires Torch MPS and MLX",
)
class TestCompiledRadix(CustomTestCase):
    def test_aot_graph_handles_strides_streams_and_missing_extension(self):
        import mlx.core as mx
        from sgl_kernel import metal

        from sglang.kernels.ops.attention.mlx.radix_attention import radix_decode

        with mx.stream(mx.new_stream(mx.gpu)):
            q = mx.ones((3, 8, 128))[:, ::2, ::2]
            k = mx.ones((3, 4, 128))[:, ::2, ::2]
            pool = mx.ones((2, 120, 64)).transpose(1, 0, 2)
            table = mx.arange(120, dtype=mx.int32).reshape(3, 40)[:, ::2]
            rows = mx.array([2, 99, 0, 99, 1, 99], dtype=mx.int64)[::2]
            lengths = mx.array([2, 0, 7, 0, 17, 0], dtype=mx.int64)[::2]
            args = q, k, k * 3, pool, pool, table, rows, lengths
            tails = k, k * 5
            with patch.object(
                mx.fast, "metal_kernel", side_effect=AssertionError("Unexpected JIT")
            ):
                result = radix_decode(*args, 0.125, tails=tails)
                with tempfile.TemporaryDirectory() as directory:
                    path = Path(directory) / "radix.dot"
                    mx.export_to_dot(str(path), result)
                    graph = path.read_text()
                    self.assertIn("AotRadixAttention", graph)
                    self.assertIn("AotRadixReduce", graph)
                run = mx.compile(lambda *x: radix_decode(*x[:-2], 0.125, tails=x[-2:]))
                pending = [run(*args, *tails), run(*args, *tails)]
                mx.async_eval(pending)
                expected = (1 + 6 / lengths.astype(mx.float32))[:, None, None]
                for output in [result, *pending]:
                    self.assertTrue(mx.allclose(output, expected, atol=1e-5).item())
                self.assertTrue(mx.all(pool == 1).item())
            with patch.object(metal, "_metal", None):
                with self.assertRaisesRegex(ImportError, "Metal build"):
                    radix_decode(*args, 0.125, tails=tails)

    def test_unused_pages_and_empty_partitions_never_poison_valid_tokens(self):
        import mlx.core as mx

        from sglang.kernels.ops.attention.mlx.radix_attention import radix_decode

        q = mx.ones((2, 4, 64))
        k = mx.ones((2, 2, 64))
        v = mx.full((2, 2, 64), 3.0)
        table = mx.full((2, 32), -123, dtype=mx.int32)
        requests = mx.array([0, 1])
        for page_size, tail in product((1, 16, 32, 64), (False, True)):
            pool = mx.full((page_size, 2, 64), float("nan"))
            with self.subTest(page_size=page_size, tail=tail):
                run = mx.compile(
                    lambda rows, lengths: radix_decode(
                        q,
                        k,
                        v,
                        pool,
                        pool,
                        table,
                        rows,
                        lengths,
                        0.125,
                        tails=(k, v) if tail else None,
                        page_size=page_size,
                    )
                )
                length = 2 if tail else 1
                self.assertTrue(
                    mx.all(run(requests, mx.array([length] * 2)) == 3).item()
                )
                for rows, lengths in (
                    (requests, mx.array([33, 0])),
                    (requests, mx.array([length + 1] * 2)),
                    (mx.array([-1, 2]), mx.array([length] * 2)),
                ):
                    self.assertTrue(mx.all(mx.isnan(run(rows, lengths))).item())
        self.assertTrue(mx.all(mx.isnan(pool)).item())
        self.assertTrue(mx.all(table == -123).item())

    def test_direct_mlx_inventory_and_shape_contract(self):
        import mlx.core as mx

        from sglang.kernels.ops.attention.mlx.radix_attention import radix_decode
        from sglang.kernels.selector import select_kernel
        from sglang.kernels.spec import KernelBackend, PlatformInfo

        q = mx.zeros((1, 4, 64))
        k = mx.zeros((1, 2, 64))
        pool = mx.zeros((16, 2, 64))
        table = mx.zeros((2, 8), dtype=mx.int32)
        rows, lengths = mx.array([1]), mx.array([2])
        spec = select_kernel("attention.mlx_radix_decode")
        self.assertEqual(spec.backend, KernelBackend.AOT)
        self.assertTrue(spec.is_available(PlatformInfo(device_type="mps")))
        self.assertFalse(spec.is_available(PlatformInfo(device_type="cuda")))
        with self.assertRaisesRegex(ValueError, "Pending K/V"):
            radix_decode(q, k, k, pool, pool, table, rows, lengths, 0.125, tails=(q, q))
        with self.assertRaisesRegex(ValueError, "dtype"):
            radix_decode(
                q, k, k, pool, pool, table.astype(mx.float32), rows, lengths, 0.125
            )
        with self.assertRaisesRegex(ValueError, "geometry"):
            radix_decode(
                q[:0], k[:0], k[:0], pool, pool, table, rows[:0], lengths[:0], 0.125
            )
        for page_size in (0, -16, 2, 32):
            with self.subTest(page_size=page_size):
                with self.assertRaisesRegex(ValueError, "page_size"):
                    radix_decode(
                        q,
                        k,
                        k,
                        pool,
                        pool,
                        table,
                        rows,
                        lengths,
                        0.125,
                        page_size=page_size,
                    )
        for base in (1, 32):
            result = radix_decode(
                q,
                k,
                k,
                pool,
                pool,
                mx.full_like(table, base),
                rows,
                lengths,
                0.125,
                page_size=16,
            )
            self.assertTrue(mx.all(mx.isnan(result)).item())

    def test_read_only_partitions_and_tail_match_committed_reference(self):
        import mlx.core as mx

        from sglang.kernels.ops.attention.mlx.radix_attention import radix_decode
        from sglang.kernels.ops.attention.mlx.radix_attention_export import (
            radix_decode as reference,
        )
        from sglang.srt.utils.tensor_bridge import mlx_call_multi

        torch.manual_seed(17)
        for dtype, dim, heads, kv_heads, sequence_lengths, page_size in (
            (torch.float32, 64, 4, 4, [2, 17, 63], 1),
            (torch.bfloat16, 128, 4, 2, [2, 513, 8193], 1),
            (torch.float16, 256, 4, 1, [2, 4095, 8193], 1),
            (torch.bfloat16, 128, 16, 8, [2, 3, 7, 31, 63, 127, 255, 513], 1),
            (torch.float32, 64, 16, 16, [2, 3, 7, 31, 63, 127, 255, 513], 1),
            (torch.bfloat16, 128, 16, 8, [2, 7, 31, 63] * 4, 1),
            (torch.float16, 256, 16, 4, [2, 7, 31, 63] * 4, 1),
            (torch.bfloat16, 128, 6, 2, [2, 17, 63], 1),
            (torch.bfloat16, 128, 16, 8, [2, 15, 16, 17, 31, 32, 33, 8193], 16),
            (torch.float32, 64, 4, 4, [2, 31, 32, 33, 63, 64, 65], 32),
            (torch.float16, 256, 6, 2, [2, 63, 64, 65, 127, 128, 129], 64),
        ):
            with (
                self.subTest(
                    dtype=dtype,
                    dim=dim,
                    heads=heads,
                    kv_heads=kv_heads,
                    batch=len(sequence_lengths),
                    page_size=page_size,
                ),
                torch.no_grad(),
            ):
                batch = len(sequence_lengths)
                width = (max(sequence_lengths) + page_size) // page_size * page_size
                q = torch.randn(batch, heads, dim, device="mps", dtype=dtype)
                k = torch.randn(batch, kv_heads, dim, device="mps", dtype=dtype)
                v, tk, tv = (torch.randn_like(k) for _ in range(3))
                slots = (batch + 1) * width + page_size
                kp = torch.randn(slots, kv_heads, dim, device="mps", dtype=dtype)
                vp = torch.randn_like(kp)
                table = (
                    (
                        (
                            torch.randperm(slots // page_size - 1, device="mps")[
                                :, None
                            ]
                            + 1
                        )
                        * page_size
                        + torch.arange(page_size, device="mps")
                    )
                    .reshape(batch + 1, width)
                    .int()
                )
                requests = torch.tensor([batch, *range(batch - 1)], device="mps")
                lengths = torch.tensor(sequence_lengths, device="mps")
                args = (q, k, v, kp, vp, table, requests, lengths)
                scale = dim**-0.5
                before = kp.clone(), vp.clone()
                tolerance = 1e-5 if dtype == torch.float32 else 0.006
                committed_k, committed_v = kp.clone(), vp.clone()
                for index in range(batch):
                    slot = table[requests[index], lengths[index] - 2]
                    committed_k[slot], committed_v[slot] = tk[index], tv[index]
                expected = reference(
                    q,
                    k,
                    v,
                    committed_k,
                    committed_v,
                    table,
                    requests,
                    lengths,
                    scale,
                    page_size=page_size,
                )
                plain_expected = reference(*args, scale, page_size=page_size)
                plain = mlx_call_multi(
                    mx.compile(
                        lambda *x: (radix_decode(*x, scale, page_size=page_size),)
                    ),
                    *args,
                )[0]
                torch.testing.assert_close(
                    plain, plain_expected, atol=tolerance, rtol=tolerance
                )
                actual = mlx_call_multi(
                    mx.compile(
                        lambda *x: (
                            radix_decode(
                                *x[:-2], scale, tails=x[-2:], page_size=page_size
                            ),
                        )
                    ),
                    *args,
                    tk,
                    tv,
                )[0]
                torch.testing.assert_close(
                    actual, expected, atol=tolerance, rtol=tolerance
                )
                torch.testing.assert_close(kp, before[0], atol=0, rtol=0)
                torch.testing.assert_close(vp, before[1], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
