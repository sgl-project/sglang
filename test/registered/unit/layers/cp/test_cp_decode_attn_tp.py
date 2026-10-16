import unittest

import torch

from sglang.srt.layers.cp.cp_decode_attn_tp import CpDecodeAttnTpContext
from sglang.srt.layers.linear import ColumnParallelLinear, RowParallelLinear
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")

N, K = 256, 256
# gfx950 block-fp8-as-MXFP8 linears carry these load-time copies next to the weight.
SCALE_COPIES = (
    "weight_scale_inv",
    "weight_scale_inv_mx",
    "weight_scale_mx_e8m0",
    "weight_bf16",
)


def _context(rank=1, size=2):
    ctx = CpDecodeAttnTpContext.__new__(CpDecodeAttnTpContext)
    ctx.decode_tp_rank, ctx.decode_tp_size = rank, size
    ctx._slice_cache = {}
    return ctx


def _linear(cls):
    linear = cls.__new__(cls)
    torch.nn.Module.__init__(linear)
    linear.weight = torch.nn.Parameter(
        torch.arange(N * K, dtype=torch.float32).view(N, K), requires_grad=False
    )
    linear.weight_scale_inv = torch.arange(4.0).view(N // 128, K // 128)
    linear.weight_scale_inv_mx = torch.arange(N * K // 32.0).view(N, K // 32)
    linear.weight_scale_mx_e8m0 = torch.arange(N * K // 1024.0).view(N // 32, K // 32)
    linear.weight_bf16 = torch.arange(N * K, dtype=torch.float32).view(N, K)
    linear.weight_scale = None
    return linear


def _data(t):
    return t.data if isinstance(t, torch.nn.Parameter) else t


class TestCpDecodeAttnTpScaleCopies(CustomTestCase):
    def test_scale_copies_are_narrowed_with_the_weight(self):
        """--enable-cp-decode-attn-tp on gfx950 narrowed the weight but left its
        MXFP8 scale copies full-size, so the decode attention GEMM read wrong scales."""
        for cls, dim in ((ColumnParallelLinear, 0), (RowParallelLinear, 1)):
            with self.subTest(cls=cls.__name__):
                ctx, linear = _context(), _linear(cls)
                originals = {n: getattr(linear, n) for n in ("weight",) + SCALE_COPIES}
                full = {n: _data(t).clone() for n, t in originals.items()}
                attrs = ctx._get_linear_attrs(linear)
                self.assertEqual({n for _, n, _ in attrs}, set(originals))
                for obj, name, d in attrs:
                    self.assertEqual(d, dim)
                    ctx._activate(obj, name, d)
                for name, t in full.items():
                    expected = t.narrow(dim, t.shape[dim] // 2, t.shape[dim] // 2)
                    self.assertTrue(
                        torch.equal(_data(getattr(linear, name)), expected), name
                    )
                for obj, name, _ in reversed(attrs):
                    ctx._restore(obj, name)
                for name, orig in originals.items():
                    self.assertIs(getattr(linear, name), orig)
                    self.assertTrue(torch.equal(_data(orig), full[name]), name)

    def test_mxfp8_native_weight_is_rejected(self):
        linear = _linear(ColumnParallelLinear)
        linear.mxfp8_native_ready = True
        with self.assertRaises(NotImplementedError):
            _context()._get_linear_attrs(linear)


if __name__ == "__main__":
    unittest.main()
