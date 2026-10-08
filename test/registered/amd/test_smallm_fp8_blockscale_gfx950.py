import os
import unittest
from unittest import mock

import torch

from sglang.kernels.ops.gemm import smallm_fp8_blockscale_gfx950 as B
from sglang.srt.layers.quantization import fp8_utils
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")

SHAPES = [(5120, 4096), (4608, 4096), (4096, 2048)]
MS = (1, 3, 4, 8, 9, 16, 17, 24, 32)


def _x(m, k):
    # per-group magnitudes differ so every 128-wide group gets its own scale
    g = torch.rand(m, k // 128, 1, device="cuda") * 8
    return (torch.randn(m, k // 128, 128, device="cuda") * g).view(m, k).bfloat16()


def _w(w):
    from aiter.ops.shuffle import shuffle_weight

    wq = shuffle_weight(w.to(torch.float8_e4m3fn), (16, 16))
    wq.is_shuffled = True
    return wq


def _quant(x):
    from sglang.srt.layers.quantization.fp8_utils import aiter_per1x128_quant as q

    return q(x, quant_dtype=torch.float8_e4m3fn, transpose_scale=False)


@unittest.skipUnless(
    torch.version.hip
    and fp8_utils._use_aiter_bpreshuffle_gfx95
    and torch.cuda.is_available()
    and torch.cuda.get_device_properties(0).gcnArchName.startswith("gfx950"),
    "gfx950 (MI35x) with AITER bpreshuffle only",
)
class TestSmallMFp8BlockscaleGfx950(CustomTestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_matches_aiter(self):
        from aiter import gemm_a8w8_blockscale_bpreshuffle as gemm

        from sglang.srt.layers.quantization.fp8_utils import (
            materialize_bpreshuffle_fp8_scale as materialize,
        )

        for n, k in SHAPES:
            wq = _w(torch.randn(n, k, device="cuda") * 0.5)
            ws = torch.rand(n // 128, k // 128, device="cuda") * 1e-2 + 1e-3
            for m in MS:
                x = _x(m, k)
                self.assertTrue(B.smallm_fp8_bs_supported(x, wq, ws), f"{n=} {m=}")
                xq, xs = _quant(x)
                ref = gemm(xq, wq, materialize(xs), ws, dtype=torch.bfloat16)
                out = B.smallm_fp8_bs_linear(x, wq, ws)
                torch.testing.assert_close(out, ref, atol=2e-2, rtol=2e-2)

    def test_quant_bit_exact(self):
        # one-hot weights, unit scales: out[m, n] = bf16(q[m, k] * xs[m, k // 128])
        for n, k in SHAPES:
            col = torch.arange(n, device="cuda") % k
            wq = _w(torch.nn.functional.one_hot(col, k).float())
            ws = torch.ones(n // 128, k // 128, device="cuda")
            for m in MS:
                x = _x(m, k)
                xq, xs = _quant(x)
                ref = (xq.float() * xs.repeat_interleave(128, 1))[:, col].bfloat16()
                out = B.smallm_fp8_bs_linear(x, wq, ws)
                self.assertTrue(torch.equal(out, ref), f"{n=} {m=}")

    def test_fallback(self):
        wq = _w(torch.zeros(4096, 2048, device="cuda"))
        ws = torch.ones(32, 16, device="cuda")
        x = _x(4, 2048)
        self.assertTrue(B.smallm_fp8_bs_supported(x, wq, ws))
        unaligned = torch.randn(4, 2052, device="cuda").bfloat16()[:, :2048]
        unshuffled = wq.view(torch.uint8).view(wq.dtype)
        for bad_x, bad_w in (
            (_x(33, 2048), wq),
            (x.half(), wq),
            (unaligned, wq),
            (x, unshuffled),
        ):
            self.assertFalse(B.smallm_fp8_bs_supported(bad_x, bad_w, ws))
        with mock.patch.dict(os.environ, {"SGLANG_ROCM_SMALLM_FP8_BS": "0"}):
            self.assertFalse(B.smallm_fp8_bs_supported(x, wq, ws))

    def test_graph_replay(self):
        wq = _w(torch.randn(5120, 4096, device="cuda") * 0.5)
        ws = torch.rand(40, 32, device="cuda") * 1e-2
        x = _x(4, 4096)
        eager = B.smallm_fp8_bs_linear(x, wq, ws)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = B.smallm_fp8_bs_linear(x, wq, ws)
        g.replay()
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(out, eager))


if __name__ == "__main__":
    unittest.main()
