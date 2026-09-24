"""gfx950 small-M W8A8 FP8 GEMM and fused output-side FP8 quant at the Qwen3.5 AttnFP8 TP4 shapes."""

import os
import types
import unittest
from unittest.mock import patch

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

_OFF = {"SGLANG_ROCM_SMALLM_FP8_PROJ": "0"}


@unittest.skipUnless(is_gfx95_supported(), "gfx950 (MI35x) only")
class TestSmallMFp8ProjGfx950(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("SGLANG_USE_AITER", "1")
        from aiter.ops.shuffle import shuffle_weight

        from sglang.kernels.ops.quantization.fp8_kernel import (
            per_token_group_quant_fp8,
        )
        from sglang.srt.layers.quantization import fp8_utils

        cls.fp8_utils, cls.quant = fp8_utils, staticmethod(per_token_group_quant_fp8)
        torch.manual_seed(0)
        # packed GDN in_proj, attention qkv_proj, GDN out_proj / attention o_proj; max M sent to the new kernel
        cls.shapes = []
        for n, k, max_m in ((5184, 4096, 28), (4608, 4096, 28), (4096, 2048, 36)):
            w = (torch.randn(n, k, device="cuda") * 0.05).to(torch.float8_e4m3fn)
            s = torch.rand(n, 1, device="cuda") * 0.01 + 1e-3
            cls.shapes.append((shuffle_weight(w, (16, 16)).t(), s, max_m))

    def linear(self, x, w, s):
        return self.fp8_utils.apply_fp8_linear(x, w, s, use_per_token_if_dynamic=True)

    def run_counted(self, *inputs):
        calls, real = [], self.fp8_utils.smallm_fp8_gemm
        with patch.object(
            self.fp8_utils,
            "smallm_fp8_gemm",
            lambda *a, **k: calls.append(1) or real(*a, **k),
        ):
            return [self.linear(*i) for i in inputs], len(calls)

    def test_gemm_matches_aiter_tuple_fallback_and_kill_switch(self):
        from aiter import gemm_a8w8_bpreshuffle

        for w, s, max_m in self.shapes:
            for m in (1, 4, 16, 24, 29, 36, 41):
                with self.subTest(n=w.shape[1], m=m):
                    x = torch.randn(m, w.shape[0], device="cuda", dtype=torch.bfloat16)
                    q, xs = self.quant(x, group_size=x.shape[1])
                    (got, got_tuple), n = self.run_counted((x, w, s), ((q, xs), w, s))
                    self.assertEqual(n, 2 if m <= max_m else 0)
                    self.assertTrue(torch.equal(got, got_tuple))
                    ref = gemm_a8w8_bpreshuffle(q, w.t(), xs, s, None, torch.bfloat16)
                    ref = ref.float()
                    ulp = torch.exp2(torch.floor(torch.log2(ref.abs() + 1e-30)) - 7)
                    err = (got.float() - ref).abs()
                    self.assertTrue((err <= ulp + 1e-5 * ref.abs().max()).all())
                    with patch.dict(os.environ, _OFF):
                        (off,), n = self.run_counted((x, w, s))
                    self.assertEqual(n, 0)
                    self.assertTrue(torch.equal(off.float(), ref))

    def test_producer_guard(self):
        from sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
            QuarkW8A8Fp8,
        )
        from sglang.srt.models import qwen3_5

        scheme = QuarkW8A8Fp8(
            {"qscheme": "per_channel"}, {"qscheme": "per_channel", "is_dynamic": True}
        )
        fwd = types.SimpleNamespace(sp_active=False)
        with patch.object(qwen3_5, "get_forward", return_value=fwd):
            fp8_in = qwen3_5._fp8_tuple_input
            self.assertTrue(fp8_in(types.SimpleNamespace(scheme=scheme), 40))
            self.assertFalse(fp8_in(types.SimpleNamespace(scheme=scheme), 41))
            self.assertFalse(fp8_in(torch.nn.Linear(8, 8), 4))  # BF16 MTP layer
            with patch.dict(os.environ, _OFF):
                self.assertFalse(fp8_in(types.SimpleNamespace(scheme=scheme), 4))

    def producers(self, t, trial=0):
        from sglang.kernels.ops.attention.fla.layernorm_gated import (
            _layer_norm_fwd,
            rms_norm_gated,
        )
        from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

        # GDN RMSNormGated over 16 heads x 128 per rank; attention gate over 8 heads x 256 per rank
        x = torch.randn(t * 16, 128, device="cuda", dtype=torch.bfloat16) * (1 + trial)
        z = torch.randn_like(x) * 3
        w = torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.5 + 1
        a = torch.randn(t, 2048, device="cuda", dtype=torch.bfloat16) * (1 + trial)
        g = torch.randn(t, 8, 512, device="cuda", dtype=torch.bfloat16)[:, :, 256:]
        fused = (
            lambda: _layer_norm_fwd(
                x, w, None, 1e-6, z=z, is_rms_norm=True, quant_heads=16
            )[0],
            lambda: fused_sigmoid_mul(a, g, quant=True),
        )
        norm = rms_norm_gated(x=x, weight=w, bias=None, z=z, eps=1e-6, is_rms_norm=True)
        unfused = (
            self.quant(norm.view(t, 2048), group_size=2048),
            self.quant(fused_sigmoid_mul(a.clone(), g), group_size=2048),
        )
        return fused, unfused

    def test_fused_quant_bit_exact(self):
        for t in (1, 4, 16, 33, 40):
            for trial in range(5):
                fused, unfused = self.producers(t, trial)
                for f, (rq, rs) in zip(fused, unfused):
                    q, s = f()
                    self.assertTrue(
                        torch.equal(q.view(torch.uint8), rq.view(torch.uint8))
                    )
                    self.assertTrue(torch.equal(s, rs))

    def test_graph_replay_matches_eager(self):
        (w_in, s_in, _), _, (w_out, s_out, _) = self.shapes
        x = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
        (norm_q, sig_q), _ = self.producers(4)

        def run():
            return (
                self.linear(x, w_in, s_in),
                self.linear(norm_q(), w_out, s_out),
                self.linear(sig_q(), w_out, s_out),
            )

        eager = [o.clone() for o in run()]
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            outs = run()
        g.replay()
        torch.cuda.synchronize()
        for e, o in zip(eager, outs):
            self.assertTrue(torch.equal(e, o))


if __name__ == "__main__":
    unittest.main()
