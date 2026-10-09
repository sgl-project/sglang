"""gfx950 small-M W8A8 FP8 GEMM and fused output-side FP8 quant, Qwen3.5 AttnFP8 TP4 and TP2 shapes."""

import itertools
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


def _per_token_scheme():
    from sglang.srt.layers.quantization.quark.schemes.quark_w8a8_fp8 import (
        QuarkW8A8Fp8,
    )

    return QuarkW8A8Fp8(
        {"qscheme": "per_channel"}, {"qscheme": "per_channel", "is_dynamic": True}
    )


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
        # packed GDN in_proj, attention qkv_proj, GDN out_proj / attention o_proj at TP4 then TP2; max M sent to
        # the new kernel
        cls.shapes = []
        for n, k, max_m in (
            (5184, 4096, 28),
            (4608, 4096, 28),
            (4096, 2048, 36),
            (10304, 4096, 32),
            (8704, 4096, 16),
            (4096, 4096, 40),
        ):
            w = (torch.randn(n, k, device="cuda") * 0.05).to(torch.float8_e4m3fn)
            s = torch.rand(n, 1, device="cuda") * 0.01 + 1e-3
            cls.shapes.append((shuffle_weight(w, (16, 16)).t(), s, max_m))

    def linear(self, x, w, s):
        x, xs = x if isinstance(x, tuple) else (x, None)
        return self.fp8_utils.apply_fp8_linear(
            x, w, s, input_scale=xs, use_per_token_if_dynamic=True
        )

    def run_counted(self, *thunks):
        calls, real = [], self.fp8_utils.smallm_fp8_gemm
        with patch.object(
            self.fp8_utils,
            "smallm_fp8_gemm",
            lambda *a, **k: calls.append(1) or real(*a, **k),
        ):
            return [f() for f in thunks], len(calls)

    def cases(self):
        """(n, m, w, s, in range, x, q, xs, aiter ref) on both sides of each shape's max M."""
        from aiter import gemm_a8w8_bpreshuffle

        for w, s, max_m in self.shapes:
            for m in (1, 4, 8, 9, 16, 17, 24, 29, 33, 36, 41):
                x = torch.randn(m, w.shape[0], device="cuda", dtype=torch.bfloat16)
                q, xs = self.quant(x, group_size=x.shape[1])
                ref = gemm_a8w8_bpreshuffle(q, w.t(), xs, s, None, torch.bfloat16)
                yield w.shape[1], m, w, s, m <= max_m, x, q, xs, ref.float()

    def test_gemm_matches_aiter(self):
        for n, m, w, s, in_range, x, _, _, ref in self.cases():
            with self.subTest(n=n, m=m):
                (got,), calls = self.run_counted(lambda: self.linear(x, w, s))
                self.assertEqual(calls, int(in_range))
                ulp = torch.exp2(torch.floor(torch.log2(ref.abs() + 1e-30)) - 7)
                err = (got.float() - ref).abs()
                self.assertTrue((err <= ulp + 1e-5 * ref.abs().max()).all())

    def test_tuple_input_matches_bf16(self):
        for n, m, w, s, in_range, x, q, xs, _ in self.cases():
            with self.subTest(n=n, m=m):
                (got, got_tuple), calls = self.run_counted(
                    lambda: self.linear(x, w, s), lambda: self.linear((q, xs), w, s)
                )
                self.assertEqual(calls, 2 * in_range)
                self.assertTrue(torch.equal(got, got_tuple))

    def test_kill_switch(self):
        for n, m, w, s, _, x, _, _, ref in self.cases():
            with self.subTest(n=n, m=m), patch.dict(os.environ, _OFF):
                (off,), calls = self.run_counted(lambda: self.linear(x, w, s))
                self.assertEqual(calls, 0)
                self.assertTrue(torch.equal(off.float(), ref))

    def test_producer_guard(self):
        from sglang.srt.models import qwen3_5

        scheme = _per_token_scheme()
        fwd = types.SimpleNamespace(sp_active=False)
        # weight is [K, N] after Quark loading; TP4 o_proj (N, K) = (4096, 2048)
        o_proj = types.SimpleNamespace(scheme=scheme, weight=torch.empty(2048, 4096))
        other = types.SimpleNamespace(scheme=scheme, weight=torch.empty(2048, 1024))
        with patch.object(qwen3_5, "get_forward", return_value=fwd):
            fp8_in = qwen3_5._fp8_tuple_input
            self.assertTrue(fp8_in(o_proj, 36))
            self.assertFalse(fp8_in(o_proj, 37))
            self.assertFalse(fp8_in(other, 4))  # shape outside SHAPES
            self.assertFalse(fp8_in(torch.nn.Linear(8, 8), 4))  # BF16 MTP layer
            with patch.dict(os.environ, _OFF):
                self.assertFalse(fp8_in(o_proj, 4))

    def producers(self, t, trial=0, tp=4):
        from sglang.kernels.ops.attention.fla.layernorm_gated import (
            _layer_norm_fwd,
            rms_norm_gated,
        )
        from sglang.kernels.ops.elementwise.elementwise import fused_sigmoid_mul

        # per rank: GDN RMSNormGated over 64 / tp heads x 128; attention gate over 32 / tp heads x 256
        nv, na = 64 // tp, 32 // tp
        x = torch.randn(t * nv, 128, device="cuda", dtype=torch.bfloat16) * (1 + trial)
        z = torch.randn_like(x) * 3
        w = torch.randn(128, device="cuda", dtype=torch.bfloat16) * 0.5 + 1
        a = torch.randn(t, na * 256, device="cuda", dtype=torch.bfloat16) * (1 + trial)
        g = torch.randn(t, na, 512, device="cuda", dtype=torch.bfloat16)[:, :, 256:]
        fused = (
            lambda: _layer_norm_fwd(
                x, w, None, 1e-6, z=z, is_rms_norm=True, quant_heads=nv
            )[0],
            lambda: fused_sigmoid_mul(a, g, quant=True),
        )
        norm = rms_norm_gated(x=x, weight=w, bias=None, z=z, eps=1e-6, is_rms_norm=True)
        plain = (norm.view(t, nv * 128), fused_sigmoid_mul(a.clone(), g))
        unfused = tuple(self.quant(p, group_size=p.shape[1]) for p in plain)
        return fused, unfused, plain

    def test_fused_quant_bit_exact(self):
        for tp, t in itertools.product((4, 2), (1, 4, 16, 33, 40)):
            for trial in range(5):
                fused, unfused, _ = self.producers(t, trial, tp)
                for f, (rq, rs) in zip(fused, unfused):
                    q, s = f()
                    self.assertTrue(
                        torch.equal(q.view(torch.uint8), rq.view(torch.uint8))
                    )
                    self.assertTrue(torch.equal(s, rs))

    def test_quark_row_linear_takes_producer_tuple(self):
        from sglang.srt.layers.linear import RowParallelLinear
        from sglang.srt.layers.quantization.quark.quark import QuarkLinearMethod
        from sglang.srt.runtime_context import get_context, get_parallel
        from sglang.test.layer_ut_utils import init_single_process_dist

        class _QuarkPerToken:
            def get_quant_method(self, layer, prefix):
                layer.scheme = _per_token_scheme()
                return QuarkLinearMethod(self)

        init_single_process_dist()
        with (
            get_parallel().override(tp_size=1, tp_rank=0),
            get_context().override_server_args(),
        ):
            # TP4 GDN out_proj / attention o_proj, loaded the way Quark checkpoints are
            proj = RowParallelLinear(
                2048,
                4096,
                bias=False,
                reduce_results=False,
                params_dtype=torch.bfloat16,
                quant_config=_QuarkPerToken(),
                tp_rank=0,
                tp_size=1,
            ).cuda()
            proj.weight.data.copy_(
                (torch.randn(4096, 2048, device="cuda") * 0.05).to(torch.float8_e4m3fn)
            )
            proj.weight_scale.data.copy_(torch.rand(4096, device="cuda") * 0.01 + 1e-3)
            proj.quant_method.process_weights_after_loading(proj)
            for t in (1, 4, 36):
                fused, _, plain = self.producers(t)
                for f, p in zip(fused, plain):
                    with self.subTest(t=t):
                        (got, ref), n = self.run_counted(
                            lambda: proj(f())[0], lambda: proj(p)[0]
                        )
                        self.assertEqual(n, 2)
                        self.assertTrue(torch.equal(got, ref))
            with patch.dict(os.environ, _OFF), self.assertRaises(AssertionError):
                proj(fused[0]())

    def test_graph_replay_matches_eager(self):
        (w_in, s_in, _), _, (w_out, s_out, _) = self.shapes[:3]
        x = torch.randn(4, 4096, device="cuda", dtype=torch.bfloat16)
        (norm_q, sig_q), *_ = self.producers(4)

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
