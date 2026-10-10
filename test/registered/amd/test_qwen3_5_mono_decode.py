"""Tests for the Qwen3.8 mono decode kernels (ROCm / FlyDSL, gfx950).

``LayoutTest`` checks, on the host, that every fp4 byte and e8m0 scale of the
routed experts is where ``layout`` says, for the layout SGLang leaves on gfx950
(quark_w4a4_mxfp4_moe.py: ``shuffle_weight(w, (16, 16))`` on the uint8
weight, ``e8m0_shuffle`` over (E N, K / 32)). K2 reads the experts in place,
so a layout change there breaks the kernel silently.

``GdnPreTest`` builds and launches K1 (``gdn_pre``) on one GPU with the state
in SGLang's mamba pool layout: conv (slots, dim, 3) dim-first, SSM
(slots, HV, V, K) fp32, slot 0 reserved and -1 pad rows. It checks that the
live rows are written and that pad rows and other slots are left alone.

Run directly with ``python test/registered/amd/test_qwen3_5_mono_decode.py``.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=120, suite="stage-b-test-1-gpu-small-amd-mi35x")

NE = 2  # experts: enough to catch a wrong expert stride
BF = torch.bfloat16


def _gfx950() -> bool:
    return (
        torch.cuda.is_available()
        and torch.version.hip is not None
        and "gfx950" in torch.cuda.get_device_properties(0).gcnArchName
    )


@unittest.skipUnless(_gfx950(), "needs aiter on gfx950")
class LayoutTest(CustomTestCase):
    def _check_weight(self, w, base_of, k):
        from aiter.ops.shuffle import shuffle_weight

        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import layout as L

        flat = shuffle_weight(w.contiguous(), (16, 16)).reshape(-1).view(torch.int32)
        lane = torch.arange(64)
        for e in range(NE):
            for rg in range(w.shape[1] // L.ROWS):
                rows = w[e, rg * L.ROWS + lane % 16]  # (64, K / 2)
                for st in range(k // L.FP4_STEP):
                    at = L.fp4_tile_dword(base_of(e), rg, k, st, lane * 0) + lane * 4
                    got = torch.stack([flat[at + q] for q in range(4)], 1)
                    b0 = st * 64 + (lane // 16) * 16  # first byte of the lane's 32 fp4
                    want = torch.stack(
                        [rows[i, b0[i] : b0[i] + 16] for i in range(64)]
                    ).view(torch.int32)
                    self.assertTrue(torch.equal(got, want), (e, rg, st))

    def test_w13_tiles(self):
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import layout as L

        g = torch.Generator().manual_seed(0)
        w = torch.randint(
            0, 256, (NE, 2 * L.RI, L.HIDDEN // 2), dtype=torch.uint8, generator=g
        )
        self._check_weight(w, L.w13_base, L.HIDDEN)

    def test_w2_tiles(self):
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import layout as L

        g = torch.Generator().manual_seed(1)
        w = torch.randint(
            0, 256, (NE, L.HIDDEN, L.RI // 2), dtype=torch.uint8, generator=g
        )
        self._check_weight(w, L.w2_base, L.RI)

    def test_scales(self):
        from aiter.utility.fp4_utils import e8m0_shuffle

        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import layout as L

        for rows, k, index in (
            (2 * L.RI, L.HIDDEN, L.w13_scale_index),
            (L.HIDDEN, L.RI, L.w2_scale_index),
        ):
            g = torch.Generator().manual_seed(2)
            s = torch.randint(
                0, 256, (NE, rows, k // 32), dtype=torch.uint8, generator=g
            )
            flat = e8m0_shuffle(s.view(NE * rows, -1)).reshape(-1)
            self.assertEqual(flat.numel(), s.numel())
            n = torch.arange(rows)[:, None]
            kb = torch.arange(k // 32)[None, :]
            for e in range(NE):
                self.assertTrue(torch.equal(flat[index(e, n, kb)], s[e]), (rows, e))


@unittest.skipUnless(_gfx950(), "needs FlyDSL on gfx950")
class GdnPreTest(CustomTestCase):
    def test_pad_rows_and_slots(self):
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl import gdn
        from sglang.kernels.ops.moe.qwen3_5_mono_flydsl.layout import (
            BA,
            CONV,
            CORE,
            HD,
            HIDDEN,
            NV,
            QKVZ,
        )

        g = torch.Generator(device="cuda").manual_seed(0)

        def rnd(*shape, scale=1.0, dtype=BF):
            return (torch.randn(*shape, generator=g, device="cuda") * scale).to(dtype)

        slots = 16
        for s, n_real, first in ((1, 1, True), (4, 3, False), (8, 8, False)):
            conv = rnd(slots, CONV, 3)
            rstate = rnd(slots, NV, HD, HD, scale=0.05, dtype=torch.float32)
            conv0, rstate0 = conv.clone(), rstate.clone()
            idx = torch.full((s,), -1, dtype=torch.int32, device="cuda")
            idx[:n_real] = torch.arange(3, 3 + n_real, dtype=torch.int32, device="cuda")
            key = gdn.K1Build(tokens=s, first=first)
            core = torch.full((s, CORE), float("nan"), dtype=BF, device="cuda")
            gdn.gdn_pre(
                key,
                hidden=rnd(s, HIDDEN, scale=2.0),
                residual=None if first else rnd(s, HIDDEN, scale=2.0),
                res_out=torch.empty(s, HIDDEN, dtype=BF, device="cuda"),
                ln_w=rnd(HIDDEN, scale=0.1),
                w_qkvz=rnd(QKVZ, HIDDEN, scale=0.02),
                w_ba=rnd(BA, HIDDEN, scale=0.05),
                conv_w=rnd(CONV, 1, 4, scale=0.4),
                conv_state=conv,
                a_log=rnd(NV, scale=0.5, dtype=torch.float32),
                dt_bias=rnd(NV, scale=0.5),
                norm_w=(1 + rnd(HD, scale=0.1, dtype=torch.float32)).to(BF),
                rstate=rstate,
                st_idx=idx,
                core=core,
                scratch=torch.zeros(
                    gdn.scratch_bytes(key), dtype=torch.uint8, device="cuda"
                ),
                epoch=torch.ones(1, dtype=torch.int32, device="cuda"),
                layer=0,
            )
            torch.cuda.synchronize()
            live = idx[:n_real].long()
            keep = torch.ones(slots, dtype=torch.bool, device="cuda")
            keep[live] = False
            with self.subTest(s=s, n_real=n_real, first=first):
                self.assertTrue(torch.isfinite(core[:n_real]).all())
                self.assertGreater(core[:n_real].abs().sum().item(), 0)
                self.assertTrue(
                    torch.equal(core[n_real:], torch.zeros_like(core[n_real:]))
                )
                self.assertFalse(torch.equal(rstate[live], rstate0[live]))
                self.assertFalse(torch.equal(conv[live], conv0[live]))
                self.assertTrue(torch.equal(rstate[keep], rstate0[keep]))
                self.assertTrue(torch.equal(conv[keep], conv0[keep]))


if __name__ == "__main__":
    unittest.main()
