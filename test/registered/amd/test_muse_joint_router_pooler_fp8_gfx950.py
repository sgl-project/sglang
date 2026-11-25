"""gfx950 FP8 joint router/pooler front (router_pooler_topk.joint_router_pooler_topk_fp8), Muse Spark shapes.

Checks the split-K partials of _joint_fp8_partial against an FP32 reference at row counts that cover every launch
bucket and both sides of every row-count rule (XCD map and M-tile thresholds), plus the routing outputs and the
gfx950 gate of joint_router_pooler_fp8_applies.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=60, suite="base-b-test-1-gpu-small-amd-mi35x")

# Every FP8_JOINT bucket and both sides of the 40 / 96 / 128 / 216 / 256 row rules, odd sizes included.
_ROWS = (
    17,
    24,
    33,
    40,
    41,
    57,
    64,
    65,
    96,
    97,
    128,
    129,
    200,
    216,
    217,
    256,
    257,
    333,
    512,
)


@unittest.skipUnless(is_gfx95_supported(), "gfx950 (MI35x) only")
class TestMuseJointRouterPoolerFp8Gfx950(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from sglang.kernels.ops.gemm.fp8w_small_m_gemm import quantize_rows
        from sglang.kernels.ops.moe import router_pooler_topk as rp

        cls.rp, cls.quantize_rows = rp, staticmethod(quantize_rows)
        gen = torch.Generator(device="cuda").manual_seed(0)
        router = torch.randn(8192, 256, device="cuda", generator=gen) * 0.02
        pooler = (torch.randn(4096, 8192, device="cuda", generator=gen) * 0.02).to(
            torch.bfloat16
        )
        cls.packed = rp.pack_joint_router_pooler_weight(router, pooler)
        cls.wq, cls.ws = rp.quantize_joint_pooler(cls.packed)
        cls.bias = torch.randn(256, device="cuda", generator=gen) * 0.1
        cls.norm = (1.0 + 0.1 * torch.randn(4096, device="cuda", generator=gen)).to(
            torch.bfloat16
        )
        cls.gen = gen

    def run_front(self, m):
        """(captured FP32 partials [SK, M, 4608], scores, ids, expert input) for ``m`` random rows."""
        x = (torch.randn(m, 8192, device="cuda", generator=self.gen) * 0.5).to(
            torch.bfloat16
        )
        xq, xs = self.quantize_rows(x)
        xq, xs = xq.contiguous(), xs.contiguous()
        self.assertTrue(
            self.rp.joint_router_pooler_fp8_applies(xq, self.packed, (self.wq, self.ws))
        )
        captured, real_finish = {}, self.rp._joint_finish

        def finish(partial, *args, **kwargs):
            captured["partial"] = partial.clone()
            return real_finish(partial, *args, **kwargs)

        with patch.object(self.rp, "_joint_finish", finish):
            scores, ids, expert_input = self.rp.joint_router_pooler_topk_fp8(
                xq,
                xs,
                x,
                self.packed,
                (self.wq, self.ws),
                self.bias,
                self.norm,
                1.0,
                1.0,
                1e-5,
            )[:3]
        return x, xq, captured["partial"], scores, ids, expert_input

    def test_partials_match_fp32_reference(self):
        wr = self.packed[4096:].float()
        for m in _ROWS:
            with self.subTest(m=m):
                x, xq, partial, _, _, _ = self.run_front(m)
                self.assertEqual(partial.dtype, torch.float32)
                self.assertEqual(tuple(partial.shape[1:]), (m, 4608))
                ref = torch.cat(
                    [xq.float() @ self.wq.float().T, x.float() @ wr.T], dim=1
                )
                torch.testing.assert_close(
                    partial.sum(0), ref, rtol=1e-4, atol=1e-4 * ref.abs().max().item()
                )

    def test_routing_outputs(self):
        for m in (24, 64, 200):
            with self.subTest(m=m):
                _, _, _, scores, ids, expert_input = self.run_front(m)
                self.assertEqual(tuple(ids.shape), (m, 8))
                self.assertTrue(bool(((ids >= 0) & (ids < 256)).all()))
                self.assertTrue(all(len(set(row.tolist())) == 8 for row in ids))
                self.assertTrue(bool(torch.isfinite(scores).all()))
                self.assertTrue(bool(torch.isfinite(expert_input.float()).all()))
                self.assertEqual(tuple(expert_input.shape), (m, 4096))

    def test_gate_off_elsewhere(self):
        x = torch.randn(40, 8192, device="cuda").to(torch.bfloat16)
        xq, _ = self.quantize_rows(x)
        with patch.object(self.rp, "_fp8_joint_runs_here", lambda: False):
            self.assertFalse(
                self.rp.joint_router_pooler_fp8_applies(
                    xq.contiguous(), self.packed, (self.wq, self.ws)
                )
            )
        self.assertTrue(self.rp._fp8_joint_runs_here())


if __name__ == "__main__":
    unittest.main()
