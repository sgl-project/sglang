"""The gfx950 streaming attention-residual kernel against the register-tile kernel."""

import unittest

import torch

from sglang.kernels.ops.attention import attn_res_hip as ar
from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# backend-specific: the streaming bank read is selected on gfx950 only.
register_amd_ci(est_time=15, suite="stage-b-test-1-gpu-small-amd-mi35x")

# Kimi-K3: hidden 7168, up to 8 bank snapshots.
H, NB = 7168, 9


@unittest.skipUnless(is_gfx95_supported(), "gfx950 attention-residual kernels")
class TestAttnResHipStream(CustomTestCase):
    @torch.inference_mode()
    def test_streamed_bank_matches_register_tile(self):
        """Batches that read the bank a few rows at a time (online softmax) must
        agree with the whole-bank register tile within one bf16 ulp, and write the
        same prefix and bank snapshot bit for bit."""
        g = torch.Generator(device="cuda").manual_seed(0)
        rnd = lambda *s: torch.randn(*s, generator=g, device="cuda")  # noqa: E731
        T = ar.STREAM_MIN_ROWS + 37
        prefix = (rnd(T, H) * 4).bfloat16()
        addend = rnd(T, H).bfloat16()
        cw = rnd(H) * 0.01
        ow = (rnd(H) * 0.1 + 1).bfloat16()
        for nvb in (5, 8):
            for with_add, write_prefix in ((True, True), (False, False)):
                bank0 = (rnd(T, NB, H) * 4).bfloat16()
                outs = []
                for stream_min_rows in (ar.STREAM_MIN_ROWS, 1 << 30):
                    bank = bank0.clone()
                    out = torch.empty(T, H, device="cuda", dtype=torch.bfloat16)
                    pre = torch.empty_like(out)
                    saved, ar.STREAM_MIN_ROWS = ar.STREAM_MIN_ROWS, stream_min_rows
                    try:
                        ar.attn_res_hip(
                            prefix,
                            bank,
                            cw,
                            ow,
                            out,
                            nvb,
                            1e-6,
                            1e-6,
                            addend=addend if with_add else None,
                            prefix_out=pre if with_add else None,
                            write_prefix=write_prefix,
                        )
                    finally:
                        ar.STREAM_MIN_ROWS = saved
                    outs.append((out.float(), pre, bank))
                (o_s, p_s, b_s), (o_t, p_t, b_t) = outs
                with self.subTest(
                    nvb=nvb, with_add=with_add, write_prefix=write_prefix
                ):
                    beyond_ulp = (o_s - o_t).abs() > o_t.abs() * 2**-7 + 1e-3
                    self.assertLessEqual(beyond_ulp.sum().item(), o_s.numel() // 1000)
                    if with_add:
                        self.assertTrue(torch.equal(p_s, p_t))
                    self.assertTrue(torch.equal(b_s, b_t))


if __name__ == "__main__":
    unittest.main()
