import unittest

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from sglang.srt.utils import is_gfx95_supported
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=30, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

if is_gfx95_supported():
    from sglang.kernels.ops.attention.dsv4.decode_attention_sm100_gluon import _load_v4
    from sglang.kernels.ops.attention.dsv4.swapab_gluon_hip import _load_v4_hip

    @gluon.jit
    def load_both(
        K,
        IDS,
        VALID,
        BASE,
        FUSED,
        PAGE: gl.constexpr,
        STRIDE: gl.constexpr,
        BT: gl.constexpr,
    ):
        layout: gl.constexpr = gl.BlockedLayout([1, 8], [8, 8], [1, 4], [1, 0])
        n = gl.arange(0, BT, gl.SliceLayout(1, layout))
        ids = gl.load(IDS + n)
        valid = gl.load(VALID + n) != 0
        d = gl.arange(0, 512, gl.SliceLayout(0, layout))
        out = n[:, None] * 512 + d[None, :]
        gl.store(BASE + out, _load_v4(K, ids, valid, PAGE, STRIDE, layout, 576, 8, 64))
        gl.store(
            FUSED + out, _load_v4_hip(K, ids, valid, PAGE, STRIDE, layout, 576, 8, 64)
        )


@unittest.skipUnless(is_gfx95_supported(), "requires AMD gfx950")
class TestSwapABDequantHip(CustomTestCase):
    def check(self, lo, hi, seed):
        page, pages, bt = 64, 9, 64
        stride = page * 584 + 256
        gen = torch.Generator(device="cuda").manual_seed(seed)
        raw = torch.randint(
            0, 256, (pages, stride), dtype=torch.uint8, device="cuda", generator=gen
        )
        scales = raw[:, page * 576 : page * 584].view(pages, page, 8)
        scales[:] = torch.randint(
            lo, hi, scales.shape, dtype=torch.uint8, device="cuda", generator=gen
        )
        ids = torch.randint(
            0, pages * page, (bt,), dtype=torch.int32, device="cuda", generator=gen
        )
        valid = torch.rand(bt, device="cuda", generator=gen) > 0.2
        base = torch.empty(bt, 512, dtype=torch.bfloat16, device="cuda")
        fused = torch.empty_like(base)
        load_both[(1,)](
            raw, ids, valid.to(torch.int8), base, fused, page, stride, bt, num_warps=4
        )

        a, b = base.view(torch.int16), fused.view(torch.int16)
        both_nan = base.isnan() & fused.isnan()
        bad = (a != b) & ~both_nan
        self.assertEqual(int(bad.sum()), 0, f"{int(bad.sum())} of {bad.numel()} differ")
        self.assertTrue(bool(valid.any()) and bool((~valid).any()))
        self.assertTrue(bool(base[~valid].eq(0).all()))

    def test_serving_exponents(self):
        self.check(100, 150, 1)

    def test_wide_exponents(self):
        self.check(1, 255, 2)

    def test_zero_exponent(self):
        self.check(0, 1, 3)


if __name__ == "__main__":
    unittest.main()
