"""The DSv4 fp8 e4m3 conversion on its own, byte for byte against torch.

Every fp8 store in the DSv4 tree quantizes through ``pack_fp8``, and none of the
callers can see when it is wrong: the value has already been divided by a
quantization scale, so a bad rounding or saturation boundary comes back as fp8 being
lossier than it should be, not as a failure. ``pack_fp8`` on ROCm used to be a
hand-written bit twiddle, and it had two: the whole top exponent segment saturated
to the max normal, and the binade under the min subnormal flushed to zero instead of
rounding up to it. Both are pinned below.

gfx950 only. gfx942 still runs the software cast, bugs and all, because the
instruction cannot produce the fnuz bytes that arch writes; that one needs its own fix.
The reference is torch's own cast.
"""

import math
import unittest

import torch

from sglang.kernels.ops.attention.dsv4.fp8_cvt import cvt_fp8_e4m3
from sglang.kernels.ops.quantization.fp8_kernel import fp8_dtype, fp8_max
from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

# the default amd runner is mi300, where pack_fp8 still takes the software path this
# does not cover
register_amd_ci(est_time=60, suite="stage-b-test-1-gpu-small-amd-mi35x")

DEVICE = torch.device("cuda")

# start of the top exponent segment, i.e. the largest power of two the format holds
TOP_BINADE = 2.0 ** math.floor(math.log2(fp8_max))


def _representable():
    vals = torch.arange(256, dtype=torch.uint8).view(fp8_dtype).float()
    return vals[vals.isfinite()]


def _domain():
    """every representable value, every midpoint between two of them, every bf16"""
    vals = _representable()
    mids = ((vals[:, None] + vals[None, :]) / 2).flatten()
    cases = torch.cat(
        [
            vals,
            mids,
            torch.linspace(-fp8_max, fp8_max, 100003),
            torch.arange(1 << 16, dtype=torch.int32).view(torch.bfloat16).float(),
        ]
    )
    # stay inside the range: past the max the two casts are allowed to disagree on
    # whether to clamp or produce NaN, which is not what this is testing
    cases = cases[cases.isfinite() & (cases.abs() <= fp8_max)].unique()
    if cases.numel() % 2:
        cases = cases[:-1]
    return cases.contiguous()


def _as_bytes(x):
    # the conversion runs two values at a time, so the length has to stay even --
    # e4m3fnuz has an odd number of representable values (only 0x80 is NaN)
    if x.numel() % 2:
        x = x[:-1]
    x = x.contiguous().to(DEVICE)
    return cvt_fp8_e4m3(x), x.to(fp8_dtype).view(torch.uint8)


@unittest.skipUnless(
    torch.cuda.is_available() and is_hip() and is_gfx95_supported(),
    "the gfx95 path of pack_fp8 is what this pins",
)
class TestDsv4Fp8Cast(CustomTestCase):
    def test_matches_torch_over_the_whole_range(self):
        cases = _domain()
        got, want = _as_bytes(cases)
        bad = got != want
        if bool(bad.any()):
            v = cases.to(DEVICE)[bad]
            worst = v.abs().argmax()
            self.fail(
                f"{int(bad.sum())} of {cases.numel()} bytes differ, "
                f"|v| in [{v.abs().min():.4e}, {v.abs().max():.4e}]; e.g. "
                f"{v[worst]:.6g} -> {got[bad][worst].item():#04x} "
                f"(torch {want[bad][worst].item():#04x})"
            )

    def test_top_exponent_segment_is_not_saturated(self):
        # testing the exponent alone here used to write every value from TOP_BINADE up
        # to the max out as the max normal
        vals = _representable()
        # pack_fp8 clips to fp8_max, so anything the format holds above it never comes
        # out of the conversion
        top = vals[(vals.abs() >= TOP_BINADE) & (vals.abs() <= fp8_max)]
        self.assertGreater(top.numel(), 2)
        got, want = _as_bytes(top)
        self.assertTrue(torch.equal(got, want))
        # and they really are distinct values, not all the same byte
        self.assertGreater(int(got.unique().numel()), 2)

    def test_binade_below_the_min_subnormal_rounds_up(self):
        min_subnormal = _representable().abs()
        min_subnormal = min_subnormal[min_subnormal > 0].min().item()
        # (midpoint, min subnormal): rounds up. the midpoint itself is a tie and goes
        # to even, i.e. to zero
        band = torch.linspace(min_subnormal / 2, min_subnormal, 2049)[1:-1]
        band = torch.cat([band, -band])
        got, want = _as_bytes(band.contiguous())
        self.assertTrue(torch.equal(got, want))
        self.assertTrue(bool((got & 0x7F).all()))


if __name__ == "__main__":
    unittest.main()
