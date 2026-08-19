"""AMD/gfx950 (MI355X) test for the ROCm fp8 range used by per-token-group quant.

ROCm fp8 is device-dependent: e4m3fnuz (max 224.0) on gfx94x (MI300) vs e4m3fn
(max 448.0) on gfx95x (MI355X). ``_per_token_group_quant_8bit_raw`` previously
hardcoded 224.0 for *all* ROCm devices, which silently halved the usable fp8
range on gfx95x -- the per-group scale was computed as ``absmax / 224`` instead
of ``absmax / 448``, wasting one binade of e4m3fn precision.

This runs the real Triton quant kernel on the actual GPU (no mocking): it feeds a
group whose absmax is exactly the e4m3fn max (448.0) and asserts the emitted
per-group scale is 1.0 (i.e. ``absmax / 448``). On the unfixed code the scale is
``448 / 224 = 2.0`` and the test fails. On gfx94x this bug does not exist (224 is
correct), so the test is gated to gfx95x where it is a genuine bug-catcher.
"""

import unittest

import torch

from sglang.srt.utils import is_gfx95_supported, is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, suite="stage-b-test-1-gpu-small-amd-mi35x")

# e4m3fn (max 448.0) only exists on gfx95x; on gfx94x fp8 is e4m3fnuz (max 224.0).
_RUNNABLE = is_hip() and is_gfx95_supported()

# e4m3fn representable max; the value the kernel must scale/clamp against on gfx95x.
E4M3FN_MAX = 448.0


@unittest.skipUnless(_RUNNABLE, "requires HIP gfx950 (MI355X, e4m3fn fp8)")
class TestPerTokenGroupQuant8BitHipMax(CustomTestCase):
    def test_hip_fp8_scale_uses_e4m3fn_max_448(self):
        from sglang.kernels.ops.quantization.fp8_kernel import (
            _per_token_group_quant_8bit_raw,
            fp8_dtype,
            fp8_max,
        )

        # Sanity: on gfx95x the module must resolve to e4m3fn / 448.
        self.assertIs(fp8_dtype, torch.float8_e4m3fn)
        self.assertEqual(fp8_max, E4M3FN_MAX)

        group_size = 8
        # One group whose absmax is exactly the e4m3fn max.
        x = torch.zeros((1, group_size), dtype=torch.bfloat16, device="cuda")
        x[0, 0] = E4M3FN_MAX

        _, x_s = _per_token_group_quant_8bit_raw(
            x, group_size=group_size, dtype=fp8_dtype
        )

        # scale == absmax / device_max. Correct: 448/448 = 1.0. Unfixed: 448/224 = 2.0.
        scale = x_s.float().flatten()[0].item()
        self.assertAlmostEqual(scale, 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
