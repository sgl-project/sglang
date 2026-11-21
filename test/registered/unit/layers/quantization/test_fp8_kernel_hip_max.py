"""AMD tests for ROCm per-token-group fp8 quant: the fp8 range and the HIP dispatch.

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

register_amd_ci(est_time=20, suite="base-b-test-1-gpu-small-amd-mi35x")

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


@unittest.skipUnless(is_hip(), "requires HIP")
class TestPerTokenGroupQuant8BitHipDispatch(CustomTestCase):
    def test_group32_quant_into_caller_buffers_matches_reference(self):
        """BF16 group-32 quant must run on HIP and match a torch reference.

        It reached the CUDA-only JIT kernel and raised NameError on ROCm, which
        broke block-32 ue8m0 FP8 linears (DeepSeek-V4.1 on gfx942).
        """
        from sglang.kernels.ops.quantization.fp8_kernel import (
            fp8_dtype,
            fp8_max,
            sglang_per_token_group_quant_fp8,
        )

        torch.manual_seed(0)
        x = (torch.randn((129, 1024), device="cuda") * 3.0).to(torch.bfloat16)
        x[0, :32] = 0  # an all-zero group takes the absmax floor
        groups = x.float().view(x.shape[0], -1, 32)
        for scale_ue8m0 in (True, False):
            with self.subTest(scale_ue8m0=scale_ue8m0):
                x_q, x_s = sglang_per_token_group_quant_fp8(
                    x, group_size=32, scale_ue8m0=scale_ue8m0
                )
                ref_s = groups.abs().amax(dim=-1).clamp_min(1e-10) / fp8_max
                if scale_ue8m0:
                    ref_s = torch.exp2(torch.ceil(torch.log2(ref_s)))
                torch.testing.assert_close(x_s, ref_s, rtol=1e-6, atol=0)
                ref_q = (groups / ref_s[..., None]).clamp(-fp8_max, fp8_max)
                ref_q = ref_q.to(fp8_dtype).view(x.shape)
                code_diff = (
                    (x_q.view(torch.uint8).int() - ref_q.view(torch.uint8).int())
                    .abs()
                    .max()
                    .item()
                )
                # Power-of-two scales divide exactly; the kernel multiplies by 1/scale.
                self.assertLessEqual(code_diff, 0 if scale_ue8m0 else 1)


if __name__ == "__main__":
    unittest.main()
