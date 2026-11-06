"""Real-AITER check of the Kimi-K3 tuned MoE front GEMM on gfx950.

The CPU tests in ``test_kimi_k3_rocm_quant.py`` fake ``tgemm.mm``;
only this file checks the keywords ``TunedFrontGemm`` passes to AITER and its output.
"""

import os
import unittest
from unittest.mock import patch

import torch

from sglang.srt.environ import envs
from sglang.srt.models.kimi_k3 import _k3_bf16_gemm
from sglang.srt.models.kimi_k3_rocm_moe import (
    _first_untuned_tokens,
    build_tuned_front_gemm,
)
from sglang.srt.utils import is_hip
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=20, stage="stage-b", runner_config="1-gpu-small-amd-mi35x")

# TP8 merged front [shared gate-up | router | latent down], the only tuned shape.
_FRONT_N, _FRONT_K = 6016, 7168
# Inside the default window; its tuned row is a split-K kernel, the noisiest one.
_IN_WINDOW_M = 64
# Last contiguous tuned row in ROCm/aiter#6411, the AITER pinned in docker/rocm.Dockerfile.
_LAST_TUNED_M = 128
# Measured 4e-3..8e-3 over 5 seeds on AITER 285579b62, from bf16 output rounding;
# a transposed weight gives ~1.5.
_MAX_REL_ERR = 2e-2


@unittest.skipUnless(is_hip(), "AITER's tuned front rows are gfx950 only")
class TestTunedFrontGemmGfx950(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx

        # A partitioned gfx950 has fewer CUs, and AITER keys its rows on the CU count.
        if (get_gfx(), get_cu_num()) != ("gfx950", 256):
            raise unittest.SkipTest("AITER's tuned front rows are 256-CU gfx950 only")
        gen = torch.Generator(device="cuda").manual_seed(0)
        cls.weight = (
            torch.randn(_FRONT_N, _FRONT_K, device="cuda", generator=gen) * 0.02
        ).to(torch.bfloat16)
        cls.x = torch.randn(_IN_WINDOW_M, _FRONT_K, device="cuda", generator=gen).to(
            torch.bfloat16
        )
        # Default window; the build probes AITER's row for every batch size in it.
        with (
            patch.dict(os.environ),
            envs.SGLANG_USE_AITER.override(True),
            envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT.override(True),
        ):
            envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT_MIN_TOKENS.clear()
            envs.SGLANG_OPT_HIP_K3_TUNED_MOE_FRONT_MAX_TOKENS.clear()
            cls.tuned = build_tuned_front_gemm(weight=cls.weight)

    def test_matches_the_default_bf16_gemm_inside_the_window(self):
        self.assertIsNotNone(self.tuned)
        self.assertTrue(self.tuned.covers(_IN_WINDOW_M))

        out = self.tuned(self.x, self.weight)
        ref = _k3_bf16_gemm(self.x, self.weight)

        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.shape, ref.shape)
        rel_err = (out.float() - ref.float()).abs().max() / ref.float().abs().max()
        self.assertLess(rel_err.item(), _MAX_REL_ERR)

    def test_default_window_is_tuned_up_to_its_last_batch(self):
        """tgemm silently runs an untuned kernel for a batch without a row; the
        default window must end at the pinned AITER's last contiguous front row (M=128)."""
        from aiter.tuned_gemm import get_GEMM_A16W16_tuned_config

        self.assertTrue(self.tuned.covers(_LAST_TUNED_M))
        self.assertFalse(self.tuned.covers(_LAST_TUNED_M + 1))
        self.assertIsNone(
            _first_untuned_tokens(
                get_tuned_config=get_GEMM_A16W16_tuned_config,
                n=_FRONT_N,
                k=_FRONT_K,
                min_tokens=1,
                max_tokens=_LAST_TUNED_M,
            )
        )


if __name__ == "__main__":
    unittest.main()
