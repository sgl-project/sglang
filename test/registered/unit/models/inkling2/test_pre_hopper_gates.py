# SPDX-License-Identifier: Apache-2.0
"""Architecture gates that make Inkling-2-Mini runnable on pre-Hopper GPUs.

The gates are pure functions of the platform, so these run on CPU with faked
architectures. The numerics of the three Triton kernels are covered by the
kernel tests that compare them against their torch references.
"""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.srt.models.inkling_common.inkling2_arch import (
    prefer_triton_fallbacks,
    supports_fa4,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _cuda(device_sm: int) -> SimpleNamespace:
    # device_sm is the two-digit form sglang stores (80, 90, 100).
    return SimpleNamespace(
        is_cuda=True, device_sm=device_sm, device_capability=(device_sm // 10, 0)
    )


def _rocm(device_sm: int) -> SimpleNamespace:
    return SimpleNamespace(
        is_cuda=False, device_sm=device_sm, device_capability=(device_sm // 10, 0)
    )


class TestInkling2PreHopperGates(CustomTestCase):
    def test_fa4_is_hopper_and_newer(self):
        for device_sm, expected in ((80, False), (90, True), (100, True), (120, True)):
            platform = _cuda(device_sm)
            with patch(
                "sglang.srt.models.inkling_common.inkling2_arch.get_platform",
                return_value=platform,
            ):
                self.assertIs(supports_fa4(), expected, f"sm{device_sm}")

    def test_fa4_is_false_without_a_capability(self):
        platform = SimpleNamespace(
            is_cuda=True, device_sm=None, device_capability=(None, None)
        )
        with patch(
            "sglang.srt.models.inkling_common.inkling2_arch.get_platform",
            return_value=platform,
        ):
            self.assertFalse(supports_fa4())

    def test_triton_fallbacks_only_on_pre_hopper_cuda(self):
        self.assertTrue(prefer_triton_fallbacks(_cuda(80)))
        for device_sm in (90, 100, 120):
            self.assertFalse(
                prefer_triton_fallbacks(_cuda(device_sm)), f"sm{device_sm}"
            )

    def test_triton_fallbacks_leave_rocm_alone(self):
        # ROCm reports a gfx major through device_sm as well, so only the CUDA
        # check keeps gfx90a on the existing CUTE kernels.
        for gfx_major in (9, 10):
            self.assertFalse(
                prefer_triton_fallbacks(_rocm(gfx_major)), f"gfx{gfx_major}"
            )

    def test_rms_norm_picks_the_triton_kernel_on_sm80(self):
        from sglang.srt.models.inkling_common.inkling2_norm import RMSNorm

        with patch(
            "sglang.srt.models.inkling_common.inkling2_arch.get_platform",
            return_value=_cuda(80),
        ):
            self.assertTrue(RMSNorm(128, pre_affine=True)._use_triton)

        with patch(
            "sglang.srt.models.inkling_common.inkling2_arch.get_platform",
            return_value=_cuda(90),
        ):
            self.assertFalse(RMSNorm(128, pre_affine=True)._use_triton)


if __name__ == "__main__":
    unittest.main()
