"""SAIL's emulated SM numbers must not enable PDL, which is NVIDIA-only."""

import unittest
from contextlib import ExitStack
from types import SimpleNamespace
from unittest.mock import patch

from sglang.kernels.jit.utils import arch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestJitPdlCapability(CustomTestCase):
    def test_pdl_platform_and_sm_boundary(self):
        # HIP/MUSA report no PDL through their own arch type, so only the SAIL
        # short-circuit and the CUDA SM boundary are exercised here.
        cases = (
            ("sail", 9, False),
            ("sail", 10, False),
            ("cuda", 8, False),
            ("cuda", 9, True),
            ("cuda", 10, True),
        )
        for platform, major, expected in cases:
            with self.subTest(platform=platform, major=major), ExitStack() as stack:
                for name in ("sail", "hip", "musa"):
                    stack.enter_context(
                        patch.object(
                            arch, f"is_{name}_runtime", return_value=platform == name
                        )
                    )
                # A local SM90+ GPU would report PDL, so the SAIL cases only pass
                # when the short-circuit wins over the capability.
                stack.enter_context(
                    patch.object(
                        arch,
                        "current_arch",
                        return_value=SimpleNamespace(supports_pdl=major >= 9),
                    )
                )
                # Bypass the process-wide static cache so every platform
                # boundary is validated independently.
                self.assertEqual(arch.is_arch_support_pdl.__wrapped__(), expected)


class TestJitTargetFlags(CustomTestCase):
    def test_sail_macro_is_restricted_to_sail(self):
        compile_arch = arch.CompileArch(8, 9, "")
        common_flags = [
            compile_arch.jit_flag,
            "-std=c++20",
            "-O3",
            "--expt-relaxed-constexpr",
        ]
        for platform in ("sail", "cuda", "hip"):
            with self.subTest(platform=platform), ExitStack() as stack:
                stack.enter_context(
                    patch.object(arch, "is_hip_runtime", return_value=platform == "hip")
                )
                stack.enter_context(
                    patch.object(
                        arch, "is_sail_runtime", return_value=platform == "sail"
                    )
                )
                flags = arch.get_default_target_flags(compile_arch)
                if platform == "sail":
                    self.assertEqual(flags, [*common_flags, "-DSGL_SAIL=1"])
                elif platform == "cuda":
                    self.assertEqual(flags, common_flags)
                else:
                    self.assertIn("-DUSE_ROCM", flags)
                    self.assertNotIn("-DSGL_SAIL=1", flags)


if __name__ == "__main__":
    unittest.main()
