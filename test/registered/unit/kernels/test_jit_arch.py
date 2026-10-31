"""SAIL's emulated SM numbers must not enable PDL, which is NVIDIA-only."""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

from sglang.kernels.jit.utils import arch
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestJitPdlCapability(CustomTestCase):
    def test_pdl_platform_and_sm_boundary(self):
        cases = (
            ("sail", 9, False),
            ("sail", 10, False),
            ("hip", 10, False),
            ("musa", 10, False),
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
                stack.enter_context(
                    patch.object(
                        arch,
                        "get_jit_cuda_arch",
                        return_value=arch.ArchInfo(major, 0, ""),
                    )
                )
                # Bypass the process-wide static cache so every platform
                # boundary is validated independently.
                self.assertEqual(arch.is_arch_support_pdl.__wrapped__(), expected)


if __name__ == "__main__":
    unittest.main()
