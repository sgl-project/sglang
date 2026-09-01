"""Regression for the architecture boundary of CuTe GDN/KDA prefill.

SM120 must use the existing Triton prefill fallback.  The CuTe prefill kernels
are validated for the SM100/SM103 family only; a broad ``major >= 10`` check
silently selects them on consumer Blackwell.
"""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.layers.attention.linear.kernels.gdn_cutedsl import CuteDSLGDNKernel
from sglang.srt.layers.attention.linear.kernels.kda_cutedsl import CuteDSLKDAKernel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class TestCuteDSLPrefillArchitectureGate(unittest.TestCase):
    def test_only_sm10x_advertises_cutedsl_prefill(self):
        cases = (
            ((9, 0), False),
            ((10, 0), True),
            ((10, 3), True),
            ((12, 0), False),
        )
        for capability, expected in cases:
            with (
                self.subTest(capability=capability),
                patch.object(torch.cuda, "is_available", return_value=True),
                patch.object(
                    torch.cuda,
                    "get_device_capability",
                    return_value=capability,
                ),
            ):
                self.assertEqual(CuteDSLGDNKernel().supports_prefill, expected)
                self.assertEqual(CuteDSLKDAKernel().supports_prefill, expected)
