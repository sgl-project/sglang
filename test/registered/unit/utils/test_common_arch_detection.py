"""CPU unit tests for the AMD architecture-detection helpers in srt/utils/common.py.

These helpers read the AMD GPU's ``gcnArchName`` and gate a numerics-affecting
dispatch: ``mxfp8_block_convert_required`` decides, in
``layers/quantization/fp8.py``, whether an MXFP8 checkpoint runs the native
MX-scaled matmul (gfx950 / CDNA4) or is first converted to block-FP8
(gfx942 / CDNA3, which has no hardware MX-scaled matmul). No server, no GPU,
no model loading. Ref: #20865.
"""

import unittest
from unittest.mock import patch

from sglang.srt.utils.common import (
    is_gfx95_supported,
    is_gfx942_supported,
    mxfp8_block_convert_required,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _clear_arch_caches():
    is_gfx942_supported.cache_clear()
    is_gfx95_supported.cache_clear()
    mxfp8_block_convert_required.cache_clear()


def _set_amd_arch(mock_torch, gcn_arch):
    mock_torch.version.hip = "6.0"
    mock_torch.cuda.get_device_properties.return_value.gcnArchName = gcn_arch


class TestMxfp8BlockConvertRequired(CustomTestCase):
    """``mxfp8_block_convert_required`` picks the fp8.py MXFP8 dispatch: True
    converts to block-FP8 (gfx942, no HW MX matmul); False keeps the native MX
    path (gfx950). Non-CDNA and non-HIP devices never convert."""

    def setUp(self):
        _clear_arch_caches()

    def tearDown(self):
        _clear_arch_caches()

    @patch("sglang.srt.utils.common.torch")
    def test_arch_dispatch_matrix(self, mock_torch):
        # gcnArchName -> mxfp8_block_convert_required(); the negative branches
        # (native-MX gfx950 and the RDNA archs) are the point of the guard.
        cases = [
            ("gfx942:sramecc+:xnack-", True),
            ("gfx950:sramecc+:xnack-", False),
            ("gfx1151", False),
            ("gfx1100", False),
        ]
        for gcn_arch, expected in cases:
            with self.subTest(gcn_arch=gcn_arch):
                _clear_arch_caches()
                _set_amd_arch(mock_torch, gcn_arch)
                self.assertIs(mxfp8_block_convert_required(), expected)

    @patch("sglang.srt.utils.common.torch")
    def test_non_hip_never_converts(self, mock_torch):
        mock_torch.version.hip = None
        self.assertIs(mxfp8_block_convert_required(), False)


class TestGfxArchDetection(CustomTestCase):
    """``is_gfx942_supported`` / ``is_gfx95_supported`` substring-match the
    suffixed gcnArchName and are mutually exclusive on a single device."""

    def setUp(self):
        _clear_arch_caches()

    def tearDown(self):
        _clear_arch_caches()

    @patch("sglang.srt.utils.common.torch")
    def test_gfx942_matches_suffixed_arch(self, mock_torch):
        _set_amd_arch(mock_torch, "gfx942:sramecc+:xnack-")
        self.assertIs(is_gfx942_supported(), True)
        self.assertIs(is_gfx95_supported(), False)

    @patch("sglang.srt.utils.common.torch")
    def test_gfx95_matches_suffixed_arch(self, mock_torch):
        _set_amd_arch(mock_torch, "gfx950:sramecc+:xnack-")
        self.assertIs(is_gfx95_supported(), True)
        self.assertIs(is_gfx942_supported(), False)

    @patch("sglang.srt.utils.common.torch")
    def test_non_hip_returns_false(self, mock_torch):
        mock_torch.version.hip = None
        self.assertIs(is_gfx942_supported(), False)
        self.assertIs(is_gfx95_supported(), False)


if __name__ == "__main__":
    unittest.main()
