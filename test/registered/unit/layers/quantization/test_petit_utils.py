"""Unit tests for srt/layers/quantization/petit_utils.py"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import importlib
import sys
import unittest
from unittest.mock import patch

from sglang.srt.layers.quantization import petit_utils
from sglang.test.test_utils import CustomTestCase


class TestPetitNotInstalled(CustomTestCase):
    def test_missing_petit_kernel_reports_install_hint(self):
        """Without petit_kernel a supported NVFP4 config is rejected with the install
        hint, instead of loading and failing later with a NameError."""
        try:
            with patch.dict(sys.modules, {"petit_kernel": None}):
                module = importlib.reload(petit_utils)
                with self.assertRaisesRegex(ValueError, "pip install petit-kernel"):
                    module.verify_petit_nvfp4_supported("NVFP4", 16)
        finally:
            importlib.reload(petit_utils)


if __name__ == "__main__":
    unittest.main()
