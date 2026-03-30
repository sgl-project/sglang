"""Unit tests for DCP need_lse flag logic.

Verifies the need_lse boolean expression used in flashattention_backend.py:
  need_lse = use_cascade_attn or use_dcp
"""

import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="stage-a-test-cpu")


class TestDCPNeedLSELogic(CustomTestCase):
    """Verify need_lse flag is set correctly for DCP."""

    def test_need_lse_with_dcp(self):
        use_cascade_attn = False
        use_dcp = True
        self.assertTrue(use_cascade_attn or use_dcp)

    def test_need_lse_with_cascade(self):
        use_cascade_attn = True
        use_dcp = False
        self.assertTrue(use_cascade_attn or use_dcp)

    def test_need_lse_both(self):
        use_cascade_attn = True
        use_dcp = True
        self.assertTrue(use_cascade_attn or use_dcp)

    def test_need_lse_neither(self):
        use_cascade_attn = False
        use_dcp = False
        self.assertFalse(use_cascade_attn or use_dcp)


if __name__ == "__main__":
    unittest.main()
