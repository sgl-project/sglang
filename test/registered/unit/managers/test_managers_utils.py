"""Unit tests for managers/utils.py."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

from sglang.srt.managers.utils import (
    is_health_check_generate_req,
    validate_input_length,
)
from sglang.test.test_utils import CustomTestCase


class TestManagersUtils(CustomTestCase):
    """Test utility functions inside managers/utils.py."""

    def test_validate_input_length_pass(self):
        """Verify that short inputs are validated correctly without truncation."""
        req = SimpleNamespace(origin_input_ids=[1, 2, 3])
        error = validate_input_length(
            req, max_req_input_len=5, allow_auto_truncate=False
        )
        self.assertIsNone(error)
        self.assertEqual(len(req.origin_input_ids), 3)

    def test_validate_input_length_truncate(self):
        """Verify that long inputs are truncated when allow_auto_truncate=True."""
        req = SimpleNamespace(origin_input_ids=[1, 2, 3, 4, 5, 6])
        error = validate_input_length(
            req, max_req_input_len=4, allow_auto_truncate=True
        )
        self.assertIsNone(error)
        self.assertEqual(len(req.origin_input_ids), 4)
        self.assertEqual(req.origin_input_ids, [1, 2, 3, 4])

    def test_validate_input_length_error(self):
        """Verify that long inputs produce an error string when truncation is disallowed."""
        req = SimpleNamespace(origin_input_ids=[1, 2, 3, 4, 5, 6])
        error = validate_input_length(
            req, max_req_input_len=4, allow_auto_truncate=False
        )
        self.assertIsNotNone(error)
        self.assertIn("exceeds the maximum allowed length", error)
        self.assertEqual(len(req.origin_input_ids), 6)  # Should not be truncated

    def test_is_health_check_generate_req_true(self):
        """Verify that a health check prefix returns True."""
        req = SimpleNamespace(rid="HEALTH_CHECK_123")
        # Ensure it matches the HEALTH_CHECK_RID_PREFIX which is "HEALTH_CHECK"
        self.assertTrue(is_health_check_generate_req(req))

    def test_is_health_check_generate_req_false_normal(self):
        """Verify that a normal request ID returns False."""
        req = SimpleNamespace(rid="normal_req_456")
        self.assertFalse(is_health_check_generate_req(req))

    def test_is_health_check_generate_req_false_none_rid(self):
        """Verify that a request with no request ID returns False."""
        req = SimpleNamespace(rid=None)
        self.assertFalse(is_health_check_generate_req(req))


if __name__ == "__main__":
    unittest.main()
