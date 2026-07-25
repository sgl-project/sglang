"""Unit tests for request_headers.py."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace

from fastapi import HTTPException

from sglang.srt.entrypoints.request_headers import apply_header_overrides
from sglang.test.test_utils import CustomTestCase


class TestRequestHeaders(CustomTestCase):
    """Test the application of header overrides on request objects."""

    def test_apply_header_overrides_success(self):
        """Verify all valid headers are correctly cast and applied."""
        # SimpleNamespace serves as a clean dummy object for setattr.
        obj = SimpleNamespace(
            rid=None,
            bootstrap_host=None,
            bootstrap_port=None,
            bootstrap_room=None,
            conversation_id=None,
            routed_dp_rank=None,
            disagg_prefill_dp_rank=None,
            priority=None,
        )
        headers = {
            "x-override-rid": "test-rid-123",
            "x-override-bootstrap-host": "localhost",
            "x-override-bootstrap-port": "8080",
            "x-override-bootstrap-room": "1",
            "x-override-conversation-id": "conv-456",
            "x-override-routed-dp-rank": "2",
            "x-override-disagg-prefill-dp-rank": "3",
            "x-override-priority": "10",
        }

        apply_header_overrides(obj, headers)

        self.assertEqual(obj.rid, "test-rid-123")
        self.assertEqual(obj.bootstrap_host, "localhost")
        self.assertEqual(obj.bootstrap_port, 8080)
        self.assertEqual(obj.bootstrap_room, 1)
        self.assertEqual(obj.conversation_id, "conv-456")
        self.assertEqual(obj.routed_dp_rank, 2)
        self.assertEqual(obj.disagg_prefill_dp_rank, 3)
        self.assertEqual(obj.priority, 10)

    def test_apply_header_overrides_partial(self):
        """Verify that only the provided headers are applied, and missing ones are ignored."""
        obj = SimpleNamespace(
            rid=None,
            priority=None,
            bootstrap_host=None,
            bootstrap_port=None,
        )
        headers = {
            "x-override-rid": "test-rid-123",
            "x-override-priority": "5",
        }

        apply_header_overrides(obj, headers)

        self.assertEqual(obj.rid, "test-rid-123")
        self.assertEqual(obj.priority, 5)
        self.assertIsNone(obj.bootstrap_host)
        self.assertIsNone(obj.bootstrap_port)

    def test_apply_header_overrides_missing(self):
        """Verify behavior when no headers are provided."""
        obj = SimpleNamespace(rid=None, priority=None)
        headers = {}

        apply_header_overrides(obj, headers)

        # Should not raise any error, and attributes should remain None
        self.assertIsNone(obj.rid)
        self.assertIsNone(obj.priority)

    def test_apply_header_overrides_invalid_type(self):
        """Verify that invalid header value types result in a 400 HTTPException."""
        obj = SimpleNamespace(priority=None)
        headers = {
            "x-override-priority": "invalid-int",
        }

        with self.assertRaises(HTTPException) as context:
            apply_header_overrides(obj, headers)

        self.assertEqual(context.exception.status_code, 400)
        self.assertIn("invalid x-override-priority header", context.exception.detail)


if __name__ == "__main__":
    unittest.main()
