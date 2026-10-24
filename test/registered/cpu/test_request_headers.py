import unittest
from types import SimpleNamespace

from fastapi import HTTPException
from starlette.datastructures import Headers

from sglang.srt.entrypoints.request_headers import (
    apply_header_overrides,
    apply_session_id_header,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=4, suite="stage-a-test-cpu-intel")


def _obj():
    return SimpleNamespace(
        rid=None,
        bootstrap_host=None,
        bootstrap_port=None,
        bootstrap_room=None,
        conversation_id=None,
        routed_dp_rank=None,
        disagg_prefill_dp_rank=None,
        priority=None,
    )


class TestApplyRoutingHeaders(CustomTestCase):
    def test_sets_all_fields_with_types(self):
        obj = _obj()
        apply_header_overrides(
            obj,
            Headers(
                {
                    "x-override-rid": "r1",
                    "x-override-bootstrap-host": "prefill1",
                    "x-override-bootstrap-port": "8998",
                    "x-override-bootstrap-room": "18446744073709551615",
                    "x-override-conversation-id": "c1",
                    "x-override-routed-dp-rank": "3",
                    "x-override-disagg-prefill-dp-rank": "5",
                    "x-override-priority": "7",
                }
            ),
        )
        self.assertEqual(obj.rid, "r1")
        self.assertEqual(obj.bootstrap_host, "prefill1")
        self.assertEqual(obj.bootstrap_port, 8998)
        self.assertEqual(obj.bootstrap_room, 18446744073709551615)
        self.assertEqual(obj.conversation_id, "c1")
        self.assertEqual(obj.routed_dp_rank, 3)
        self.assertEqual(obj.disagg_prefill_dp_rank, 5)
        self.assertEqual(obj.priority, 7)

    def test_absent_headers_leave_obj_unchanged(self):
        obj = _obj()
        apply_header_overrides(obj, Headers({}))
        self.assertIsNone(obj.rid)
        self.assertIsNone(obj.bootstrap_host)
        self.assertIsNone(obj.routed_dp_rank)

    def test_header_overrides_existing_value(self):
        obj = _obj()
        obj.rid = "from-body"
        apply_header_overrides(obj, Headers({"x-override-rid": "from-header"}))
        self.assertEqual(obj.rid, "from-header")

    def test_partial_headers_set_only_present(self):
        obj = _obj()
        apply_header_overrides(
            obj, Headers({"x-override-rid": "r1", "x-override-routed-dp-rank": "2"})
        )
        self.assertEqual(obj.rid, "r1")
        self.assertEqual(obj.routed_dp_rank, 2)
        self.assertIsNone(obj.bootstrap_host)

    def test_invalid_int_fails_loud(self):
        obj = _obj()
        with self.assertRaises(HTTPException):
            apply_header_overrides(
                obj, Headers({"x-override-bootstrap-port": "not-an-int"})
            )

    def test_priority_header_overrides_body_value(self):
        # The scheduler reads obj.priority, so the header value must be the one
        # that ends up on the object even when the body already set a priority.
        obj = _obj()
        obj.priority = 1
        apply_header_overrides(obj, Headers({"x-override-priority": "5"}))
        self.assertEqual(obj.priority, 5)

    def test_negative_priority_header_is_applied(self):
        obj = _obj()
        obj.priority = 1
        apply_header_overrides(obj, Headers({"x-override-priority": "-3"}))
        self.assertEqual(obj.priority, -3)

    def test_priority_body_value_preserved_when_header_absent(self):
        obj = _obj()
        obj.priority = 2
        apply_header_overrides(obj, Headers({}))
        self.assertEqual(obj.priority, 2)

    def test_invalid_priority_fails_loud(self):
        obj = _obj()
        with self.assertRaises(HTTPException):
            apply_header_overrides(obj, Headers({"x-override-priority": "high"}))


def _session_obj():
    return SimpleNamespace(session_id=None, session_params=None)


class TestApplySessionIdHeader(CustomTestCase):
    """``X-Session-ID`` is the same header the vLLM OpenAI server accepts."""

    def test_header_fills_missing_session_id(self):
        obj = _session_obj()
        apply_session_id_header(obj, Headers({"x-session-id": "conv-1"}))
        self.assertEqual(obj.session_id, "conv-1")

    def test_body_session_id_wins(self):
        obj = _session_obj()
        obj.session_id = "from-body"
        apply_session_id_header(obj, Headers({"x-session-id": "from-header"}))
        self.assertEqual(obj.session_id, "from-body")

    def test_absent_header_leaves_session_id_none(self):
        obj = _session_obj()
        apply_session_id_header(obj, Headers({}))
        self.assertIsNone(obj.session_id)

    def test_empty_header_is_ignored(self):
        obj = _session_obj()
        apply_session_id_header(obj, Headers({"x-session-id": ""}))
        self.assertIsNone(obj.session_id)

    def test_session_params_disables_header(self):
        # The scheduler treats session_params and session_id as mutually
        # exclusive, so the header must not create the invalid combination.
        obj = _session_obj()
        obj.session_params = {"id": "managed-session"}
        apply_session_id_header(obj, Headers({"x-session-id": "conv-1"}))
        self.assertIsNone(obj.session_id)


if __name__ == "__main__":
    unittest.main()
