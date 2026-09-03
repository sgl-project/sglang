import unittest
from types import SimpleNamespace

from fastapi import HTTPException
from starlette.datastructures import Headers

from sglang.srt.entrypoints.request_headers import (
    apply_header_overrides,
    resolve_rid_from_headers,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-b-test-cpu")


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


class TestApplyRoutingHeaders(unittest.TestCase):
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


class TestResolveRidFromHeaders(unittest.TestCase):
    """x-request-id is list-valued, so both wire forms must parse the same way."""

    def test_absent_or_blank_header_names_no_identity(self):
        self.assertIsNone(resolve_rid_from_headers(Headers({})))
        self.assertIsNone(resolve_rid_from_headers(Headers({"x-request-id": "  "})))
        self.assertIsNone(resolve_rid_from_headers(Headers({"x-request-id": " , "})))

    def test_one_value_is_a_string(self):
        self.assertEqual(
            resolve_rid_from_headers(Headers({"x-request-id": "abc"})), "abc"
        )

    def test_blank_entries_are_dropped(self):
        for value in ("abc,", ",abc", "abc, "):
            self.assertEqual(
                resolve_rid_from_headers(Headers({"x-request-id": value})), "abc"
            )

    def test_comma_separated_values_become_a_list(self):
        self.assertEqual(
            resolve_rid_from_headers(Headers({"x-request-id": "a, b ,c"})),
            ["a", "b", "c"],
        )

    def test_repeated_lines_are_equivalent_to_one_comma_joined_line(self):
        """RFC 9110 5.3: repeated lines of a list-valued field carry one list.

        The documented way to name every sample of a batch is one line per batch
        item, so the merged and split spellings must resolve identically.
        """
        two_lines = Headers(raw=[(b"x-request-id", b"a"), (b"x-request-id", b"b")])
        self.assertEqual(resolve_rid_from_headers(two_lines), ["a", "b"])

        per_batch_item = Headers(
            raw=[(b"x-request-id", b"a-0, a-1"), (b"x-request-id", b"b-0, b-1")]
        )
        self.assertEqual(
            resolve_rid_from_headers(per_batch_item), ["a-0", "a-1", "b-0", "b-1"]
        )
        self.assertEqual(
            resolve_rid_from_headers(
                Headers({"x-request-id": "a-0, a-1, b-0, b-1"}),
            ),
            resolve_rid_from_headers(per_batch_item),
        )

    def test_wire_order_is_preserved(self):
        """Position is the only thing mapping a rid to its batch item."""
        headers = Headers(raw=[(b"x-request-id", b"b, a"), (b"x-request-id", b"c")])
        self.assertEqual(resolve_rid_from_headers(headers), ["b", "a", "c"])


if __name__ == "__main__":
    unittest.main()
