import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from fastapi import HTTPException
from starlette.datastructures import Headers

from sglang.srt.entrypoints.http_server import generate_request
from sglang.srt.entrypoints.request_headers import (
    apply_header_overrides,
    extract_routed_dp_rank,
)
from sglang.srt.managers.io_struct import GenerateReqInput
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="stage-a-test-cpu-intel")


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


if __name__ == "__main__":
    unittest.main()


class TestExtractRoutedDpRank(CustomTestCase):
    def test_no_header_keeps_body_value(self):
        self.assertIsNone(extract_routed_dp_rank(Headers({}), None))
        self.assertEqual(extract_routed_dp_rank(Headers({}), 2), 2)
        self.assertEqual(extract_routed_dp_rank(None, 2), 2)

    def test_header_is_read_case_insensitively(self):
        self.assertEqual(
            extract_routed_dp_rank(Headers({"X-Data-Parallel-Rank": "3"}), None), 3
        )

    def test_header_overrides_body(self):
        self.assertEqual(
            extract_routed_dp_rank(Headers({"x-data-parallel-rank": "3"}), 1), 3
        )

    def test_non_integer_header_is_rejected(self):
        with self.assertRaises(HTTPException) as ctx:
            extract_routed_dp_rank(Headers({"x-data-parallel-rank": "abc"}), None)
        self.assertEqual(ctx.exception.status_code, 400)


class TestGenerateReadsRoutedDpRankHeader(CustomTestCase):
    """A header pin on /generate is a 200 whether or not it is honoured; only
    the routed_dp_rank handed to the tokenizer manager shows the difference."""

    def _generate(self, obj, headers):
        captured = {}

        async def fake_generate_request(req, raw_request):
            captured["obj"] = req
            yield {"text": "ok", "meta_info": {}}

        state = SimpleNamespace(
            tokenizer_manager=SimpleNamespace(generate_request=fake_generate_request)
        )
        request = SimpleNamespace(headers=Headers(headers))
        with patch("sglang.srt.entrypoints.http_server._global_state", state):
            asyncio.run(generate_request(obj, request))
        return captured["obj"]

    def test_header_pins_the_rank(self):
        obj = GenerateReqInput(text="hi", routed_dp_rank=0)
        self.assertEqual(
            self._generate(obj, {"x-data-parallel-rank": "3"}).routed_dp_rank, 3
        )

    def test_no_header_keeps_body_rank(self):
        obj = GenerateReqInput(text="hi", routed_dp_rank=1)
        self.assertEqual(self._generate(obj, {}).routed_dp_rank, 1)
