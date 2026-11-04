"""HTTP parsing and bounded ACK-race recovery; transport is entirely mocked."""

import io
import json
import tempfile
import unittest
import urllib.error
from pathlib import Path
from unittest.mock import patch

import lifecycle_bench as bench
from router_bench import Router
from session_kv_bench import HTTP, BenchError, HTTPBenchError, Journal


class Response(io.BytesIO):
    status = 200


class RouterDrainTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.path = Path(self.directory.name, "raw.jsonl")
        self.journal = Journal(self.path)
        self.clock, self.calls = 0, []
        self.views = [True, False, True]
        self.refusal = {
            "success": False,
            "error": {"code": "rejected", "reason": "operation/claim not quiescent"},
        }
        self.accepted = {"success": True, "data": {"acknowledged": True}}
        self.acks = [(400, self.refusal), (200, self.accepted)]

    def tearDown(self):
        self.journal.file.close()
        self.directory.cleanup()

    def urlopen(self, request, timeout):
        self.clock += 0.01
        if request.data is None:
            value = {
                "success": True,
                "data": {"enabled": True, "lifecycle": "explicit"},
            }
            return Response(json.dumps(value).encode())
        command = json.loads(request.data)
        kind = command["kind"]
        self.calls.append(kind)
        if len(self.calls) > 30:
            raise AssertionError(
                "cleanup retry is not bounded by the original deadline"
            )
        code = 200
        if kind == "query":
            ready = self.views.pop(0) if len(self.views) > 1 else self.views[0]
            value = {
                "success": True,
                "data": {"quiescent": ready, "operations": [{"quiescent": ready}]},
            }
        elif kind == "ack":
            code, value = self.acks.pop(0) if len(self.acks) > 1 else self.acks[0]
        else:
            self.assertEqual(kind, "cancel")
            value = {"success": True, "data": {"quiescent": False}}
        raw = (
            json.dumps(value).encode() if not isinstance(value, str) else value.encode()
        )
        if code != 200:
            raise urllib.error.HTTPError(
                request.full_url, code, "fixture", {}, io.BytesIO(raw)
            )
        return Response(raw)

    def sleep(self, seconds):
        self.clock += seconds

    def drain(self, timeout=0.2):
        with (
            patch("urllib.request.urlopen", side_effect=self.urlopen),
            patch.object(bench.time, "monotonic", side_effect=lambda: self.clock),
            patch.object(bench.time, "sleep", side_effect=self.sleep),
        ):
            http = HTTP("http://fixture", self.journal, 1)
            router = Router(http, http, poll_ms=1)
            return bench.drain_router(
                router, {"session_id": "task"}, "prepare", "trial", timeout
            )

    def test_transient_ack_refusal_requeries_and_preserves_raw_failure(self):
        view = self.drain()
        self.assertTrue(view["quiescent"])
        self.assertEqual(
            self.calls, ["cancel", "query", "ack", "query", "query", "ack"]
        )
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        failures = [row for row in rows if row.get("status_code") == 400]
        self.assertEqual(len(failures), 1)
        self.assertEqual(json.loads(failures[0]["response"]), self.refusal)
        self.assertEqual(failures[0]["request"]["kind"], "ack")
        cancel = next(
            row for row in rows if (row.get("request") or {}).get("kind") == "cancel"
        )
        self.assertGreaterEqual(
            rows[-1]["monotonic"] - cancel["started_monotonic"], 0.06
        )
        self.assertGreater(self.clock, 0.07)

    def test_persistent_ack_refusal_expires_without_recancelling(self):
        self.views, self.acks = [True], [(400, self.refusal)]
        with self.assertRaisesRegex(BenchError, "did not become quiescent"):
            self.drain(timeout=0.055)
        self.assertEqual(self.calls.count("cancel"), 1)
        self.assertGreaterEqual(self.calls.count("ack"), 2)
        self.assertLessEqual(len(self.calls), 9)
        rows = [json.loads(line) for line in self.path.read_text().splitlines()]
        self.assertEqual(
            sum(row.get("status_code") == 400 for row in rows), self.calls.count("ack")
        )

    def test_other_errors_and_invalid_ack_do_not_retry(self):
        cases = [
            (
                400,
                {
                    "success": False,
                    "error": {"code": "rejected", "reason": "unknown intent"},
                },
            ),
            (
                400,
                {
                    "success": False,
                    "error": {
                        "code": "capacity",
                        "reason": "operation/claim not quiescent",
                    },
                },
            ),
            (500, self.refusal),
            (400, "operation/claim not quiescent"),
            (
                400,
                {
                    "success": False,
                    "error": {
                        "code": "rejected",
                        "reason": "operation/claim not quiescent yet",
                    },
                },
            ),
            (200, self.refusal),
            (200, {"success": True, "data": {"acknowledged": False}}),
            (200, {"success": True, "data": {"acknowledged": 1}}),
            (200, {"success": True, "data": None}),
        ]
        for code, reply in cases:
            with self.subTest(code=code, reply=reply):
                self.calls, self.views, self.acks = [], [True], [(code, reply)]
                with self.assertRaises(BenchError) as raised:
                    self.drain()
                self.assertEqual(self.calls, ["cancel", "query", "ack"])
                if code != 200:
                    self.assertIsInstance(raised.exception, HTTPBenchError)
                    self.assertEqual(raised.exception.status_code, code)


if __name__ == "__main__":
    unittest.main()
