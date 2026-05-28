"""Unit tests for the lightweight request lifecycle tracer."""

import logging
import os
import unittest

from sglang.srt.utils import request_tracer
from sglang.test.test_utils import CustomTestCase


class TestRequestTracer(CustomTestCase):
    ENV_VAR = "SGLANG_TRACE_REQUEST_LIFECYCLE"

    def setUp(self):
        self._prev_env = os.environ.pop(self.ENV_VAR, None)
        request_tracer._reset_cache_for_testing()

    def tearDown(self):
        if self._prev_env is None:
            os.environ.pop(self.ENV_VAR, None)
        else:
            os.environ[self.ENV_VAR] = self._prev_env
        request_tracer._reset_cache_for_testing()

    def _set_enabled(self, value: bool) -> None:
        os.environ[self.ENV_VAR] = "1" if value else "0"
        request_tracer._reset_cache_for_testing()

    def _capture(self):
        records: list[logging.LogRecord] = []

        class _Handler(logging.Handler):
            def emit(self, record):  # noqa: D401
                records.append(record)

        handler = _Handler(level=logging.DEBUG)
        request_tracer.logger.addHandler(handler)
        request_tracer.logger.setLevel(logging.DEBUG)
        return records, handler

    def test_disabled_by_default(self):
        self.assertFalse(request_tracer.is_request_trace_enabled())
        records, handler = self._capture()
        try:
            request_tracer.trace_req_event("rid-1", "received")
            self.assertEqual(records, [])
        finally:
            request_tracer.logger.removeHandler(handler)

    def test_enabled_emits_structured_event(self):
        self._set_enabled(True)
        self.assertTrue(request_tracer.is_request_trace_enabled())
        records, handler = self._capture()
        try:
            request_tracer.trace_req_event(
                "rid-abc",
                "prefill_start",
                stage="scheduler",
                batch_size=4,
            )
        finally:
            request_tracer.logger.removeHandler(handler)

        self.assertEqual(len(records), 1)
        msg = records[0].getMessage()
        self.assertIn("req_trace", msg)
        self.assertIn("event=prefill_start", msg)
        self.assertIn("rid=rid-abc", msg)
        self.assertIn("stage=scheduler", msg)
        self.assertIn("batch_size=4", msg)
        self.assertIn("ts=", msg)

    def test_none_fields_are_omitted(self):
        self._set_enabled(True)
        records, handler = self._capture()
        try:
            request_tracer.trace_req_event(
                "rid-x", "finished", finish_reason=None, num_output_tokens=7
            )
        finally:
            request_tracer.logger.removeHandler(handler)

        self.assertEqual(len(records), 1)
        msg = records[0].getMessage()
        self.assertNotIn("finish_reason", msg)
        self.assertIn("num_output_tokens=7", msg)

    def test_enabled_value_is_cached(self):
        self._set_enabled(True)
        self.assertTrue(request_tracer.is_request_trace_enabled())
        # Mutating the env after first read should not change the cached value.
        os.environ[self.ENV_VAR] = "0"
        self.assertTrue(request_tracer.is_request_trace_enabled())
        # Explicit reset reflects the new value.
        request_tracer._reset_cache_for_testing()
        self.assertFalse(request_tracer.is_request_trace_enabled())


if __name__ == "__main__":
    unittest.main()
