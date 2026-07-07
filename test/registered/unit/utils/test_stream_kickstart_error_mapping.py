"""Unit tests for the streaming-kickstart exception → HTTP status mapping.

Covers create_error_response_for_stream_kickstart(): the helper that maps
a pre-first-chunk exception to a proper HTTP status code so streaming
endpoints don't emit an SSE-body error under a HTTP 200 header when the
engine wants to signal 503 / 429 / etc.

Usage:
    python3 -m pytest test/registered/unit/utils/test_stream_kickstart_error_mapping.py -v
"""

import unittest
from unittest.mock import MagicMock

from fastapi import HTTPException

from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _TestServing(OpenAIServingBase):
    # OpenAIServingBase is abstract; stub the two required methods so we
    # can instantiate it for testing the error-mapping helper.
    def _convert_to_internal_request(self, request):
        raise NotImplementedError

    def _request_id_prefix(self):
        return "test"


class TestStreamKickstartErrorMapping(unittest.TestCase):
    def _make_serving(self):
        return _TestServing(MagicMock())

    def test_value_error_maps_to_400(self):
        # Existing behavior: ValueError → HTTP 400 (backward compat with
        # the narrow kickstart that only handled ValueError).
        serving = self._make_serving()
        resp = serving.create_error_response_for_stream_kickstart(
            ValueError("bad prompt shape")
        )
        self.assertEqual(resp.status_code, 400)

    def test_http_exception_preserves_status_code_503(self):
        # The whole point: engine raises HTTPException(503) pre-first-chunk,
        # we return a real HTTP 503 instead of a 200 + SSE-body error.
        serving = self._make_serving()
        resp = serving.create_error_response_for_stream_kickstart(
            HTTPException(status_code=503, detail="engine overloaded")
        )
        self.assertEqual(resp.status_code, 503)

    def test_http_exception_preserves_429(self):
        # Rate limiting is another common pre-first-chunk case.
        serving = self._make_serving()
        resp = serving.create_error_response_for_stream_kickstart(
            HTTPException(status_code=429, detail="rate limit exceeded")
        )
        self.assertEqual(resp.status_code, 429)

    def test_generic_exception_falls_back_to_500(self):
        # Anything we didn't anticipate should still be a real HTTP error,
        # not silently become an SSE-body message under HTTP 200.
        serving = self._make_serving()
        resp = serving.create_error_response_for_stream_kickstart(
            RuntimeError("unexpected engine state")
        )
        self.assertEqual(resp.status_code, 500)

    def test_stop_async_iteration_maps_to_500(self):
        # Edge case: generator terminates without yielding anything. Old
        # narrow ``except ValueError`` let this bubble to FastAPI's default
        # 500. ``except Exception`` catches it too (StopAsyncIteration is
        # Exception-derived), and we return the same 500 status with a
        # structured body — strict improvement, not a regression.
        serving = self._make_serving()
        resp = serving.create_error_response_for_stream_kickstart(StopAsyncIteration())
        self.assertEqual(resp.status_code, 500)


if __name__ == "__main__":
    unittest.main()
