"""OpenAIServingBase.handle_request conversion-offload tests.

Covers python/sglang/srt/entrypoints/openai/serving_base.py: the
_convert_to_internal_request call is awaited through
loop.run_in_executor so the event loop is not blocked by jinja rendering
and full-prompt tokenization. All heavy collaborators are stubbed; the
test verifies the offload actually happens on a worker thread and that
both the streaming and non-streaming downstream branches still receive
the converted request.
"""

import asyncio
import sys
import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[4]
PYTHON_DIR = REPO_ROOT / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.test.test_utils import CustomTestCase

_NON_STREAMING = object()
_STREAMING = object()


class _StubServing(OpenAIServingBase):
    """Concrete stub: only the collaborators handle_request touches."""

    def __init__(self):
        self.tokenizer_manager = SimpleNamespace(
            request_logger=SimpleNamespace(log_requests=False, log_requests_level=0),
        )
        self.convert_calls = []

    def _validate_request(self, request):
        return None

    @property
    def _request_id_prefix(self) -> str:
        return "test-"

    def _convert_to_internal_request(self, request, raw_request):
        self.convert_calls.append((request, raw_request, threading.get_ident()))
        return "adapted", "processed"

    async def _handle_streaming_request(
        self, adapted_request, processed_request, raw_request
    ):
        return _STREAMING

    async def _handle_non_streaming_request(
        self, adapted_request, processed_request, raw_request
    ):
        return _NON_STREAMING


class TestHandleRequestConversionOffload(CustomTestCase):
    def _run(self, request):
        serving = _StubServing()
        raw_request = SimpleNamespace()
        result = asyncio.run(serving.handle_request(request, raw_request))
        return serving, result, raw_request

    def test_non_streaming_offloads_conversion_to_worker_thread(self):
        request = SimpleNamespace()  # no `stream` attribute
        serving, result, raw_request = self._run(request)

        self.assertIs(result, _NON_STREAMING)
        self.assertEqual(len(serving.convert_calls), 1)
        req, raw, thread_id = serving.convert_calls[0]
        self.assertIs(req, request)
        self.assertIs(raw, raw_request)
        # The conversion must run off the event-loop thread.
        self.assertNotEqual(thread_id, threading.get_ident())

    def test_streaming_request_uses_converted_request(self):
        request = SimpleNamespace(stream=True)
        serving, result, _ = self._run(request)

        self.assertIs(result, _STREAMING)
        self.assertEqual(len(serving.convert_calls), 1)

    def test_event_loop_stays_responsive_during_conversion(self):
        """While the executor runs the conversion, loop-side timers still fire."""

        class _SlowConvert(_StubServing):
            def _convert_to_internal_request(self, request, raw_request):
                self.convert_calls.append((request, raw_request, threading.get_ident()))
                import time

                time.sleep(0.2)  # simulate tokenization blocking the worker
                return "adapted", "processed"

        serving = _SlowConvert()
        ticks = []

        async def main():
            task = asyncio.ensure_future(
                serving.handle_request(SimpleNamespace(), SimpleNamespace())
            )
            # A timer co-scheduled with the offloaded conversion must fire
            # well within the 0.2 s the worker is blocked.
            loop = asyncio.get_running_loop()
            t0 = loop.time()
            await asyncio.sleep(0.05)
            ticks.append(loop.time() - t0)
            return await task

        result = asyncio.run(main())
        self.assertIs(result, _NON_STREAMING)
        self.assertEqual(len(ticks), 1)
        self.assertLess(ticks[0], 0.15)


if __name__ == "__main__":
    unittest.main()
