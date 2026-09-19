import asyncio
import json
import threading
import unittest
from contextvars import ContextVar
from types import SimpleNamespace

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase
from sglang.srt.managers.tokenizer_manager import TokenizerManager
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class Handler(OpenAIServingBase):
    def _request_id_prefix(self):
        return "test"

    def _validate_request(self, request):
        return getattr(request, "validation_error", None)

    def _convert_to_internal_request(self, request, raw_request=None):
        return SimpleNamespace(), request.convert()

    async def _handle_non_streaming_request(self, adapted, processed, raw_request):
        return processed


class TestRequestPreprocessing(CustomTestCase):
    def setUp(self):
        self.manager = TokenizerManager.__new__(TokenizerManager)
        self.manager.init_request_preprocessor()
        self.addCleanup(self.manager._request_preprocessor_executor.shutdown)
        self.manager.server_args = None
        self.manager.request_logger = SimpleNamespace(log_requests=False)

    def test_conversion_keeps_event_loop_responsive(self):
        """A blocked conversion must not prevent another coroutine from running."""
        release = threading.Event()
        context = ContextVar("request_context", default=None)

        async def run():
            started = asyncio.Event()
            loop = asyncio.get_running_loop()

            def convert():
                loop.call_soon_threadsafe(started.set)
                if not release.wait(2):
                    raise TimeoutError("conversion blocked the event loop")
                return context.get()

            context.set("request-value")
            task = asyncio.create_task(
                Handler(self.manager).handle_request(
                    SimpleNamespace(stream=False, convert=convert), None
                )
            )
            try:
                await asyncio.wait_for(started.wait(), 3)
                release.set()
                self.assertEqual(await task, "request-value")
            finally:
                release.set()
                await task

        asyncio.run(run())

    def test_cancellation_preserves_serialization(self):
        """Cancelling a waiter must not let the next job overlap its running work."""
        release = threading.Event()
        order = []

        async def run():
            loop = asyncio.get_running_loop()
            started = asyncio.Event()

            def first():
                loop.call_soon_threadsafe(started.set)
                if not release.wait(2):
                    raise TimeoutError("worker not released")
                order.append("first")
                raise ValueError("cancelled request failed")

            task = asyncio.create_task(self.manager.run_in_request_preprocessor(first))
            try:
                await asyncio.wait_for(started.wait(), 3)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                queued = asyncio.create_task(
                    self.manager.run_in_request_preprocessor(order.append, "cancelled")
                )
                await asyncio.sleep(0)
                queued.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await queued
                second = asyncio.create_task(
                    self.manager.run_in_request_preprocessor(order.append, "second")
                )
                await asyncio.sleep(0)
                release.set()
                await second
                self.assertEqual(order, ["first", "second"])
            finally:
                release.set()

        asyncio.run(run())

    def test_validation_and_conversion_errors_remain_bad_requests(self):
        def invalid_conversion():
            raise ValueError("invalid conversion")

        for request, message in (
            (SimpleNamespace(validation_error="invalid schema"), "invalid schema"),
            (SimpleNamespace(convert=invalid_conversion), "invalid conversion"),
        ):
            with self.subTest(message=message):
                response = asyncio.run(
                    Handler(self.manager).handle_request(request, None)
                )
                self.assertEqual(response.status_code, 400)
                self.assertEqual(json.loads(response.body)["message"], message)

    def test_fallback_tokenization_runs_off_loop(self):
        main_thread = threading.get_ident()

        def encode(text):
            self.assertNotEqual(threading.get_ident(), main_thread)
            return [len(text)]

        class Tokenizer:
            def __call__(self, texts, **kwargs):
                return {"input_ids": [encode(text) for text in texts]}

        self.manager.async_dynamic_batch_tokenizer = None
        self.manager.model_config = SimpleNamespace(is_embedding_gemma=False)
        self.manager.tokenizer = Tokenizer()
        self.manager.tokenizer.encode = encode
        for is_fast in (False, True):
            with self.subTest(is_fast=is_fast):
                self.manager.tokenizer.is_fast = is_fast
                self.assertEqual(
                    asyncio.run(self.manager._tokenize_texts("hello")), ([5], None)
                )


if __name__ == "__main__":
    unittest.main()
