import asyncio
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

    def test_conversion_stays_off_loop_after_cancellation(self):
        """A cancelled HTTP waiter must not interrupt or overlap its conversion."""
        release = threading.Event()
        self.addCleanup(release.set)
        context = ContextVar("request_context", default=None)
        completed = []

        async def run():
            loop = asyncio.get_running_loop()
            started = asyncio.Event()

            def convert():
                loop.call_soon_threadsafe(started.set)
                if not release.wait(2):
                    raise TimeoutError("conversion blocked the event loop")
                completed.append(context.get())

            context.set("first")
            task = asyncio.create_task(
                Handler(self.manager).handle_request(
                    SimpleNamespace(stream=False, convert=convert), None
                )
            )
            try:
                await asyncio.wait_for(started.wait(), 3)
                task.cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await task
                second = asyncio.create_task(
                    self.manager.run_in_request_preprocessor(completed.append, "second")
                )
                await asyncio.sleep(0)
                release.set()
                await second
                self.assertEqual(completed, ["first", "second"])
            finally:
                release.set()
                await asyncio.gather(task, return_exceptions=True)

        asyncio.run(run())

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
