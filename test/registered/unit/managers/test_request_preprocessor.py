"""Request preprocessing runs off the HTTP event loop without reordering requests."""

import asyncio
import threading
import unittest
from contextvars import ContextVar
from types import SimpleNamespace

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase  # noqa: E402
from sglang.srt.managers.request_preprocessor import (  # noqa: E402
    RequestPreprocessor,
)
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LONG_PROMPT = "x" * 100_000
SHORT_PROMPT = "hi"


class _GatedTokenizer:
    """Blocks on long prompts until released; records finish order and overlap."""

    is_fast = True

    def __init__(self):
        self.long_started = threading.Event()
        self.release_long = threading.Event()
        self.finished = []
        self.num_active = 0
        self.overlapped = False

    def __call__(self, texts, **kwargs):
        self.num_active += 1
        self.overlapped |= self.num_active > 1
        released = True
        if texts[0] == LONG_PROMPT:
            self.long_started.set()
            released = self.release_long.wait(timeout=2)
        self.num_active -= 1
        self.finished.append(texts[0])
        return {"input_ids": [[int(released)]]}


class _Handler(OpenAIServingBase):
    def _request_id_prefix(self):
        return "test-"

    def _convert_to_internal_request(self, request, raw_request=None):
        return SimpleNamespace(), request.convert()

    async def _handle_non_streaming_request(self, adapted, processed, raw_request):
        return processed


class TestRequestPreprocessor(CustomTestCase):
    def setUp(self):
        self.tokenizer = _GatedTokenizer()
        self.addCleanup(self.tokenizer.release_long.set)
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.tokenizer = self.tokenizer
        manager.async_dynamic_batch_tokenizer = None
        manager.model_config = SimpleNamespace(is_embedding_gemma=False)
        manager.request_preprocessor = RequestPreprocessor()
        self.manager = manager

    def test_long_prompt_tokenization_keeps_event_loop_responsive(self):
        """Tokenizing a long prompt must not stall other coroutines on the loop."""

        async def run():
            task = asyncio.create_task(self.manager._tokenize_texts(LONG_PROMPT))
            await asyncio.sleep(0)
            # Reached while tokenization is in flight only if the loop is free.
            self.tokenizer.release_long.set()
            return await task

        input_ids, _ = asyncio.run(run())
        self.assertEqual(input_ids, [1], "tokenization blocked the event loop")

    def test_chat_conversion_keeps_event_loop_responsive(self):
        """Chat template rendering must not stall the loop, and keeps contextvars."""
        release = threading.Event()
        self.addCleanup(release.set)
        request_context = ContextVar("request_context")
        handler = _Handler(
            SimpleNamespace(
                server_args=None,
                request_logger=SimpleNamespace(log_requests=False),
                request_preprocessor=self.manager.request_preprocessor,
            )
        )

        def convert():
            return release.wait(timeout=2), request_context.get()

        async def run():
            request_context.set("request")
            request = SimpleNamespace(stream=False, convert=convert)
            task = asyncio.create_task(handler.handle_request(request, None))
            await asyncio.sleep(0)
            release.set()
            return await task

        released, context = asyncio.run(run())
        self.assertTrue(released, "request conversion blocked the event loop")
        self.assertEqual(context, "request")

    def test_short_prompt_waits_behind_earlier_long_prompt(self):
        """A short prompt tokenized inline must not overtake an earlier long prompt
        still being tokenized, or it would reach the scheduler first."""

        released = []

        async def tokenize(prompt):
            await self.manager._tokenize_texts(prompt)
            released.append(prompt)

        async def run():
            threading.Timer(0.2, self.tokenizer.release_long.set).start()
            long_task = asyncio.create_task(tokenize(LONG_PROMPT))
            await asyncio.sleep(0)
            await asyncio.gather(long_task, tokenize(SHORT_PROMPT))

        asyncio.run(run())
        self.assertEqual(released, [LONG_PROMPT, SHORT_PROMPT])

    def test_cancelled_request_keeps_tokenizer_serialized(self):
        """Cancelling a request cannot stop its running tokenization; later
        requests must still wait for it instead of using the tokenizer
        concurrently from the event loop."""

        async def run():
            long_task = asyncio.create_task(self.manager._tokenize_texts(LONG_PROMPT))
            started = await asyncio.to_thread(self.tokenizer.long_started.wait, 2)
            self.assertTrue(started)
            long_task.cancel()
            with self.assertRaises(asyncio.CancelledError):
                await long_task
            short_task = asyncio.create_task(self.manager._tokenize_texts(SHORT_PROMPT))
            await asyncio.sleep(0.05)
            self.tokenizer.release_long.set()
            return await short_task

        self.assertEqual(asyncio.run(run()), ([1], None))
        self.assertFalse(self.tokenizer.overlapped)
        self.assertEqual(self.tokenizer.finished, [LONG_PROMPT, SHORT_PROMPT])


if __name__ == "__main__":
    unittest.main()
