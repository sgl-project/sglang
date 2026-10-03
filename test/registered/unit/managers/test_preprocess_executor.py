"""Request preprocessing runs off the HTTP event loop without reordering requests."""

import asyncio
import threading
import unittest
from contextvars import ContextVar
from types import SimpleNamespace
from unittest import mock

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.openai.protocol import ChatCompletionRequest  # noqa: E402
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase  # noqa: E402
from sglang.srt.entrypoints.openai.serving_chat import (  # noqa: E402
    OpenAIServingChat,
)
from sglang.srt.managers.preprocess_executor import PreprocessExecutor  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

LONG_PROMPT = "x" * 100_000
SHORT_PROMPT = "hi"


class _GatedTokenizer:
    """Blocks on long prompts until released; records finish order."""

    is_fast = True

    def __init__(self):
        self.long_started = threading.Event()
        self.release_long = threading.Event()
        self.finished = []

    def __call__(self, texts, **kwargs):
        released = True
        if texts[0] == LONG_PROMPT:
            self.long_started.set()
            released = self.release_long.wait(timeout=2)
        self.finished.append(texts[0])
        return {"input_ids": [[int(released)]]}


def _fast_tokenizer():
    backend = Tokenizer(models.WordLevel({"[UNK]": 0, "hi": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[UNK]"
    )


class _Handler(OpenAIServingBase):
    def _request_id_prefix(self):
        return "test-"

    def _convert_to_internal_request(self, request, raw_request=None):
        return SimpleNamespace(), request.convert()

    async def _handle_non_streaming_request(self, adapted, processed, raw_request):
        return processed


def _make_manager(tokenizer):
    manager = TokenizerManager.__new__(TokenizerManager)
    manager.tokenizer = tokenizer
    manager.async_dynamic_batch_tokenizer = None
    manager.model_config = SimpleNamespace(is_embedding_gemma=False)
    manager.preprocess_executor = PreprocessExecutor(
        get_shared_tokenizer=lambda: manager._tokenizer
    )
    return manager


class TestPreprocessExecutor(CustomTestCase):
    def setUp(self):
        self.tokenizer = _GatedTokenizer()
        self.addCleanup(self.tokenizer.release_long.set)
        self.manager = _make_manager(self.tokenizer)

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
                preprocess_executor=self.manager.preprocess_executor,
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

    def test_cancelled_request_keeps_requests_ordered(self):
        """Cancelling a request cannot stop its running tokenization; a later
        short request must still wait for it instead of tokenizing inline and
        reaching the scheduler first."""

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
        self.assertEqual(self.tokenizer.finished, [LONG_PROMPT, SHORT_PROMPT])

    def test_jobs_never_borrow_the_event_loops_backend(self):
        """The event loop keeps calling the shared tokenizer (multimodal processor
        calls with padding=True) while jobs run. A fast tokenizer's Rust backend
        raises "Already borrowed", or blocks the loop, when two threads use it at
        once, so jobs must run on a backend of their own."""
        tokenizer = _fast_tokenizer()
        manager = _make_manager(tokenizer)

        async def run():
            return await manager.preprocess_executor.run(
                lambda: (manager.tokenizer.backend_tokenizer, manager.tokenizer("hi"))
            )

        job_backend, encoded = asyncio.run(run())
        self.assertIsNot(job_backend, tokenizer.backend_tokenizer)
        self.assertEqual(encoded["input_ids"], tokenizer("hi")["input_ids"])
        self.assertIs(manager.tokenizer, tokenizer)

    def test_jobs_see_tokenizer_attributes_set_after_startup(self):
        """Startup sets the chat template after TokenizerManager init, so a job's
        tokenizer must not be a snapshot taken at construction."""
        tokenizer = _fast_tokenizer()
        manager = _make_manager(tokenizer)
        tokenizer.chat_template = "{{ messages }}"

        async def run():
            return await manager.preprocess_executor.run(
                lambda: manager.tokenizer.chat_template
            )

        self.assertEqual(asyncio.run(run()), "{{ messages }}")

    def test_requests_never_copy_a_tokenizer_backend(self):
        """Copying a real backend takes hundreds of ms and holds the GIL; doing it
        for a request, even only the first one, stalls the event loop."""
        manager = _make_manager(_fast_tokenizer())

        async def run():
            return [
                await manager.preprocess_executor.run(
                    lambda: threading.current_thread().name
                )
                for _ in range(2)
            ]

        with mock.patch("copy.deepcopy", side_effect=AssertionError("copied")):
            threads = asyncio.run(run())
        for name in threads:
            self.assertTrue(name.startswith("sglang-preprocess"))

    def test_only_short_plain_text_chats_skip_the_worker(self):
        """A chat is converted on the loop only when its rendering is cheap; a
        long conversation, or one whose template output its text length does
        not bound (tools, content parts), must go to the worker."""

        def cheap(**fields):
            request = ChatCompletionRequest(model="m", **fields)
            return OpenAIServingChat._is_cheap_to_preprocess(None, request)

        short = [{"role": "user", "content": SHORT_PROMPT}]
        tool = {"type": "function", "function": {"name": "f", "parameters": {}}}
        parts = [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]
        self.assertTrue(cheap(messages=short))
        self.assertFalse(cheap(messages=[{"role": "user", "content": LONG_PROMPT}]))
        self.assertFalse(cheap(messages=short, tools=[tool]))
        self.assertFalse(cheap(messages=parts))


if __name__ == "__main__":
    unittest.main()
