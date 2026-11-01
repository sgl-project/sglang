"""Event-loop responsiveness tests for TokenizerManager tokenization."""

import asyncio
import threading
import types
import unittest
from contextvars import ContextVar
from unittest.mock import patch

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _BlockingFastTokenizer(PreTrainedTokenizerFast):
    def __init__(self, tokenization_started, release_tokenization):
        backend = Tokenizer(
            WordLevel({"[UNK]": 0, "hello": 1, "[PAD]": 2}, unk_token="[UNK]")
        )
        backend.pre_tokenizer = Whitespace()
        super().__init__(tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]")
        self.tokenization_started = tokenization_started
        self.release_tokenization = release_tokenization

    def __call__(self, texts, **kwargs):
        self.tokenization_started.set()
        if not self.release_tokenization.wait(timeout=5):
            raise TimeoutError("test did not release tokenization")
        return super().__call__(texts, **kwargs)


class TestTokenizerManagerEventLoop(CustomTestCase):
    def test_regular_tokenizer_does_not_block_event_loop(self):
        tokenization_started = threading.Event()
        loop_progressed = threading.Event()
        release_tokenization = threading.Event()
        loop_was_starved = threading.Event()
        tokenization_did_not_start = threading.Event()

        manager = TokenizerManager.__new__(TokenizerManager)
        manager.tokenizer = _BlockingFastTokenizer(
            tokenization_started, release_tokenization
        )
        manager.async_dynamic_batch_tokenizer = None
        # `_tokenize_texts` consults model_config for the EmbeddingGemma EOS
        # fix-up; this test builds a bare manager, so stub the flag it reads.
        manager.model_config = types.SimpleNamespace(is_embedding_gemma=False)
        if hasattr(manager, "init_tokenizer_executor"):
            manager.init_tokenizer_executor()

        def release_after_loop_progress():
            if not tokenization_started.wait(timeout=2):
                tokenization_did_not_start.set()
                release_tokenization.set()
                return
            if not loop_progressed.wait(timeout=1):
                loop_was_starved.set()
            release_tokenization.set()

        watchdog = threading.Thread(target=release_after_loop_progress)
        watchdog.start()

        async def tokenize_and_probe_loop():
            tokenize_task = asyncio.create_task(manager._tokenize_texts("hello"))
            await asyncio.sleep(0)
            loop_progressed.set()
            return await tokenize_task

        try:
            input_ids, token_type_ids = asyncio.run(tokenize_and_probe_loop())
        finally:
            release_tokenization.set()
            watchdog.join(timeout=2)
            executor = getattr(manager, "_tokenizer_executor", None)
            if executor is not None:
                executor.shutdown(wait=True)

        self.assertEqual(input_ids, [1])
        self.assertIsNone(token_type_ids)
        self.assertFalse(tokenization_did_not_start.is_set())
        self.assertFalse(watchdog.is_alive())
        self.assertFalse(
            loop_was_starved.is_set(),
            "synchronous tokenization blocked the API event loop",
        )

    @unittest.skipUnless(
        hasattr(TokenizerManager, "_regular_tokenizer_for_worker"),
        "requires isolated regular tokenization",
    )
    def test_worker_backend_is_isolated_and_wrapper_settings_are_current(self):
        started, release = threading.Event(), threading.Event()
        release.set()
        tokenizer = _BlockingFastTokenizer(started, release)
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.tokenizer = tokenizer
        manager.async_dynamic_batch_tokenizer = None
        manager.model_config = types.SimpleNamespace(is_embedding_gemma=False)
        manager.init_tokenizer_executor()
        self.addCleanup(manager._tokenizer_executor.shutdown, wait=True)
        # Configuration applied after manager startup must reach the worker.
        tokenizer.padding_side = "left"
        with patch("copy.deepcopy", side_effect=AssertionError("request-time copy")):
            worker = manager._regular_tokenizer_for_worker()
            self.assertIsNot(worker.backend_tokenizer, tokenizer.backend_tokenizer)
            self.assertEqual(worker.padding_side, "left")
            worker(["hello", "hello hello"], padding=True)
            self.assertIsNone(tokenizer.backend_tokenizer.padding)
            self.assertEqual(
                asyncio.run(manager._tokenize_texts(["hello", "hello hello"])),
                ([[1], [1, 1]], None),
            )
        self.assertIs(manager.tokenizer, tokenizer)
        # A replacement must not keep using the old vocabulary.
        replacement = Tokenizer(WordLevel({"[UNK]": 0, "hello": 3}, unk_token="[UNK]"))
        manager.tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=replacement, unk_token="[UNK]"
        )
        self.assertIsNone(manager._regular_tokenizer_for_worker())
        self.assertEqual(asyncio.run(manager._tokenize_texts("hello")), ([3], None))

    def test_custom_tokenizers_keep_the_synchronous_path(self):
        loop_thread = threading.get_ident()
        calls = []

        def encode(text):
            calls.append(threading.get_ident())
            return [1]

        def batch_encode(texts, **kwargs):
            calls.append(threading.get_ident())
            return {"input_ids": [[1] for _ in texts]}

        for is_fast in (False, True):
            with self.subTest(is_fast=is_fast):

                class CustomTokenizer:
                    def __call__(self, texts, **kwargs):
                        return batch_encode(texts, **kwargs)

                tokenizer = CustomTokenizer()
                tokenizer.is_fast = is_fast
                tokenizer.encode = encode
                manager = TokenizerManager.__new__(TokenizerManager)
                manager.tokenizer = tokenizer
                manager.async_dynamic_batch_tokenizer = None
                manager.model_config = types.SimpleNamespace(is_embedding_gemma=False)
                if hasattr(manager, "init_tokenizer_executor"):
                    manager.init_tokenizer_executor()
                    self.addCleanup(manager._tokenizer_executor.shutdown, wait=True)
                self.assertEqual(
                    asyncio.run(manager._tokenize_texts("hello")), ([1], None)
                )
        self.assertEqual(calls, [loop_thread, loop_thread])

    @unittest.skipUnless(
        hasattr(TokenizerManager, "init_tokenizer_executor"),
        "requires the tokenizer executor",
    )
    def test_request_preprocessing_is_serialized(self):
        first_started = threading.Event()
        release_first = threading.Event()
        second_started = threading.Event()
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.tokenizer = None
        manager.init_tokenizer_executor()

        def first_call():
            first_started.set()
            if not release_first.wait(timeout=5):
                raise TimeoutError("test did not release first preprocessing call")

        def second_call():
            second_started.set()

        async def run_concurrent_calls():
            first_task = asyncio.create_task(
                manager.run_in_tokenizer_executor(first_call)
            )
            started = await asyncio.to_thread(first_started.wait, 1)
            self.assertTrue(started)

            second_task = asyncio.create_task(
                manager.run_in_tokenizer_executor(second_call)
            )
            await asyncio.sleep(0.05)
            self.assertFalse(second_started.is_set())

            release_first.set()
            await asyncio.gather(first_task, second_task)

        try:
            asyncio.run(run_concurrent_calls())
        finally:
            release_first.set()
            manager._tokenizer_executor.shutdown(wait=True)

        self.assertTrue(second_started.is_set())

    @unittest.skipUnless(
        hasattr(TokenizerManager, "init_tokenizer_executor"),
        "requires the tokenizer executor",
    )
    def test_request_preprocessing_preserves_context(self):
        request_context = ContextVar("request_context")
        manager = TokenizerManager.__new__(TokenizerManager)
        manager.tokenizer = None
        manager.init_tokenizer_executor()

        async def read_context_from_preprocessor():
            request_context.set("request-value")
            return await manager.run_in_tokenizer_executor(request_context.get)

        try:
            result = asyncio.run(read_context_from_preprocessor())
        finally:
            manager._tokenizer_executor.shutdown(wait=True)

        self.assertEqual(result, "request-value")


if __name__ == "__main__":
    unittest.main()
