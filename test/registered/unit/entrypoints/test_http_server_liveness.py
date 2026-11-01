"""ASGI integration coverage for HTTP liveness during regular tokenization."""

import asyncio
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import httpx
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerFast

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints import http_server  # noqa: E402
from sglang.srt.managers.tokenizer_manager import (  # noqa: E402
    ServerStatus,
    TokenizerManager,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _BlockingTokenizer(PreTrainedTokenizerFast):
    def __init__(self, started, release):
        backend = Tokenizer(WordLevel({"[UNK]": 0, "hello": 1}, unk_token="[UNK]"))
        super().__init__(tokenizer_object=backend, unk_token="[UNK]")
        self.started = started
        self.release = release

    def __call__(self, texts, **kwargs):
        self.started.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release tokenization")
        return super().__call__(texts, **kwargs)


class _TokenizingManager(TokenizerManager):
    """Exercise real tokenization without launching an inference engine."""

    def __init__(self, started, release):
        self.tokenizer = _BlockingTokenizer(started, release)
        self.async_dynamic_batch_tokenizer = None
        self.model_config = SimpleNamespace(is_embedding_gemma=False)
        self.server_status = ServerStatus.Up
        self.gracefully_exit = False
        if hasattr(self, "init_tokenizer_executor"):
            self.init_tokenizer_executor()

    async def generate_request(self, obj, request):
        input_ids, _ = await self._tokenize_texts(obj.text)
        yield {"input_ids": input_ids}


class TestHTTPServerLiveness(CustomTestCase):
    def test_ping_responds_while_generate_tokenization_is_blocked(self):
        tokenization_started = threading.Event()
        ping_completed = threading.Event()
        release_tokenization = threading.Event()
        ping_was_starved = threading.Event()
        tokenization_did_not_start = threading.Event()
        manager = _TokenizingManager(tokenization_started, release_tokenization)

        def release_after_ping():
            if not tokenization_started.wait(timeout=2):
                tokenization_did_not_start.set()
                release_tokenization.set()
                return
            if not ping_completed.wait(timeout=1):
                ping_was_starved.set()
            release_tokenization.set()

        watchdog = threading.Thread(target=release_after_ping)
        watchdog.start()

        async def call_generate_and_ping():
            transport = httpx.ASGITransport(app=http_server.app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                generate_task = asyncio.create_task(
                    client.post("/generate", json={"text": "hello"})
                )
                # Queue liveness before tokenization gets its first loop turn.
                ping_task = asyncio.create_task(client.get("/ping"))
                ping_response = await ping_task
                ping_completed.set()
                generate_response = await generate_task
                return ping_response, generate_response

        try:
            with patch.object(
                http_server,
                "_global_state",
                http_server._GlobalState(manager, None, {}),
            ):
                ping_response, generate_response = asyncio.run(call_generate_and_ping())
        finally:
            release_tokenization.set()
            watchdog.join(timeout=2)
            executor = getattr(manager, "_tokenizer_executor", None)
            if executor is not None:
                executor.shutdown(wait=True)

        self.assertEqual(ping_response.status_code, 200)
        self.assertEqual(generate_response.status_code, 200)
        self.assertEqual(generate_response.json(), {"input_ids": [1]})
        self.assertFalse(tokenization_did_not_start.is_set())
        self.assertFalse(watchdog.is_alive())
        self.assertFalse(
            ping_was_starved.is_set(),
            "/ping was starved by synchronous regular tokenization",
        )


if __name__ == "__main__":
    unittest.main()
