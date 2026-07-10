"""ASGI integration coverage for HTTP liveness during request preprocessing."""

import asyncio
import threading
import unittest
from types import SimpleNamespace

import httpx

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.entrypoints.http_server import app  # noqa: E402
from sglang.srt.entrypoints.openai.serving_base import OpenAIServingBase  # noqa: E402
from sglang.srt.managers.tokenizer_manager import TokenizerManager  # noqa: E402
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _BlockingChatServing(OpenAIServingBase):
    def __init__(self, conversion_started, release_conversion):
        tokenizer_manager = TokenizerManager.__new__(TokenizerManager)
        if hasattr(tokenizer_manager, "init_request_preprocessor"):
            tokenizer_manager.init_request_preprocessor()
        tokenizer_manager.request_logger = SimpleNamespace(
            log_requests=False, log_requests_level=0
        )
        tokenizer_manager.server_args = SimpleNamespace(
            tokenizer_metrics_allowed_custom_labels=None,
        )
        super().__init__(tokenizer_manager)
        self.conversion_started = conversion_started
        self.release_conversion = release_conversion

    def _request_id_prefix(self):
        return "test-"

    def _convert_to_internal_request(self, request, raw_request=None):
        self.conversion_started.set()
        if not self.release_conversion.wait(timeout=5):
            raise TimeoutError("test did not release request conversion")
        return object(), request

    async def _handle_non_streaming_request(
        self, adapted_request, request, raw_request
    ):
        return {"status": "ok"}


class TestHTTPServerLiveness(CustomTestCase):
    def test_ping_responds_while_chat_preprocessing_is_blocked(self):
        conversion_started = threading.Event()
        ping_completed = threading.Event()
        release_conversion = threading.Event()
        ping_was_starved = threading.Event()
        conversion_did_not_start = threading.Event()
        serving = _BlockingChatServing(conversion_started, release_conversion)

        def release_after_ping():
            if not conversion_started.wait(timeout=2):
                conversion_did_not_start.set()
                release_conversion.set()
                return
            if not ping_completed.wait(timeout=1):
                ping_was_starved.set()
            release_conversion.set()

        watchdog = threading.Thread(target=release_after_ping)
        watchdog.start()

        had_previous_serving = hasattr(app.state, "openai_serving_chat")
        previous_serving = getattr(app.state, "openai_serving_chat", None)
        app.state.openai_serving_chat = serving

        async def call_chat_and_ping():
            transport = httpx.ASGITransport(app=app)
            async with httpx.AsyncClient(
                transport=transport, base_url="http://test"
            ) as client:
                chat_task = asyncio.create_task(
                    client.post(
                        "/v1/chat/completions",
                        json={
                            "model": "test-model",
                            "messages": [{"role": "user", "content": "hello"}],
                        },
                    )
                )
                await asyncio.sleep(0)
                ping_response = await client.get("/ping")
                ping_completed.set()
                chat_response = await chat_task
                return ping_response, chat_response

        try:
            ping_response, chat_response = asyncio.run(call_chat_and_ping())
        finally:
            release_conversion.set()
            watchdog.join(timeout=2)
            executor = getattr(
                serving.tokenizer_manager, "_request_preprocessor_executor", None
            )
            if executor is not None:
                executor.shutdown(wait=True)
            if had_previous_serving:
                app.state.openai_serving_chat = previous_serving
            else:
                delattr(app.state, "openai_serving_chat")

        self.assertEqual(ping_response.status_code, 200)
        self.assertEqual(chat_response.status_code, 200)
        self.assertEqual(chat_response.json(), {"status": "ok"})
        self.assertFalse(conversion_did_not_start.is_set())
        self.assertFalse(watchdog.is_alive())
        self.assertFalse(
            ping_was_starved.is_set(),
            "/ping was starved by synchronous chat request preprocessing",
        )


if __name__ == "__main__":
    unittest.main()
