"""Preserve SSE errors returned after successful HTTP response headers."""

import asyncio
import json
import unittest
from argparse import Namespace
from unittest.mock import AsyncMock, Mock, patch

from sglang.benchmark import serving
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestBenchServingStreamErrors(unittest.TestCase):
    def _run(self, backend, chunks):
        serving.set_global_args(
            Namespace(
                disable_stream=False,
                disable_ignore_eos=True,
                return_logprob=False,
                print_requests=False,
                tokenizer="",
                header=None,
            )
        )

        async def frames():
            for chunk in chunks:
                yield b"data: " + json.dumps(chunk).encode() + b"\n\n"
            yield b"data: [DONE]\n\n"

        response = Mock(status=200, content=frames())
        post = AsyncMock()
        post.__aenter__.return_value = response
        session = AsyncMock()
        session.__aenter__.return_value = session
        session.post = Mock(return_value=post)
        progress = Mock()
        request = serving.RequestFuncInput(
            prompt="hello",
            api_url=(
                "http://fixture/v1/chat/completions"
                if backend is serving.async_request_openai_chat_completions
                else "http://fixture/v1/completions"
            ),
            prompt_len=1,
            output_len=64,
            model="fixture",
            lora_name="",
            image_data=None,
            extra_request_body={},
        )
        with patch.object(
            serving, "_create_bench_client_session", return_value=session
        ):
            result = asyncio.run(backend(request, progress))
        progress.update.assert_called_once_with(1)
        return result

    def test_stream_error_is_a_failed_request_with_server_details(self):
        error = {
            "error": {
                "object": "error",
                "message": "The request queue is full.",
                "type": "SERVICE_UNAVAILABLE",
                "code": 503,
            }
        }
        for backend, token in self._backends():
            for chunks in ([error], [token, error]):
                with self.subTest(backend=backend.__name__, partial=len(chunks) > 1):
                    result = self._run(backend, chunks)
                    self.assertFalse(result.success)
                    self.assertIn("The request queue is full.", result.error)
                    self.assertIn("SERVICE_UNAVAILABLE", result.error)
                    self.assertIn("503", result.error)
                    self.assertNotIn("KeyError", result.error)

    def test_successful_stream_is_unchanged(self):
        for backend, token in self._backends():
            with self.subTest(backend=backend.__name__):
                result = self._run(backend, [token])
                self.assertTrue(result.success, result.error)
                self.assertEqual(result.generated_text, "hello")
                self.assertEqual(result.output_len, 1)
                self.assertEqual(result.error, "")

    @staticmethod
    def _backends():
        return [
            (
                serving.async_request_openai_completions,
                {"choices": [{"text": "hello"}], "usage": {"completion_tokens": 1}},
            ),
            (
                serving.async_request_openai_chat_completions,
                {
                    "choices": [{"delta": {"content": "hello"}}],
                    "usage": {"completion_tokens": 1},
                },
            ),
        ]


if __name__ == "__main__":
    unittest.main()
