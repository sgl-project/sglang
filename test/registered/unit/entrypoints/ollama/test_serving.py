"""Unit tests for the Ollama-compatible serving handlers."""

import asyncio
import unittest
from types import SimpleNamespace

from fastapi.responses import StreamingResponse

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from sglang.srt.entrypoints.ollama.protocol import (  # noqa: E402
    OllamaChatRequest,
    OllamaGenerateRequest,
)
from sglang.srt.entrypoints.ollama.serving import OllamaServing  # noqa: E402

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class _FakeTokenizer:
    def __init__(self):
        self.chat_template_kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.chat_template_kwargs = kwargs
        if kwargs.get("return_dict") is False:
            return [1, 2, 3]
        return {"input_ids": [1, 2, 3]}

    def encode(self, text):
        return [1, 2, 3]


class _FakeTokenizerManager:
    served_model_name = "test-model"
    context_len = 10
    num_reserved_tokens = 1

    def __init__(self):
        self.tokenizer = _FakeTokenizer()
        self.last_request = None

    def generate_request(self, request, raw_request):
        request.normalize_batch_and_arguments()
        self.last_request = request

        async def _generate():
            yield {
                "text": "answer",
                "meta_info": {
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                },
            }

        return _generate()


class TestOllamaServing(CustomTestCase):
    def setUp(self):
        self.tokenizer_manager = _FakeTokenizerManager()
        self.serving = OllamaServing(self.tokenizer_manager)

    def test_chat_uses_flat_input_ids_and_context_aware_default(self):
        request = OllamaChatRequest(
            model="test-model",
            messages=[{"role": "user", "content": "hello"}],
            stream=False,
        )

        asyncio.run(self.serving.handle_chat(request, raw_request=object()))

        self.assertEqual(
            self.tokenizer_manager.tokenizer.chat_template_kwargs["return_dict"],
            False,
        )
        self.assertEqual(self.tokenizer_manager.last_request.input_ids, [1, 2, 3])
        self.assertEqual(
            self.tokenizer_manager.last_request.sampling_params["max_new_tokens"],
            6,
        )

    def test_generate_uses_remaining_context_for_default(self):
        request = OllamaGenerateRequest(
            model="test-model",
            prompt="hello",
            stream=False,
        )

        asyncio.run(self.serving.handle_generate(request, raw_request=object()))

        self.assertEqual(
            self.tokenizer_manager.last_request.sampling_params["max_new_tokens"],
            6,
        )

    def test_explicit_num_predict_is_preserved(self):
        request = OllamaGenerateRequest(
            model="test-model",
            prompt="hello",
            options={"num_predict": 2},
            stream=False,
        )

        asyncio.run(self.serving.handle_generate(request, raw_request=object()))

        self.assertEqual(
            self.tokenizer_manager.last_request.sampling_params["max_new_tokens"],
            2,
        )

    def test_default_is_capped_at_2048(self):
        self.tokenizer_manager.context_len = 8192
        self.assertEqual(self.serving._get_default_max_new_tokens(3), 2048)

    def test_default_is_zero_when_context_is_exhausted(self):
        self.assertEqual(self.serving._get_default_max_new_tokens(9), 0)
        self.assertEqual(self.serving._get_default_max_new_tokens(12), 0)

    def test_default_uses_model_config_when_context_len_is_unavailable(self):
        self.tokenizer_manager.context_len = None
        self.tokenizer_manager.model_config = SimpleNamespace(context_len=8)
        self.assertEqual(self.serving._get_default_max_new_tokens(3), 4)

    def test_default_is_preserved_without_context_metadata(self):
        self.tokenizer_manager.context_len = None
        self.assertEqual(self.serving._get_default_max_new_tokens(3), 2048)

    def test_streaming_chat_uses_the_same_input_normalization(self):
        async def consume():
            response = await self.serving.handle_chat(
                OllamaChatRequest(
                    model="test-model",
                    messages=[{"role": "user", "content": "hello"}],
                    stream=True,
                ),
                raw_request=object(),
            )
            self.assertIsInstance(response, StreamingResponse)
            async for _ in response.body_iterator:
                pass

        asyncio.run(consume())
        self.assertEqual(self.tokenizer_manager.last_request.input_ids, [1, 2, 3])
        self.assertEqual(
            self.tokenizer_manager.last_request.sampling_params["max_new_tokens"], 6
        )


if __name__ == "__main__":
    unittest.main()
