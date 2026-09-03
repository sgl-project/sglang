"""Unit tests for the Ollama-compatible serving handlers."""

import asyncio
import unittest

from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import maybe_stub_sgl_kernel

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
        return [1, 2, 3]

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


class TestOllamaServing(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
