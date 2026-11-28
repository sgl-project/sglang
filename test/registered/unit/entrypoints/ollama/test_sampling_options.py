"""Ollama-compatible endpoints must forward documented sampling options.

The option conversion table in OllamaServing used to silently drop ``min_p``,
so a client that set the documented relative-cutoff option got SGLang's
default ``min_p=0`` sampling behavior instead. This regression pins the
mapping at the handler boundary (both /api/chat and /api/generate, both
streaming and non-streaming) without duplicating sampler-level tests.
"""

import asyncio
import json
import unittest
from types import SimpleNamespace

from sglang.srt.entrypoints.ollama.protocol import (
    OllamaChatRequest,
    OllamaGenerateRequest,
)
from sglang.srt.entrypoints.ollama.serving import OllamaServing
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=15, suite="base-a-test-cpu")


class _FakeTokenizerManager:
    """Engine boundary: captures the request SGLang would schedule."""

    served_model_name = "fixture"
    tokenizer = SimpleNamespace(apply_chat_template=lambda *args, **kwargs: [1, 2, 3])

    def __init__(self):
        self.captured = None

    async def generate_request(self, request, raw_request):
        self.captured = request
        yield {
            "text": "ok",
            "meta_info": {
                "finish_reason": {"type": "stop"},
                "prompt_tokens": 3,
                "completion_tokens": 1,
            },
        }


def _make_serving():
    tokenizer_manager = _FakeTokenizerManager()
    return tokenizer_manager, OllamaServing(tokenizer_manager)


async def _call(endpoint, stream, options):
    tokenizer_manager, serving = _make_serving()
    if endpoint == "chat":
        request = OllamaChatRequest.model_validate_json(
            json.dumps(
                {
                    "model": "fixture",
                    "stream": stream,
                    "messages": [{"role": "user", "content": "Hello"}],
                    "options": options,
                }
            )
        )
    else:
        request = OllamaGenerateRequest.model_validate_json(
            json.dumps(
                {
                    "model": "fixture",
                    "stream": stream,
                    "prompt": "Hello",
                    "options": options,
                }
            )
        )
    response = await getattr(serving, f"handle_{endpoint}")(request, None)
    if stream:
        async for _ in response.body_iterator:
            pass
    return tokenizer_manager.captured.sampling_params


class TestOllamaSamplingOptions(CustomTestCase):
    def assert_min_p_forwarded(self, endpoint, stream):
        sampling_params = asyncio.run(
            _call(endpoint, stream, {"min_p": 0.5, "temperature": 0.7})
        )
        self.assertEqual(sampling_params["min_p"], 0.5)
        # Existing mappings must keep working alongside min_p.
        self.assertEqual(sampling_params["temperature"], 0.7)

    def assert_min_p_disabled_by_default(self, endpoint, stream):
        for options in ({}, {"min_p": 0.0}, {"min_p": None}):
            with self.subTest(endpoint=endpoint, stream=stream, options=options):
                sampling_params = asyncio.run(_call(endpoint, stream, options))
                # SamplingParams normalizes a missing/zero/None min_p to 0.0.
                self.assertIn(sampling_params.get("min_p", 0.0) or 0.0, (0.0, None))

    def test_chat_forwards_min_p(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.assert_min_p_forwarded("chat", stream)

    def test_generate_forwards_min_p(self):
        for stream in (False, True):
            with self.subTest(stream=stream):
                self.assert_min_p_forwarded("generate", stream)

    def test_min_p_not_enabled_by_default(self):
        for endpoint in ("chat", "generate"):
            for stream in (False, True):
                self.assert_min_p_disabled_by_default(endpoint, stream)


if __name__ == "__main__":
    unittest.main(verbosity=2)
