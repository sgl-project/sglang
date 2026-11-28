"""Unit tests for srt/entrypoints/ollama/serving.py — the Ollama `format`
field (Structured Outputs) must reach the native grammar constraint instead
of being silently dropped before inference."""

import asyncio
import json
import unittest

from sglang.srt.entrypoints.ollama.protocol import (
    OllamaChatRequest,
    OllamaGenerateRequest,
    OllamaMessage,
)
from sglang.srt.entrypoints.ollama.serving import OllamaServing
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class _StubTokenizer:
    def apply_chat_template(self, messages, **kwargs):
        return [101, 202, 303]


class _StubTokenizerManager:
    served_model_name = "stub-model"

    def __init__(self):
        self.tokenizer = _StubTokenizer()
        self.captured = None

    def generate_request(self, gen_request, raw_request):
        self.captured = gen_request

        async def gen():
            yield {
                "text": "{}",
                "meta_info": {
                    "finish_reason": "stop",
                    "prompt_tokens": 3,
                    "completion_tokens": 1,
                },
            }

        return gen()


class TestOllamaFormatForwardToJsonSchema(CustomTestCase):
    def setUp(self):
        self.tm = _StubTokenizerManager()
        self.serving = OllamaServing(self.tm)

    def _chat(self, **kwargs):
        request = OllamaChatRequest(
            model="stub-model",
            messages=[OllamaMessage(role="user", content="Give me JSON.")],
            stream=False,
            **kwargs,
        )
        return self.serving.handle_chat(request, raw_request=None)

    def _generate(self, **kwargs):
        request = OllamaGenerateRequest(
            model="stub-model", prompt="Give me JSON.", stream=False, **kwargs
        )
        return self.serving.handle_generate(request, raw_request=None)

    def test_chat_format_json_uses_object_schema(self):
        asyncio.run(self._chat(format="json"))
        sp = self.tm.captured.sampling_params
        self.assertEqual(sp["json_schema"], '{"type": "object"}')

    def test_chat_format_dict_is_serialized_verbatim(self):
        schema = {
            "type": "object",
            "properties": {
                "name": {"type": "string", "enum": ["日本", "Éire"]},
                "age": {"type": "integer"},
            },
            "required": ["name"],
            "additionalProperties": False,
            "$defs": {"name": {"type": "string"}},
        }
        snapshot = json.loads(json.dumps(schema))

        asyncio.run(self._chat(format=schema, options={"temperature": 0.7}))
        sp = self.tm.captured.sampling_params
        self.assertEqual(json.loads(sp["json_schema"]), snapshot)
        # Existing options still map natively alongside the constraint.
        self.assertEqual(sp["temperature"], 0.7)
        self.assertEqual(sp["max_new_tokens"], 2048)
        # The request payload is not mutated by serialization.
        self.assertEqual(schema, snapshot)

    def test_chat_format_empty_dict_is_a_real_constraint(self):
        asyncio.run(self._chat(format={}))
        sp = self.tm.captured.sampling_params
        self.assertEqual(json.loads(sp["json_schema"]), {})
        # A truthiness guard (`if format:`) would drop this case and is wrong.

    def test_chat_format_array_schema_is_not_forced_to_object(self):
        asyncio.run(self._chat(format={"type": "array", "items": {"type": "integer"}}))
        sp = self.tm.captured.sampling_params
        self.assertEqual(
            json.loads(sp["json_schema"]),
            {"type": "array", "items": {"type": "integer"}},
        )

    def test_generate_format_dict_is_serialized(self):
        schema = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }

        asyncio.run(self._generate(format=schema, options={"num_predict": 32}))
        sp = self.tm.captured.sampling_params
        self.assertEqual(json.loads(sp["json_schema"]), schema)
        self.assertEqual(sp["max_new_tokens"], 32)

    def test_generate_format_json_and_empty_dict(self):
        asyncio.run(self._generate(format="json"))
        self.assertEqual(
            self.tm.captured.sampling_params["json_schema"], '{"type": "object"}'
        )

        asyncio.run(self._generate(format={}))
        self.assertEqual(
            json.loads(self.tm.captured.sampling_params["json_schema"]), {}
        )

    def test_omitted_or_null_format_leaves_no_constraint(self):
        for kwargs in ({}, {"format": None}):
            with self.subTest(kwargs=kwargs):
                asyncio.run(self._chat(**kwargs))
                sp = self.tm.captured.sampling_params
                self.assertIsNone(sp.get("json_schema"))
                self.assertEqual(sp["max_new_tokens"], 2048)

                asyncio.run(self._generate(**kwargs))
                sp = self.tm.captured.sampling_params
                self.assertIsNone(sp.get("json_schema"))


if __name__ == "__main__":
    unittest.main()
