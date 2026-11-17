"""Structured outputs for Muse Glimmer through the OpenAI server.

Muse ends reasoning with `<|eom|>` and then writes `<|start|>assistant to=user<|message|>`
before the answer. The JSON grammar must start after that header; started at `<|eom|>`,
the model can only pad whitespace or write the header text into a JSON key.
"""

import json
import os
import unittest

import jsonschema
import openai

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    terminate_and_kill_process_tree,
)

register_cuda_ci(est_time=600, stage="nightly", runner_config="1-gpu-large")

MODEL = os.environ.get("MUSE_GLIMMER_MODEL", "meta-models/Muse-Glimmer-30B")
COUNTRY_SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "capital": {"type": "string"},
        "continent": {"type": "string"},
        "population_millions": {"type": "number"},
    },
    "required": ["name", "capital", "continent", "population_millions"],
    "additionalProperties": False,
}


# GSM8K questions where, with the grammar started at `<|eom|>`, greedy decoding pads
# whitespace until max tokens instead of answering.
GSM8K_QUESTIONS = [
    "Miguel uses 2 pads of paper a week for his drawing. If there are 30 sheets of paper on a pad of paper, how many sheets of paper does he use every month?",
    "At a certain grade level, three-fourths of students have a desktop computer at home. If 20 students do not have desktop computers, how many students are there at that grade level?",
    "Comet Halley orbits the sun every 75 years. Bill's dad saw the Comet when he was 30 years old. Bill saw the comet a second time when he was three times the age his father was when he saw the Comet. How old was Bill when he saw the Comet for the first time?",
]
RESULT_SCHEMA = {
    "type": "object",
    "properties": {"result": {"type": "integer"}},
    "required": ["result"],
    "additionalProperties": False,
}
WEATHER_SCHEMA = {
    "type": "object",
    "properties": {"city": {"type": "string"}, "temperature_c": {"type": "number"}},
    "required": ["city", "temperature_c"],
    "additionalProperties": False,
}


class TestMuseStructuredOutput(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--reasoning-parser",
                "muse",
                "--tool-call-parser",
                "muse",
                "--language-model-only",
            ],
        )
        cls.client = openai.Client(api_key="EMPTY", base_url=f"{cls.base_url}/v1")

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            terminate_and_kill_process_tree(cls.process)

    def _complete(self, stream, **kwargs):
        content, finish_reason, _ = self._complete_with_calls(stream, **kwargs)
        return content, finish_reason

    def _complete_with_calls(self, stream, **kwargs):
        kwargs = dict(model=MODEL, temperature=0, max_tokens=4096, **kwargs)
        if not stream:
            choice = self.client.chat.completions.create(**kwargs).choices[0]
            calls = [c.function.name for c in choice.message.tool_calls or []]
            return choice.message.content or "", choice.finish_reason, calls
        content, finish_reason, calls = "", None, []
        for chunk in self.client.chat.completions.create(stream=True, **kwargs):
            if chunk.choices:
                delta = chunk.choices[0].delta
                content += delta.content or ""
                calls += [
                    c.function.name
                    for c in delta.tool_calls or []
                    if c.function and c.function.name
                ]
                finish_reason = chunk.choices[0].finish_reason or finish_reason
        return content, finish_reason, calls

    def test_json_schema_answer(self):
        # The prompt does not ask for JSON: only the grammar can make the answer JSON.
        for stream in (False, True):
            with self.subTest(stream=stream):
                content, finish_reason = self._complete(
                    stream,
                    messages=[{"role": "user", "content": "What is France?"}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "country", "schema": COUNTRY_SCHEMA},
                    },
                )
                self.assertEqual(finish_reason, "stop", content)
                jsonschema.validate(json.loads(content), COUNTRY_SCHEMA)

    def test_json_schema_answer_ends(self):
        for question in GSM8K_QUESTIONS:
            with self.subTest(question=question[:40]):
                content, finish_reason = self._complete(
                    False,
                    messages=[{"role": "user", "content": question}],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": "answer", "schema": RESULT_SCHEMA},
                    },
                )
                self.assertEqual(finish_reason, "stop", content[:200])
                jsonschema.validate(json.loads(content), RESULT_SCHEMA)

    def test_json_object_keys_are_not_reasoning(self):
        # With free keys, a grammar started before the header lets the model write its
        # header and reasoning into the first key.
        for stream in (False, True):
            with self.subTest(stream=stream):
                content, finish_reason = self._complete(
                    stream,
                    messages=[
                        {
                            "role": "user",
                            "content": 'What is 17 * 23? Reply as JSON with the key "result".',
                        }
                    ],
                    response_format={"type": "json_object"},
                )
                self.assertEqual(finish_reason, "stop", content)
                jsonschema.validate(json.loads(content), RESULT_SCHEMA)

    def test_responses_text_format_answer(self):
        # /v1/responses carries the schema in `text.format` instead of response_format.
        response = self.client.responses.create(
            model=MODEL,
            input="What is France?",
            text={
                "format": {
                    "type": "json_schema",
                    "name": "country",
                    "schema": COUNTRY_SCHEMA,
                }
            },
            temperature=0,
            max_output_tokens=4096,
        )
        self.assertEqual(response.status, "completed", response.output_text)
        jsonschema.validate(json.loads(response.output_text), COUNTRY_SCHEMA)

    def test_json_schema_with_tools(self):
        # A turn ends in a tool call or in the answer; only the answer is constrained.
        tools = [
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "Get the current weather for a city",
                    "parameters": {
                        "type": "object",
                        "properties": {"location": {"type": "string"}},
                        "required": ["location"],
                    },
                },
            }
        ]
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "weather", "schema": WEATHER_SCHEMA},
        }
        question = {
            "role": "user",
            "content": "Look up the weather in Paris, then report it.",
        }
        call = {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"location": "Paris"}',
                    },
                }
            ],
        }
        result = {
            "role": "tool",
            "tool_call_id": "call_1",
            "content": '{"temperature_c": 18, "condition": "cloudy"}',
        }
        for stream in (False, True):
            with self.subTest(stream=stream):
                content, finish_reason, calls = self._complete_with_calls(
                    stream,
                    messages=[question],
                    tools=tools,
                    response_format=response_format,
                )
                self.assertEqual(calls, ["get_weather"], content)
                self.assertEqual(finish_reason, "tool_calls", content)
                content, finish_reason, calls = self._complete_with_calls(
                    stream,
                    messages=[question, call, result],
                    tools=tools,
                    response_format=response_format,
                )
                self.assertEqual((finish_reason, calls), ("stop", []), content)
                report = json.loads(content)
                jsonschema.validate(report, WEATHER_SCHEMA)
                self.assertEqual(report["temperature_c"], 18, content)


if __name__ == "__main__":
    unittest.main(verbosity=3)
