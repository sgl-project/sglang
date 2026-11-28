import json
import unittest

import openai

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=120, stage="extra-a", runner_config="1-gpu-large")


class TestResponseTemplateReasoningTools(DefaultServerBase):
    model = "Qwen/Qwen3.6-27B"
    other_args = [
        "--revision",
        "1cda14dd71a3f47115850eb790f416f165da1cb5",
        "--reasoning-parser",
        "response_template",
        "--tool-call-parser",
        "response_template",
        "--context-length",
        "4096",
        "--max-running-requests",
        "4",
        "--disable-prefill-cuda-graph",
        "--disable-flashinfer-autotune",
    ]

    def setUp(self):
        self.client = openai.OpenAI(
            base_url=self.base_url + "/v1", api_key=self.api_key
        )
        self.addCleanup(self.client.close)
        self.weather = {
            "type": "function",
            "function": {
                "name": "get_weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string", "enum": ["Paris"]}},
                    "required": ["city"],
                    "additionalProperties": False,
                },
            },
        }
        self.request = dict(
            model=self.model,
            messages=[{"role": "user", "content": "Call get_weather for Paris."}],
            temperature=0,
            max_completion_tokens=96,
        )

    def test_auto_tool_replays_implicit_end_and_counts_streamed_reasoning(self):
        # Bias toward "plan" until its frequency penalty makes <tool_call> win.
        # The supplied grammar accepts exactly one opener and the complete call.
        tag = {
            "type": "structural_tag",
            "format": {
                "type": "sequence",
                "elements": [
                    {"type": "token", "token": "<tool_call>"},
                    {
                        "type": "const_string",
                        "value": "\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n</function>\n",
                    },
                    {"type": "token", "token": "</tool_call>"},
                ],
            },
        }
        extra = {
            "chat_template_kwargs": {"enable_thinking": True},
            "response_format": tag,
        }
        request = dict(
            **self.request,
            tools=[self.weather],
            tool_choice="auto",
            frequency_penalty=2,
            logit_bias={"10083": 100, "248058": 85, "248069": -100},
        )
        result = self.client.chat.completions.create(
            **request, extra_body={**extra, "return_token_ids": True}
        )
        choice = result.choices[0]
        ids = choice.response_token_ids
        self.assertNotIn(248069, ids)
        self.assertEqual(ids.count(248058), 1)
        count = result.usage.completion_tokens_details.reasoning_tokens
        self.assertGreater(count, 0)
        self.assertEqual(ids[: ids.index(248058)], [10083] * count)
        self.assertEqual(choice.finish_reason, "tool_calls")
        (call,) = choice.message.tool_calls
        self.assertEqual(call.function.name, "get_weather")
        self.assertEqual(json.loads(call.function.arguments), {"city": "Paris"})

        chunks = list(
            self.client.chat.completions.create(
                **request,
                extra_body=extra,
                stream=True,
                stream_options={"include_usage": True},
            )
        )
        choices = [choice for chunk in chunks for choice in chunk.choices]
        streamed_count = chunks[-1].usage.completion_tokens_details.reasoning_tokens
        reasoning = "".join(
            getattr(choice.delta, "reasoning_content", None) or "" for choice in choices
        )
        self.assertGreater(streamed_count, 0)
        self.assertEqual(reasoning, "plan" * streamed_count)
        self.assertEqual(choices[-1].finish_reason, "tool_calls")
        calls = [call for choice in choices for call in choice.delta.tool_calls or []]
        self.assertEqual({call.index for call in calls}, {0})
        self.assertEqual(
            "".join(call.function.name or "" for call in calls), "get_weather"
        )
        arguments = "".join(call.function.arguments or "" for call in calls)
        self.assertEqual(json.loads(arguments), {"city": "Paris"})

    def test_required_and_named_strict_tools_after_explicit_end(self):
        self.weather["function"]["strict"] = True
        named = {"type": "function", "function": {"name": "get_weather"}}
        other = {
            "type": "function",
            "function": {**self.weather["function"], "name": "get_time"},
        }
        for choice, tools in (
            ("required", [self.weather]),
            (named, [other, self.weather]),
        ):
            with self.subTest(choice=choice):
                result = self.client.chat.completions.create(
                    **self.request,
                    tools=tools,
                    tool_choice=choice,
                    parallel_tool_calls=False,
                    logit_bias={"248069": 100},
                    extra_body={
                        "chat_template_kwargs": {"enable_thinking": True},
                        "return_token_ids": True,
                    },
                )
                answer = result.choices[0]
                self.assertEqual(answer.response_token_ids[0], 248069)
                self.assertEqual(
                    result.usage.completion_tokens_details.reasoning_tokens, 1
                )
                self.assertEqual(answer.finish_reason, "tool_calls")
                (call,) = answer.message.tool_calls
                self.assertEqual(call.function.name, "get_weather")
                self.assertEqual(json.loads(call.function.arguments), {"city": "Paris"})


if __name__ == "__main__":
    unittest.main()
