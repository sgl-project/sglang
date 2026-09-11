"""Regression coverage for Anthropic deferred-tool compatibility.

CALLING SPEC:
    python3 test/registered/unit/entrypoints/anthropic/test_tool_reference.py

Runs CPU-only tests for native template expansion, generic-template deferred
tool routing, and Qwen-style strict content rendering. No external services or
model weights are required.
"""

import unittest
from types import SimpleNamespace

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()  # must precede imports that may pull in sgl_kernel

from jinja2 import Environment  # noqa: E402

from sglang.srt.entrypoints.anthropic.protocol import (  # noqa: E402
    AnthropicMessagesRequest,
)
from sglang.srt.entrypoints.anthropic.serving import AnthropicServing  # noqa: E402
from sglang.srt.entrypoints.anthropic.tool_reference import (  # noqa: E402
    template_supports_deferred_tool_loading,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


NATIVE_TEMPLATE = """
{%- for tool in tools if not tool.function.defer_loading -%}
{{ tool.function.name }}
{%- endfor -%}
{%- for message in messages if message.role == "tool" -%}
{%- if message.content is not string and message.content.0.type == "tool_reference" -%}
{%- for reference in message.content -%}
{%- for tool in tools if tool.function.name == reference.name -%}
{{ tool.function.name }}
{%- endfor -%}
{%- endfor -%}
{%- endif -%}
{%- endfor -%}
"""

QWEN_LIKE_TEMPLATE = """
{%- macro render_content(content) -%}
    {%- if content is string -%}
        {{- content -}}
    {%- else -%}
        {%- for item in content -%}
            {%- if item.type == "text" -%}
                {{- item.text -}}
            {%- else -%}
                {{- raise_exception("Unexpected item type in content.") -}}
            {%- endif -%}
        {%- endfor -%}
    {%- endif -%}
{%- endmacro -%}
{%- for tool in tools -%}{{ tool.function.name }} {% endfor -%}
{%- for message in messages -%}{{ render_content(message.content) }}{% endfor -%}
"""


class _FakeOpenAIServingChat:
    def __init__(self, chat_template):
        self.tokenizer_manager = SimpleNamespace(
            tokenizer=SimpleNamespace(chat_template=chat_template)
        )


def _tool(name: str, *, defer_loading: bool) -> dict:
    return {
        "name": name,
        "description": f"The {name} tool",
        "input_schema": {"type": "object", "properties": {}},
        "defer_loading": defer_loading,
    }


def _request(
    *,
    references: list[str] | None = None,
    tool_choice: dict | None = None,
) -> AnthropicMessagesRequest:
    if references is None:
        messages = [{"role": "user", "content": "Find a useful tool"}]
    else:
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool_search_1",
                        "content": [
                            {
                                "type": "tool_reference",
                                "tool_name": name,
                            }
                            for name in references
                        ],
                    }
                ],
            }
        ]
    request = {
        "model": "test-model",
        "max_tokens": 16,
        "messages": messages,
        "tools": [
            _tool("ToolSearch", defer_loading=False),
            _tool("Bash", defer_loading=True),
            _tool("Read", defer_loading=True),
        ],
    }
    if tool_choice is not None:
        request["tool_choice"] = tool_choice
    return AnthropicMessagesRequest.model_validate(request)


def _convert(chat_template: str, *, references: list[str] | None = None) -> dict:
    serving = AnthropicServing(_FakeOpenAIServingChat(chat_template))
    request = serving._convert_to_chat_completion_request(
        _request(references=references)
    )
    return request.model_dump(exclude_none=True)


def _raise_exception(message: str) -> None:
    raise ValueError(message)


class TestTemplateCapabilityDetection(unittest.TestCase):
    def test_detects_native_deferred_tool_expansion(self):
        self.assertTrue(template_supports_deferred_tool_loading(NATIVE_TEMPLATE))

    def test_ignores_jinja_comment(self):
        self.assertFalse(
            template_supports_deferred_tool_loading(
                "{# tool_reference and defer_loading are unsupported #}"
            )
        )

    def test_reference_text_rendering_is_not_native_expansion(self):
        marker_only_template = """
        {% if item.type == "tool_reference" %}
        [tool reference: {{ item.name }}]
        {% endif %}
        """
        self.assertFalse(template_supports_deferred_tool_loading(marker_only_template))

    def test_checks_each_named_template(self):
        self.assertTrue(
            template_supports_deferred_tool_loading(
                {"default": "{{ message.content }}", "tool_use": NATIVE_TEMPLATE}
            )
        )


class TestGenericTemplateDeferredTools(unittest.TestCase):
    def test_hides_deferred_tools_before_discovery(self):
        payload = _convert(QWEN_LIKE_TEMPLATE)

        self.assertEqual(
            [tool["function"]["name"] for tool in payload["tools"]],
            ["ToolSearch"],
        )

    def test_forwards_only_referenced_deferred_tools(self):
        payload = _convert(QWEN_LIKE_TEMPLATE, references=["Bash"])

        self.assertEqual(
            [tool["function"]["name"] for tool in payload["tools"]],
            ["ToolSearch", "Bash"],
        )
        self.assertNotIn("defer_loading", payload["tools"][1])
        self.assertNotIn("defer_loading", payload["tools"][1]["function"])

    def test_qwen_style_render_receives_text_and_unlocked_schema(self):
        payload = _convert(QWEN_LIKE_TEMPLATE, references=["Bash"])
        environment = Environment()
        environment.globals["raise_exception"] = _raise_exception

        prompt = environment.from_string(QWEN_LIKE_TEMPLATE).render(
            messages=payload["messages"],
            tools=payload["tools"],
        )

        self.assertIn("ToolSearch Bash", prompt)
        self.assertNotIn("Read", prompt)
        self.assertIn("[tool reference: Bash]", prompt)

    def test_unknown_reference_does_not_unlock_a_tool(self):
        payload = _convert(QWEN_LIKE_TEMPLATE, references=["Unknown"])

        self.assertEqual(
            [tool["function"]["name"] for tool in payload["tools"]],
            ["ToolSearch"],
        )
        self.assertEqual(
            payload["messages"][0]["content"],
            "[tool reference: Unknown]",
        )

    def test_forced_deferred_tool_requires_prior_discovery(self):
        serving = AnthropicServing(_FakeOpenAIServingChat(QWEN_LIKE_TEMPLATE))
        request = _request(tool_choice={"type": "tool", "name": "Bash"})

        with self.assertRaisesRegex(ValueError, "not in the forwarded tools list"):
            serving._convert_to_chat_completion_request(request)

    def test_discovered_deferred_tool_can_be_forced(self):
        serving = AnthropicServing(_FakeOpenAIServingChat(QWEN_LIKE_TEMPLATE))
        request = _request(
            references=["Bash"],
            tool_choice={"type": "tool", "name": "Bash"},
        )

        payload = serving._convert_to_chat_completion_request(request).model_dump(
            exclude_none=True
        )

        self.assertEqual(payload["tool_choice"]["function"]["name"], "Bash")


class TestNativeTemplateDeferredTools(unittest.TestCase):
    def test_preserves_structured_reference_and_full_catalog(self):
        payload = _convert(NATIVE_TEMPLATE, references=["Bash"])

        self.assertEqual(
            [tool["function"]["name"] for tool in payload["tools"]],
            ["ToolSearch", "Bash", "Read"],
        )
        self.assertTrue(payload["tools"][1]["function"]["defer_loading"])
        self.assertEqual(
            payload["messages"][0]["content"],
            [{"type": "tool_reference", "name": "Bash"}],
        )


if __name__ == "__main__":
    unittest.main()
