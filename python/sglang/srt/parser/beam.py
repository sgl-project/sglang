"""Beam response-template fallback profile."""

from typing import Any

from sglang.srt.function_call.response_template_structural_tag import (
    answer_rest_structural_tag,
    json_answer_structural_tag,
)
from sglang.srt.parser.chat_parsing.response_parser import ResponseParser
from sglang.srt.parser.response_template import (
    _CONTENT_FIELD,
    ResponseTemplateReasoningDetector,
)

BEAM_RESPONSE_TEMPLATE: dict[str, Any] = {
    "defaults": {"role": "assistant"},
    "start_anchor_pattern": r"<\|start\|>assistant",
    "fields": {
        "thinking": {
            "open": "<|channel|>analysis<|message|>",
            # Consuming the next message's `<|start|>assistant` makes reasoning
            # end exactly where the answer or a call opens.
            "close": "<|end|><|start|>assistant",
            "content": "text",
        },
        "content": {
            "open": "<|channel|>final<|message|>",
            # No grammar can keep the `<|call|>` stop token from ending the
            # answer, and the output keeps it as a possible tool call closer.
            "close": ["<|return|>", "<|end|>", "<|call|>"],
            "content": "text",
        },
        "tool_calls": {
            # The prompt and every closer before a call end with
            # `<|start|>assistant`, so each call opens at ` to=`. Like the
            # Harmony recipient, the name is any run without whitespace or `<`.
            "open_pattern": (
                r" to=(?P<name>[^\s<]+)"
                r"<\|channel\|>analysis <\|constrain\|>xml<\|message\|>"
            ),
            "close": ["<|end|><|start|>assistant", "<|call|>"],
            "repeats": True,
            "content": "xml-inline",
            "content_args": {
                "tag_pattern": (
                    r'<rfl:parameter name="(?P<key>[^"]+)">'
                    r"\n(?P<value>.*?)\n</rfl:parameter>"
                ),
                "value_parser": {
                    "name": "text",
                    "args": {"strip": False},
                },
            },
            "transform": {
                "type": "function",
                "function": {
                    "name": "{name}",
                    "arguments": "{content}",
                },
            },
        },
    },
}


class BeamReasoningDetector(ResponseTemplateReasoningDetector):
    response_template = BEAM_RESPONSE_TEMPLATE
    reasoning_default = "always"

    def answer_rest_structural_tag(self, prefix: str) -> dict[str, Any] | None:
        parser = ResponseParser(self.response_template, prefix=prefix)
        regions = [
            (event.get("field"), event["type"])
            for event in parser.initial_events
            if event["type"] != "region_chunk"
        ]
        if regions[-1:] != [(_CONTENT_FIELD, "region_open")]:
            return None
        return answer_rest_structural_tag(self._content_field)

    def json_answer_structural_tag(
        self,
        json_schema: Any,
        *,
        tool_names: list[str],
        message_start: bool = False,
    ) -> dict[str, Any] | None:
        if not self.think_end_token or self._content_field is None:
            return None
        return json_answer_structural_tag(
            self._content_field,
            json_schema,
            self._tool_field,
            tool_names,
            thinking=self._thinking_field,
            message_start=self._message_start if message_start else None,
            grammar_requires_reasoning=True,
        )
