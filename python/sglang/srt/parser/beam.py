"""Beam response-template fallback profile."""

from typing import Any

from sglang.srt.parser.response_template import ResponseTemplateReasoningDetector

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
    grammar_requires_reasoning = True
