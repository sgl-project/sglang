"""Beam response-template fallback profile."""

from typing import Any

from sglang.srt.parser.response_template import ResponseTemplateReasoningDetector

BEAM_RESPONSE_TEMPLATE: dict[str, Any] = {
    "defaults": {"role": "assistant"},
    "start_anchor_pattern": r"<\|start\|>assistant",
    "fields": {
        "thinking": {
            "open": "<|channel|>analysis<|message|>",
            "close": "<|end|><|start|>assistant",
            "content": "text",
        },
        "content": {
            "open": "<|channel|>final<|message|>",
            # `<|call|>` is a stop token, so a grammar that lets the answer end
            # cannot keep it from ending the answer.
            "close": ["<|return|>", "<|end|>", "<|call|>"],
            "content": "text",
        },
        "tool_calls": {
            "open_pattern": (
                r"(?:<\|start\|>assistant)? to=(?P<name>[a-zA-Z_][a-zA-Z0-9_.-]*)"
                r"<\|channel\|>analysis <\|constrain\|>xml<\|message\|>"
            ),
            "close": ["<|end|>", "<|call|>"],
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
