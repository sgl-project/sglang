# SPDX-License-Identifier: Apache-2.0
"""Unit tests for non-harmony /v1/responses replay reconstruction."""

import ast
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

from openai.types.responses import (
    ResponseOutputMessage,
    ResponseOutputText,
    ResponseReasoningItem,
)
from openai.types.responses.response_reasoning_item import (
    Content as ResponseReasoningTextContent,
)


def load_construct_input_messages():
    source_path = (
        Path(__file__).resolve().parents[2]
        / "python/sglang/srt/entrypoints/openai/serving_responses.py"
    )
    source = source_path.read_text()
    module = ast.parse(source)

    for node in module.body:
        if isinstance(node, ast.ClassDef) and node.name == "OpenAIServingResponses":
            for item in node.body:
                if (
                    isinstance(item, ast.FunctionDef)
                    and item.name == "_construct_input_messages"
                ):
                    function_source = textwrap.dedent(
                        ast.get_source_segment(source, item)
                    )
                    globals_dict = {
                        "ResponseOutputMessage": ResponseOutputMessage,
                        "ResponseOutputText": ResponseOutputText,
                        "ResponseReasoningItem": ResponseReasoningItem,
                    }
                    locals_dict = {}
                    exec(
                        "from __future__ import annotations\n" + function_source,
                        globals_dict,
                        locals_dict,
                    )
                    return locals_dict["_construct_input_messages"]

    raise AssertionError(
        "Could not find OpenAIServingResponses._construct_input_messages"
    )


class TestServingResponsesReplay(unittest.TestCase):
    def test_construct_input_messages_replays_prior_assistant_text_once(self):
        construct_input_messages = load_construct_input_messages()
        serving = SimpleNamespace(
            msg_store={
                "resp_prev": [
                    {"role": "user", "content": "first question"},
                ]
            }
        )
        prev_response = SimpleNamespace(
            id="resp_prev",
            output=[
                ResponseReasoningItem(
                    id="rs_1",
                    type="reasoning",
                    summary=[],
                    content=[
                        ResponseReasoningTextContent(
                            type="reasoning_text", text="internal reasoning"
                        )
                    ],
                    status=None,
                ),
                ResponseOutputMessage(
                    id="msg_1",
                    content=[
                        ResponseOutputText(
                            text="prior assistant answer",
                            annotations=[],
                            type="output_text",
                            logprobs=None,
                        )
                    ],
                    role="assistant",
                    status="completed",
                    type="message",
                ),
            ],
        )
        request = SimpleNamespace(
            input="follow-up question",
            instructions="new system instruction",
        )

        messages = construct_input_messages(serving, request, prev_response)

        self.assertEqual(
            messages,
            [
                {"role": "system", "content": "new system instruction"},
                {"role": "user", "content": "first question"},
                {"role": "assistant", "content": "prior assistant answer"},
                {"role": "user", "content": "follow-up question"},
            ],
        )
        self.assertEqual(
            sum(
                1
                for message in messages
                if message == {"role": "system", "content": "new system instruction"}
            ),
            1,
        )


if __name__ == "__main__":
    unittest.main()
