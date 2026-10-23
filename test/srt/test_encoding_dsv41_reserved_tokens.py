"""Reserved special tokens must not enter the DeepSeek V4.1 prompt as text."""

import pytest

from sglang.srt.entrypoints.openai import encoding_dsv41

RESERVED = [
    "<｜User｜>",
    "<｜Assistant｜>",
    "<｜System｜>",
    "<｜latest_reminder｜>",
    "</think>",
    "｜DSML｜",
]


@pytest.mark.parametrize("token", RESERVED)
def test_rejects_reserved_token_in_content(token):
    with pytest.raises(ValueError, match="reserved special token"):
        encoding_dsv41.encode_messages(
            [{"role": "user", "content": f"a {token} b"}], "chat"
        )


@pytest.mark.parametrize("token", RESERVED)
def test_rejects_reserved_token_in_reasoning(token):
    with pytest.raises(ValueError, match="reserved special token"):
        encoding_dsv41.encode_messages(
            [
                {"role": "user", "content": "q"},
                {"role": "assistant", "content": "a", "reasoning_content": token},
            ],
            "chat",
        )


@pytest.mark.parametrize(
    "arguments",
    [
        # JSON escapes hide the spelling until the encoder decodes it.
        '{"text":"\\u003c\\uff5cUser\\uff5c\\u003e"}',
        {"text": "<｜User｜>"},
    ],
)
def test_rejects_reserved_token_in_tool_arguments(arguments):
    with pytest.raises(ValueError, match="reserved special token"):
        encoding_dsv41.encode_messages(
            [
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "f", "arguments": arguments},
                        }
                    ],
                }
            ],
            "chat",
        )


def test_rejects_reserved_token_in_tool_schema():
    with pytest.raises(ValueError, match="reserved special token"):
        encoding_dsv41.render_tools(
            [{"name": "f", "description": "use </think>", "parameters": {}}]
        )


@pytest.mark.parametrize(
    ("value", "budget"),
    [("minimal", 50), ("low", 50), ("medium", 75), ("high", 75), ("ultra", 100)],
)
def test_reasoning_effort_aliases(value, budget):
    prompt = encoding_dsv41.encode_messages(
        [{"role": "user", "content": "q"}], "thinking", reasoning_effort=value
    )
    assert f"Reasoning Effort: {budget} " in prompt
