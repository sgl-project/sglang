from sglang.srt.entrypoints.openai.encoding_dsv32 import encode_messages


def test_tool_call_thinking_end_logic():
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What's the weather in Tokyo?"},
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I need to check the weather...",
            "tool_calls": [
                {
                    "id": "call_xxx",
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "arguments": '{"city": "Tokyo"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call_xxx", "content": "Sunny, 72°F"},
    ]

    final_prompt = encode_messages(messages, thinking_mode="thinking")

    assert final_prompt.strip().endswith(
        "</think>"
    ), f"Expected prompt to end with </think>, but got: {final_prompt[-20:]!r}"

    assert not final_prompt.strip().endswith(
        "<think>"
    ), "Prompt incorrectly ends with <think> after tool response"
