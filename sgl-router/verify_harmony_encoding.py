#!/usr/bin/env python3
"""
Verify Harmony encoding matches between Python (openai-harmony) and Rust implementation.

This script encodes sample messages using openai-harmony and outputs the token IDs
so they can be compared with Rust's implementation.
"""

import json
from openai_harmony import (
    load_harmony_encoding,
    HarmonyEncodingName,
    Role,
    Message,
    Author,
    TextContent,
    Content,
    Conversation,
    SystemContent,
    DeveloperContent,
    ToolDescription,
)


def main():
    # Load Harmony encoding
    encoding = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)

    print("=" * 80)
    print("Harmony Encoding Verification - Python Implementation")
    print("=" * 80)
    print()

    # Test 1: Simple user message
    print("Test 1: Simple user message")
    print("-" * 80)

    messages = [
        Message.from_role_and_content(Role.USER, "Hello"),
    ]

    conversation = Conversation.from_messages(messages)
    token_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

    print(f"Input: User message 'Hello'")
    print(f"Token IDs: {token_ids}")
    print(f"Token count: {len(token_ids)}")
    print()

    # Test 2: Multi-turn conversation
    print("Test 2: Multi-turn conversation")
    print("-" * 80)

    msg1 = Message.from_role_and_content(Role.USER, "What's the weather?")
    msg2 = Message.from_role_and_content(Role.ASSISTANT, '{"location": "SF"}')
    msg2 = msg2.with_channel("commentary")
    msg2 = msg2.with_recipient("functions.get_weather")
    msg2 = msg2.with_content_type("json")

    messages = [msg1, msg2]

    conversation = Conversation.from_messages(messages)
    token_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

    print(f"Input: User + Tool call")
    print(f"Token IDs: {token_ids}")
    print(f"Token count: {len(token_ids)}")
    print()

    # Test 3: Tool response
    print("Test 3: Tool response")
    print("-" * 80)

    msg1 = Message.from_role_and_content(Role.USER, "What's the weather?")

    msg2 = Message.from_role_and_content(Role.ASSISTANT, '{"location": "SF"}')
    msg2 = msg2.with_channel("commentary")
    msg2 = msg2.with_recipient("functions.get_weather")
    msg2 = msg2.with_content_type("json")

    msg3 = Message.from_author_and_content(
        Author(role=Role.TOOL, name="functions.get_weather"),
        '{"temperature": 72}',
    )
    msg3 = msg3.with_channel("commentary")

    messages = [msg1, msg2, msg3]

    conversation = Conversation.from_messages(messages)
    token_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

    print(f"Input: User + Tool call + Tool response")
    print(f"Token IDs: {token_ids}")
    print(f"Token count: {len(token_ids)}")
    print()

    # Test 4: Assistant with final channel
    print("Test 4: Assistant final response")
    print("-" * 80)

    msg1 = Message.from_role_and_content(Role.USER, "Hello")
    msg2 = Message.from_role_and_content(Role.ASSISTANT, "Hi there")
    msg2 = msg2.with_channel("final")

    messages = [msg1, msg2]

    conversation = Conversation.from_messages(messages)
    token_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

    print(f"Input: User + Assistant final")
    print(f"Token IDs: {token_ids}")
    print(f"Token count: {len(token_ids)}")
    print()

    # Test 5: With tools (system + developer + user)
    print("Test 5: With tool definitions")
    print("-" * 80)

    # System message
    sys_msg = Message.from_role_and_content(Role.SYSTEM, SystemContent.new())

    # Developer message with tools
    tool_desc = ToolDescription.new(
        "get_weather",
        "Get the current weather for a location",
        {"type": "object", "properties": {"location": {"type": "string"}}}
    )
    dev_content = DeveloperContent.new().with_function_tools([tool_desc])
    dev_msg = Message.from_role_and_content(Role.DEVELOPER, dev_content)

    # User message
    user_msg = Message.from_role_and_content(Role.USER, "What's the weather in SF?")

    messages = [sys_msg, dev_msg, user_msg]
    conversation = Conversation.from_messages(messages)
    token_ids = encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

    print(f"Input: System + Developer (with tools) + User")
    print(f"Token IDs: {token_ids}")
    print(f"Token count: {len(token_ids)}")
    print()

    # Output JSON for easy comparison
    print("=" * 80)
    print("JSON Output for Comparison")
    print("=" * 80)

    test_cases = {
        "simple_user": {
            "messages": [
                {"role": "user", "content": "Hello"}
            ],
            "token_ids": encoding.render_conversation_for_completion(
                Conversation.from_messages([Message.from_role_and_content(Role.USER, "Hello")]),
                Role.ASSISTANT
            ),
        },
        "multi_turn": {
            "messages": [
                {"role": "user", "content": "What's the weather?"},
                {
                    "role": "assistant",
                    "channel": "commentary",
                    "recipient": "functions.get_weather",
                    "content_type": "json",
                    "content": '{"location": "SF"}',
                },
            ],
            "token_ids": encoding.render_conversation_for_completion(
                Conversation.from_messages([
                    Message.from_role_and_content(Role.USER, "What's the weather?"),
                    Message.from_role_and_content(Role.ASSISTANT, '{"location": "SF"}')
                        .with_channel("commentary")
                        .with_recipient("functions.get_weather")
                        .with_content_type("json"),
                ]),
                Role.ASSISTANT
            ),
        },
    }

    print(json.dumps(test_cases, indent=2))


if __name__ == "__main__":
    main()
