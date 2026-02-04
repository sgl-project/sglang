#!/usr/bin/env python3
"""
Test script for DeepSeek-V3.2 chat template.
Tests multi-turn conversations with chat template formatting.
"""

import os
from transformers import AutoTokenizer


def load_chat_template(template_path: str) -> str:
    """Load chat template from file."""
    with open(template_path, 'r', encoding='utf-8') as f:
        return f.read()


def test_chat_template():
    """Test DeepSeek-V3.2 chat template with multi-turn conversations."""

    # Path to custom chat template
    template_path = "/mnt/data/wjh/sglang/examples/chat_template/tool_chat_template_deepseekv32.jinja"

    # Load the custom chat template
    custom_template = load_chat_template(template_path)

    print("=" * 80)
    print("DeepSeek-V3.2 Chat Template Test")
    print("=" * 80)
    print("\n📋 Chat Template Content:")
    print("-" * 80)
    print(custom_template[:500] + "..." if len(custom_template) > 500 else custom_template)
    print("-" * 80)

    # Initialize tokenizer (use a compatible tokenizer, e.g., deepseek-v3)
    # You may need to adjust the model path
    model_path = "/mnt/data/models/DeepSeek-V3.2"  # Adjust if needed

    print(f"\n🔧 Loading tokenizer from: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


    # Set the custom chat template
    tokenizer.chat_template = custom_template

    # Test Case 1: Simple single turn conversation
    print("\n" + "=" * 80)
    print("Test Case 1: Simple Single Turn Conversation")
    print("=" * 80)

    messages_simple = [
        {"role": "user", "content": "Hello, how are you?"}
    ]

    print("\n📝 Input messages:")
    for msg in messages_simple:
        print(f"  {msg['role']}: {msg['content']}")

    formatted_simple = tokenizer.apply_chat_template(
        messages_simple,
        tokenize=False,
        add_generation_prompt=True
    )

    print("\n✨ Formatted output:")
    print("-" * 80)
    print(repr(formatted_simple))
    print("-" * 80)

    # Test Case 2: Multi-turn conversation with system prompt
    print("\n" + "=" * 80)
    print("Test Case 2: Multi-Turn Conversation with System Prompt")
    print("=" * 80)

    messages_multi_turn = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "And what about Germany?"},
        {"role": "assistant", "content": "1111"},
    ]

    print("\n📝 Input messages:")
    for msg in messages_multi_turn:
        print(f"  {msg['role']}: {msg['content']}")

    formatted_multi = tokenizer.apply_chat_template(
        messages_multi_turn,
        tokenize=False,
        add_generation_prompt=True,
        thinking=True
    )

    print("\n✨ Formatted output:")
    print("-" * 80)
    print(repr(formatted_multi))
    print("-" * 80)

    # Test Case 3: Multi-turn conversation with tool calls
    print("\n" + "=" * 80)
    print("Test Case 3: Multi-Turn Conversation with Tool Calls")
    print("=" * 80)

    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather in a given location",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "The city and state, e.g. San Francisco, CA"
                        }
                    },
                    "required": ["location"]
                }
            }
        }
    ]

    messages_with_tools = [
        {"role": "system", "content": "You are a helpful assistant with access to tools."},
        {"role": "user", "content": "What's the weather in Beijing?"},
        {"role": "assistant", "content": None, "tool_calls": [
            {
                "function": {
                    "name": "get_weather",
                    "arguments": {"location": "Beijing"}
                }
            }
        ]},
        {"role": "tool", "content": "The weather in Beijing is sunny, 25°C"},
        {"role": "user", "content": "What about Shanghai?"},
    ]

    print("\n📝 Input messages:")
    for msg in messages_with_tools:
        print(f"  {msg['role']}: {msg}")

    formatted_with_tools = tokenizer.apply_chat_template(
        messages_with_tools,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True
    )

    print("\n✨ Formatted output:")
    print("-" * 80)
    print(repr(formatted_with_tools))
    print("-" * 80)

    # Test Case 4: With tokenization
    print("\n" + "=" * 80)
    print("Test Case 4: Tokenized Output")
    print("=" * 80)

    messages_tokenize = [
        {"role": "system", "content": "You are a helpful AI assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there! How can I help you today?"},
        {"role": "user", "content": "Can you explain what you can do?"},
    ]

    print("\n📝 Input messages:")
    for msg in messages_tokenize:
        print(f"  {msg['role']}: {msg['content']}")

    # Tokenize the formatted conversation
    tokenized = tokenizer.apply_chat_template(
        messages_tokenize,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    print("\n✨ Tokenized shape:", tokenized.shape)
    print("✨ Number of tokens:", tokenized.shape[-1])

    # Decode back to text for verification
    decoded = tokenizer.decode(tokenized[0], skip_special_tokens=False)
    print("\n✨ Decoded text:")
    print("-" * 80)
    print(decoded)
    print("-" * 80)

    print("\n" + "=" * 80)
    print("✅ All tests completed!")
    print("=" * 80)


if __name__ == "__main__":
    test_chat_template()
