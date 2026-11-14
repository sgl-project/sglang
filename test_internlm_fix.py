#!/usr/bin/env python3
"""Test script to verify the InternlmDetector fix for issue #46661."""

import json
import sys

# Add the python directory to the path
sys.path.insert(0, '/Users/depend/sglang/python')

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.internlm_detector import InternlmDetector

def test_internlm_detector():
    """Test that InternlmDetector correctly handles multiple tool calls in streaming mode."""
    tools = [
        Tool(type="function", function=Function(
            name=n,
            parameters={"type": "object", "properties": {"city": {"type": "string"}}}
        )) for n in ("get_weather", "get_time")
    ]

    call1 = json.dumps({"name": "get_weather", "parameters": {"city": "Tokyo"}})
    call2 = json.dumps({"name": "get_time", "parameters": {"city": "Paris"}})

    # Token chunks as they would arrive from the model
    tok = [
        "Checking.",
        "<|action_start|>",
        " ",
        "<|plugin|>",
        "\n",
        call1,
        "<|action_end|>",
        "\n",
        "<|action_start|>",
        " ",
        "<|plugin|>",
        "\n",
        call2,
        "<|action_end|>"
    ]
    full = "".join(tok)

    def stream(chunks):
        d, calls = InternlmDetector(), []
        for c in chunks:
            result = d.parse_streaming_increment(c, tools)
            calls += [(x.tool_index, x.name) for x in result.calls]
        return calls

    # Test non-streaming
    non_streaming_result = InternlmDetector().detect_and_parse(full, tools)
    non_streaming_calls = [c.name for c in non_streaming_result.calls]
    print(f"non-streaming: {non_streaming_calls}")
    assert non_streaming_calls == ['get_weather', 'get_time'], f"Expected ['get_weather', 'get_time'], got {non_streaming_calls}"

    # Test token chunks
    token_chunks_result = stream(tok)
    print(f"token chunks: {token_chunks_result}")
    assert token_chunks_result == [(0, 'get_weather'), (1, 'get_time')], f"Expected [(0, 'get_weather'), (1, 'get_time')], got {token_chunks_result}"

    # Test one char per chunk
    one_char_result = stream(list(full))
    print(f"one char per chunk: {one_char_result}")
    assert one_char_result == [(0, 'get_weather'), (1, 'get_time')], f"Expected [(0, 'get_weather'), (1, 'get_time')], got {one_char_result}"

    # Test call 2 in one chunk
    call2_in_one_chunk_result = stream(tok[:8] + ["".join(tok[8:])])
    print(f"call 2 in one chunk: {call2_in_one_chunk_result}")
    assert call2_in_one_chunk_result == [(0, 'get_weather'), (1, 'get_time')], f"Expected [(0, 'get_weather'), (1, 'get_time')], got {call2_in_one_chunk_result}"

    print("\n✓ All tests passed!")
    return True

if __name__ == "__main__":
    try:
        success = test_internlm_detector()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
