#!/usr/bin/env python3
"""
Standalone script to reproduce MiniMax M2 detector bugs.
Run: python3 test_minimax_m2_bugs.py
"""

import os
import sys

# Add python directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "python"))

import json

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.minimax_m2 import MinimaxM2Detector


def test_anyof_null_bug():
    """
    BUG #1: anyOf [string, null] incorrectly converts "hi" to null
    Issue: https://github.com/sgl-project/sglang/issues/16057
    """
    print("=" * 70)
    print("TEST 1: anyOf with null - String value incorrectly becomes null")
    print("=" * 70)

    detector = MinimaxM2Detector()
    tools = [
        Tool(
            type="function",
            function=Function(
                name="the_tool",
                description="The Tool",
                parameters={
                    "type": "object",
                    "required": ["args"],
                    "properties": {
                        "args": {
                            "anyOf": [{"type": "string"}, {"type": "null"}],
                            "description": "Args for The Tool.",
                        }
                    },
                },
            ),
        )
    ]

    text = '<minimax:tool_call><invoke name="the_tool"><parameter name="args">hi</parameter></invoke></minimax:tool_call>'

    result = detector.detect_and_parse(text, tools)

    print(f"Input parameter value: 'hi'")
    print(f"Schema: anyOf [{{type: string}}, {{type: null}}]")
    print(f"\nResult calls: {len(result.calls)}")

    if result.calls:
        params = json.loads(result.calls[0].parameters)
        print(f"Parsed parameters: {params}")
        print(f"params['args'] = {params['args']!r}")
        print(f"Type: {type(params['args']).__name__}")

        if params["args"] is None:
            print("\n❌ BUG CONFIRMED: 'hi' was converted to None!")
            print("   Root cause: Line 148-149 uses OR logic instead of AND")
            print(
                "   if 'null' in normalized_types or value.lower() in ('null', 'none', 'nil'):"
            )
            print("      return None")
            return False
        else:
            print(f"\n✅ PASS: Value correctly preserved as {params['args']!r}")
            return True
    else:
        print("\n❌ ERROR: No tool calls detected")
        return False


def test_content_leak_bug():
    """
    BUG #2: Tool call content leaks into normal_text when </minimax:tool_call> is missing
    """
    print("\n" + "=" * 70)
    print("TEST 2: Incomplete tool call leaks into normal_text")
    print("=" * 70)

    detector = MinimaxM2Detector()
    tools = [
        Tool(
            type="function",
            function=Function(
                name="context_info",
                description="Get context",
                parameters={
                    "type": "object",
                    "properties": {"metadata_only": {"type": "boolean"}},
                },
            ),
        )
    ]

    # Missing </minimax:tool_call> end tag
    text = (
        '[Pasted ~4 lines]<minimax:tool_call><invoke name="context_info"><parameter name="metadata_only">false</parameter></invoke>'
        # NOTE: Missing </minimax:tool_call>
    )

    result = detector.detect_and_parse(text, tools)

    print(f"Input: '{text[:30]}...'")
    print(f"Note: Missing </minimax:tool_call> end tag")
    print(f"\nNormal text: {result.normal_text!r}")
    print(f"Tool calls detected: {len(result.calls)}")

    has_leak = (
        "<minimax:tool_call>" in result.normal_text
        or "<invoke" in result.normal_text
        or "<parameter" in result.normal_text
    )

    if has_leak:
        print("\n❌ BUG CONFIRMED: Tool call content leaked into normal_text!")
        print("   Root cause: _extract() Line 470")
        print("   if e == -1:")
        print(
            "       normal_parts.append(text[s:])  # BUG: Adds tool call to normal text"
        )
        return False
    else:
        print("\n✅ PASS: No tool call markers in normal_text")
        return True


def test_streaming_leak_bug():
    """
    BUG #3: Streaming doesn't properly buffer incomplete tool calls
    """
    print("\n" + "=" * 70)
    print("TEST 3: Streaming incomplete tool call handling")
    print("=" * 70)

    detector = MinimaxM2Detector()
    tools = [
        Tool(
            type="function",
            function=Function(
                name="test_func",
                description="Test",
                parameters={"type": "object", "properties": {}},
            ),
        )
    ]

    chunks = [
        "Normal text ",
        "<minimax:tool_call>",
        '<invoke name="test_func">',
        # Stream ends without closing tags
    ]

    all_normal_text = ""
    for chunk in chunks:
        result = detector.parse_streaming_increment(chunk, tools)
        all_normal_text += result.normal_text

    print(f"Streamed chunks: {chunks}")
    print(f"Combined normal_text: {all_normal_text!r}")

    has_leak = "<minimax:tool_call>" in all_normal_text or "<invoke" in all_normal_text

    if has_leak:
        print("\n❌ BUG CONFIRMED: Tool call markers leaked during streaming!")
        return False
    else:
        print("\n✅ PASS: Tool call properly buffered, no leakage")
        return True


def test_regex_closing_tag_bug():
    """
    BUG #4: Regex cannot handle parameter values containing </parameter> substring
    """
    print("\n" + "=" * 70)
    print("TEST 4: Regex bug - Parameter value contains </parameter> substring")
    print("=" * 70)

    detector = MinimaxM2Detector()
    tools = [
        Tool(
            type="function",
            function=Function(
                name="test_func",
                description="Test",
                parameters={
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                },
            ),
        )
    ]

    text = '<minimax:tool_call><invoke name="test_func"><parameter name="content">This text contains </parameter> in it</parameter></invoke></minimax:tool_call>'

    result = detector.detect_and_parse(text, tools)

    print(
        "Input: '<parameter name=\"content\">This text contains </parameter> in it</parameter>'"
    )
    print(f"\nResult calls: {len(result.calls)}")

    if result.calls:
        params = json.loads(result.calls[0].parameters)
        print(f"Parsed parameters: {params}")
        print(f"params['content'] = {params['content']!r}")

        expected = "This text contains </parameter> in it"
        actual = params.get("content", "")

        if actual == expected:
            print(f"\n✅ PASS: Value correctly preserved as {expected!r}")
            return True
        else:
            print("\n❌ BUG CONFIRMED: Regex truncated at first </parameter>!")
            print(f"   Expected: {expected!r}")
            print(f"   Actual:   {actual!r}")
            print(
                "   Root cause: regex r'<parameter name=\"(.*?)</parameter>' uses non-greedy match"
            )
            return False
    else:
        print("\n❌ ERROR: No tool calls detected")
        return False


def main():
    print("\n" + "=" * 70)
    print("MiniMax M2 Detector - Bug Reproduction Tests")
    print("Related Issue: https://github.com/sgl-project/sglang/issues/16057")
    print("=" * 70)

    results = []

    try:
        results.append(("anyOf null conversion", test_anyof_null_bug()))
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("anyOf null conversion", False))

    try:
        results.append(("content leak", test_content_leak_bug()))
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("content leak", False))

    try:
        results.append(("streaming leak", test_streaming_leak_bug()))
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("streaming leak", False))

    try:
        results.append(("regex closing tag bug", test_regex_closing_tag_bug()))
    except Exception as e:
        print(f"\n❌ Test failed with exception: {e}")
        import traceback

        traceback.print_exc()
        results.append(("regex closing tag bug", False))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for test_name, passed in results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {test_name}")

    failed_count = sum(1 for _, passed in results if not passed)
    print(f"\nTotal: {len(results)} tests, {failed_count} failures")

    if failed_count > 0:
        print("\n🔧 These bugs need to be fixed in minimax_m2.py")
        return 1
    else:
        print("\n🎉 All tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
