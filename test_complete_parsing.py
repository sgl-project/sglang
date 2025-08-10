#\!/usr/bin/env python3
"""Test complete GPT-OSS pattern parsing with reasoning and tool calls"""

from sglang.srt.reasoning_parser import GPTOSSDetector

# Your example response
test_response = '''<|channel|>analysis<|message|>The user asks: "What is the sales made by John for Jan 2020?" We have a function sales_search that takes a firstName. Likely returns sales data. Possibly we need to query for John and then filter for Jan 2020. The function only takes firstName, no date parameter. We'll call sales_search with firstName "John". Then we need to inspect the returned data and find Jan 2020 sales.<|end|><|start|>assistant<|channel|>commentary to=functions.sales_search <|constrain|>json<|message|>{
  "firstName": "John"
}<|call|><|start|>assistant<|channel|>analysis<|message|>We need to see the result.<|end|><|start|>assistant<|channel|>commentary<|message|>We need to see the function output.<|end|><|start|>assistant<|channel|>final<|message|>The sales records for John in January 2020 show a total of **$23,450** in sales. If you need a more detailed breakdown (by day, product, or region), just let me know\!'''

# Test the detector
detector = GPTOSSDetector(stream_reasoning=False)
result = detector.detect_and_parse(test_response)

print("=" * 80)
print("REASONING TEXT:")
print("-" * 40)
print(repr(result.reasoning_text))
print()
print("=" * 80)
print("NORMAL TEXT (includes tool calls for further processing):")
print("-" * 40)
print(repr(result.normal_text))
print()

# Now test if the tool call parser can handle the normal_text
from sglang.srt.function_call.gpt_oss_detector import GPTOSSDetector as ToolCallDetector
from sglang.srt.entrypoints.openai.protocol import Tool, ToolFunction

# Create a dummy tool for testing
tools = [
    Tool(
        type="function",
        function=ToolFunction(
            name="sales_search",
            description="Allows to search for sales rep data.",
            parameters={
                "type": "object",
                "properties": {
                    "firstName": {
                        "type": "string",
                        "description": "FirstName of the sales rep.",
                    }
                },
                "required": ["firstName"],
            }
        )
    )
]

# Test tool call detection
tool_detector = ToolCallDetector()
if tool_detector.has_tool_call(result.normal_text):
    print("=" * 80)
    print("TOOL CALL DETECTION:")
    print("-" * 40)
    tool_result = tool_detector.detect_and_parse(result.normal_text, tools)
    print(f"Tool calls found: {len(tool_result.calls)}")
    for call in tool_result.calls:
        print(f"  - Function: {call.name}")
        print(f"    Parameters: {call.parameters}")
    print()
    print("Final text after tool extraction:")
    print(repr(tool_result.normal_text))
else:
    print("No tool calls detected in normal_text")
