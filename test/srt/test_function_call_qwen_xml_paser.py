import json
import unittest

from xgrammar import GrammarCompiler, TokenizerInfo

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector

# from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.function_call.qwen3_coder_new_detector import Qwen3CoderDetector
from sglang.srt.function_call.qwen25_detector import Qwen25Detector
from sglang.srt.hf_transformers_utils import get_tokenizer
from sglang.test.test_utils import DEFAULT_SMALL_MODEL_NAME_FOR_TEST


class TestQwen3CoderDetector(unittest.TestCase):
    def setUp(self):
        # Create sample tools for testing
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_current_weather",
                    description="Get the current weather",
                    parameters={
                        "properties": {
                            "city": {"type": "string", "description": "The city name"},
                            "state": {
                                "type": "string",
                                "description": "The state code",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["fahrenheit", "celsius"],
                            },
                        },
                        "required": ["city", "state"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="calculate_area",
                    description="Calculate area of a shape",
                    parameters={
                        "properties": {
                            "shape": {"type": "string"},
                            "dimensions": {"type": "object"},
                            "precision": {"type": "integer"},
                        }
                    },
                ),
            ),
        ]
        self.detector = Qwen3CoderDetector()

    def test_has_tool_call(self):
        """Test detection of tool call markers."""
        self.assertTrue(self.detector.has_tool_call("<tool_call>test</tool_call>"))
        self.assertFalse(self.detector.has_tool_call("No tool call here"))

    def test_detect_and_parse_no_tools(self):
        """Test parsing text without tool calls."""
        model_output = "This is a test response without any tool calls"
        result = self.detector.detect_and_parse(model_output, tools=[])
        self.assertEqual(result.normal_text, model_output)
        self.assertEqual(result.calls, [])

    def test_detect_and_parse_single_tool(self):
        """Test parsing a single tool call."""
        model_output = """<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>"""

        result = self.detector.detect_and_parse(model_output, tools=self.tools)

        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_current_weather")

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Dallas")
        self.assertEqual(params["state"], "TX")
        self.assertEqual(params["unit"], "fahrenheit")

    def test_detect_and_parse_with_content(self):
        """Test parsing tool call with surrounding text."""
        model_output = """Sure! Let me check the weather for you.<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>"""

        result = self.detector.detect_and_parse(model_output, tools=self.tools)

        self.assertEqual(result.normal_text, "Sure! Let me check the weather for you.")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_current_weather")

    def test_detect_and_parse_multiline_param(self):
        """Test parsing tool call with multiline parameter values."""
        model_output = """<tool_call>
<function=calculate_area>
<parameter=shape>
rectangle
</parameter>
<parameter=dimensions>
{"width": 10,
 "height": 20}
</parameter>
<parameter=precision>
2
</parameter>
</function>
</tool_call>"""

        result = self.detector.detect_and_parse(model_output, tools=self.tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "calculate_area")

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["shape"], "rectangle")
        self.assertEqual(params["dimensions"], {"width": 10, "height": 20})
        self.assertEqual(params["precision"], 2)

    def test_detect_and_parse_parallel_tools(self):
        """Test parsing multiple tool calls."""
        model_output = """<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>
<tool_call>
<function=get_current_weather>
<parameter=city>
Orlando
</parameter>
<parameter=state>
FL
</parameter>
<parameter=unit>
fahrenheit
</parameter>
</function>
</tool_call>"""

        result = self.detector.detect_and_parse(model_output, tools=self.tools)

        self.assertEqual(result.normal_text, "\n")
        self.assertEqual(len(result.calls), 2)

        # First call
        self.assertEqual(result.calls[0].name, "get_current_weather")
        params1 = json.loads(result.calls[0].parameters)
        self.assertEqual(params1["city"], "Dallas")
        self.assertEqual(params1["state"], "TX")

        # Second call
        self.assertEqual(result.calls[1].name, "get_current_weather")
        params2 = json.loads(result.calls[1].parameters)
        self.assertEqual(params2["city"], "Orlando")
        self.assertEqual(params2["state"], "FL")

    def test_edge_case_no_parameters(self):
        """Test tool call without parameters."""
        model_output = """<tool_call>
<function=get_current_weather>
</function>
</tool_call>"""

        result = self.detector.detect_and_parse(model_output, tools=self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_current_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {})

    def test_edge_case_special_chars_in_value(self):
        """Test parameter with special characters in value."""
        model_output = """<tool_call>
<function=get_current_weather>
<parameter=city>
Dallas->TX
</parameter>
</function>
</tool_call>"""

        result = self.detector.detect_and_parse(model_output, tools=self.tools)
        self.assertEqual(len(result.calls), 1)

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["city"], "Dallas->TX")

    def test_extract_tool_calls_fallback_no_tags(self):
        """Test fallback parsing when XML tags are missing (just function without tool_call wrapper)."""
        model_output = """<function=get_current_weather>
<parameter=city>
Dallas
</parameter>
<parameter=state>
TX
</parameter>
</function>"""

        result = self.detector.detect_and_parse(model_output, tools=self.tools)

        self.assertIsNotNone(result)

    def test_extract_tool_calls_type_conversion(self):
        """Test parameter type conversion based on tool schema."""
        test_tool = Tool(
            type="function",
            function=Function(
                name="test_types",
                parameters={
                    "type": "object",
                    "properties": {
                        "int_param": {"type": "integer"},
                        "float_param": {"type": "float"},
                        "bool_param": {"type": "boolean"},
                        "str_param": {"type": "string"},
                        "obj_param": {"type": "object"},
                    },
                },
            ),
        )

        model_output = """<tool_call>
<function=test_types>
<parameter=int_param>
42
</parameter>
<parameter=float_param>
3.14
</parameter>
<parameter=bool_param>
true
</parameter>
<parameter=str_param>
hello world
</parameter>
<parameter=obj_param>
{"key": "value"}
</parameter>
</function>
</tool_call>"""

        result = self.detector.detect_and_parse(model_output, tools=[test_tool])

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["int_param"], 42)
        self.assertEqual(params["float_param"], 3.14)
        self.assertEqual(params["bool_param"], True)
        self.assertEqual(params["str_param"], "hello world")
        self.assertEqual(params["obj_param"], {"key": "value"})

    def test_parse_streaming_simple(self):
        """Test basic streaming parsing."""
        chunks = [
            "Sure! ",
            "Let me check ",
            "the weather.",
            "<tool_call>",
            "\n<function=get_current_weather>",
            "\n<parameter=city>",
            "\nDallas",
            "\n</parameter>",
            "\n<parameter=state>",
            "\nTX",
            "\n</parameter>",
            "\n</function>",
            "\n</tool_call>",
        ]

        accumulated_text = ""
        accumulated_calls = []
        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, tools=self.tools)
            accumulated_text += result.normal_text

            # Track calls by tool_index to handle streaming properly
            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters
        self.assertEqual(accumulated_text, "Sure! Let me check the weather.")
        self.assertEqual(len(tool_calls_by_index), 1)

        # Get the complete tool call
        tool_call = tool_calls_by_index[0]
        self.assertEqual(tool_call["name"], "get_current_weather")

        # Parse the accumulated parameters
        params = json.loads(tool_call["parameters"])
        self.assertEqual(params["city"], "Dallas")
        self.assertEqual(params["state"], "TX")

    def test_parse_streaming_simple_list_value(self):
        """Test basic streaming parsing while parameter has list."""

        tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_current_weather",
                    description="Get the current weather",
                    parameters={
                        "properties": {
                            "city": {"type": "string", "description": "The city name"},
                            "state": {
                                "type": "array",
                                "items": {
                                    "type": "string",
                                },
                                "description": "The state codes",
                            },
                            "unit": {
                                "type": "string",
                                "enum": ["fahrenheit", "celsius"],
                            },
                        },
                        "required": ["city", "state"],
                    },
                ),
            )
        ]
        chunks = [
            "Sure! ",
            "Let me check ",
            "the weather.",
            "<tool_call>",
            "\n<function=get_current_weather>",
            "\n<parameter=city>",
            "\nDallas",
            "\n</parameter>",
            "\n<parameter=state>",
            "\n['SEATTLE/WA','LA.KO']\n",
            "\n</parameter>",
            "\n</function>",
            "\n</tool_call>",
        ]

        accumulated_text = ""
        accumulated_calls = []
        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, tools=tools)
            accumulated_text += result.normal_text

            # Track calls by tool_index to handle streaming properly
            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters
        self.assertEqual(accumulated_text, "Sure! Let me check the weather.")
        self.assertEqual(len(tool_calls_by_index), 1)

        # Get the complete tool call
        tool_call = tool_calls_by_index[0]
        self.assertEqual(tool_call["name"], "get_current_weather")

        # Parse the accumulated parameters
        params = json.loads(tool_call["parameters"])
        self.assertEqual(params["city"], "Dallas")
        self.assertIsInstance(params["state"], list)
        self.assertEqual(params["state"][0], "SEATTLE/WA")
        self.assertEqual(params["state"][1], "LA.KO")

    def test_parse_streaming_iregular_blocks(self):
        """Test streaming parsing using iregular blocks."""
        chunks = [
            "Sure! ",
            "Let me check ",
            "the weather.",
            "<tool_call",
            ">\n<function=",
            "get_current_weather>\n<parameter=city>",
            "\nDallas",
            "\n<",
            "/parameter>\n<parameter=state",
            ">\nTX",
            "\n</parameter>",
            "\n</function>",
            "\n</tool_call>",
        ]

        accumulated_text = ""
        accumulated_calls = []
        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, tools=self.tools)
            accumulated_text += result.normal_text

            # Track calls by tool_index to handle streaming properly
            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters
        self.assertEqual(accumulated_text, "Sure! Let me check the weather.")
        self.assertEqual(len(tool_calls_by_index), 1)

        # Get the complete tool call
        tool_call = tool_calls_by_index[0]
        self.assertEqual(tool_call["name"], "get_current_weather")

        # Parse the accumulated parameters
        params = json.loads(tool_call["parameters"])
        self.assertEqual(params["city"], "Dallas")
        self.assertEqual(params["state"], "TX")

    def test_parse_streaming_incomplete(self):
        """Test streaming with incomplete tool call."""
        # Send incomplete tool call
        chunks = [
            "<tool_call>",
            "\n<function=get_current_weather>",
            "\n<parameter=city>",
            "\nDallas",
            "\n</parameter>",
            "\n<parameter=state>",
            "\nTX",
            # Missing </parameter>, </function>, </tool_call>
        ]

        tool_calls_by_index = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, tools=self.tools)

            # Track calls by tool_index to handle streaming properly
            for call in result.calls:
                if call.tool_index is not None:
                    if call.tool_index not in tool_calls_by_index:
                        tool_calls_by_index[call.tool_index] = {
                            "name": "",
                            "parameters": "",
                        }

                    if call.name:
                        tool_calls_by_index[call.tool_index]["name"] = call.name
                    if call.parameters:
                        tool_calls_by_index[call.tool_index][
                            "parameters"
                        ] += call.parameters

        # Should have partial tool call with name but incomplete parameters
        self.assertGreater(len(tool_calls_by_index), 0)
        self.assertEqual(tool_calls_by_index[0]["name"], "get_current_weather")

        # Parameters should be incomplete (no closing brace)
        params_str = tool_calls_by_index[0]["parameters"]
        self.assertTrue(params_str.startswith('{"city": "Dallas"'))
        self.assertFalse(params_str.endswith("}"))

        # Now complete it
        result = self.detector.parse_streaming_increment(
            "\n</parameter>\n</function>\n</tool_call>", tools=self.tools
        )

        # Update the accumulated parameters
        for call in result.calls:
            if call.tool_index is not None and call.parameters:
                tool_calls_by_index[call.tool_index]["parameters"] += call.parameters

        # Now should have complete parameters
        final_params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(final_params["city"], "Dallas")
        self.assertEqual(final_params["state"], "TX")

    def test_parse_streaming_incremental(self):
        """Test that streaming is truly incremental with very small chunks."""
        model_output = """I'll check the weather.<tool_call>
        <function=get_current_weather>
        <parameter=city>
        Dallas
        </parameter>
        <parameter=state>
        TX
        </parameter>
        </function>
        </tool_call>"""

        # Simulate more realistic token-based chunks where <tool_call> is a single token
        chunks = [
            "I'll check the weather.",
            "<tool_call>",
            "\n<function=get_current_weather>\n",
            "<parameter=city>\n",
            "Dallas\n",
            "</parameter>\n",
            "<parameter=state>\n",
            "TX\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>",
        ]

        accumulated_text = ""
        tool_calls = []
        chunks_count = 0

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            accumulated_text += result.normal_text
            chunks_count += 1
            for tool_call_chunk in result.calls:
                if (
                    hasattr(tool_call_chunk, "tool_index")
                    and tool_call_chunk.tool_index is not None
                ):
                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})
                    tc = tool_calls[tool_call_chunk.tool_index]
                    if tool_call_chunk.name:
                        tc["name"] = tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters

        self.assertGreater(chunks_count, 3)

        # Verify the accumulated results
        self.assertIn("I'll check the weather.", accumulated_text)
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "get_current_weather")

        params = json.loads(tool_calls[0]["parameters"])
        self.assertEqual(params, {"city": "Dallas", "state": "TX"})

    def test_parse_streaming_multiple_tools(self):
        """Test streaming with multiple tool calls."""
        model_output = """<tool_call>
        <function=get_current_weather>
        <parameter=city>
        Dallas
        </parameter>
        <parameter=state>
        TX
        </parameter>
        </function>
        </tool_call>
        Some text in between.
        <tool_call>
        <function=calculate_area>
        <parameter=shape>
        circle
        </parameter>
        <parameter=dimensions>
        {"radius": 5}
        </parameter>
        </function>
        </tool_call>"""

        # Simulate streaming by chunks
        chunk_size = 20
        chunks = [
            model_output[i : i + chunk_size]
            for i in range(0, len(model_output), chunk_size)
        ]

        accumulated_text = ""
        tool_calls = []
        chunks_count = 0

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            accumulated_text += result.normal_text
            chunks_count += 1
            for tool_call_chunk in result.calls:
                if (
                    hasattr(tool_call_chunk, "tool_index")
                    and tool_call_chunk.tool_index is not None
                ):
                    while len(tool_calls) <= tool_call_chunk.tool_index:
                        tool_calls.append({"name": "", "parameters": ""})
                    tc = tool_calls[tool_call_chunk.tool_index]
                    if tool_call_chunk.name:
                        tc["name"] = tool_call_chunk.name
                    if tool_call_chunk.parameters:
                        tc["parameters"] += tool_call_chunk.parameters

        self.assertIn("Some text in between.", accumulated_text)
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["name"], "get_current_weather")
        self.assertEqual(tool_calls[1]["name"], "calculate_area")

        # Verify parameters
        params1 = json.loads(tool_calls[0]["parameters"])
        self.assertEqual(params1, {"city": "Dallas", "state": "TX"})

        params2 = json.loads(tool_calls[1]["parameters"])
        self.assertEqual(params2, {"shape": "circle", "dimensions": {"radius": 5}})


if __name__ == "__main__":
    unittest.main()
