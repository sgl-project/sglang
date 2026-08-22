import json
import unittest

from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.deepseekv4_detector import DeepSeekV4Detector
from sglang.srt.function_call.deepseekv32_detector import DeepSeekV32Detector
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cpu_ci(est_time=61, suite="base-c-test-cpu")


class TestBaseFormatDetector(unittest.TestCase):
    """Test buffer management and sequential tool index assignment in BaseFormatDetector."""

    def setUp(self):
        """Set up test detector and tools."""

        # Create a concrete implementation of BaseFormatDetector for testing
        class TestFormatDetector(BaseFormatDetector):
            def __init__(self):
                super().__init__()
                self.bot_token = "<tool_call>"
                self.eot_token = "</tool_call>"

            def detect_and_parse(self, text, tools):
                # Not used in streaming tests
                pass

            def has_tool_call(self, text):
                return "<tool_call>" in text

            def structure_info(self):
                # Not used in streaming tests
                pass

        self.detector = TestFormatDetector()
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_tourist_attractions",
                    description="Get tourist attractions",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "City name",
                            }
                        },
                        "required": ["city"],
                    },
                ),
            ),
        ]

    def test_sequential_tool_index_assignment(self):
        """Test that multiple tool calls get sequential tool_index values (0, 1, 2, ...)."""
        # Simulate streaming chunks for two consecutive tool calls
        chunks = [
            "<tool_call>",
            '{"name": "get_weather", ',
            '"arguments": {"city": "Paris"}}',
            ", ",
            '{"name": "get_tourist_attractions", ',
            '"arguments": {"city": "London"}}',
            "</tool_call>",
        ]

        tool_indices_seen = []

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

            if result.calls:
                for call in result.calls:
                    if call.tool_index is not None:
                        tool_indices_seen.append(call.tool_index)

        # Verify we got sequential tool indices
        unique_indices = sorted(set(tool_indices_seen))
        self.assertEqual(
            unique_indices,
            [0, 1],
            f"Expected sequential tool indices [0, 1], got {unique_indices}",
        )

    def test_buffer_content_preservation(self):
        """Test that buffer correctly preserves unprocessed content when tool completes."""
        # Test simpler scenario: tool completion followed by new tool start
        chunks = [
            "<tool_call>",
            '{"name": "get_weather", ',
            '"arguments": {"city": "Paris"}}',
            ", ",
            '{"name": "get_tourist_attractions", ',
            '"arguments": {"city": "London"}} </tool_call>',
        ]

        tool_calls_seen = []

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                for call in result.calls:
                    if (
                        call.name
                    ):  # Only count calls with names (not just parameter updates)
                        tool_calls_seen.append(call.name)

        # Should see both tool names
        self.assertIn("get_weather", tool_calls_seen, "Should process first tool")
        self.assertIn(
            "get_tourist_attractions", tool_calls_seen, "Should process second tool"
        )

    def test_current_tool_id_increment_on_completion(self):
        """Test that current_tool_id increments when a tool completes."""
        # Initial state
        self.assertEqual(
            self.detector.current_tool_id, -1, "Should start with current_tool_id=-1"
        )

        # Process first tool completely
        chunks = [
            "<tool_call>",
            '{"name": "get_weather", ',
        ]

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)

        self.assertEqual(
            self.detector.current_tool_id, 0, "current_tool_id should be 0"
        )
        self.assertEqual(
            result.calls[0].name, "get_weather", "The first tool should be get_weather"
        )
        self.assertEqual(
            result.calls[0].tool_index, 0, "The first tool index should be 0"
        )

        # Complete second tool name - this should show that current_tool_id is now 1
        result = self.detector.parse_streaming_increment(
            '"arguments": {"city": "Paris"}}, {"name": "get_', self.tools
        )
        self.assertEqual(result.calls[0].parameters, '{"city": "Paris"}')

        self.assertEqual(
            self.detector.current_tool_id,
            1,
            "current_tool_id should be 1 after first tool completes and second tool starts",
        )

        result = self.detector.parse_streaming_increment(
            'tourist_attractions", ', self.tools
        )

        # Second tool should have tool_index=1
        tourist_calls = [
            call for call in result.calls if call.name == "get_tourist_attractions"
        ]
        self.assertEqual(
            tourist_calls[0].tool_index, 1, "Second tool should have tool_index=1"
        )

    def test_buffer_reset_on_invalid_tool(self):
        """Test that buffer and state are reset when an invalid tool name is encountered."""
        # Start fresh with an invalid tool name from the beginning
        result = self.detector.parse_streaming_increment(
            '<tool_call>{"name": "invalid_tool", ', self.tools
        )

        # Should return empty result and reset state
        self.assertEqual(result.calls, [], "Should return no calls for invalid tool")
        self.assertEqual(
            self.detector.current_tool_id,
            -1,
            "current_tool_id should remain -1 for invalid tool",
        )
        self.assertEqual(
            self.detector._buffer, "", "Buffer should be cleared for invalid tool"
        )

    def test_chinese_characters_not_double_escaped(self):
        """Test that Chinese characters in tool call parameters are not double-escaped."""
        # Test with Chinese city name "杭州" (Hangzhou)
        chunks = [
            "<tool_call>",
            '{"name": "get_weather", ',
            '"arguments": {"city": "杭州"}}',
            "</tool_call>",
        ]

        accumulated_parameters = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                for call in result.calls:
                    if call.parameters:
                        tool_idx = call.tool_index if call.tool_index is not None else 0
                        if tool_idx not in accumulated_parameters:
                            accumulated_parameters[tool_idx] = ""
                        accumulated_parameters[tool_idx] += call.parameters

        # Verify that Chinese characters are preserved (not escaped as \uXXXX)
        self.assertGreater(
            len(accumulated_parameters), 0, "Should have parsed parameters"
        )
        final_params_str = accumulated_parameters[0]

        # The parameters string should contain the actual Chinese characters, not escaped Unicode
        self.assertIn(
            "杭州", final_params_str, "Should contain actual Chinese characters"
        )
        self.assertNotIn(
            "\\u676d", final_params_str, "Should not contain escaped Unicode sequences"
        )
        self.assertNotIn(
            "\\u5dde", final_params_str, "Should not contain escaped Unicode sequences"
        )

        # Verify the JSON can be parsed and contains the correct value
        params = json.loads(final_params_str)
        self.assertEqual(
            params["city"], "杭州", "Should correctly parse Chinese city name"
        )

    def test_chinese_characters_incremental_streaming(self):
        """Test that Chinese characters work correctly with incremental streaming."""
        # Test incremental streaming with Chinese characters
        chunks = [
            "<tool_call>",
            '{"name": "get_weather", ',
            '"arguments": {"city": "',
            "杭州",
            '"}}',
            "</tool_call>",
        ]

        accumulated_parameters = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            if result.calls:
                for call in result.calls:
                    if call.parameters:
                        tool_idx = call.tool_index if call.tool_index is not None else 0
                        if tool_idx not in accumulated_parameters:
                            accumulated_parameters[tool_idx] = ""
                        accumulated_parameters[tool_idx] += call.parameters

        # Verify Chinese characters are preserved throughout streaming
        self.assertGreater(
            len(accumulated_parameters), 0, "Should have parsed parameters"
        )
        final_params_str = accumulated_parameters[0]

        # Should contain actual Chinese characters, not escaped
        self.assertIn(
            "杭州", final_params_str, "Should contain actual Chinese characters"
        )

        # Parse and verify
        params = json.loads(final_params_str)
        self.assertEqual(
            params["city"], "杭州", "Should correctly parse Chinese city name"
        )


class TestDeepSeekV32Detector(unittest.TestCase):
    def setUp(self):
        """Set up test tools and detector for DeepSeekV32 format testing."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Searches for information related to query and displays topn results.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string",
                            },
                            "topn": {
                                "type": "integer",
                                "description": "Number of top results to display",
                                "default": 10,
                            },
                            "source": {
                                "type": "string",
                                "description": "Source to search within",
                                "enum": ["web", "news"],
                                "default": "web",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_favorite_tourist_spot",
                    description="Return the favorite tourist spot for a given city.",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = DeepSeekV32Detector()
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        self.tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3.2")
        self.interval = 1

    def test_detect_and_parse_xml_format(self):
        """Test parsing standard XML format (DSML)"""
        text = """I'll help you with information about San Francisco and get its favorite tourist spot for you.\n\n
        <｜DSML｜function_calls>\n
            <｜DSML｜invoke name="get_favorite_tourist_spot">\n
                <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>\n
            </｜DSML｜invoke>\n
            <｜DSML｜invoke name="search">
                <｜DSML｜parameter name="query" string="true">WebNav benchmark</｜DSML｜parameter>
                <｜DSML｜parameter name="topn" string="false">10</｜DSML｜parameter>
                <｜DSML｜parameter name="source" string="true">web</｜DSML｜parameter>
            </｜DSML｜invoke>
        </｜DSML｜function_calls>
        """
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertIn("I'll help you with information", result.normal_text)
        self.assertEqual(len(result.calls), 2)

        # Check first call
        call1 = result.calls[0]
        self.assertEqual(call1.name, "get_favorite_tourist_spot")
        params1 = json.loads(call1.parameters)
        self.assertEqual(params1["city"], "San Francisco")

        # Check second call
        call2 = result.calls[1]
        self.assertEqual(call2.name, "search")
        params2 = json.loads(call2.parameters)
        self.assertEqual(params2["query"], "WebNav benchmark")
        self.assertEqual(params2["topn"], 10)
        self.assertEqual(params2["source"], "web")

    def test_detect_and_parse_json_format(self):
        """Test parsing JSON format inside invoke tags"""
        text = """I'll help you with information about San Francisco and get its favorite tourist spot for you.

        <｜DSML｜function_calls>
            <｜DSML｜invoke name="get_favorite_tourist_spot">
            {
                "city": "San Francisco"
            }
        </｜DSML｜invoke>
            <｜DSML｜invoke name="search">
            {
                "query": "WebNav benchmark",
                "topn": 10,
                "source": "web"
            }
        </｜DSML｜invoke>
        </｜DSML｜function_calls>
        """
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertIn("I'll help you with information", result.normal_text)
        self.assertEqual(len(result.calls), 2)

        # Check first call
        call1 = result.calls[0]
        self.assertEqual(call1.name, "get_favorite_tourist_spot")
        params1 = json.loads(call1.parameters)
        self.assertEqual(params1["city"], "San Francisco")

        # Check second call
        call2 = result.calls[1]
        self.assertEqual(call2.name, "search")
        params2 = json.loads(call2.parameters)
        self.assertEqual(params2["query"], "WebNav benchmark")
        self.assertEqual(params2["topn"], 10)
        self.assertEqual(params2["source"], "web")

    def test_streaming_xml_format(self):
        """Test streaming parsing of XML format"""
        text = """<｜DSML｜function_calls>
            <｜DSML｜invoke name="get_favorite_tourist_spot">
                <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
                <｜DSML｜parameter name="another_city" string="true">London</｜DSML｜parameter>
                <｜DSML｜parameter name="topn" string="false">10</｜DSML｜parameter>
                <｜DSML｜parameter name="obj" string="false">{"name": "John", "age": 30}</｜DSML｜parameter>
            </｜DSML｜invoke>
        </｜DSML｜function_calls>"""

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            input_ids[i : i + self.interval]
            for i in range(0, len(input_ids), self.interval)
        ]
        chunks = [self.tokenizer.decode(chunk_id) for chunk_id in chunk_ids]

        tool_calls_by_index = {}

        num_tool_call_chunks = 0
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for call in result.calls:
                num_tool_call_chunks += 1
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

        self.assertGreater(num_tool_call_chunks, 8)

        self.assertEqual(len(tool_calls_by_index), 1)
        self.assertEqual(tool_calls_by_index[0]["name"], "get_favorite_tourist_spot")
        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["city"], "San Francisco")
        self.assertEqual(params["another_city"], "London")
        self.assertEqual(params["topn"], 10)
        self.assertEqual(params["obj"]["name"], "John")
        self.assertEqual(params["obj"]["age"], 30)

    def test_streaming_json_format(self):
        """Test streaming parsing of JSON format"""
        text = """<｜DSML｜function_calls>
            <｜DSML｜invoke name="get_favorite_tourist_spot">
            {
                "city": "San Francisco",
                "another_city": "London",
                "topn": 10,
                "obj": {
                    "name": "John",
                    "age": 30
                }
            }
            </｜DSML｜invoke>
        </｜DSML｜function_calls>"""

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            input_ids[i : i + self.interval]
            for i in range(0, len(input_ids), self.interval)
        ]
        chunks = [self.tokenizer.decode(chunk_id) for chunk_id in chunk_ids]

        tool_calls_by_index = {}

        num_tool_call_chunks = 0
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for call in result.calls:
                num_tool_call_chunks += 1
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

        self.assertGreater(num_tool_call_chunks, 8)
        self.assertEqual(len(tool_calls_by_index), 1)
        self.assertEqual(tool_calls_by_index[0]["name"], "get_favorite_tourist_spot")

        # Clean up parameters string if needed (trim whitespace)
        params_str = tool_calls_by_index[0]["parameters"].strip()
        params = json.loads(params_str)
        self.assertEqual(params["city"], "San Francisco")

    def test_detect_and_parse_no_parameters(self):
        """Test parsing function calls with no parameters (non-streaming)"""
        # Add a no-parameter tool
        tools_with_no_param = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_date",
                    description="Get the current date.",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

        text = """Let me get the current date for you.

<｜DSML｜function_calls>
<｜DSML｜invoke name="get_date">
</｜DSML｜invoke>
</｜DSML｜function_calls>"""

        result = self.detector.detect_and_parse(text, tools_with_no_param)

        self.assertIn("Let me get the current date", result.normal_text)
        self.assertEqual(len(result.calls), 1)

        call = result.calls[0]
        self.assertEqual(call.name, "get_date")
        params = json.loads(call.parameters)
        self.assertEqual(params, {})

    def test_streaming_no_parameters(self):
        """Test streaming parsing of function calls with no parameters.

        This test verifies the fix for the bug where functions with no parameters
        were being silently skipped in streaming mode.
        """
        # Add a no-parameter tool
        tools_with_no_param = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_date",
                    description="Get the current date.",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

        text = """<｜DSML｜function_calls>
<｜DSML｜invoke name="get_date">
</｜DSML｜invoke>
</｜DSML｜function_calls>"""

        # Reset detector state
        self.detector = DeepSeekV32Detector()

        # Simulate streaming by splitting into small chunks
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            input_ids[i : i + self.interval]
            for i in range(0, len(input_ids), self.interval)
        ]
        chunks = [self.tokenizer.decode(chunk_id) for chunk_id in chunk_ids]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, tools_with_no_param)
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

        # Verify that the no-parameter function was correctly parsed
        self.assertEqual(
            len(tool_calls_by_index), 1, "Should have exactly one tool call"
        )
        self.assertEqual(tool_calls_by_index[0]["name"], "get_date")

        # Parameters should be empty JSON object
        params_str = tool_calls_by_index[0]["parameters"].strip()
        params = json.loads(params_str)
        self.assertEqual(params, {})

    def test_streaming_no_parameters_with_whitespace(self):
        """Test streaming parsing when invoke content has only whitespace (newlines)."""
        tools_with_no_param = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="get_date",
                    description="Get the current date.",
                    parameters={"type": "object", "properties": {}},
                ),
            ),
        ]

        # This format has newlines inside the invoke tag (common model output)
        text = """<｜DSML｜function_calls>
<｜DSML｜invoke name="get_date">

</｜DSML｜invoke>
</｜DSML｜function_calls>"""

        # Reset detector state
        self.detector = DeepSeekV32Detector()

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            input_ids[i : i + self.interval]
            for i in range(0, len(input_ids), self.interval)
        ]
        chunks = [self.tokenizer.decode(chunk_id) for chunk_id in chunk_ids]

        tool_calls_by_index = {}

        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, tools_with_no_param)
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

        # Should still parse correctly even with whitespace-only content
        self.assertEqual(
            len(tool_calls_by_index), 1, "Should have exactly one tool call"
        )
        self.assertEqual(tool_calls_by_index[0]["name"], "get_date")
        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params, {})

    def test_get_model_structural_tag(self):
        import xgrammar as xgr

        self.assertEqual(self.detector.get_structural_tag_name(), "deepseek_v3_2")

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)
        serialized = structural_tag.model_dump_json()
        self.assertIn("</｜DSML｜invoke>\\n", serialized)
        self.assertNotIn("</｜DSML｜invoke>\\n\\n", serialized)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True, tool_choice="required"
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False, tool_choice="required"
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        tool_choice_name = ToolChoiceFuncName(name="search")
        tool_choice = ToolChoice(function=tool_choice_name)
        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True, tool_choice=tool_choice
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False, tool_choice=tool_choice
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

    def test_self_closing_zero_arg_invoke(self):
        """V32 inherits the same regex; verify self-closing parses to empty
        params here too (V32 model rarely emits this shape, but the parser
        must agree with V4 since V4 inherits from V32)."""
        submit_tool = Tool(
            type="function",
            function=Function(
                name="submit",
                parameters={"type": "object", "properties": {}},
            ),
        )
        text = (
            '<｜DSML｜function_calls>\n<｜DSML｜invoke name="submit"/>\n'
            "</｜DSML｜function_calls>"
        )
        result = self.detector.detect_and_parse(text, [submit_tool])
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "submit")
        self.assertEqual(json.loads(result.calls[0].parameters), {})


class TestDeepSeekV4Detector(unittest.TestCase):
    def setUp(self):
        """Set up test tools and detector for DeepSeekV4 format testing."""
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Searches for information related to query and displays topn results.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "The search query string",
                            },
                            "topn": {
                                "type": "integer",
                                "description": "Number of top results to display",
                                "default": 10,
                            },
                            "source": {
                                "type": "string",
                                "description": "Source to search within",
                                "enum": ["web", "news"],
                                "default": "web",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_favorite_tourist_spot",
                    description="Return the favorite tourist spot for a given city.",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                ),
            ),
        ]
        self.detector = DeepSeekV4Detector()
        from sglang.srt.utils.hf_transformers_utils import get_tokenizer

        self.tokenizer = get_tokenizer("deepseek-ai/DeepSeek-V3.2")
        self.interval = 1

    def test_detect_and_parse_xml_format(self):
        """Test parsing standard XML format (DSML)"""
        text = """I'll help you with information about San Francisco and get its favorite tourist spot for you.\n\n
        <｜DSML｜tool_calls>\n
            <｜DSML｜invoke name="get_favorite_tourist_spot">\n
                <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>\n
            </｜DSML｜invoke>\n
            <｜DSML｜invoke name="search">
                <｜DSML｜parameter name="query" string="true">WebNav benchmark</｜DSML｜parameter>
                <｜DSML｜parameter name="topn" string="false">10</｜DSML｜parameter>
                <｜DSML｜parameter name="source" string="true">web</｜DSML｜parameter>
            </｜DSML｜invoke>
        </｜DSML｜tool_calls>
        """
        result = self.detector.detect_and_parse(text, self.tools)

        self.assertIn("I'll help you with information", result.normal_text)
        self.assertEqual(len(result.calls), 2)

        # Check first call
        call1 = result.calls[0]
        self.assertEqual(call1.name, "get_favorite_tourist_spot")
        params1 = json.loads(call1.parameters)
        self.assertEqual(params1["city"], "San Francisco")

        # Check second call
        call2 = result.calls[1]
        self.assertEqual(call2.name, "search")
        params2 = json.loads(call2.parameters)
        self.assertEqual(params2["query"], "WebNav benchmark")
        self.assertEqual(params2["topn"], 10)
        self.assertEqual(params2["source"], "web")

    def test_streaming_xml_format(self):
        """Test streaming parsing of XML format"""
        text = """<｜DSML｜tool_calls>
            <｜DSML｜invoke name="get_favorite_tourist_spot">
                <｜DSML｜parameter name="city" string="true">San Francisco</｜DSML｜parameter>
                <｜DSML｜parameter name="another_city" string="true">London</｜DSML｜parameter>
                <｜DSML｜parameter name="topn" string="false">10</｜DSML｜parameter>
                <｜DSML｜parameter name="obj" string="false">{"name": "John", "age": 30}</｜DSML｜parameter>
            </｜DSML｜invoke>
        </｜DSML｜tool_calls>"""

        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunk_ids = [
            input_ids[i : i + self.interval]
            for i in range(0, len(input_ids), self.interval)
        ]
        chunks = [self.tokenizer.decode(chunk_id) for chunk_id in chunk_ids]

        tool_calls_by_index = {}

        num_tool_call_chunks = 0
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
            for call in result.calls:
                num_tool_call_chunks += 1
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

        self.assertGreater(num_tool_call_chunks, 8)

        self.assertEqual(len(tool_calls_by_index), 1)
        self.assertEqual(tool_calls_by_index[0]["name"], "get_favorite_tourist_spot")
        params = json.loads(tool_calls_by_index[0]["parameters"])
        self.assertEqual(params["city"], "San Francisco")
        self.assertEqual(params["another_city"], "London")
        self.assertEqual(params["topn"], 10)
        self.assertEqual(params["obj"]["name"], "John")
        self.assertEqual(params["obj"]["age"], 30)

    def test_get_model_structural_tag(self):
        import xgrammar as xgr

        self.assertEqual(self.detector.get_structural_tag_name(), "deepseek_v4")

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)
        serialized = structural_tag.model_dump_json()
        self.assertIn("</｜DSML｜invoke>\\n", serialized)
        self.assertNotIn("</｜DSML｜invoke>\\n\\n", serialized)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True, tool_choice="required"
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False, tool_choice="required"
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        tool_choice_name = ToolChoiceFuncName(name="search")
        tool_choice = ToolChoice(function=tool_choice_name)
        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=True, tool_choice=tool_choice
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

        structural_tag = self.detector.get_structural_tag(
            self.tools, thinking_mode=False, tool_choice=tool_choice
        )
        self.assertIsInstance(structural_tag, xgr.StructuralTag)
        grammar = xgr.Grammar.from_structural_tag(structural_tag)
        self.assertIsInstance(grammar, xgr.Grammar)

    def test_self_closing_zero_arg_invoke(self):
        """V4 emits `<｜DSML｜invoke name="x"/>` for zero-arg tools; the
        detector must parse it as a complete tool call with empty params
        instead of leaking the raw markup back into normal_text."""
        submit_tool = Tool(
            type="function",
            function=Function(
                name="submit",
                description="Submit the final answer.",
                parameters={"type": "object", "properties": {}},
            ),
        )

        text = (
            "Final answer.\n"
            '<｜DSML｜tool_calls>\n<｜DSML｜invoke name="submit"/>\n'
            "</｜DSML｜tool_calls>"
        )
        result = self.detector.detect_and_parse(text, [submit_tool])
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "submit")
        self.assertEqual(json.loads(result.calls[0].parameters), {})
        self.assertNotIn("DSML", result.normal_text)

    def test_self_closing_mixed_with_long_form(self):
        """Mix of long-form (with params) and self-closing tags in one block."""
        submit_tool = Tool(
            type="function",
            function=Function(
                name="submit",
                parameters={"type": "object", "properties": {}},
            ),
        )
        text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="get_favorite_tourist_spot">\n'
            '<｜DSML｜parameter name="city" string="true">SF</｜DSML｜parameter>\n'
            "</｜DSML｜invoke>\n"
            '<｜DSML｜invoke name="submit"/>\n'
            "</｜DSML｜tool_calls>"
        )
        result = self.detector.detect_and_parse(text, self.tools + [submit_tool])
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_favorite_tourist_spot")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "SF"})
        self.assertEqual(result.calls[1].name, "submit")
        self.assertEqual(json.loads(result.calls[1].parameters), {})

    def test_streaming_self_closing_invoke(self):
        """Self-closing invoke must terminate cleanly even when `/>` arrives
        after the `name=` attribute crosses chunk boundaries."""
        submit_tool = Tool(
            type="function",
            function=Function(
                name="submit",
                parameters={"type": "object", "properties": {}},
            ),
        )
        # Build the prompt and feed it through the tokenizer to exercise the
        # same chunk shapes the runtime sees.
        text = (
            "<｜DSML｜tool_calls>\n"
            '<｜DSML｜invoke name="submit"/>\n'
            "</｜DSML｜tool_calls>"
        )
        self.detector = DeepSeekV4Detector()
        input_ids = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = [
            self.tokenizer.decode(input_ids[i : i + self.interval])
            for i in range(0, len(input_ids), self.interval)
        ]

        tool_calls_by_index = {}
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, [submit_tool])
            for call in result.calls:
                if call.tool_index is None:
                    continue
                slot = tool_calls_by_index.setdefault(
                    call.tool_index, {"name": "", "parameters": ""}
                )
                if call.name:
                    slot["name"] = call.name
                if call.parameters:
                    slot["parameters"] += call.parameters

        self.assertEqual(len(tool_calls_by_index), 1)
        self.assertEqual(tool_calls_by_index[0]["name"], "submit")
        self.assertEqual(json.loads(tool_calls_by_index[0]["parameters"]), {})


class TestJsonArrayParser(unittest.TestCase):
    def setUp(self):
        # Create sample tools for testing
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "properties": {
                            "location": {
                                "type": "string",
                                "description": "Location to get weather for",
                            },
                            "unit": {
                                "type": "string",
                                "description": "Temperature unit",
                                "enum": ["celsius", "fahrenheit"],
                            },
                        },
                        "required": ["location"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search for information",
                    parameters={
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query",
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ),
        ]
        self.detector = JsonArrayParser()

    def test_parse_streaming_increment_malformed_json(self):
        """Test parsing with malformed JSON"""
        # Test with malformed JSON
        text = '[{"name": "get_weather", "parameters": {"location": "Tokyo"'
        result = self.detector.parse_streaming_increment(text, self.tools)

        # Should not crash and return a valid result
        self.assertIsInstance(result, StreamingParseResult)

        text = "[{}}}]"
        result = self.detector.parse_streaming_increment(text, self.tools)

        self.assertIsInstance(result, StreamingParseResult)

    def test_parse_streaming_increment_empty_input(self):
        """Test parsing with empty input"""
        result = self.detector.parse_streaming_increment("", self.tools)
        self.assertEqual(len(result.calls), 0)
        self.assertEqual(result.normal_text, "")

    def test_braces_in_strings(self):
        """Test that JSON with } characters inside strings works correctly"""
        # Test case: JSON array with } inside string values - streamed across chunks
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "has } inside"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = "}}"
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)
        self.assertGreater(
            len(result2.calls), 0, "Should parse tool call with } in string"
        )

        # Test with separator (streaming in progress)
        chunk3 = '[{"name": "get_weather", "parameters": {"location": "has } inside"}'
        result3 = self.detector.parse_streaming_increment(chunk3, self.tools)
        self.assertIsInstance(result3, StreamingParseResult)
        chunk4 = "},"
        result4 = self.detector.parse_streaming_increment(chunk4, self.tools)
        self.assertIsInstance(result4, StreamingParseResult)
        chunk5 = '{"name": "get_weather"'
        result5 = self.detector.parse_streaming_increment(chunk5, self.tools)
        self.assertIsInstance(result5, StreamingParseResult)
        self.assertGreater(
            len(result5.calls),
            0,
            "Should parse tool calls with separator and } in string",
        )

    def test_separator_in_same_chunk(self):
        """Test that separator already present in chunk works correctly"""
        # Test case: separator already in the chunk (streaming in progress) with 2+ chunks per tool call
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tokyo"'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = '}},{"name": "get_weather"'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)
        self.assertGreater(
            len(result2.calls),
            0,
            "Should parse tool calls with separator in same chunk",
        )

    def test_nested_objects_with_commas(self):
        """Test that nested objects with commas inside work correctly"""
        # Test with nested objects that have commas - should work with json.loads()
        chunk1 = '[{"name": "get_weather", "parameters": {"location": "Tok'
        result1 = self.detector.parse_streaming_increment(chunk1, self.tools)
        self.assertIsInstance(result1, StreamingParseResult)
        chunk2 = 'yo", "unit": "celsius"}}'
        result2 = self.detector.parse_streaming_increment(chunk2, self.tools)
        self.assertIsInstance(result2, StreamingParseResult)
        self.assertGreater(
            len(result2.calls), 0, "Should parse tool call with nested objects"
        )

    def test_three_tool_calls_separate_chunks_with_commas(self):
        """Test parsing 3 tool calls in separate chunks with commas at the end"""
        # First tool call: 2 chunks
        chunk1_1 = '[{"name": "get_weather", "parameters": '
        result1_1 = self.detector.parse_streaming_increment(chunk1_1, self.tools)
        chunk1_2 = '{"location": "Tokyo"}},'
        result1_2 = self.detector.parse_streaming_increment(chunk1_2, self.tools)
        self.assertIsInstance(result1_2, StreamingParseResult)
        self.assertGreater(len(result1_2.calls), 0, "Should parse first tool call")

        # Second tool call: 2 chunks
        chunk2_1 = '{"name": "search", "parameters": '
        result2_1 = self.detector.parse_streaming_increment(chunk2_1, self.tools)
        chunk2_2 = '{"query": "restaurants"}},'
        result2_2 = self.detector.parse_streaming_increment(chunk2_2, self.tools)
        self.assertIsInstance(result2_2, StreamingParseResult)
        self.assertGreater(len(result2_2.calls), 0, "Should parse second tool call")

        # Third tool call: 2 chunks
        chunk3_1 = '{"name": "get_weather", "parameters": '
        result3_1 = self.detector.parse_streaming_increment(chunk3_1, self.tools)
        chunk3_2 = '{"location": "Paris"}}]'
        result3_2 = self.detector.parse_streaming_increment(chunk3_2, self.tools)
        self.assertIsInstance(result3_2, StreamingParseResult)
        self.assertGreater(len(result3_2.calls), 0, "Should parse third tool call")
        # Verify all tool calls were parsed correctly
        total_calls = len(result1_2.calls) + len(result2_2.calls) + len(result3_2.calls)
        self.assertEqual(total_calls, 3, "Should have parsed exactly 3 tool calls")


class TestGetStructureConstraint(unittest.TestCase):
    """Tests for FunctionCallParser.get_structure_constraint() logic.

    Verifies that detectors supporting structural_tag use it for required/named
    tool_choice, and that the generic json_schema fallback is used otherwise.
    """

    def _make_tools(self, strict=False):
        return [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather",
                    parameters={
                        "type": "object",
                        "properties": {"city": {"type": "string"}},
                        "required": ["city"],
                    },
                    strict=strict,
                ),
            ),
        ]

    def _make_parser(self, parser_name, strict=False):
        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        return FunctionCallParser(self._make_tools(strict=strict), parser_name)

    def _constraint_json(self, result):
        return result[1].model_dump_json()

    # --- structural_tag detectors (kimi_k2, deepseekv3, qwen25, etc.) ---

    def test_kimi_required_strict_returns_structural_tag(self):
        import xgrammar as xgr

        parser = self._make_parser("kimi_k2", strict=True)
        result = parser.get_structure_constraint("required")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "structural_tag")
        self.assertIsInstance(result[1], xgr.StructuralTag)
        self.assertIn("<|tool_calls_section_begin|>", self._constraint_json(result))

    def test_kimi_required_no_strict_returns_structural_tag(self):
        """required should use structural_tag even without strict, to preserve native format."""
        import xgrammar as xgr

        parser = self._make_parser("kimi_k2", strict=False)
        result = parser.get_structure_constraint("required")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "structural_tag")
        self.assertIsInstance(result[1], xgr.StructuralTag)
        self.assertIn("<|tool_calls_section_begin|>", self._constraint_json(result))

    def test_kimi_auto_strict_returns_structural_tag(self):
        import xgrammar as xgr

        parser = self._make_parser("kimi_k2", strict=True)
        result = parser.get_structure_constraint("auto")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "structural_tag")
        self.assertIsInstance(result[1], xgr.StructuralTag)
        serialized = self._constraint_json(result)
        self.assertIn('"type":"triggered_tags"', serialized)
        self.assertIn("<|tool_calls_section_begin|>", serialized)

    def test_kimi_auto_no_strict_returns_none(self):
        """auto without strict should not constrain."""
        parser = self._make_parser("kimi_k2", strict=False)
        result = parser.get_structure_constraint("auto")
        self.assertIsNone(result)

    def test_inkling_auto_constrains_json_after_tool_trigger(self):
        import xgrammar as xgr

        from sglang.srt.parser.inkling_tokenizer import INKLING_SPECIAL_TOKEN_IDS

        parser = self._make_parser("inkling", strict=False)
        result = parser.get_structure_constraint("auto")

        self.assertIsNotNone(result)
        self.assertEqual(result[0], "structural_tag")
        self.assertIsInstance(result[1], xgr.StructuralTag)
        format_ = result[1].model_dump()["format"]
        self.assertEqual(format_["type"], "token_triggered_tags")
        self.assertEqual(
            format_["trigger_tokens"],
            [INKLING_SPECIAL_TOKEN_IDS["<|content_invoke_tool_json|>"]],
        )
        tag = format_["tags"][0]
        self.assertEqual(
            tag["end"]["token"], INKLING_SPECIAL_TOKEN_IDS["<|end_message|>"]
        )
        schema = tag["content"]["json_schema"]
        self.assertEqual(schema["required"], ["name", "args"])
        self.assertFalse(schema["additionalProperties"])

    def test_kimi_named_tool_choice_returns_structural_tag(self):
        from sglang.srt.entrypoints.openai.protocol import (
            ToolChoice,
            ToolChoiceFuncName,
        )

        parser = self._make_parser("kimi_k2", strict=False)
        tool_choice = ToolChoice(function=ToolChoiceFuncName(name="get_weather"))
        result = parser.get_structure_constraint(tool_choice)
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "structural_tag")

    def test_deepseekv3_required_no_strict_returns_structural_tag(self):
        parser = self._make_parser("deepseekv3", strict=False)
        result = parser.get_structure_constraint("required")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "structural_tag")

    def test_qwen25_required_no_strict_returns_structural_tag(self):
        parser = self._make_parser("qwen25", strict=False)
        result = parser.get_structure_constraint("required")
        self.assertIsNotNone(result)
        self.assertEqual(result[0], "structural_tag")

    # --- structural_tag content verification ---

    def test_kimi_structural_tag_has_kimi_tokens(self):
        """Verify structural_tag contains kimi-specific special tokens."""
        parser = self._make_parser("kimi_k2", strict=True)
        result = parser.get_structure_constraint("required")
        serialized = self._constraint_json(result)
        self.assertIn("<|tool_calls_section_begin|>", serialized)
        self.assertIn("functions.get_weather:", serialized)
        self.assertIn('"pattern":"\\\\d+"', serialized)
        self.assertIn("<|tool_call_end|>", serialized)
        self.assertIn("<|tool_calls_section_end|>", serialized)

    def test_kimi_required_no_strict_uses_loose_object_schema(self):
        """Kimi required calls keep non-strict arguments object-shaped but loose."""
        parser = self._make_parser("kimi_k2", strict=False)
        result = parser.get_structure_constraint("required")
        serialized = self._constraint_json(result)
        self.assertIn('"json_schema":{"type":"object"}', serialized)
        self.assertNotIn('"additionalProperties":false', serialized)
        self.assertNotIn('"properties"', serialized)

    def test_kimi_required_strict_uses_tool_schema(self):
        """With strict, native xgrammar should include the tool's parameter schema."""
        parser = self._make_parser("kimi_k2", strict=True)
        result = parser.get_structure_constraint("required")
        serialized = self._constraint_json(result)
        self.assertIn('"properties"', serialized)
        self.assertIn('"city"', serialized)

    # --- reasoning-prefix ownership ---

    def test_default_thinking_mode_is_false(self):
        """Default must be False so callers don't silently get a reasoning
        prefix added to their grammar (only relevant for detectors routed
        through the xgrammar builtin)."""
        import inspect

        from sglang.srt.function_call.function_call_parser import FunctionCallParser

        sig = inspect.signature(FunctionCallParser.get_structure_constraint)
        self.assertIs(sig.parameters["thinking_mode"].default, False)


if __name__ == "__main__":
    unittest.main()
