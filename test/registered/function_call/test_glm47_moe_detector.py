import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.core_types import StreamingParseResult
from sglang.srt.function_call.glm4_moe_detector import Glm4MoeDetector
from sglang.srt.function_call.glm47_moe_detector import (
    Glm47MoeDetector,
    get_argument_type,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "default")


class TestGlm47MoeDetector(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"},
                            "date": {"type": "string", "description": "Date"},
                        },
                        "required": ["city", "date"],
                    },
                ),
            ),
        ]
        self.detector = Glm47MoeDetector()

    # ==================== Basic Parsing Tests (5) ====================

    def test_single_tool_call(self):
        """
        Test basic single tool call parsing.

        Scenario: Parse a complete tool call with two string parameters in a single text block.
        Purpose: Verify the detector can correctly identify and extract function name and parameters
                from a simple, well-formed tool call.
        """
        text = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            result.calls[0].parameters, '{"city": "Beijing", "date": "2024-06-27"}'
        )
        self.assertEqual(result.normal_text, "")

    def test_multiple_tool_calls(self):
        """
        Test parsing multiple consecutive tool calls.

        Scenario: Parse two complete tool calls back-to-back without any text in between.
        Purpose: Verify the detector correctly handles multiple tool calls and resets state
                between calls to avoid parameter leakage or ID conflicts.
        """
        text = (
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>"
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>"
            "</tool_call>"
            "<tool_call>get_weather"
            "<arg_key>city</arg_key><arg_value>Shanghai</arg_value>"
            "<arg_key>date</arg_key><arg_value>2024-06-28</arg_value>"
            "</tool_call>"
        )
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 2)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(
            result.calls[0].parameters, '{"city": "Beijing", "date": "2024-06-27"}'
        )
        self.assertEqual(result.calls[1].name, "get_weather")
        self.assertEqual(
            result.calls[1].parameters, '{"city": "Shanghai", "date": "2024-06-28"}'
        )
        self.assertEqual(result.normal_text, "")

    def test_no_arg_function_non_streaming(self):
        """
        Test no-argument function call without streaming.

        Scenario: Parse a tool call for a function that has no parameters (empty properties).
        Purpose: Verify the detector generates a single empty object "{}" for no-argument functions
                and does not duplicate empty parameter objects.
        """
        tools_with_no_args = [
            Tool(
                type="function",
                function=Function(
                    name="list_filenames",
                    description="List filenames",
                    parameters={
                        "type": "object",
                        "properties": {},
                    },
                ),
            ),
        ]

        text = "<tool_call>list_filenames</tool_call>"
        result = self.detector.detect_and_parse(text, tools_with_no_args)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "list_filenames")
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params, {})

    def test_invalid_tool_call(self):
        """
        Test handling of invalid tool calls.

        Scenario: Attempt to parse a tool call with a function name that doesn't exist in the tool list.
        Purpose: Verify the detector gracefully rejects invalid function calls and returns no calls
                rather than throwing an error or accepting invalid input.
        """
        text = "<tool_call>invalid_func<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)
        self.assertEqual(len(result.calls), 0)

    def test_array_argument_with_escaped_json(self):
        """
        Test array arguments containing escaped JSON strings.

        Scenario: Parse tool calls with array parameters containing nested JSON objects with
                 escaped quotes (both backslash-escaped and raw escaped strings).
        Purpose: Verify the detector properly handles JSON escaping without double-escaping,
                preserving special characters like backslashes in paths and newline sequences.
        """
        tools_with_array = [
            Tool(
                type="function",
                function=Function(
                    name="todo_write",
                    description="Write todos",
                    parameters={
                        "type": "object",
                        "properties": {
                            "todos": {
                                "type": "array",
                                "description": "The updated todo list",
                            }
                        },
                        "required": ["todos"],
                    },
                ),
            ),
        ]

        def check_params(result):
            self.assertEqual(1, len(result.calls))
            self.assertEqual("todo_write", result.calls[0].name)
            params = json.loads(result.calls[0].parameters)
            self.assertIsInstance(params["todos"], list)
            self.assertEqual(4, len(params["todos"]))
            self.assertEqual("1", params["todos"][0]["id"])
            self.assertEqual(
                "Check for hard-coded issues in the backend code",
                params["todos"][0]["task"],
            )
            self.assertEqual("in_progress", params["todos"][0]["status"])
            self.assertEqual("2", params["todos"][1]["id"])
            self.assertEqual(
                "Check for hard-coded issues in the frontend code",
                params["todos"][1]["task"],
            )
            self.assertEqual("pending", params["todos"][1]["status"])
            self.assertEqual("3", params["todos"][2]["id"])
            self.assertEqual(
                "Check for code violating the Single Responsibility Principle",
                params["todos"][2]["task"],
            )
            self.assertEqual("pending", params["todos"][2]["status"])
            self.assertEqual("4", params["todos"][3]["id"])
            self.assertEqual(
                "Generate a rectification proposal report", params["todos"][3]["task"]
            )
            self.assertEqual("pending", params["todos"][3]["status"])

        # Test with normal escaped JSON in XML
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Check for hard-coded issues in the backend code\", \"status\": \"in_progress\"}, {\"id\": \"2\", \"task\": \"Check for hard-coded issues in the frontend code\", \"status\": \"pending\"}, {\"id\": \"3\", \"task\": \"Check for code violating the Single Responsibility Principle\", \"status\": \"pending\"}, {\"id\": \"4\", \"task\": \"Generate a rectification proposal report\", \"status\": \"pending\"}]</arg_value>
</tool_call>""",
            tools_with_array,
        )
        check_params(result)

        # Test with raw string escaped JSON
        result = self.detector.detect_and_parse(
            r"""<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Check for hard-coded issues in the backend code\", \"status\": \"in_progress\"}, {\"id\": \"2\", \"task\": \"Check for hard-coded issues in the frontend code\", \"status\": \"pending\"}, {\"id\": \"3\", \"task\": \"Check for code violating the Single Responsibility Principle\", \"status\": \"pending\"}, {\"id\": \"4\", \"task\": \"Generate a rectification proposal report\", \"status\": \"pending\"}]</arg_value>
</tool_call>""",
            tools_with_array,
        )
        check_params(result)

        def check_single_todos(tool_result, expected):
            self.assertEqual(1, len(tool_result.calls))
            self.assertEqual("todo_write", tool_result.calls[0].name)
            params = json.loads(tool_result.calls[0].parameters)
            self.assertIsInstance(params["todos"], list)
            self.assertEqual(1, len(params["todos"]))
            self.assertEqual("1", params["todos"][0]["id"])
            self.assertEqual(expected, params["todos"][0]["task"])
            self.assertEqual("pending", params["todos"][0]["status"])

        # Test with escaped backslashes (Windows paths)
        expected_path = r"Check file at C:\Users\test.txt"
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Check file at C:\\\\Users\\\\test.txt\", \"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_path)

        # Test with literal backslash-n (not newline)
        expected_output = r"Print \n to see newline"
        result = self.detector.detect_and_parse(
            """<tool_call>todo_write<arg_key>todos</arg_key><arg_value>[{\"id\": \"1\", \"task\": \"Print \\\\n to see newline\",\"status\": \"pending\"}]</arg_value></tool_call>""",
            tools_with_array,
        )
        check_single_todos(result, expected_output)

    # ==================== MTP Core Scenarios (3) ====================

    def test_mtp_func_and_string_split(self):
        """
        Test MTP-style function name and string parameter value splitting across chunks.

        Scenario: Simulate Model Token Provider (MTP) behavior where function names and string
                 parameter values are split mid-word across multiple chunks.
        Purpose: This is the MOST CRITICAL test - verify the detector correctly reassembles:
                - Function name split as "create_ta" + "sk"
                - String values split as "Go to Bei" + "jing" and "San Fran" + "cisco"
                These splits mimic real MTP output where tokenization breaks words arbitrarily.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="create_task",
                    parameters={
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "location": {"type": "string"},
                        },
                    },
                ),
            ),
        ]

        chunks = [
            "I'll create a task.",  # normal text before tool call
            "<tool_call>create_ta",  # function name split mid-word
            "sk<arg_key>title</arg_key><arg_value>Go to Bei",  # function name completes, param value splits
            "jing</arg_value>",  # first parameter value completes
            "<arg_key>location</arg_key><arg_value>San Fran",  # second parameter value splits
            "cisco</arg_value></tool_call>",  # second parameter and tool call complete
        ]

        detector = Glm47MoeDetector()
        all_calls = []
        all_normal_text = ""

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, tools)
            all_calls.extend(result.calls)
            all_normal_text += result.normal_text

        # Verify normal text is preserved
        self.assertEqual(all_normal_text, "I'll create a task.")

        # Verify function call
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(
            func_calls[0].name, "create_task"
        )  # "create_ta" + "sk" reassembled

        # Verify parameter reassembly
        full_params = "".join([c.parameters for c in all_calls if c.parameters])
        params = json.loads(full_params)
        self.assertEqual(
            params["title"], "Go to Beijing"
        )  # "Go to Bei" + "jing" reassembled
        self.assertEqual(
            params["location"], "San Francisco"
        )  # "San Fran" + "cisco" reassembled

    def test_mtp_noarg_and_multiple_calls(self):
        """
        Test MTP-style no-argument function and multiple tool calls with state reset.

        Scenario: Stream a no-argument function call followed by a regular function call,
                 simulating MTP's output pattern where function completion triggers state reset.
        Purpose: Verify:
                - No-argument functions emit exactly ONE empty object "{}", not duplicates
                - State properly resets between consecutive tool calls (tool_index increments)
                - Second tool call doesn't inherit parameters from first call
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="list_files",
                    parameters={
                        "type": "object",
                        "properties": {},
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                        },
                    },
                ),
            ),
        ]

        chunks = [
            "<tool_call>list_files</tool_call>",  # no-arg function, complete in one chunk
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing</arg_value></tool_call>",
        ]

        detector = Glm47MoeDetector()
        all_calls = []

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, tools)
            all_calls.extend(result.calls)

        # Verify two distinct tool calls
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 2)
        self.assertEqual(func_calls[0].name, "list_files")
        self.assertEqual(func_calls[1].name, "get_weather")

        # Verify no duplicate empty objects for no-arg function
        empty_object_calls = [c for c in all_calls if c.parameters == "{}"]
        self.assertLessEqual(
            len(empty_object_calls),
            1,
            "No-argument function should emit at most one empty object",
        )

        # Verify second call has correct parameters
        weather_params = [
            c.parameters for c in all_calls if c.parameters and c.parameters != "{}"
        ]
        if weather_params:
            full_params = "".join(weather_params)
            params = json.loads(full_params)
            self.assertEqual(params["city"], "Beijing")

    def test_mtp_number_and_complex_json(self):
        """
        Test MTP-style number parameters and complex JSON array splitting.

        Scenario: Parse tool calls with number parameters (int and float) and JSON arrays
                 split across chunks, including splits within JSON structure.
        Purpose: Verify:
                - Number types (5.5, 10) are preserved as numbers, not strings
                - JSON array content split as "description" + ": \"" maintains validity
                - Nested JSON objects in arrays are correctly reconstructed
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="create_todos",
                    parameters={
                        "type": "object",
                        "properties": {
                            "priority": {"type": "number"},
                            "count": {"type": "integer"},
                            "items": {"type": "array"},
                        },
                    },
                ),
            ),
        ]

        chunks = [
            "<tool_call>create_todos",
            "<arg_key>priority</arg_key><arg_value>5.5</arg_value>",  # float number
            "<arg_key>count</arg_key><arg_value>10</arg_value>",  # integer number
            '<arg_key>items</arg_key><arg_value>[{"description',  # JSON array splits mid-key
            '": "Test',  # key completes, value starts
            'Todo 1"}, {"description": "TestTodo 2"}]</arg_value></tool_call>',
        ]

        detector = Glm47MoeDetector()
        all_calls = []

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, tools)
            all_calls.extend(result.calls)

        # Verify function name
        func_calls = [c for c in all_calls if c.name]
        self.assertEqual(len(func_calls), 1)
        self.assertEqual(func_calls[0].name, "create_todos")

        # Verify parameters - numbers and JSON array
        full_params = "".join([c.parameters for c in all_calls if c.parameters])
        params = json.loads(full_params)

        # Number types should be preserved
        self.assertIsInstance(params["priority"], (int, float))
        self.assertEqual(params["priority"], 5.5)
        self.assertIsInstance(params["count"], int)
        self.assertEqual(params["count"], 10)

        # JSON array should be correctly reconstructed
        self.assertIsInstance(params["items"], list)
        self.assertEqual(len(params["items"]), 2)
        self.assertEqual(params["items"][0]["description"], "TestTodo 1")
        self.assertEqual(params["items"][1]["description"], "TestTodo 2")

    # ==================== Streaming Basics (3) ====================

    def test_streaming_tool_call(self):
        """
        Test basic streaming incremental parsing of a single tool call.

        Scenario: Parse a tool call split across 4 chunks with natural boundaries
                 (function name, first param, second param, closing tag).
        Purpose: Verify basic streaming functionality works correctly and accumulates
                parameters progressively across chunks.
        """
        chunks = [
            "<tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>",
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>",
            "</tool_call>",
        ]
        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
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
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(
            tool_calls[0]["parameters"], '{"city": "Beijing", "date": "2024-06-27"}'
        )

    def test_streaming_multiple_tool_calls(self):
        """
        Test streaming incremental parsing of multiple consecutive tool calls.

        Scenario: Stream two complete tool calls with the transition "</tool_call><tool_call>"
                 occurring within a single chunk.
        Purpose: Verify streaming correctly handles multiple tool calls and properly increments
                tool_index for each new call.
        """
        chunks = [
            "<tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Beijing</arg_value>",
            "<arg_key>date</arg_key><arg_value>2024-06-27</arg_value>",
            "</tool_call><tool_call>get_weather",  # two tool calls transition in same chunk
            "<arg_key>city</arg_key><arg_value>Shanghai</arg_value>",
            "<arg_key>date</arg_key><arg_value>2024-06-28</arg_value>",
            "</tool_call>",
        ]
        tool_calls = []
        for chunk in chunks:
            result = self.detector.parse_streaming_increment(chunk, self.tools)
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
        self.assertEqual(len(tool_calls), 2)
        self.assertEqual(tool_calls[0]["name"], "get_weather")
        self.assertEqual(
            tool_calls[0]["parameters"], '{"city": "Beijing", "date": "2024-06-27"}'
        )
        self.assertEqual(tool_calls[1]["name"], "get_weather")
        self.assertEqual(
            tool_calls[1]["parameters"], '{"city": "Shanghai", "date": "2024-06-28"}'
        )

    def test_normal_text_before_tool_call(self):
        """
        Test preservation of normal text (including punctuation) before tool calls.

        Scenario: Parse chunks containing normal text with various punctuation marks
                 (English and Chinese) immediately followed by tool call tags.
        Purpose: Verify normal text is preserved in result.normal_text and not lost when
                tool call parsing begins. This consolidates 6 previous Chinese punctuation tests.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="list_dir",
                    parameters={
                        "type": "object",
                        "properties": {
                            "path": {"type": "string"},
                        },
                    },
                ),
            ),
        ]

        test_cases = [
            ("Sure, let me help.<tool_call>list_dir", "English with period"),
            ("结构：<tool_call>list_dir", "Chinese colon"),
            ("问题。<tool_call>list_dir", "Chinese period"),
            ("Complete!<tool_call>list_dir", "English exclamation"),
            ("说明；<tool_call>list_dir", "Chinese semicolon"),
        ]

        for text, description in test_cases:
            with self.subTest(description=description):
                detector = Glm47MoeDetector()
                result = detector.parse_streaming_increment(text, tools)

                before_token = text.split("<tool_call>")[0]
                self.assertIn(
                    before_token,
                    result.normal_text,
                    f"Should preserve '{before_token}' in '{description}'",
                )

    # ==================== Boundary Cases (9) ====================

    def test_boundary_empty_param_value(self):
        """
        Test handling of empty parameter values.

        Scenario: Parse a tool call where a parameter value is an empty string.
        Purpose: Verify the detector correctly handles empty strings as valid parameter values
                and doesn't skip or error on them.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="create_note",
                    parameters={
                        "type": "object",
                        "properties": {
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                        },
                    },
                ),
            ),
        ]

        text = "<tool_call>create_note<arg_key>title</arg_key><arg_value>Test</arg_value><arg_key>content</arg_key><arg_value></arg_value></tool_call>"
        result = self.detector.detect_and_parse(text, tools)

        self.assertEqual(len(result.calls), 1)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["title"], "Test")
        self.assertEqual(params["content"], "")  # empty string should be preserved

    def test_boundary_param_value_extreme_split(self):
        """
        Test extreme parameter value splitting - one character per chunk.

        Scenario: Stream a parameter value where each character arrives in a separate chunk,
                 representing worst-case MTP tokenization.
        Purpose: Stress test the buffer reassembly mechanism to ensure it can handle
                extremely granular chunk boundaries without data loss or corruption.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                        },
                    },
                ),
            ),
        ]

        chunks = [
            "<tool_call>search<arg_key>query</arg_key><arg_value>N",
            "e",
            "w ",
            "Y",
            "o",
            "rk</arg_value></tool_call>",
        ]

        detector = Glm47MoeDetector()
        all_calls = []

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, tools)
            all_calls.extend(result.calls)

        full_params = "".join([c.parameters for c in all_calls if c.parameters])
        params = json.loads(full_params)
        self.assertEqual(
            params["query"], "New York"
        )  # all characters correctly reassembled

    def test_boundary_param_value_with_special_chars(self):
        """
        Test parameter values containing special characters and escape sequences.

        Scenario: Parse parameter values with quotes, backslashes, newlines, and other
                 special characters that require JSON escaping.
        Purpose: Verify special characters are properly escaped/unescaped and preserved
                through the parsing pipeline without corruption.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="execute_command",
                    parameters={
                        "type": "object",
                        "properties": {
                            "command": {"type": "string"},
                        },
                    },
                ),
            ),
        ]

        # Test with single quotes (no escaping needed)
        text = "<tool_call>execute_command<arg_key>command</arg_key><arg_value>echo 'Hello World'</arg_value></tool_call>"
        result = self.detector.detect_and_parse(text, tools)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["command"], "echo 'Hello World'")

        # Test with spaces and special chars that don't need escaping
        text = "<tool_call>execute_command<arg_key>command</arg_key><arg_value>echo Hello & World</arg_value></tool_call>"
        result = self.detector.detect_and_parse(text, tools)
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["command"], "echo Hello & World")

    def test_boundary_json_deeply_nested(self):
        """
        Test deeply nested JSON structures in parameter values.

        Scenario: Parse a parameter containing a deeply nested JSON object with multiple levels.
        Purpose: Verify the detector can handle complex nested structures without stack overflow
                or parsing errors.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="process_data",
                    parameters={
                        "type": "object",
                        "properties": {
                            "data": {"type": "object"},
                        },
                    },
                ),
            ),
        ]

        nested_json = (
            '{"level1": {"level2": {"level3": {"level4": {"value": "deep"}}}}}'
        )
        text = f"<tool_call>process_data<arg_key>data</arg_key><arg_value>{nested_json}</arg_value></tool_call>"

        result = self.detector.detect_and_parse(text, tools)
        params = json.loads(result.calls[0].parameters)

        # Navigate through nested structure
        self.assertEqual(
            params["data"]["level1"]["level2"]["level3"]["level4"]["value"], "deep"
        )

    def test_boundary_json_empty_structures(self):
        """
        Test empty JSON structures (empty objects and arrays) in parameters.

        Scenario: Parse parameters containing empty objects {} and empty arrays [].
        Purpose: Verify empty structures are preserved and not confused with no-argument
                function empty parameter generation.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="create_structure",
                    parameters={
                        "type": "object",
                        "properties": {
                            "empty_obj": {"type": "object"},
                            "empty_arr": {"type": "array"},
                        },
                    },
                ),
            ),
        ]

        text = "<tool_call>create_structure<arg_key>empty_obj</arg_key><arg_value>{}</arg_value><arg_key>empty_arr</arg_key><arg_value>[]</arg_value></tool_call>"
        result = self.detector.detect_and_parse(text, tools)

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["empty_obj"], {})
        self.assertEqual(params["empty_arr"], [])

    def test_boundary_multi_tags_one_chunk(self):
        """
        Test multiple XML tags appearing in a single chunk.

        Scenario: Parse chunks where multiple complete tags (arg_key, arg_value, etc.)
                 appear together without any chunk boundaries between them.
        Purpose: Verify the regex-based tag extraction correctly handles multiple tags
                in one chunk and processes them in the correct order.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="multi_param",
                    parameters={
                        "type": "object",
                        "properties": {
                            "a": {"type": "string"},
                            "b": {"type": "string"},
                            "c": {"type": "string"},
                        },
                    },
                ),
            ),
        ]

        # All three parameters in one chunk
        text = "<tool_call>multi_param<arg_key>a</arg_key><arg_value>1</arg_value><arg_key>b</arg_key><arg_value>2</arg_value><arg_key>c</arg_key><arg_value>3</arg_value></tool_call>"
        result = self.detector.detect_and_parse(text, tools)

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["a"], "1")
        self.assertEqual(params["b"], "2")
        self.assertEqual(params["c"], "3")

    def test_boundary_normal_text_mixed_with_tool(self):
        """
        Test normal text interleaved with tool calls.

        Scenario: Parse text with normal text before and after tool calls.
        Purpose: Verify normal text segments are correctly separated from tool call parsing
                and preserved in the normal_text output.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="action",
                    parameters={
                        "type": "object",
                        "properties": {},
                    },
                ),
            ),
        ]

        text = "First I'll do this.<tool_call>action</tool_call>Then I'll do that."
        result = self.detector.detect_and_parse(text, tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "action")
        # Verify both text before and after tool calls are preserved
        self.assertIn("First I'll do this.", result.normal_text)
        self.assertIn("Then I'll do that.", result.normal_text)

    def test_boundary_number_edge_values(self):
        """
        Test edge-case number values (zero, negative, scientific notation).

        Scenario: Parse parameters with various numeric edge cases to ensure proper type handling.
        Purpose: Verify the detector correctly preserves number types for edge values and doesn't
                convert them to strings or lose precision.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="calculate",
                    parameters={
                        "type": "object",
                        "properties": {
                            "zero": {"type": "number"},
                            "negative": {"type": "number"},
                            "large": {"type": "number"},
                        },
                    },
                ),
            ),
        ]

        text = "<tool_call>calculate<arg_key>zero</arg_key><arg_value>0</arg_value><arg_key>negative</arg_key><arg_value>-42.5</arg_value><arg_key>large</arg_key><arg_value>1e10</arg_value></tool_call>"
        result = self.detector.detect_and_parse(text, tools)

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["zero"], 0)
        self.assertEqual(params["negative"], -42.5)
        self.assertEqual(params["large"], 1e10)

    def test_boundary_type_string_with_numeric_content(self):
        """
        Test string parameters that contain numeric-looking content.

        Scenario: Parse string parameters with values like "123" or "45.67" that look like
                 numbers but should remain strings based on parameter schema.
        Purpose: Verify type preservation based on schema definition, not content appearance.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="store_data",
                    parameters={
                        "type": "object",
                        "properties": {
                            "id": {
                                "type": "string"
                            },  # string type despite numeric content
                            "code": {"type": "string"},
                        },
                    },
                ),
            ),
        ]

        text = "<tool_call>store_data<arg_key>id</arg_key><arg_value>12345</arg_value><arg_key>code</arg_key><arg_value>67.89</arg_value></tool_call>"
        result = self.detector.detect_and_parse(text, tools)

        params = json.loads(result.calls[0].parameters)
        # Should be strings, not numbers
        self.assertIsInstance(params["id"], str)
        self.assertIsInstance(params["code"], str)
        self.assertEqual(params["id"], "12345")
        self.assertEqual(params["code"], "67.89")

    # ==================== Error Handling (2) ====================

    def test_error_undefined_tool(self):
        """
        Test error handling for undefined tool names.

        Scenario: Attempt to call a function that doesn't exist in the provided tools list.
        Purpose: Verify the detector gracefully handles undefined tools by returning an empty
                call list rather than crashing or producing malformed output.
        """
        text = "<tool_call>nonexistent_function<arg_key>param</arg_key><arg_value>value</arg_value></tool_call>"
        result = self.detector.detect_and_parse(text, self.tools)

        # Should not crash, should return empty calls
        self.assertEqual(len(result.calls), 0)

    def test_error_incomplete_buffer_at_end(self):
        """
        Test handling of incomplete tool calls at end of stream.

        Scenario: Streaming ends with an incomplete tool call (e.g., missing closing tag).
        Purpose: Verify the detector handles incomplete buffers gracefully without throwing
                exceptions, as streaming may end mid-parse in real scenarios.
        """
        chunks = [
            "<tool_call>get_weather<arg_key>city</arg_key><arg_value>Beijing",
            # Stream ends here, no closing tags
        ]

        detector = Glm47MoeDetector()

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            # Should not crash
            self.assertIsInstance(result, StreamingParseResult)

        # Incomplete call should not be in results
        # (or may be partially present - main thing is no exception)

    # ==================== Streamed Raw Length Bug Tests (3) ====================

    def test_streamed_raw_length_incomplete_xml_tag(self):
        """
        Test that _streamed_raw_length is updated even when json_increment is empty.

        Scenario: Stream XML content that is split at an incomplete tag boundary,
                 causing the state machine to buffer without producing JSON output.
        Purpose: Verify that _streamed_raw_length is updated regardless of whether
                json_increment is empty, preventing reprocessing of the same input.

        This tests the bug where:
        1. raw_increment is extracted from func_args_raw[self._streamed_raw_length:]
        2. _process_xml_to_json_streaming() returns empty string (buffering state)
        3. If _streamed_raw_length is NOT updated before the early return,
           the next call will reprocess the same raw_increment
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    parameters={
                        "type": "object",
                        "properties": {
                            "city": {"type": "string"},
                            "temperature": {"type": "number"},
                        },
                    },
                ),
            ),
        ]

        # Simulate streaming chunks where XML tags are split
        chunks = [
            "<tool_call>get_weather",
            "<arg_key>city</arg_key><arg_value>Bei",  # Split in middle of value
            "jing</arg_value>",  # Complete the value
            "<arg_key>temperature</arg_key><arg_value>2",  # Split numeric value
            "5</arg_value></tool_call>",
        ]

        detector = Glm47MoeDetector()
        all_calls = []
        collected_params = ""

        for i, chunk in enumerate(chunks):
            result = detector.parse_streaming_increment(chunk, tools)
            all_calls.extend(result.calls)

            # Collect parameters
            for call in result.calls:
                if call.parameters:
                    collected_params += call.parameters

        # Verify complete parameters were collected without duplication
        if collected_params:
            params = json.loads(collected_params)
            self.assertEqual(params["city"], "Beijing")
            self.assertEqual(params["temperature"], 25)

            # Critical: Verify no duplicate JSON output due to reprocessing
            # Count occurrences of "city" key - should appear exactly once
            city_count = collected_params.count('"city"')
            self.assertEqual(
                city_count,
                1,
                f"'city' key appears {city_count} times, expected 1. "
                f"This indicates input reprocessing bug.",
            )

    def test_streamed_raw_length_tag_split_across_chunks(self):
        """
        Test _streamed_raw_length update when tag is split across chunk boundaries.

        Scenario: XML tags themselves are split across chunks (e.g., "<arg_k" + "ey>").
        Purpose: Verify that even when the state machine is buffering partial tags,
                _streamed_raw_length is correctly updated to prevent reprocessing.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {"type": "string"},
                            "limit": {"type": "integer"},
                        },
                    },
                ),
            ),
        ]

        # Split tags in extreme positions
        chunks = [
            "<tool_call>search<arg_",  # Split tag name
            "key>query</arg_key><arg_value>Python progra",  # Complete tag, split value
            "mming</arg_value><arg_",  # Complete value, split next tag
            "key>limit</arg_key><arg_value>10</arg_value></tool_call>",
        ]

        detector = Glm47MoeDetector()
        all_params = ""

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, tools)
            for call in result.calls:
                if call.parameters:
                    all_params += call.parameters

        # Verify correct reassembly
        params = json.loads(all_params)
        self.assertEqual(params["query"], "Python programming")
        self.assertEqual(params["limit"], 10)

        # Verify no duplication in output
        query_count = all_params.count('"query"')
        limit_count = all_params.count('"limit"')
        self.assertEqual(query_count, 1, "query key duplicated - reprocessing bug")
        self.assertEqual(limit_count, 1, "limit key duplicated - reprocessing bug")

    def test_streamed_raw_length_buffer_only_partial_tag(self):
        """
        Test that _streamed_raw_length updates even when state machine returns empty.

        Scenario: Send increment that is ONLY a partial opening tag that state machine
                 must buffer completely without producing any JSON output.
        Purpose: Force json_increment to be empty string to expose the bug where
                _streamed_raw_length is not updated before early return.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="test_func",
                    parameters={
                        "type": "object",
                        "properties": {
                            "key1": {"type": "string"},
                        },
                    },
                ),
            ),
        ]

        # Manually call _process_arguments_streaming to have precise control
        detector = Glm47MoeDetector()
        detector.current_tool_id = 0
        detector.current_tool_name_sent = True
        detector._reset_streaming_state()
        detector.streamed_args_for_tool = [""]
        detector._streamed_raw_length = 0

        # First call: Complete tag that produces JSON output
        func_args_1 = "<arg_key>key1</arg_key><arg_value>va"
        result_1 = detector._process_arguments_streaming(
            "test_func", func_args_1, tools
        )

        # Should produce JSON output: {"key1": "va (partial)
        self.assertIsNotNone(result_1)
        self.assertGreater(len(result_1.parameters), 0)
        initial_length = detector._streamed_raw_length
        self.assertEqual(initial_length, len(func_args_1))

        # Second call: Add just partial closing tag - state machine will buffer this
        # without producing JSON (it's waiting to see if </arg_value> is complete)
        func_args_2 = func_args_1 + "<"  # Add partial tag
        result_2 = detector._process_arguments_streaming(
            "test_func", func_args_2, tools
        )

        # This is the critical test: if _streamed_raw_length is NOT updated when
        # json_increment is empty, then detector._streamed_raw_length will still be
        # at initial_length, and the next call will reprocess the "<" character

        # Check if length was updated (bug test)
        updated_length = detector._streamed_raw_length

        # BUG: If code has bug, updated_length will equal initial_length
        # FIXED: If code is correct, updated_length should equal len(func_args_2)
        self.assertEqual(
            updated_length,
            len(func_args_2),
            "Bug detected: _streamed_raw_length not updated when json_increment is empty. "
            f"Expected {len(func_args_2)}, got {updated_length}",
        )

    def test_streamed_raw_length_multiple_empty_returns(self):
        """
        Test consecutive chunks that produce empty json_increment.

        Scenario: Multiple consecutive chunks that all result in empty json_increment
                 as the state machine buffers complex nested structures.
        Purpose: Verify _streamed_raw_length advances correctly through multiple
                empty-return cycles without getting stuck or reprocessing.
        """
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="update_settings",
                    parameters={
                        "type": "object",
                        "properties": {
                            "name": {"type": "string"},
                            "value": {"type": "string"},
                        },
                    },
                ),
            ),
        ]

        # Split XML at positions that may cause state machine buffering
        chunks = [
            "<tool_call>update_settings<arg_key>na",  # Split in tag name
            "me</arg_key><arg_val",  # Complete tag, split next tag
            "ue>co",  # Complete tag start, split value  # codespell:ignore ue
            "nf",  # Continue value
            "ig_v1</arg_value><arg_key>val",  # Complete value, split next key
            "ue</arg_key><arg_value>ena",  # Complete key name, split value  # codespell:ignore ue
            "bled</arg_value></tool_call>",  # Complete everything
        ]

        detector = Glm47MoeDetector()
        all_params = ""

        for i, chunk in enumerate(chunks):
            result = detector.parse_streaming_increment(chunk, tools)

            for call in result.calls:
                if call.parameters:
                    all_params += call.parameters

        # Verify final output is correct
        self.assertGreater(len(all_params), 0, "Should have generated some parameters")
        params = json.loads(all_params)
        self.assertEqual(params["name"], "config_v1")
        self.assertEqual(params["value"], "enabled")

        # Verify no duplicate keys due to reprocessing
        name_count = all_params.count('"name"')
        value_count = all_params.count('"value"')
        self.assertEqual(
            name_count,
            1,
            f"'name' appears {name_count} times - indicates reprocessing bug",
        )
        self.assertEqual(
            value_count,
            1,
            f"'value' appears {value_count} times - indicates reprocessing bug",
        )


class TestGlm4ComplexJsonSchema(unittest.TestCase):
    """Test complex JSON Schema type inference for GLM function call parsers."""

    def setUp(self):
        """Set up test tools with complex JSON schemas."""
        self.tools_with_complex_schema = [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search for information",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "description": "Search query, can be a string or a complex object",
                                "anyOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"},
                                            "filters": {"type": "object"},
                                        },
                                    },
                                ],
                            },
                            "priority": {"enum": ["low", "medium", "high"]},
                            "options": {
                                "oneOf": [{"type": "string"}, {"type": "number"}]
                            },
                            "config": {
                                "allOf": [
                                    {"type": "object"},
                                    {"properties": {"timeout": {"type": "number"}}},
                                ]
                            },
                            "tags": {"type": ["string", "null"]},
                            "data": {
                                "type": "object",
                                "properties": {
                                    "nested": {
                                        "anyOf": [
                                            {"type": "string"},
                                            {
                                                "type": "object",
                                                "properties": {
                                                    "value": {"type": "string"}
                                                },
                                            },
                                        ]
                                    }
                                },
                            },
                        },
                        "required": ["query"],
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather information",
                    parameters={
                        "type": "object",
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
        ]
        self.glm4_detector = Glm4MoeDetector()
        self.glm47_detector = Glm47MoeDetector()

    def test_get_argument_type_simple_type(self):
        """Test that get_argument_type correctly handles simple type fields."""
        result = get_argument_type(
            "get_weather", "location", self.tools_with_complex_schema
        )
        self.assertEqual(result, "string")

    def test_get_argument_type_enum_type(self):
        """Test that get_argument_type correctly identifies enum as string type."""
        result = get_argument_type(
            "get_weather", "unit", self.tools_with_complex_schema
        )
        # Current implementation returns the direct type field, which is "string" for the enum parameter
        # But it doesn't handle enum-only schemas properly (without type field)
        self.assertEqual(result, "string")

    def test_get_argument_type_anyof_type(self):
        """Test that get_argument_type correctly handles anyOf type fields."""
        result = get_argument_type("search", "query", self.tools_with_complex_schema)
        # anyOf with [{"type": "string"}, {"type": "object", ...}] should return "string"
        self.assertEqual(result, "string")  # Returns first common type

    def test_get_argument_type_oneof_type(self):
        """Test that get_argument_type correctly handles oneOf type fields."""
        result = get_argument_type("search", "options", self.tools_with_complex_schema)
        # oneOf with [{"type": "string"}, {"type": "number"}] should return "string" (prioritizes string)
        self.assertEqual(result, "string")

    def test_get_argument_type_allof_type(self):
        """Test that get_argument_type correctly handles allOf type fields."""
        result = get_argument_type("search", "config", self.tools_with_complex_schema)
        # allOf with [{"type": "object"}, ...] should return "object"
        self.assertEqual(result, "object")

    def test_get_argument_type_type_array(self):
        """Test that get_argument_type correctly handles type arrays."""
        result = get_argument_type("search", "tags", self.tools_with_complex_schema)
        # Type arrays should return the first non-null type
        self.assertEqual(
            result, "string"
        )  # ["string", "null"] -> "string" (non-null type)

    def test_glm4_detector_with_complex_schema_anyof(self):
        """Test GLM4 detector with anyOf schema - should demonstrate current issues."""
        # This test shows the current behavior with complex schemas
        text = (
            "<tool_call>search\n"
            "<arg_key>query</arg_key>\n<arg_value>Hello world</arg_value>\n"
            "<arg_key>priority</arg_key>\n<arg_value>medium</arg_value>\n"
            "</tool_call>"
        )
        result = self.glm4_detector.detect_and_parse(
            text, self.tools_with_complex_schema
        )

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")

        # Parse parameters to check if they are correctly handled
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "Hello world")
        self.assertEqual(params["priority"], "medium")

    def test_glm47_detector_with_complex_schema_anyof(self):
        """Test GLM47 detector with anyOf schema - should demonstrate current issues."""
        # This test shows the current behavior with complex schemas
        text = (
            "<tool_call>search"
            "<arg_key>query</arg_key><arg_value>Hello world</arg_value>"
            "<arg_key>priority</arg_key><arg_value>medium</arg_value>"
            "</tool_call>"
        )
        result = self.glm47_detector.detect_and_parse(
            text, self.tools_with_complex_schema
        )

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")

        # Parse parameters to check if they are correctly handled
        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "Hello world")
        self.assertEqual(params["priority"], "medium")

    def test_glm4_detector_with_enum_values(self):
        """Test GLM4 detector with enum values in complex schema."""
        text = (
            "<tool_call>search\n"
            "<arg_key>query</arg_key>\n<arg_value>test query</arg_value>\n"
            "<arg_key>priority</arg_key>\n<arg_value>high</arg_value>\n"
            "</tool_call>"
        )
        result = self.glm4_detector.detect_and_parse(
            text, self.tools_with_complex_schema
        )

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "test query")
        self.assertEqual(params["priority"], "high")

    def test_glm47_detector_with_enum_values(self):
        """Test GLM47 detector with enum values in complex schema."""
        text = (
            "<tool_call>search"
            "<arg_key>query</arg_key><arg_value>test query</arg_value>"
            "<arg_key>priority</arg_key><arg_value>high</arg_value>"
            "</tool_call>"
        )
        result = self.glm47_detector.detect_and_parse(
            text, self.tools_with_complex_schema
        )

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "search")

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "test query")
        self.assertEqual(params["priority"], "high")

    def test_glm4_detector_streaming_with_complex_schema(self):
        """Test GLM4 detector streaming with complex schema."""
        chunks = [
            "<tool_call>search\n",
            "<arg_key>query</arg_key>\n<arg_value>nested object</arg_value>\n",
            "<arg_key>priority</arg_key>\n<arg_value>low</arg_value>\n",
            "</tool_call>",
        ]
        tool_calls = []
        for chunk in chunks:
            result = self.glm4_detector.parse_streaming_increment(
                chunk, self.tools_with_complex_schema
            )
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

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "search")

        params = json.loads(tool_calls[0]["parameters"])
        self.assertEqual(params["query"], "nested object")
        self.assertEqual(params["priority"], "low")

    def test_glm47_detector_streaming_with_complex_schema(self):
        """Test GLM47 detector streaming with complex schema."""
        chunks = [
            "<tool_call>search",
            "<arg_key>query</arg_key><arg_value>nested object</arg_value>",
            "<arg_key>priority</arg_key><arg_value>low</arg_value>",
            "</tool_call>",
        ]
        tool_calls = []
        for chunk in chunks:
            result = self.glm47_detector.parse_streaming_increment(
                chunk, self.tools_with_complex_schema
            )
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

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "search")

        params = json.loads(tool_calls[0]["parameters"])
        self.assertEqual(params["query"], "nested object")
        self.assertEqual(params["priority"], "low")

    def test_type_inference_issue_reproduction(self):
        """Reproduce the issue where complex JSON schemas are not properly handled."""
        # This test demonstrates the current limitations
        complex_tools = [
            Tool(
                type="function",
                function=Function(
                    name="complex_function",
                    parameters={
                        "type": "object",
                        "properties": {
                            "complex_param": {
                                "anyOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {"value": {"type": "string"}},
                                    },
                                ]
                            },
                            "enum_param": {"enum": ["option1", "option2", "option3"]},
                        },
                    },
                ),
            )
        ]

        # Test that get_argument_type returns appropriate types for complex schemas
        anyof_result = get_argument_type(
            "complex_function", "complex_param", complex_tools
        )
        enum_result = get_argument_type("complex_function", "enum_param", complex_tools)

        # Verify complex schema types are correctly inferred
        self.assertEqual(anyof_result, "string")  # anyOf prioritizes string type
        self.assertEqual(enum_result, "string")  # enum values are strings

    def test_expected_behavior_for_complex_schemas(self):
        """Test cases that should work but currently fail - demonstrating the issue."""
        # This test shows what the behavior SHOULD be after the fix
        complex_tools = [
            Tool(
                type="function",
                function=Function(
                    name="complex_function",
                    parameters={
                        "type": "object",
                        "properties": {
                            "complex_param": {
                                "anyOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {"value": {"type": "string"}},
                                    },
                                ]
                            },
                            "enum_param": {"enum": ["option1", "option2", "option3"]},
                            "oneof_param": {
                                "oneOf": [{"type": "string"}, {"type": "number"}]
                            },
                            "allof_param": {
                                "allOf": [
                                    {"type": "object"},
                                    {"properties": {"timeout": {"type": "number"}}},
                                ]
                            },
                        },
                    },
                ),
            )
        ]

        # These assertions represent the EXPECTED behavior after implementing RFC improvements
        # Currently they will fail, demonstrating the issue
        anyof_result = get_argument_type(
            "complex_function", "complex_param", complex_tools
        )
        enum_result = get_argument_type("complex_function", "enum_param", complex_tools)
        oneof_result = get_argument_type(
            "complex_function", "oneof_param", complex_tools
        )
        allof_result = get_argument_type(
            "complex_function", "allof_param", complex_tools
        )

        # These should pass after implementing the RFC improvements, but will currently fail
        # This demonstrates the issue exists
        self.assertIsNotNone(
            anyof_result, "anyOf should return a type after RFC implementation"
        )
        self.assertEqual(
            enum_result,
            "string",
            "enum should return 'string' type after RFC implementation",
        )
        self.assertIsNotNone(
            oneof_result, "oneOf should return a type after RFC implementation"
        )
        self.assertIsNotNone(
            allof_result, "allOf should return a type after RFC implementation"
        )

    def test_complex_schema_type_inference_scenarios(self):
        """Test various complex schema scenarios mentioned in the RFC."""
        # Create tools with different complex schema structures
        complex_schema_tools = [
            Tool(
                type="function",
                function=Function(
                    name="search_complex",
                    parameters={
                        "type": "object",
                        "properties": {
                            # anyOf example - parameter can be string or object
                            "query": {
                                "description": "Search query, can be a string or a complex object",
                                "anyOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {
                                            "text": {"type": "string"},
                                            "filters": {"type": "object"},
                                        },
                                    },
                                ],
                            },
                            # oneOf example - parameter must be one of the specified types
                            "priority": {
                                "oneOf": [{"type": "string"}, {"type": "integer"}]
                            },
                            # enum example - parameter must be one of the enum values
                            "category": {"enum": ["news", "sports", "tech"]},
                            # allOf example - parameter must satisfy all schemas
                            "config": {
                                "allOf": [
                                    {"type": "object"},
                                    {"properties": {"timeout": {"type": "number"}}},
                                ]
                            },
                            # Type array example
                            "tags": {"type": ["string", "null"]},
                        },
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="get_data",
                    parameters={
                        "type": "object",
                        "properties": {
                            # Complex nested anyOf
                            "input": {
                                "anyOf": [
                                    {"type": "string"},
                                    {"type": "number"},
                                    {
                                        "type": "object",
                                        "properties": {
                                            "type": {"type": "string"},
                                            "value": {},
                                        },
                                    },
                                ]
                            }
                        },
                    },
                ),
            ),
        ]

        # Test each complex type scenario
        query_type = get_argument_type("search_complex", "query", complex_schema_tools)
        priority_type = get_argument_type(
            "search_complex", "priority", complex_schema_tools
        )
        category_type = get_argument_type(
            "search_complex", "category", complex_schema_tools
        )
        config_type = get_argument_type(
            "search_complex", "config", complex_schema_tools
        )
        tags_type = get_argument_type("search_complex", "tags", complex_schema_tools)
        input_type = get_argument_type("get_data", "input", complex_schema_tools)

        # All of these should return appropriate types according to RFC
        self.assertEqual(query_type, "string")  # anyOf: string | object -> string
        self.assertEqual(priority_type, "string")  # oneOf: string | integer -> string
        self.assertEqual(
            category_type, "string"
        )  # enum: ["news", "sports", "tech"] -> string
        self.assertEqual(config_type, "object")  # allOf with object -> object
        self.assertEqual(
            tags_type, "string"
        )  # type array: ["string", "null"] -> string
        self.assertEqual(
            input_type, "string"
        )  # nested anyOf: string | number | object -> string

    def test_glm4_detector_type_handling_with_complex_schema(self):
        """Test how GLM4 detector handles type inference for complex schemas in practice."""
        complex_tools = [
            Tool(
                type="function",
                function=Function(
                    name="complex_search",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "anyOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {"text": {"type": "string"}},
                                    },
                                ]
                            },
                            "category": {"enum": ["tech", "news", "sports"]},
                        },
                    },
                ),
            )
        ]

        # Test with string value for anyOf parameter
        text = (
            "<tool_call>complex_search\n"
            "<arg_key>query</arg_key>\n<arg_value>test search</arg_value>\n"
            "<arg_key>category</arg_key>\n<arg_value>tech</arg_value>\n"
            "</tool_call>"
        )
        result = self.glm4_detector.detect_and_parse(text, complex_tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "complex_search")

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "test search")
        self.assertEqual(params["category"], "tech")

    def test_glm47_detector_type_handling_with_complex_schema(self):
        """Test how GLM47 detector handles type inference for complex schemas in practice."""
        complex_tools = [
            Tool(
                type="function",
                function=Function(
                    name="complex_search",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "anyOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {"text": {"type": "string"}},
                                    },
                                ]
                            },
                            "category": {"enum": ["tech", "news", "sports"]},
                        },
                    },
                ),
            )
        ]

        # Test with string value for anyOf parameter
        text = (
            "<tool_call>complex_search"
            "<arg_key>query</arg_key><arg_value>test search</arg_value>"
            "<arg_key>category</arg_key><arg_value>tech</arg_value>"
            "</tool_call>"
        )
        result = self.glm47_detector.detect_and_parse(text, complex_tools)

        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "complex_search")

        params = json.loads(result.calls[0].parameters)
        self.assertEqual(params["query"], "test search")
        self.assertEqual(params["category"], "tech")

    def test_streaming_with_complex_schema_type_inference(self):
        """Test streaming behavior with complex schema type inference."""
        complex_tools = [
            Tool(
                type="function",
                function=Function(
                    name="stream_test",
                    parameters={
                        "type": "object",
                        "properties": {
                            "data": {
                                "anyOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {"value": {"type": "string"}},
                                    },
                                ]
                            },
                            "status": {"enum": ["active", "inactive"]},
                        },
                    },
                ),
            )
        ]

        # Test GLM4 detector streaming
        chunks = [
            "<tool_call>stream_test\n",
            "<arg_key>data</arg_key>\n<arg_value>nested data</arg_value>\n",
            "<arg_key>status</arg_key>\n<arg_value>active</arg_value>\n",
            "</tool_call>",
        ]
        tool_calls = []
        for chunk in chunks:
            result = self.glm4_detector.parse_streaming_increment(chunk, complex_tools)
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

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "stream_test")

        params = json.loads(tool_calls[0]["parameters"])
        self.assertEqual(params["data"], "nested data")
        self.assertEqual(params["status"], "active")

    def test_streaming_with_complex_schema_type_inference_glm47(self):
        """Test GLM47 streaming behavior with complex schema type inference."""
        complex_tools = [
            Tool(
                type="function",
                function=Function(
                    name="stream_test",
                    parameters={
                        "type": "object",
                        "properties": {
                            "data": {
                                "anyOf": [
                                    {"type": "string"},
                                    {
                                        "type": "object",
                                        "properties": {"value": {"type": "string"}},
                                    },
                                ]
                            },
                            "status": {"enum": ["active", "inactive"]},
                        },
                    },
                ),
            )
        ]

        # Test GLM47 detector streaming
        chunks = [
            "<tool_call>stream_test",
            "<arg_key>data</arg_key><arg_value>nested data</arg_value>",
            "<arg_key>status</arg_key><arg_value>active</arg_value>",
            "</tool_call>",
        ]
        tool_calls = []
        for chunk in chunks:
            result = self.glm47_detector.parse_streaming_increment(chunk, complex_tools)
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

        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0]["name"], "stream_test")

        params = json.loads(tool_calls[0]["parameters"])
        self.assertEqual(params["data"], "nested data")
        self.assertEqual(params["status"], "active")

    if __name__ == "__main__":
        unittest.main()
