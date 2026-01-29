import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool

# from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.function_call.qwen3_coder_new_detector import Qwen3CoderDetector


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

        self.assertEqual(result.normal_text, "")
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

    def test_parse_streaming_complex_example(self):
        """Test basic streaming parsing while parameter has list."""

        tools = [
            Tool(
                type="function",
                function=Function(
                    name="list_directory",
                    description="Lists the names of files and subdirectories directly within a specified directory path. Can optionally ignore entries matching provided glob patterns.",
                    parameters={
                        "properties": {
                            "path": {
                                "description": "The absolute path to the directory to list (must be absolute, not relative)",
                                "type": "string",
                            },
                            "ignore": {
                                "description": "List of glob patterns to ignore",
                                "items": {"type": "string"},
                                "type": "array",
                            },
                            "respect_git_ignore": {
                                "description": "Optional: Whether to respect .gitignore patterns when listing files. Only available in git repositories. Defaults to true.",
                                "type": "boolean",
                            },
                        },
                        "required": ["path"],
                        "type": "object",
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="read_file",
                    description="Reads and returns the content of a specified file from the local filesystem. Handles text, images (PNG, JPG, GIF, WEBP, SVG, BMP), and PDF files. For text files, it can read specific line ranges.",
                    parameters={
                        "properties": {
                            "absolute_path": {
                                "description": "The absolute path to the file to read (e.g., '/home/user/project/file.txt'). Relative paths are not supported. You must provide an absolute path.",
                                "type": "string",
                            },
                            "offset": {
                                "description": "Optional: For text files, the 0-based line number to start reading from. Requires 'limit' to be set. Use for paginating through large files.",
                                "type": "number",
                            },
                            "limit": {
                                "description": "Optional: For text files, maximum number of lines to read. Use with 'offset' to paginate through large files. If omitted, reads the entire file (if feasible, up to a default limit).",
                                "type": "number",
                            },
                        },
                        "required": ["absolute_path"],
                        "type": "object",
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="search_file_content",
                    description="Searches for a regular expression pattern within the content of files in a specified directory (or current working directory). Can filter files by a glob pattern. Returns the lines containing matches, along with their file paths and line numbers.",
                    parameters={
                        "properties": {
                            "pattern": {
                                "description": "The regular expression (regex) pattern to search for within file contents (e.g., 'function\\s+myFunction', 'import\\s+\\{.*\\}\\s+from\\s+.*').",
                                "type": "string",
                            },
                            "path": {
                                "description": "Optional: The absolute path to the directory to search within. If omitted, searches the current working directory.",
                                "type": "string",
                            },
                            "include": {
                                "description": "Optional: A glob pattern to filter which files are searched (e.g., '*.js', '*.{ts,tsx}', 'src/**'). If omitted, searches all files (respecting potential global ignores).",
                                "type": "string",
                            },
                        },
                        "required": ["pattern"],
                        "type": "object",
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="glob",
                    description="Efficiently finds files matching specific glob patterns (e.g., `src/**/*.ts`, `**/*.md`), returning absolute paths sorted by modification time (newest first). Ideal for quickly locating files based on their name or path structure, especially in large codebases.",
                    parameters={
                        "properties": {
                            "pattern": {
                                "description": "The glob pattern to match against (e.g., '**/*.py', 'docs/*.md').",
                                "type": "string",
                            },
                            "path": {
                                "description": "Optional: The absolute path to the directory to search within. If omitted, searches the root directory.",
                                "type": "string",
                            },
                            "case_sensitive": {
                                "description": "Optional: Whether the search should be case-sensitive. Defaults to false.",
                                "type": "boolean",
                            },
                            "respect_git_ignore": {
                                "description": "Optional: Whether to respect .gitignore patterns when finding files. Only available in git repositories. Defaults to true.",
                                "type": "boolean",
                            },
                        },
                        "required": ["pattern"],
                        "type": "object",
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="replace",
                    description="Replaces text within a file. By default, replaces a single occurrence, but can replace multiple occurrences when `expected_replacements` is specified. This tool requires providing significant context around the change to ensure precise targeting. Always use the read_file tool to examine the file's current content before attempting a text replacement.\n\n      The user has the ability to modify the `new_string` content. If modified, this will be stated in the response.\n\nExpectation for required parameters:\n1. `file_path` MUST be an absolute path; otherwise an error will be thrown.\n2. `old_string` MUST be the exact literal text to replace (including all whitespace, indentation, newlines, and surrounding code etc.).\n3. `new_string` MUST be the exact literal text to replace `old_string` with (also including all whitespace, indentation, newlines, and surrounding code etc.). Ensure the resulting code is correct and idiomatic.\n4. NEVER escape `old_string` or `new_string`, that would break the exact literal text requirement.\n**Important:** If ANY of the above are not satisfied, the tool will fail. CRITICAL for `old_string`: Must uniquely identify the single instance to change. Include at least 3 lines of context BEFORE and AFTER the target text, matching whitespace and indentation precisely. If this string matches multiple locations, or does not match exactly, the tool will fail.\n**Multiple replacements:** Set `expected_replacements` to the number of occurrences you want to replace. The tool will replace ALL occurrences that match `old_string` exactly. Ensure the number of replacements matches your expectation.",
                    parameters={
                        "properties": {
                            "file_path": {
                                "description": "The absolute path to the file to modify. Must start with '/'.",
                                "type": "string",
                            },
                            "old_string": {
                                "description": "The exact literal text to replace, preferably unescaped. For single replacements (default), include at least 3 lines of context BEFORE and AFTER the target text, matching whitespace and indentation precisely. For multiple replacements, specify expected_replacements parameter. If this string is not the exact literal text (i.e. you escaped it) or does not match exactly, the tool will fail.",
                                "type": "string",
                            },
                            "new_string": {
                                "description": "The exact literal text to replace `old_string` with, preferably unescaped. Provide the EXACT text. Ensure the resulting code is correct and idiomatic.",
                                "type": "string",
                            },
                            "expected_replacements": {
                                "type": "number",
                                "description": "Number of replacements expected. Defaults to 1 if not specified. Use when you want to replace multiple occurrences.",
                                "minimum": 1,
                            },
                        },
                        "required": ["file_path", "old_string", "new_string"],
                        "type": "object",
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="write_file",
                    description="Writes content to a specified file in the local filesystem. \n      \n      The user has the ability to modify `content`. If modified, this will be stated in the response.",
                    parameters={
                        "properties": {
                            "file_path": {
                                "description": "The absolute path to the file to write to (e.g., '/home/user/project/file.txt'). Relative paths are not supported.",
                                "type": "string",
                            },
                            "content": {
                                "description": "The content to write to the file.",
                                "type": "string",
                            },
                        },
                        "required": ["file_path", "content"],
                        "type": "object",
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="run_shell_command",
                    description="This tool executes a given shell command as `bash -c <command>`. Command can start  \
        background processes using `&`. Command is executed as a subprocess that leads its own process group. Command  \
        process group can be terminated as `kill -- -PGID` or signaled as `kill -s SIGNAL -- -PGID`.\n\nThe following  \
        information is returned:\n\nCommand: Executed command.\nDirectory: Directory (relative to project root) where  \
        command was executed, or `(root)`.\nStdout: Output on stdout stream. Can be `(empty)` or partial on error and  \
        for any unwaited background processes.\nStderr: Output on stderr stream. Can be `(empty)` or partial on error  \
        and for any unwaited background processes.\nError: Error or `(none)` if no error was reported for the          \
        subprocess.\nExit Code: Exit code or `(none)` if terminated by signal.\nSignal: Signal number or `(none)` if   \
        no signal was received.\nBackground PIDs: List of background processes started or `(none)`.\nProcess Group     \
        PGID: Process group started or `(none)`",
                    parameters={
                        "type": "object",
                        "properties": {
                            "command": {
                                "type": "string",
                                "description": "Exact bash command to execute as `bash -c <command>`",
                            },
                            "description": {
                                "type": "string",
                                "description": "Brief description of the command for the user. Be specific and concise.  \
        Ideally a single sentence. Can be up to 3 sentences for clarity. No line breaks.",
                            },
                            "directory": {
                                "type": "string",
                                "description": "(OPTIONAL) Directory to run the command in, if not the project root  \
        directory. Must be relative to the project root directory and must already exist.",
                            },
                        },
                        "required": ["command"],
                    },
                ),
            ),
        ]
        model_output = """  I'll solve this step by
step to create a command line tool for MNIST inference.
First, let me check what we have in the directory
and understand the existing files:
<tool_call>
<function=list_directory>
<parameter=path>
"app"
</parameter>
</function>
</tool_call>
"""

        # Simulate streaming by chunks
        chunk_size = 1
        chunks = [
            model_output[i : i + chunk_size]
            for i in range(0, len(model_output), chunk_size)
        ]
        print(chunks)

        accumulated_text = ""
        tool_calls = []
        chunks_count = 0

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
        self.assertEqual(
            accumulated_text,
            "  I'll solve this step by \n\
step to create a command line tool for MNIST inference.\n\
First, let me check what we have in the directory \n\
and understand the existing files:\n",
        )
        self.assertEqual(len(tool_calls_by_index), 1)

        # Get the complete tool call
        self.assertIn(0, tool_calls_by_index)
        tool_call = tool_calls_by_index[0]
        self.assertEqual(tool_call["name"], "list_directory")

        # Parse the accumulated parameters
        params = json.loads(tool_call["parameters"])
        self.assertEqual(params["path"], '"app"')


if __name__ == "__main__":
    unittest.main()
