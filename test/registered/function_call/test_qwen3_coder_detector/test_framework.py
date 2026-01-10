from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(1.0, "default")


import json
import re
import random
from typing import Dict, List, Optional, Generator, Tuple
from dataclasses import dataclass

from sglang.srt.entrypoints.openai.protocol import Tool

@dataclass
class StreamingTestResult:
    """Result of a streaming test execution."""
    test_name: str
    full_parsed_text: str
    final_reconstructed: Dict
    step_count: int
    non_streaming_result: any

    error_message: Optional[str] = None
    success: bool = True
    streaming_match: Optional[bool] = None
    streaming_errors: Optional[List[str]] = None
    non_streaming_match: Optional[bool] = None
    non_streaming_errors: Optional[List[str]] = None

class StreamingTestRunner:
    def __init__(self, detector, tools: Optional[List[Tool]] = None, parser_mode: str = "both", verbose: bool = False):
        self.detector = detector
        self.parser_mode = parser_mode
        self.default_tools = tools
        self.verbose = verbose

    def run_test(
        self,
        test_name: str,
        response_text: str,
        mode: str = "atomic_tags",
        tools: Optional[List[Tool]] = None,
        verbose: Optional[bool] = None,
        compare_with_non_streaming: Optional[bool] = None,
        expected: Optional[Dict] = None,
    ) -> StreamingTestResult:
        """
        Run a single streaming test

        Args:
            test_name: Test name
            response_text: Complete response text
            mode: Streaming generation mode ('char', 'atomic_tags', or other)
            tools: Tool list, if None use default tools
            verbose: Whether to print detailed logs, if None use instance attribute self.verbose
            compare_with_non_streaming: Whether to compare with non-streaming parse result, if None auto-decide based on parser_mode
            expected: Expected parse result, containing 'text' and 'tools' fields

        Returns:
            StreamingTestResult: Test result
        """
        # If verbose not specified, use instance attribute
        if verbose is None:
            verbose = self.verbose

        # Decide whether to run streaming and non-streaming parsing based on parser_mode
        run_streaming = self.parser_mode in ("both", "streaming")
        run_non_streaming = self.parser_mode in ("both", "nonstream")

        # If compare_with_non_streaming not specified, decide based on parser_mode
        if compare_with_non_streaming is None:
            compare_with_non_streaming = run_non_streaming

        if verbose:
            self._print_test_header(test_name, response_text, mode, run_streaming, run_non_streaming)

        tools = tools or self.default_tools
        if tools is None:
            raise ValueError("tools must be provided either in __init__ or run_test")

        full_parsed_text = ""  # Accumulate final parsed plain text
        final_reconstructed = {}  # Accumulate final parsed tool calls

        step = 0
        error_message = None

        # Only run streaming parsing when needed
        if run_streaming:
            try:
                for chunk in token_stream_generator(response_text, mode=mode):
                    step += 1
                    result = self.detector.parse_streaming_increment(chunk, tools)

                    # If parser returns normal_text, append it to total text
                    if result.normal_text:
                        full_parsed_text += result.normal_text

                    # Collect tool call information
                    if result.calls:
                        for call in result.calls:
                            if call.tool_index not in final_reconstructed:
                                final_reconstructed[call.tool_index] = {"name": "", "args": ""}

                            if call.name:
                                final_reconstructed[call.tool_index]["name"] = call.name
                            if call.parameters:
                                final_reconstructed[call.tool_index]["args"] += call.parameters

                    if verbose:
                        display_chunk = chunk.replace("\n", "\\n")
                        log_msg = f"Step {step:03d} | Input: '{display_chunk}'"
                        if result.normal_text:
                            log_msg += f" | Text Found: {result.normal_text}"
                        print(log_msg)

            except Exception as e:
                error_message = str(e)
                if verbose:
                    print(f"ERROR: {error_message}")

        # Get non-streaming parse result for comparison
        non_streaming_result = None
        if run_non_streaming:
            try:
                non_streaming_result = self.detector.detect_and_parse(response_text, tools)
            except Exception as e:
                if verbose:
                    print(f"WARNING: Non-streaming parse failed: {e}")

        if verbose:
            self._print_test_summary(full_parsed_text, final_reconstructed, non_streaming_result, run_streaming, run_non_streaming)

        # Validate results
        streaming_match = None
        non_streaming_match = None
        streaming_errors = []
        non_streaming_errors = []

        if expected is not None:
            # Only validate streaming parse result when running streaming parsing
            if run_streaming:
                streaming_match, streaming_errors = self._validate_result(expected=expected, actual_text=full_parsed_text, actual_tools=final_reconstructed, result_type="streaming")

            # Only validate non-streaming parse result when running non-streaming parsing
            if run_non_streaming and non_streaming_result is not None:
                non_streaming_tools = {}
                for idx, call in enumerate(non_streaming_result.calls):
                    try:
                        args = json.loads(call.parameters) if call.parameters else {}
                    except:
                        args = call.parameters
                    non_streaming_tools[call.tool_index] = {"name": call.name, "args": args}
                non_streaming_match, non_streaming_errors = self._validate_result(
                    expected=expected,
                    actual_text=non_streaming_result.normal_text or "",
                    actual_tools=non_streaming_tools,
                    result_type="non-streaming",
                )

        return StreamingTestResult(
            test_name=test_name,
            full_parsed_text=full_parsed_text,
            final_reconstructed=final_reconstructed,
            non_streaming_result=non_streaming_result,
            step_count=step,
            success=error_message is None,
            error_message=error_message,
            streaming_match=streaming_match,
            non_streaming_match=non_streaming_match,
            streaming_errors=streaming_errors if streaming_errors else None,
            non_streaming_errors=non_streaming_errors if non_streaming_errors else None,
        )

    def _print_test_header(self, test_name: str, response_text: str, mode: str, run_streaming: bool, run_non_streaming: bool):
        """Print test header information"""
        print(f"\n{'='*60}")
        mode_info = []
        if run_streaming:
            mode_info.append(f"streaming ({mode})")
        if run_non_streaming:
            mode_info.append("non-streaming")
        mode_str = " + ".join(mode_info) if mode_info else "none"
        print(f"TEST: {test_name} (Mode: {mode_str})")
        print(f"Response Text: ")
        print(response_text)
        print(f"{'-'*60}")

    def _print_test_summary(self, full_parsed_text: str, final_reconstructed: Dict, non_streaming_result: any, run_streaming: bool, run_non_streaming: bool):
        """Print test result summary"""
        print("Parse Result Summary (FINAL RESULTS):")

        section_num = 1
        if run_streaming:
            print(f"\n[{section_num}] Streaming parse result:")
            print(f"Extracted plain text:\n\033[94m<bos>{full_parsed_text}<eos>\033[0m")
            print(f"Extracted tool calls:")
            print(json.dumps(final_reconstructed, indent=2, ensure_ascii=False))
            section_num += 1

        if run_non_streaming and non_streaming_result:
            print(f"\n[{section_num}] Non-streaming parse result:")
            print(f"Text: {non_streaming_result.normal_text}")
            for idx, call in enumerate(non_streaming_result.calls):
                print(f" - (T{idx}) Tool call: {call.name}")
                print(f"\tP-string: {call.parameters}")
                try:
                    print(f"\t  P-dict: {json.loads(call.parameters)}")
                except:
                    print(f"\t  P-dict: (parse error)")

    def _validate_result(self, expected: Dict, actual_text: str, actual_tools: Dict, result_type: str) -> Tuple[bool, List[str]]:
        """
        Validate if parse result matches expected result

        Args:
            expected: Expected result, containing 'text' and 'tools' fields
            actual_text: Actual parsed text
            actual_tools: Actual parsed tool calls, format: {tool_index: {"name": str, "args": str}}
            result_type: Result type ('streaming' or 'non-streaming')

        Returns:
            (whether matched, error list)
        """
        errors = []
        match = True

        # Validate text
        if "text" in expected:
            expected_text = expected["text"]

            if actual_text.rstrip() != expected_text.rstrip():
                match = False
                errors.append(f"{result_type} text mismatch: expected '{expected_text}', got '{actual_text}'")

        # Validate tool calls
        if "tools" in expected:
            expected_tools = expected["tools"]
            actual_tools_list = []

            # Convert actual_tools to list format for comparison
            for tool_idx in sorted(actual_tools.keys()):
                tool = actual_tools[tool_idx]
                tool_name = tool.get("name", "")
                tool_args_str = tool.get("args", "")

                # Try to parse args as JSON
                try:
                    if isinstance(tool_args_str, str):
                        tool_args = json.loads(tool_args_str)
                    else:
                        tool_args = tool_args_str
                except:
                    tool_args = tool_args_str

                actual_tools_list.append({"name": tool_name, "args": tool_args})

            # Compare tool count
            if len(actual_tools_list) != len(expected_tools):
                match = False
                errors.append(f"{result_type} tool count mismatch: expected {len(expected_tools)}, got {len(actual_tools_list)}")

            # Compare each tool
            for i, expected_tool in enumerate(expected_tools):
                if i >= len(actual_tools_list):
                    match = False
                    errors.append(f"{result_type} tool {i} missing")
                    continue

                actual_tool = actual_tools_list[i]
                if actual_tool["name"] != expected_tool["name"]:
                    match = False
                    errors.append(f"{result_type} tool {i} name mismatch: expected '{expected_tool['name']}', got '{actual_tool['name']}'")

                # Deep compare args
                if not self._deep_compare(actual_tool.get("args", {}), expected_tool.get("args", {})):
                    match = False
                    errors.append(f"{result_type} tool {i} args mismatch: expected {expected_tool.get('args')}, got {actual_tool.get('args')}")

        return match, errors

    def _deep_compare(self, actual: any, expected: any) -> bool:
        """Deep compare two values for equality"""
        if type(actual) != type(expected):
            return False

        if isinstance(actual, dict):
            if set(actual.keys()) != set(expected.keys()):
                return False
            for key in actual.keys():
                if not self._deep_compare(actual[key], expected[key]):
                    return False
            return True
        elif isinstance(actual, list):
            if len(actual) != len(expected):
                return False
            for i in range(len(actual)):
                if not self._deep_compare(actual[i], expected[i]):
                    return False
            return True
        else:
            return actual == expected

def token_stream_generator(text: str, mode: str = "atomic_tags") -> Generator[str, None, None]:
    """
    Generator to simulate streaming output.

    Args:
        text: Complete LLM output string.
        mode:
            'char': Randomly output 1-5 characters at a time (stress test).
            'atomic_tags': Ensure tags like <function=...> are not split, other parts are random.
            'whole': Output entire text at once (for debugging).

    Yields:
        str: Streaming text chunk
    """
    if mode == "char":
        idx = 0
        while idx < len(text):
            # Randomly simulate different token lengths
            step = random.randint(1, 5)
            yield text[idx : idx + step]
            idx += step

    elif mode == "atomic_tags":
        # Protect specific tags from being split, simulating ideal Tokenizer behavior
        protected_tokens = [
            "<function=",
            "</function>",
            "<parameter=",
            "</parameter>",
            ">",
        ]

        # Simple regex split
        pattern = "|".join(map(re.escape, protected_tokens))
        parts = re.split(f"({pattern})", text)

        for part in parts:
            if not part:
                continue
            if part in protected_tokens:
                yield part
            else:
                # Content between tags, like parameter values, can be randomly fragmented
                idx = 0
                while idx < len(part):
                    step = random.randint(1, 4)
                    yield part[idx : idx + step]
                    idx += step

    elif mode == "whole":
        # Output entire text at once (for debugging)
        yield text

    else:  # default simple iteration
        yield text
