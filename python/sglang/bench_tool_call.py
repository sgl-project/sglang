"""
Tool Call Benchmark for SGLang.

This module provides utilities for benchmarking tool/function calling capabilities
of LLM models. It supports multiple tool call parsers and provides metrics for:
- Tool call success rate
- Latency (TTFT, E2E)
- Throughput
- Argument parsing accuracy

Supported parsers: llama3, llama4, qwen, qwen3_coder, deepseekv3, deepseekv31,
                   glm, gpt-oss, kimi_k2, mistral, pythonic, step3

Usage:
    python -m sglang.bench_tool_call --base-url http://localhost:30000/v1 --parser llama3
"""

import argparse
import json
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Union

import openai
from pydantic import BaseModel


class ToolCallParser(str, Enum):
    """Supported tool call parsers."""

    LLAMA3 = "llama3"
    LLAMA4 = "llama4"
    QWEN = "qwen"
    QWEN25 = "qwen25"  # Legacy alias for qwen
    QWEN3_CODER = "qwen3_coder"
    DEEPSEEK_V3 = "deepseekv3"
    DEEPSEEK_V31 = "deepseekv31"
    GLM = "glm"
    GPT_OSS = "gpt-oss"
    KIMI_K2 = "kimi_k2"
    MISTRAL = "mistral"
    PYTHONIC = "pythonic"
    STEP3 = "step3"


class ToolChoiceMode(str, Enum):
    """Tool choice modes."""

    AUTO = "auto"
    NONE = "none"
    REQUIRED = "required"
    SPECIFIC = "specific"


@dataclass
class ToolDefinition:
    """Definition of a tool for benchmarking."""

    name: str
    description: str
    parameters: Dict[str, Any]
    strict: bool = False
    expected_args: Optional[Dict[str, Any]] = None  # For validation

    def to_openai_format(self) -> Dict[str, Any]:
        """Convert to OpenAI tool format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
                "strict": self.strict,
            },
        }


@dataclass
class ToolCallTestCase:
    """A single test case for tool call benchmarking."""

    name: str
    description: str
    messages: List[Dict[str, str]]
    tools: List[ToolDefinition]
    tool_choice: Union[str, Dict[str, Any]] = "auto"
    expected_function_name: Optional[str] = None
    expected_args: Optional[Dict[str, Any]] = None
    expected_args_subset: Optional[Dict[str, Any]] = None  # Partial match
    streaming: bool = False
    n_choices: int = 1
    max_tokens: int = 1024
    temperature: float = 0.0
    multi_turn: bool = False  # Whether this is a multi-turn test
    tool_result: Optional[str] = None  # Result to return if multi-turn


class ToolCallResult(BaseModel):
    """Result of a single tool call benchmark."""

    test_name: str
    success: bool
    function_name: Optional[str] = None
    arguments: Optional[Dict[str, Any]] = None
    arguments_raw: Optional[str] = None
    error_message: Optional[str] = None

    # Timing metrics (in seconds)
    time_to_first_token: Optional[float] = None
    end_to_end_latency: float = 0.0

    # Validation results
    function_name_correct: bool = False
    arguments_correct: bool = False
    arguments_parseable: bool = False

    # Response details
    finish_reason: Optional[str] = None
    content: Optional[str] = None
    streaming: bool = False
    n_choices: int = 1

    def to_markdown_row(self) -> str:
        """Convert result to markdown table row."""
        status = "PASS" if self.success else "FAIL"
        ttft = (
            f"{self.time_to_first_token * 1000:.2f}"
            if self.time_to_first_token
            else "N/A"
        )
        e2e = f"{self.end_to_end_latency * 1000:.2f}"
        args_status = (
            "PASS"
            if self.arguments_correct
            else ("PARSE" if self.arguments_parseable else "FAIL")
        )

        return (
            f"| {self.test_name} | {status} | {self.function_name or 'N/A'} | "
            f"{args_status} | {ttft} | {e2e} | {self.finish_reason or 'N/A'} |"
        )


class ToolCallBenchmarkResult(BaseModel):
    """Aggregated results of tool call benchmark."""

    model_path: str
    parser: str
    total_tests: int
    passed_tests: int
    failed_tests: int
    success_rate: float

    # Aggregate timing metrics
    avg_ttft_ms: Optional[float] = None
    avg_e2e_latency_ms: float = 0.0
    p50_e2e_latency_ms: Optional[float] = None
    p95_e2e_latency_ms: Optional[float] = None
    p99_e2e_latency_ms: Optional[float] = None

    # Detailed results
    results: List[ToolCallResult] = []

    def to_markdown_report(self) -> str:
        """Generate markdown report."""
        lines = [
            f"### Tool Call Benchmark: {self.model_path}",
            f"**Parser**: {self.parser}",
            f"**Success Rate**: {self.success_rate:.1%} ({self.passed_tests}/{self.total_tests})",
            "",
            "#### Timing Metrics",
            (
                f"- Avg TTFT: {self.avg_ttft_ms:.2f} ms"
                if self.avg_ttft_ms
                else "- Avg TTFT: N/A"
            ),
            f"- Avg E2E Latency: {self.avg_e2e_latency_ms:.2f} ms",
            (
                f"- P50 Latency: {self.p50_e2e_latency_ms:.2f} ms"
                if self.p50_e2e_latency_ms
                else ""
            ),
            (
                f"- P95 Latency: {self.p95_e2e_latency_ms:.2f} ms"
                if self.p95_e2e_latency_ms
                else ""
            ),
            (
                f"- P99 Latency: {self.p99_e2e_latency_ms:.2f} ms"
                if self.p99_e2e_latency_ms
                else ""
            ),
            "",
            "#### Detailed Results",
            "| Test | Status | Function | Args | TTFT (ms) | E2E (ms) | Finish |",
            "|------|--------|----------|------|-----------|----------|--------|",
        ]

        for result in self.results:
            lines.append(result.to_markdown_row())

        return "\n".join(lines)

    def to_json(self) -> str:
        """Export results to JSON."""
        return self.model_dump_json(indent=2)


# Pre-defined tool definitions for benchmarking
BENCHMARK_TOOLS = {
    "get_current_weather": ToolDefinition(
        name="get_current_weather",
        description="Get the current weather in a given location",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The city to find the weather for, e.g. 'San Francisco'",
                },
                "state": {
                    "type": "string",
                    "description": "The two-letter abbreviation for the state, e.g. 'CA'",
                },
                "unit": {
                    "type": "string",
                    "description": "The unit to fetch the temperature in",
                    "enum": ["celsius", "fahrenheit"],
                },
            },
            "required": ["city", "state", "unit"],
        },
    ),
    "add": ToolDefinition(
        name="add",
        description="Compute the sum of two numbers",
        parameters={
            "type": "object",
            "properties": {
                "a": {"type": "integer", "description": "First number"},
                "b": {"type": "integer", "description": "Second number"},
            },
            "required": ["a", "b"],
        },
    ),
    "subtract": ToolDefinition(
        name="subtract",
        description="Compute the difference of two integers",
        parameters={
            "type": "object",
            "properties": {
                "int_a": {"type": "integer", "description": "First integer"},
                "int_b": {"type": "integer", "description": "Second integer"},
            },
            "required": ["int_a", "int_b"],
        },
        strict=True,
    ),
    "get_tourist_attractions": ToolDefinition(
        name="get_tourist_attractions",
        description="Get a list of top tourist attractions for a given city",
        parameters={
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "The name of the city to find attractions for",
                },
            },
            "required": ["city"],
        },
    ),
    "search_web": ToolDefinition(
        name="search_web",
        description="Search the web for information",
        parameters={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "The search query",
                },
                "num_results": {
                    "type": "integer",
                    "description": "Number of results to return",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    ),
}


def get_standard_test_cases(parser: ToolCallParser) -> List[ToolCallTestCase]:
    """
    Get standard test cases for a given parser.

    These test cases are based on the tool_parser.ipynb examples and cover:
    - Basic function calling (non-streaming)
    - Streaming function calling
    - Tool choice: required
    - Tool choice: specific function
    - Multi-turn conversation
    - Strict mode
    - Multiple tools

    Args:
        parser: The tool call parser being tested

    Returns:
        List of test cases appropriate for the parser
    """
    # System message varies by parser
    system_messages = {
        ToolCallParser.LLAMA3: (
            "You are a helpful assistant with tool calling capabilities. "
            "Only reply with a tool call if the function exists in the library provided by the user. "
            "If it doesn't exist, just reply directly in natural language. "
            "When you receive a tool call response, use the output to format an answer to the original user question. "
            "You have access to the following functions. "
            "To call a function, please respond with JSON for a function call. "
            'Respond in the format {"name": function name, "parameters": dictionary of argument name and its value}. '
            "Do not use variables.\n\n"
        ),
        ToolCallParser.PYTHONIC: (
            "You are a travel assistant. "
            "When asked to call functions, ALWAYS respond ONLY with a python list of function calls, "
            "using this format: [func_name1(param1=value1, param2=value2), func_name2(param=value)]. "
            "Do NOT use JSON, do NOT use variables, do NOT use any other format. "
            "Here is an example:\n"
            '[get_weather(location="Paris"), get_tourist_attractions(city="Paris")]'
        ),
    }

    default_system = "You are a helpful assistant with tool calling capabilities."
    system_message = system_messages.get(parser, default_system)

    test_cases = [
        # Test 1: Basic function calling - weather
        ToolCallTestCase(
            name="basic_weather",
            description="Basic weather function call",
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": "What's the weather like in Boston today? Use the tools to help you.",
                },
            ],
            tools=[BENCHMARK_TOOLS["get_current_weather"]],
            expected_function_name="get_current_weather",
            expected_args_subset={"city": "Boston"},
        ),
        # Test 2: Basic function calling - math
        ToolCallTestCase(
            name="basic_add",
            description="Basic addition function call",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": "Compute (3+5)"},
            ],
            tools=[BENCHMARK_TOOLS["add"]],
            expected_function_name="add",
            expected_args={"a": 3, "b": 5},
        ),
        # Test 3: Streaming function call
        ToolCallTestCase(
            name="streaming_weather",
            description="Streaming weather function call",
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": "What is the temperature in Paris in celsius?",
                },
            ],
            tools=[BENCHMARK_TOOLS["get_current_weather"]],
            streaming=True,
            expected_function_name="get_current_weather",
            expected_args_subset={"city": "Paris", "unit": "celsius"},
        ),
        # Test 4: Streaming with argument parsing
        ToolCallTestCase(
            name="streaming_add",
            description="Streaming addition with argument parsing",
            messages=[
                {"role": "system", "content": system_message},
                {
                    "role": "user",
                    "content": "Please sum 5 and 7, just call the function.",
                },
            ],
            tools=[
                ToolDefinition(
                    name="add",
                    description="Compute the sum of two integers",
                    parameters={
                        "type": "object",
                        "properties": {
                            "a": {"type": "integer", "description": "First integer"},
                            "b": {"type": "integer", "description": "Second integer"},
                        },
                        "required": ["a", "b"],
                    },
                    strict=True,
                )
            ],
            streaming=True,
            expected_function_name="add",
            expected_args={"a": 5, "b": 7},
        ),
        # Test 5: Tool choice = required
        ToolCallTestCase(
            name="tool_choice_required",
            description="Force tool call with tool_choice=required",
            messages=[
                {"role": "user", "content": "What is the capital of France?"},
            ],
            tools=[BENCHMARK_TOOLS["get_current_weather"]],
            tool_choice="required",
            expected_function_name="get_current_weather",
        ),
        # Test 6: Tool choice = specific function
        ToolCallTestCase(
            name="tool_choice_specific",
            description="Force specific function with tool_choice",
            messages=[
                {
                    "role": "user",
                    "content": "What are the most attractive places in France?",
                },
            ],
            tools=[BENCHMARK_TOOLS["get_current_weather"]],
            tool_choice={
                "type": "function",
                "function": {"name": "get_current_weather"},
            },
            expected_function_name="get_current_weather",
        ),
        # Test 7: Strict mode
        ToolCallTestCase(
            name="strict_mode",
            description="Strict mode function calling",
            messages=[
                {"role": "user", "content": "Please compute 5 - 7, using your tool."},
            ],
            tools=[BENCHMARK_TOOLS["subtract"]],
            expected_function_name="subtract",
            expected_args={"int_a": 5, "int_b": 7},
        ),
        # Test 8: Multiple tools selection
        ToolCallTestCase(
            name="multiple_tools",
            description="Select correct tool from multiple options",
            messages=[
                {"role": "system", "content": system_message},
                {"role": "user", "content": "What's the weather in Seattle?"},
            ],
            tools=[
                BENCHMARK_TOOLS["add"],
                BENCHMARK_TOOLS["get_current_weather"],
                BENCHMARK_TOOLS["search_web"],
            ],
            expected_function_name="get_current_weather",
            expected_args_subset={"city": "Seattle"},
        ),
        # Test 9: No tool call expected (tool_choice=none)
        ToolCallTestCase(
            name="no_tool_call",
            description="No tool call when tool_choice=none",
            messages=[
                {"role": "user", "content": "Who are you?"},
            ],
            tools=[BENCHMARK_TOOLS["get_current_weather"]],
            tool_choice="none",
            expected_function_name=None,  # Expect no tool call
        ),
        # Test 10: Multiple choices (n > 1)
        ToolCallTestCase(
            name="multiple_choices",
            description="Multiple choices with n=2",
            messages=[
                {"role": "user", "content": "What is the weather like in Los Angeles?"},
            ],
            tools=[BENCHMARK_TOOLS["get_current_weather"]],
            tool_choice="required",
            streaming=True,
            n_choices=2,
            expected_function_name="get_current_weather",
        ),
    ]

    # Add pythonic-specific test case
    if parser == ToolCallParser.PYTHONIC:
        test_cases.append(
            ToolCallTestCase(
                name="pythonic_parallel",
                description="Pythonic parallel tool calls",
                messages=[
                    {"role": "system", "content": system_message},
                    {
                        "role": "user",
                        "content": (
                            "I'm planning a trip to Tokyo next week. What's the weather like and "
                            "what are some top tourist attractions? "
                            "Propose parallel tool calls at once."
                        ),
                    },
                ],
                tools=[
                    BENCHMARK_TOOLS["get_current_weather"],
                    BENCHMARK_TOOLS["get_tourist_attractions"],
                ],
            )
        )

    return test_cases


class ToolCallBenchmark:
    """
    Tool call benchmark runner.

    This class runs tool call benchmarks against a model server and collects
    metrics on success rate, latency, and argument parsing accuracy.
    """

    def __init__(
        self,
        base_url: str,
        api_key: str = "test-key",
        model: Optional[str] = None,
        parser: ToolCallParser = ToolCallParser.LLAMA3,
        timeout: float = 60.0,
    ):
        """
        Initialize the benchmark runner.

        Args:
            base_url: Base URL for the server (e.g., "http://localhost:30000/v1")
            api_key: API key for authentication
            model: Model name (if None, will be auto-detected)
            parser: Tool call parser being used
            timeout: Request timeout in seconds
        """
        self.base_url = base_url
        self.api_key = api_key
        self.parser = parser
        self.timeout = timeout

        # Initialize client
        self.client = openai.Client(
            api_key=api_key,
            base_url=base_url,
            timeout=timeout,
        )

        # Auto-detect model if not provided
        if model is None:
            try:
                models = self.client.models.list()
                self.model = models.data[0].id if models.data else "unknown"
            except Exception:
                self.model = "unknown"
        else:
            self.model = model

    def run_single_test(self, test_case: ToolCallTestCase) -> ToolCallResult:
        """
        Run a single test case.

        Args:
            test_case: The test case to run

        Returns:
            ToolCallResult with metrics and validation results
        """
        result = ToolCallResult(
            test_name=test_case.name,
            success=False,
            streaming=test_case.streaming,
            n_choices=test_case.n_choices,
        )

        tools = [tool.to_openai_format() for tool in test_case.tools]
        start_time = time.perf_counter()

        try:
            if test_case.streaming:
                result = self._run_streaming_test(test_case, tools, result, start_time)
            else:
                result = self._run_non_streaming_test(
                    test_case, tools, result, start_time
                )

        except Exception as e:
            result.error_message = str(e)
            result.end_to_end_latency = time.perf_counter() - start_time

        return result

    def _run_non_streaming_test(
        self,
        test_case: ToolCallTestCase,
        tools: List[Dict],
        result: ToolCallResult,
        start_time: float,
    ) -> ToolCallResult:
        """Run a non-streaming test case."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=test_case.messages,
            tools=tools,
            tool_choice=test_case.tool_choice,
            max_tokens=test_case.max_tokens,
            temperature=test_case.temperature,
            stream=False,
            n=test_case.n_choices,
        )

        result.end_to_end_latency = time.perf_counter() - start_time

        choice = response.choices[0]
        result.finish_reason = choice.finish_reason
        result.content = choice.message.content

        tool_calls = choice.message.tool_calls

        # Handle case where no tool call is expected
        if test_case.expected_function_name is None:
            result.success = tool_calls is None or len(tool_calls) == 0
            if not result.success:
                result.error_message = "Expected no tool call but got one"
            return result

        if not tool_calls or len(tool_calls) == 0:
            result.error_message = "No tool calls returned"
            return result

        tool_call = tool_calls[0]
        result.function_name = tool_call.function.name
        result.arguments_raw = tool_call.function.arguments

        # Validate function name
        result.function_name_correct = (
            result.function_name == test_case.expected_function_name
        )

        # Parse and validate arguments
        result = self._validate_arguments(test_case, result)

        # Overall success
        result.success = result.function_name_correct and (
            result.arguments_correct or test_case.expected_args is None
        )

        return result

    def _run_streaming_test(
        self,
        test_case: ToolCallTestCase,
        tools: List[Dict],
        result: ToolCallResult,
        start_time: float,
    ) -> ToolCallResult:
        """Run a streaming test case."""
        response_stream = self.client.chat.completions.create(
            model=self.model,
            messages=test_case.messages,
            tools=tools,
            tool_choice=test_case.tool_choice,
            max_tokens=test_case.max_tokens,
            temperature=test_case.temperature,
            stream=True,
            n=test_case.n_choices,
        )

        function_name = None
        argument_fragments = []
        first_token_time = None
        content_fragments = []
        finish_reasons = {}

        for chunk in response_stream:
            if first_token_time is None:
                first_token_time = time.perf_counter()

            for choice in chunk.choices:
                if choice.finish_reason is not None:
                    finish_reasons[choice.index] = choice.finish_reason

                if choice.delta.content:
                    content_fragments.append(choice.delta.content)

                if choice.delta.tool_calls:
                    tool_call = choice.delta.tool_calls[0]
                    if tool_call.function.name:
                        function_name = tool_call.function.name
                    if tool_call.function.arguments:
                        argument_fragments.append(tool_call.function.arguments)

        result.end_to_end_latency = time.perf_counter() - start_time
        if first_token_time:
            result.time_to_first_token = first_token_time - start_time

        result.finish_reason = finish_reasons.get(0)
        result.content = "".join(content_fragments) if content_fragments else None
        result.function_name = function_name
        result.arguments_raw = (
            "".join(argument_fragments) if argument_fragments else None
        )

        # Handle case where no tool call is expected
        if test_case.expected_function_name is None:
            result.success = function_name is None
            if not result.success:
                result.error_message = "Expected no tool call but got one"
            return result

        if function_name is None:
            result.error_message = "No function name found in streaming response"
            return result

        # Validate function name
        result.function_name_correct = function_name == test_case.expected_function_name

        # Parse and validate arguments
        result = self._validate_arguments(test_case, result)

        # For multiple choices, verify all got finish_reason
        if test_case.n_choices > 1:
            if len(finish_reasons) != test_case.n_choices:
                result.error_message = f"Expected {test_case.n_choices} finish reasons, got {len(finish_reasons)}"

        # Overall success
        result.success = result.function_name_correct and (
            result.arguments_correct or test_case.expected_args is None
        )

        return result

    def _validate_arguments(
        self,
        test_case: ToolCallTestCase,
        result: ToolCallResult,
    ) -> ToolCallResult:
        """Validate parsed arguments against expected values."""
        if not result.arguments_raw:
            return result

        try:
            result.arguments = json.loads(result.arguments_raw)
            result.arguments_parseable = True
        except json.JSONDecodeError:
            result.arguments_parseable = False
            result.error_message = "Arguments are not valid JSON"
            return result

        # Check exact match
        if test_case.expected_args is not None:
            # Allow string/int flexibility for certain values
            result.arguments_correct = self._args_match(
                result.arguments, test_case.expected_args
            )

        # Check subset match
        elif test_case.expected_args_subset is not None:
            result.arguments_correct = self._args_subset_match(
                result.arguments, test_case.expected_args_subset
            )
        else:
            # No expected args to validate
            result.arguments_correct = True

        return result

    def _args_match(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """Check if arguments match (with type flexibility)."""
        if set(actual.keys()) != set(expected.keys()):
            return False

        for key, expected_val in expected.items():
            actual_val = actual.get(key)
            # Allow string/int flexibility
            if str(actual_val) != str(expected_val):
                return False

        return True

    def _args_subset_match(
        self, actual: Dict[str, Any], expected_subset: Dict[str, Any]
    ) -> bool:
        """Check if expected subset is contained in actual args."""
        for key, expected_val in expected_subset.items():
            actual_val = actual.get(key)
            if actual_val is None:
                return False
            # Case-insensitive string matching for flexibility
            if isinstance(expected_val, str) and isinstance(actual_val, str):
                if expected_val.lower() not in actual_val.lower():
                    return False
            elif str(actual_val) != str(expected_val):
                return False

        return True

    def run_benchmark(
        self,
        test_cases: Optional[List[ToolCallTestCase]] = None,
    ) -> ToolCallBenchmarkResult:
        """
        Run the full benchmark suite.

        Args:
            test_cases: List of test cases to run (if None, uses standard test cases)

        Returns:
            ToolCallBenchmarkResult with aggregated metrics
        """
        if test_cases is None:
            test_cases = get_standard_test_cases(self.parser)

        results = []
        for test_case in test_cases:
            result = self.run_single_test(test_case)
            results.append(result)

        return self._aggregate_results(results)

    def _aggregate_results(
        self, results: List[ToolCallResult]
    ) -> ToolCallBenchmarkResult:
        """Aggregate individual results into benchmark result."""
        passed = sum(1 for r in results if r.success)
        total = len(results)

        # Calculate latency metrics
        e2e_latencies = [r.end_to_end_latency * 1000 for r in results]
        ttft_values = [
            r.time_to_first_token * 1000
            for r in results
            if r.time_to_first_token is not None
        ]

        avg_e2e = sum(e2e_latencies) / len(e2e_latencies) if e2e_latencies else 0
        avg_ttft = sum(ttft_values) / len(ttft_values) if ttft_values else None

        # Calculate percentiles using linear interpolation
        def _calculate_percentile(
            sorted_values: List[float], percentile: float
        ) -> Optional[float]:
            if not sorted_values:
                return None
            if len(sorted_values) == 1:
                return sorted_values[0]
            index = (len(sorted_values) - 1) * percentile
            lower = int(index)
            upper = min(lower + 1, len(sorted_values) - 1)
            weight = index - lower
            return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

        sorted_e2e = sorted(e2e_latencies)
        p50 = _calculate_percentile(sorted_e2e, 0.5)
        p95 = _calculate_percentile(sorted_e2e, 0.95)
        p99 = _calculate_percentile(sorted_e2e, 0.99)

        return ToolCallBenchmarkResult(
            model_path=self.model,
            parser=self.parser.value,
            total_tests=total,
            passed_tests=passed,
            failed_tests=total - passed,
            success_rate=passed / total if total > 0 else 0,
            avg_ttft_ms=avg_ttft,
            avg_e2e_latency_ms=avg_e2e,
            p50_e2e_latency_ms=p50,
            p95_e2e_latency_ms=p95,
            p99_e2e_latency_ms=p99,
            results=results,
        )


def run_tool_call_benchmark(
    base_url: str,
    parser: Union[str, ToolCallParser],
    api_key: str = "test-key",
    model: Optional[str] = None,
    test_cases: Optional[List[ToolCallTestCase]] = None,
) -> ToolCallBenchmarkResult:
    """
    Convenience function to run tool call benchmark.

    Args:
        base_url: Server base URL (e.g., "http://localhost:30000/v1")
        parser: Tool call parser name or enum
        api_key: API key for authentication
        model: Model name (auto-detected if None)
        test_cases: Custom test cases (uses standard if None)

    Returns:
        ToolCallBenchmarkResult with metrics and details
    """
    if isinstance(parser, str):
        parser = ToolCallParser(parser)

    benchmark = ToolCallBenchmark(
        base_url=base_url,
        api_key=api_key,
        model=model,
        parser=parser,
    )

    return benchmark.run_benchmark(test_cases)


class ToolCallBenchmarkRunner:
    """
    Helper class for running tool call benchmarks in nightly tests.

    This class follows the same pattern as NightlyBenchmarkRunner and provides:
    - Server lifecycle management (launch/kill)
    - Report generation in markdown format
    - Integration with GitHub CI summaries

    Usage in nightly tests:
        runner = ToolCallBenchmarkRunner("TestName", base_url)
        result, success = runner.run_benchmark_for_model(
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            parser=ToolCallParser.LLAMA3,
            other_args=["--tp", "1"],
        )
        runner.add_report(result)
        runner.write_final_report()
    """

    def __init__(
        self,
        test_name: str,
        base_url: str,
        gpu_config: Optional[str] = None,
    ):
        """
        Initialize the tool call benchmark runner.

        Args:
            test_name: Name of the test (used for reporting)
            base_url: Base URL for the server (without /v1)
            gpu_config: Optional GPU configuration string
        """
        import os

        self.test_name = test_name
        self.base_url = base_url
        self.gpu_config = gpu_config or os.environ.get("GPU_CONFIG", "")

        # Initialize report
        header = f"## {test_name} - Tool Call Benchmark"
        if self.gpu_config:
            header += f" ({self.gpu_config})"
        header += "\n\n"
        self.full_report = header

    def run_benchmark_for_model(
        self,
        model_path: str,
        parser: Union[str, ToolCallParser],
        other_args: Optional[List[str]] = None,
        test_cases: Optional[List[ToolCallTestCase]] = None,
        api_key: str = "test-key",
        timeout: int = 600,
        success_threshold: float = 0.7,
    ) -> tuple:
        """
        Run tool call benchmark for a model with server management.

        This method handles the full lifecycle:
        1. Launch server with tool-call-parser enabled
        2. Run benchmark tests
        3. Kill server
        4. Return results

        Args:
            model_path: Path to the model
            parser: Tool call parser to use
            other_args: Additional server launch arguments
            test_cases: Custom test cases (uses standard if None)
            api_key: API key for authentication
            timeout: Server launch timeout in seconds
            success_threshold: Minimum success rate to consider benchmark passed

        Returns:
            Tuple of (ToolCallBenchmarkResult, success_bool)
        """
        from sglang.srt.utils import kill_process_tree
        from sglang.test.test_utils import popen_launch_server

        if isinstance(parser, str):
            parser = ToolCallParser(parser)

        # Build server args
        server_args = list(other_args) if other_args else []
        server_args.extend(["--tool-call-parser", parser.value])

        # Launch server
        process = popen_launch_server(
            model=model_path,
            base_url=self.base_url,
            other_args=server_args,
            timeout=timeout,
            api_key=api_key,
        )

        try:
            # Run benchmark
            benchmark = ToolCallBenchmark(
                base_url=f"{self.base_url}/v1",
                api_key=api_key,
                model=model_path,
                parser=parser,
            )

            result = benchmark.run_benchmark(test_cases)
            success = result.success_rate >= success_threshold

            return result, success

        finally:
            # Always clean up server process
            kill_process_tree(process.pid)

    def run_benchmark_with_server(
        self,
        base_url: str,
        model_path: str,
        parser: Union[str, ToolCallParser],
        test_cases: Optional[List[ToolCallTestCase]] = None,
        api_key: str = "test-key",
        success_threshold: float = 0.7,
    ) -> tuple:
        """
        Run tool call benchmark against an already running server.

        Use this when the server is already launched by another benchmark.

        Args:
            base_url: Server URL with /v1 endpoint
            model_path: Model name for reporting
            parser: Tool call parser being used
            test_cases: Custom test cases (uses standard if None)
            api_key: API key for authentication
            success_threshold: Minimum success rate to consider benchmark passed

        Returns:
            Tuple of (ToolCallBenchmarkResult, success_bool)
        """
        if isinstance(parser, str):
            parser = ToolCallParser(parser)

        benchmark = ToolCallBenchmark(
            base_url=base_url,
            api_key=api_key,
            model=model_path,
            parser=parser,
        )

        result = benchmark.run_benchmark(test_cases)
        success = result.success_rate >= success_threshold

        return result, success

    def add_report(self, result: ToolCallBenchmarkResult) -> None:
        """Add benchmark result to the full report."""
        self.full_report += result.to_markdown_report() + "\n\n"

    def write_final_report(self) -> None:
        """Write the final report to GitHub summary if in CI."""
        from sglang.test.test_utils import is_in_ci, write_github_step_summary

        if is_in_ci():
            write_github_step_summary(self.full_report)
        print(self.full_report)

    def get_full_report(self) -> str:
        """Get the accumulated full report."""
        return self.full_report


# Model to parser mapping for convenience
MODEL_PARSER_MAP: Dict[str, ToolCallParser] = {
    # Llama models
    "meta-llama/Llama-3.1": ToolCallParser.LLAMA3,
    "meta-llama/Llama-3.2": ToolCallParser.LLAMA3,
    "meta-llama/Llama-3.3": ToolCallParser.LLAMA3,
    "meta-llama/Llama-4": ToolCallParser.LLAMA4,
    # Qwen models
    "Qwen/Qwen2": ToolCallParser.QWEN,
    "Qwen/Qwen2.5": ToolCallParser.QWEN,
    "Qwen/Qwen3": ToolCallParser.QWEN,
    "Qwen/Qwen3-Coder": ToolCallParser.QWEN3_CODER,
    # DeepSeek models
    "deepseek-ai/DeepSeek-V3-0324": ToolCallParser.DEEPSEEK_V3,
    "deepseek-ai/DeepSeek-V3.1": ToolCallParser.DEEPSEEK_V31,
    "deepseek-ai/DeepSeek-V3.2": ToolCallParser.DEEPSEEK_V31,
    # Other models
    "THUDM/GLM": ToolCallParser.GLM,
    "openai/gpt-oss": ToolCallParser.GPT_OSS,
    "moonshotai/Kimi-K2": ToolCallParser.KIMI_K2,
    "mistralai/Mistral": ToolCallParser.MISTRAL,
}


def get_parser_for_model(model_path: str) -> Optional[ToolCallParser]:
    """
    Get the appropriate tool call parser for a model.

    Args:
        model_path: Path to the model

    Returns:
        ToolCallParser if model supports tool calling, None otherwise
    """
    for prefix, parser in MODEL_PARSER_MAP.items():
        if model_path.startswith(prefix):
            return parser
    return None


def run_tool_call_benchmark_for_nightly(
    model_path: str,
    base_url: str,
    other_args: Optional[List[str]] = None,
    parser: Optional[Union[str, ToolCallParser]] = None,
    api_key: str = "test-key",
    timeout: int = 600,
    test_cases: Optional[List[ToolCallTestCase]] = None,
) -> tuple:
    """
    Convenience function to run tool call benchmark in nightly tests.

    This is the main entry point for adding tool call benchmarks to existing
    nightly tests. It handles parser detection and server management.

    Args:
        model_path: Path to the model
        base_url: Server base URL (without /v1)
        other_args: Additional server launch arguments
        parser: Tool call parser (auto-detected if None)
        api_key: API key for authentication
        timeout: Server launch timeout
        test_cases: Custom test cases (uses standard if None)

    Returns:
        Tuple of (ToolCallBenchmarkResult, success_bool)
        Returns (None, False) if model doesn't support tool calling

    Example:
        # In a nightly test
        result, success = run_tool_call_benchmark_for_nightly(
            model_path="meta-llama/Llama-3.1-8B-Instruct",
            base_url=DEFAULT_URL_FOR_TEST,
            other_args=["--tp", "1"],
        )
        if result:
            print(result.to_markdown_report())
    """
    # Auto-detect parser if not provided
    if parser is None:
        parser = get_parser_for_model(model_path)
        if parser is None:
            print(f"No tool call parser found for model: {model_path}")
            return None, False

    runner = ToolCallBenchmarkRunner(
        test_name=f"Tool Call - {model_path}",
        base_url=base_url,
    )

    return runner.run_benchmark_for_model(
        model_path=model_path,
        parser=parser,
        other_args=other_args,
        test_cases=test_cases,
        api_key=api_key,
        timeout=timeout,
    )


@dataclass
class BenchArgs:
    """Command line arguments for bench_tool_call."""

    base_url: str = "http://localhost:30000/v1"
    api_key: str = "test-key"
    model: Optional[str] = None
    parser: str = "llama3"
    output_file: Optional[str] = None
    show_report: bool = True
    json_output: bool = False

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
        """Add CLI arguments to parser."""
        parser.add_argument(
            "--base-url",
            type=str,
            default="http://localhost:30000/v1",
            help="Base URL for the server",
        )
        parser.add_argument(
            "--api-key",
            type=str,
            default="test-key",
            help="API key for authentication",
        )
        parser.add_argument(
            "--model",
            type=str,
            default=None,
            help="Model name (auto-detected if not provided)",
        )
        parser.add_argument(
            "--parser",
            type=str,
            default="llama3",
            choices=[p.value for p in ToolCallParser],
            help="Tool call parser to use",
        )
        parser.add_argument(
            "--output-file",
            type=str,
            default=None,
            help="Output file path for results",
        )
        parser.add_argument(
            "--show-report",
            action="store_true",
            default=True,
            help="Show markdown report",
        )
        parser.add_argument(
            "--json-output",
            action="store_true",
            default=False,
            help="Output results as JSON",
        )
        return parser

    @classmethod
    def from_cli_args(cls, args: argparse.Namespace) -> "BenchArgs":
        """Create from parsed CLI arguments."""
        return cls(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            parser=args.parser,
            output_file=args.output_file,
            show_report=args.show_report,
            json_output=args.json_output,
        )


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="Tool Call Benchmark for SGLang",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run benchmark with default settings
  python -m sglang.bench_tool_call --base-url http://localhost:30000/v1 --parser llama3

  # Run with specific model and JSON output
  python -m sglang.bench_tool_call --base-url http://localhost:30000/v1 --parser qwen --json-output

  # Save results to file
  python -m sglang.bench_tool_call --base-url http://localhost:30000/v1 --parser deepseekv3 --output-file results.json
        """,
    )

    BenchArgs.add_cli_args(parser)
    args = parser.parse_args()
    bench_args = BenchArgs.from_cli_args(args)

    # Run benchmark
    result = run_tool_call_benchmark(
        base_url=bench_args.base_url,
        parser=bench_args.parser,
        api_key=bench_args.api_key,
        model=bench_args.model,
    )

    # Output results
    if bench_args.json_output:
        output = result.to_json()
    else:
        output = result.to_markdown_report()

    if bench_args.show_report:
        print(output)

    if bench_args.output_file:
        with open(bench_args.output_file, "w") as f:
            if bench_args.json_output:
                f.write(result.to_json())
            else:
                f.write(result.to_markdown_report())
        print(f"\nResults saved to {bench_args.output_file}")

    # Return exit code based on success rate
    return 0 if result.success_rate >= 0.8 else 1


if __name__ == "__main__":
    exit(main())
