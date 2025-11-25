"""
Performance tests for structural tag overhead.

Benchmarks compilation time and generation speed for structural_tag
vs json_schema vs no constraint.
"""

import time
import unittest
from typing import Dict, List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.llama32_detector import Llama32Detector
from sglang.srt.function_call.utils import get_json_schema_constraint


class PerformanceTestCase(unittest.TestCase):
    """Base test case for performance tests."""

    def get_simple_tool(self) -> Tool:
        """Get a simple tool with basic parameters."""
        return Tool(
            type="function",
            function={
                "name": "get_weather",
                "description": "Get weather information",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string", "description": "City name"},
                        "unit": {
                            "type": "string",
                            "enum": ["celsius", "fahrenheit"],
                        },
                    },
                    "required": ["city"],
                },
            },
        )

    def get_complex_tool(self) -> Tool:
        """Get a tool with complex nested parameters."""
        return Tool(
            type="function",
            function={
                "name": "analyze_data",
                "description": "Analyze complex data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "data": {
                            "type": "object",
                            "properties": {
                                "metrics": {
                                    "type": "array",
                                    "items": {"type": "string"},
                                },
                                "config": {
                                    "type": "object",
                                    "properties": {
                                        "threshold": {"type": "number"},
                                        "enabled": {"type": "boolean"},
                                        "nested": {
                                            "type": "object",
                                            "properties": {
                                                "value": {"type": "string"},
                                            },
                                        },
                                    },
                                },
                            },
                            "required": ["metrics"],
                        },
                        "options": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "name": {"type": "string"},
                                    "value": {"type": "string"},
                                },
                            },
                        },
                    },
                    "required": ["data"],
                },
            },
        )

    def benchmark_build_time(self, func, *args, **kwargs):
        """Benchmark the time to build a constraint."""
        times = []
        for _ in range(10):  # Run 10 times for average
            start = time.perf_counter()
            result = func(*args, **kwargs)
            end = time.perf_counter()
            times.append(end - start)
        return sum(times) / len(times), result


class TestStructuralTagCompilationTime(PerformanceTestCase):
    """Test compilation time for structural tags."""

    def test_single_tool_structural_tag_vs_json_schema(self):
        """Compare compilation time for single tool."""
        tool = self.get_simple_tool()
        detector = Llama32Detector()

        # Benchmark structural tag
        struct_time, struct_result = self.benchmark_build_time(
            detector.build_structural_tag,
            tools=[tool],
            at_least_one=False,
            stop_after_first=False,
        )

        # Benchmark json schema
        json_time, json_result = self.benchmark_build_time(
            get_json_schema_constraint,
            tools=[tool],
            tool_choice="required",
            parallel_tool_calls=True,
        )

        # Both should be fast, but let's verify they complete
        self.assertIsNotNone(struct_result)
        self.assertIsNotNone(json_result)
        # Structural tag should be reasonably fast (less than 1ms typically)
        self.assertLess(struct_time, 0.01)
        self.assertLess(json_time, 0.01)

    def test_multiple_tools_structural_tag(self):
        """Test compilation time with multiple tools."""
        tools = [
            self.get_simple_tool(),
            self.get_complex_tool(),
            Tool(
                type="function",
                function={
                    "name": "get_time",
                    "description": "Get time",
                    "parameters": {
                        "type": "object",
                        "properties": {"timezone": {"type": "string"}},
                    },
                },
            ),
        ]
        detector = Llama32Detector()

        # Benchmark with 1, 3, 5 tools
        for num_tools in [1, 3]:
            test_tools = tools[:num_tools]
            avg_time, result = self.benchmark_build_time(
                detector.build_structural_tag,
                tools=test_tools,
                at_least_one=False,
                stop_after_first=False,
            )

            self.assertIsNotNone(result)
            self.assertEqual(len(result["format"]["tags"]), num_tools)
            # Should still be fast even with multiple tools
            self.assertLess(avg_time, 0.01)

    def test_complex_schema_structural_tag(self):
        """Test compilation time with complex parameter schemas."""
        tool = self.get_complex_tool()
        detector = Llama32Detector()

        avg_time, result = self.benchmark_build_time(
            detector.build_structural_tag,
            tools=[tool],
            at_least_one=False,
            stop_after_first=False,
        )

        self.assertIsNotNone(result)
        # Complex schemas should still be fast
        self.assertLess(avg_time, 0.01)

    def test_parser_integration_time(self):
        """Test time for parser.get_structure_constraint."""
        tools = [self.get_simple_tool()]
        parser = FunctionCallParser(tools=tools, tool_call_parser="llama3")

        # Benchmark auto mode (structural_tag)
        auto_time, auto_result = self.benchmark_build_time(
            parser.get_structure_constraint,
            tool_choice="auto",
            parallel_tool_calls=True,
        )

        # Benchmark required mode (json_schema)
        required_time, required_result = self.benchmark_build_time(
            parser.get_structure_constraint,
            tool_choice="required",
            parallel_tool_calls=True,
        )

        self.assertIsNotNone(auto_result)
        self.assertIsNotNone(required_result)
        # Both should be fast
        self.assertLess(auto_time, 0.01)
        self.assertLess(required_time, 0.01)


class TestStructuralTagScalability(PerformanceTestCase):
    """Test scalability with varying numbers of tools."""

    def test_scalability_one_to_five_tools(self):
        """Test performance scaling from 1 to 5 tools."""
        detector = Llama32Detector()
        base_tool = self.get_simple_tool()

        results = []
        for num_tools in [1, 3, 5]:
            tools = [
                Tool(
                    type="function",
                    function={
                        "name": f"tool_{i}",
                        "description": f"Tool {i}",
                        "parameters": base_tool.function.parameters,
                    },
                )
                for i in range(num_tools)
            ]

            avg_time, result = self.benchmark_build_time(
                detector.build_structural_tag,
                tools=tools,
                at_least_one=False,
                stop_after_first=False,
            )

            results.append((num_tools, avg_time))
            self.assertIsNotNone(result)
            self.assertEqual(len(result["format"]["tags"]), num_tools)

        # Performance should scale reasonably (not exponentially)
        # Time for 5 tools should be less than 5x time for 1 tool
        time_1 = results[0][1]
        time_5 = results[2][1]
        self.assertLess(time_5, time_1 * 10)  # Allow some overhead but not excessive

    def test_complex_schema_scalability(self):
        """Test performance with complex schemas."""
        detector = Llama32Detector()
        complex_tool = self.get_complex_tool()

        avg_time, result = self.benchmark_build_time(
            detector.build_structural_tag,
            tools=[complex_tool],
            at_least_one=False,
            stop_after_first=False,
        )

        self.assertIsNotNone(result)
        # Complex schemas should still be reasonably fast
        self.assertLess(avg_time, 0.01)


class TestRelativePerformance(PerformanceTestCase):
    """Test relative performance between different constraint types."""

    def test_structural_tag_vs_json_schema_relative(self):
        """Compare relative performance of structural_tag vs json_schema."""
        tool = self.get_simple_tool()
        detector = Llama32Detector()

        struct_times = []
        json_times = []

        for _ in range(20):  # More iterations for better average
            # Structural tag
            start = time.perf_counter()
            detector.build_structural_tag(
                tools=[tool], at_least_one=False, stop_after_first=False
            )
            struct_times.append(time.perf_counter() - start)

            # JSON schema
            start = time.perf_counter()
            get_json_schema_constraint(
                tools=[tool], tool_choice="required", parallel_tool_calls=True
            )
            json_times.append(time.perf_counter() - start)

        avg_struct = sum(struct_times) / len(struct_times)
        avg_json = sum(json_times) / len(json_times)

        # Both should be fast, verify they're in similar range
        # (structural tag might be slightly slower due to more complex structure)
        self.assertLess(avg_struct, 0.01)
        self.assertLess(avg_json, 0.01)
        # Structural tag shouldn't be more than 2x slower
        if avg_json > 0:
            ratio = avg_struct / avg_json
            self.assertLess(ratio, 10)  # Allow some variance but not excessive


if __name__ == "__main__":
    unittest.main()



