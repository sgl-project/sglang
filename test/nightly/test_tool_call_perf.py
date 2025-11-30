"""
Standalone Tool Call Benchmark Test.

This test provides a quick CI validation for tool call functionality
using a small model. For comprehensive model-specific tool call testing,
see the individual nightly performance tests (test_text_models_perf.py,
test_qwen3_235b_perf.py, etc.) which run tool call benchmarks alongside
performance benchmarks.

This test is useful for:
- Quick validation of tool call infrastructure
- Testing with small models in CI
- Standalone tool call testing without perf benchmarks
"""

import unittest

from sglang.bench_tool_call import (
    ToolCallBenchmark,
    ToolCallParser,
    run_tool_call_benchmark_for_nightly,
)
from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    popen_launch_server,
)


class TestToolCallStandalone(unittest.TestCase):
    """
    Standalone tool call test for quick CI validation.

    Uses a small model to verify tool call infrastructure works correctly.
    """

    @classmethod
    def setUpClass(cls):
        """Set up test fixtures."""
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.api_key = "test-key"
        cls.parser = ToolCallParser.LLAMA3

        # Launch server
        cls.process = popen_launch_server(
            model=cls.model,
            base_url=cls.base_url,
            other_args=["--tool-call-parser", cls.parser.value],
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            api_key=cls.api_key,
        )

    @classmethod
    def tearDownClass(cls):
        """Clean up server process."""
        kill_process_tree(cls.process.pid)

    def test_tool_call_benchmark(self):
        """Run standard tool call benchmark suite."""
        benchmark = ToolCallBenchmark(
            base_url=f"{self.base_url}/v1",
            api_key=self.api_key,
            model=self.model,
            parser=self.parser,
        )

        result = benchmark.run_benchmark()

        # Print detailed results
        print(result.to_markdown_report())

        # Consider success if at least 70% of tests pass
        self.assertGreaterEqual(
            result.success_rate,
            0.7,
            f"Tool call benchmark failed: {result.passed_tests}/{result.total_tests} "
            f"tests passed ({result.success_rate:.1%})",
        )


class TestToolCallNightlyHelper(unittest.TestCase):
    """
    Test the nightly helper function for tool call benchmarks.

    This validates that run_tool_call_benchmark_for_nightly() works correctly.
    """

    def test_nightly_helper_function(self):
        """Test run_tool_call_benchmark_for_nightly helper."""
        result, success = run_tool_call_benchmark_for_nightly(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            parser=ToolCallParser.LLAMA3,
            base_url=DEFAULT_URL_FOR_TEST,
        )

        # Print results
        print(result.to_markdown_report())

        # Verify result structure
        self.assertIsNotNone(result)
        self.assertGreater(result.total_tests, 0)
        self.assertIsNotNone(result.model)
        self.assertEqual(result.parser, ToolCallParser.LLAMA3.value)


if __name__ == "__main__":
    unittest.main()
