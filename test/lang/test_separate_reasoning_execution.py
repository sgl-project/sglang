"""
Tests for the execution of separate_reasoning functionality in sglang.

Usage:
python3 -m unittest test/lang/test_separate_reasoning_execution.py
"""

import threading
import unittest
from unittest.mock import MagicMock, patch

from sglang.lang.interpreter import StreamExecutor
from sglang.lang.ir import SglGen, SglSeparateReasoning
from sglang.test.test_utils import CustomTestCase


# Helper function to create events that won't block program exit
def create_daemon_event():
    event = threading.Event()
    return event


class MockReasoningParser:
    def __init__(self, model_type):
        self.model_type = model_type
        self.parse_non_stream_called = False
        self.parse_stream_chunk_called = False

    def parse_non_stream(self, full_text):
        self.parse_non_stream_called = True
        # Simulate parsing by adding a prefix to indicate reasoning
        reasoning = f"[REASONING from {self.model_type}]: {full_text}"
        normal_text = f"[NORMAL from {self.model_type}]: {full_text}"
        return reasoning, normal_text

    def parse_stream_chunk(self, chunk_text):
        self.parse_stream_chunk_called = True
        # Simulate parsing by adding a prefix to indicate reasoning
        reasoning = f"[REASONING from {self.model_type}]: {chunk_text}"
        normal_text = f"[NORMAL from {self.model_type}]: {chunk_text}"
        return reasoning, normal_text


class TestSeparateReasoningExecution(CustomTestCase):
    def setUp(self):
        """Set up for the test."""
        super().setUp()
        # Store any events created during the test
        self.events = []

    def tearDown(self):
        """Clean up any threads that might have been created during the test."""
        super().tearDown()

        # Set all events to ensure any waiting threads are released
        for event in self.events:
            event.set()

    def tearDown(self):
        super().tearDown()
        # wake up all threads
        for ev in self.events:
            ev.set()

    @patch("sglang.srt.parser.reasoning_parser.ReasoningParser")
    def test_execute_separate_reasoning(self, mock_parser_class):
        """Test that _execute_separate_reasoning correctly calls the ReasoningParser."""
        # Setup mock parser
        mock_parser = MockReasoningParser("deepseek-r1")
        mock_parser_class.return_value = mock_parser

        # Create a mock backend to avoid AttributeError in __del__
        mock_backend = MagicMock()

        # Create a StreamExecutor with necessary setup
        executor = StreamExecutor(
            backend=mock_backend,
            arguments={},
            default_sampling_para={},
            chat_template={
                "role_map": {"user": "user", "assistant": "assistant"}
            },  # Simple chat template
            stream=False,
            use_thread=False,
        )

        # Set up the executor with a variable and its value
        var_name = "test_var"
        reasoning_name = f"{var_name}_reasoning_content"
        var_value = "Test content"
        executor.variables = {var_name: var_value}

        # Create events and track them for cleanup
        var_event = create_daemon_event()
        reasoning_event = create_daemon_event()
        self.events.extend([var_event, reasoning_event])

        executor.variable_event = {var_name: var_event, reasoning_name: reasoning_event}
        executor.variable_event[var_name].set()  # Mark as ready

        # Set up the current role
        executor.cur_role = "assistant"
        executor.cur_role_begin_pos = 0
        executor.text_ = var_value

        # Create a gen expression and a separate_reasoning expression
        gen_expr = SglGen(var_name)
        expr = SglSeparateReasoning("deepseek-r1", expr=gen_expr)

        # Execute separate_reasoning
        executor._execute_separate_reasoning(expr)

        # Verify that the parser was created with the correct model type
        mock_parser_class.assert_called_once_with("deepseek-r1")

        # Verify that parse_non_stream was called
        self.assertTrue(mock_parser.parse_non_stream_called)

        # Verify that the variables were updated correctly
        reasoning_name = f"{var_name}_reasoning_content"
        self.assertIn(reasoning_name, executor.variables)
        self.assertEqual(
            executor.variables[reasoning_name],
            f"[REASONING from deepseek-r1]: {var_value}",
        )
        self.assertEqual(
            executor.variables[var_name], f"[NORMAL from deepseek-r1]: {var_value}"
        )

        # Verify that the variable event was set
        self.assertIn(reasoning_name, executor.variable_event)
        self.assertTrue(executor.variable_event[reasoning_name].is_set())

        # Verify that the text was updated
        self.assertEqual(executor.text_, f"[NORMAL from deepseek-r1]: {var_value}")

    @patch("sglang.srt.parser.reasoning_parser.ReasoningParser")
    def test_reasoning_parser_integration(self, mock_parser_class):
        """Test the integration between separate_reasoning and ReasoningParser."""
        # Setup mock parsers for different model types
        deepseek_parser = MockReasoningParser("deepseek-r1")
        qwen_parser = MockReasoningParser("qwen3")

        # Configure the mock to return different parsers based on model type
        def get_parser(model_type):
            if model_type == "deepseek-r1":
                return deepseek_parser
            elif model_type == "qwen3":
                return qwen_parser
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        mock_parser_class.side_effect = get_parser

        # Test with DeepSeek-R1 model
        test_text = "This is a test"
        reasoning, normal_text = deepseek_parser.parse_non_stream(test_text)

        self.assertEqual(reasoning, f"[REASONING from deepseek-r1]: {test_text}")
        self.assertEqual(normal_text, f"[NORMAL from deepseek-r1]: {test_text}")

        # Test with Qwen3 model
        reasoning, normal_text = qwen_parser.parse_non_stream(test_text)

        self.assertEqual(reasoning, f"[REASONING from qwen3]: {test_text}")
        self.assertEqual(normal_text, f"[NORMAL from qwen3]: {test_text}")

    @patch("sglang.srt.parser.reasoning_parser.ReasoningParser")
    def test_reasoning_parser_invalid_model(self, mock_parser_class):
        """Test that ReasoningParser raises an error for invalid model types."""

        # Configure the mock to raise an error for invalid model types
        def get_parser(model_type):
            if model_type in ["deepseek-r1", "qwen3"]:
                return MockReasoningParser(model_type)
            elif model_type is None:
                raise ValueError("Model type must be specified")
            else:
                raise ValueError(f"Unsupported model type: {model_type}")

        mock_parser_class.side_effect = get_parser

        with self.assertRaises(ValueError) as context:
            mock_parser_class("invalid-model")
        self.assertIn("Unsupported model type", str(context.exception))

        with self.assertRaises(ValueError) as context:
            mock_parser_class(None)
        self.assertIn("Model type must be specified", str(context.exception))


if __name__ == "__main__":
    unittest.main()
