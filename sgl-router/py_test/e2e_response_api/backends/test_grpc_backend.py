"""
gRPC backend tests for Response API (including Harmony).

Run with:
    python3 -m pytest py_test/e2e_response_api/backends/test_grpc_backend.py -v
    python3 -m unittest e2e_response_api.backends.test_grpc_backend.TestGrpcBackend
"""

import json
import sys
import unittest
from pathlib import Path

# Add e2e_response_api directory for imports
_TEST_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_TEST_DIR))

# Import local modules
from mixins.function_call import FunctionCallingBaseTest
from mixins.mcp import MCPTests
from mixins.state_management import StateManagementTests
from mixins.streaming_events import HarmonyStreamingEventsTests, StreamingEventsTests
from mixins.structured_output import StructuredOutputBaseTest
from mixins.tool_choice import ToolChoiceTests
from router_fixtures import popen_launch_workers_and_router
from util import kill_process_tree


class TestGrpcBackend(
    StateManagementTests, MCPTests, StreamingEventsTests, StructuredOutputBaseTest
):
    """End to end tests for gRPC backend (Regular backend with Llama)."""

    @classmethod
    def setUpClass(cls):
        cls.model = "/home/ubuntu/models/meta-llama/Llama-3.1-8B-Instruct"
        cls.base_url_port = "http://127.0.0.1:30030"

        cls.cluster = popen_launch_workers_and_router(
            cls.model,
            cls.base_url_port,
            timeout=90,
            num_workers=1,
            tp_size=2,
            policy="round_robin",
            worker_args=[
                "--context-length=1000",
            ],
            router_args=[
                "--history-backend",
                "memory",
                "--tool-call-parser",
                "llama",
            ],
        )

        cls.base_url = cls.cluster["base_url"]

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.cluster["router"].pid)
        for worker in cls.cluster.get("workers", []):
            kill_process_tree(worker.pid)

    def test_previous_response_id_chaining(self):
        super().test_previous_response_id_chaining()

    @unittest.skip("TODO: return 501 Not Implemented")
    def test_conversation_with_multiple_turns(self):
        super().test_conversation_with_multiple_turns()

    @unittest.skip("TODO: decode error message")
    def test_mutually_exclusive_parameters(self):
        super().test_mutually_exclusive_parameters()

    def test_mcp_basic_tool_call_streaming(self):
        return super().test_mcp_basic_tool_call_streaming()

    def test_structured_output_json_schema(self):
        """Override with simpler schema for Llama model (complex schemas not well supported)."""
        data = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": "You are a math solver. Return ONLY a JSON object that matches the schema—no extra text.",
                },
                {
                    "role": "user",
                    "content": "What is 1 + 1?",
                },
            ],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "math_answer",
                    "schema": {
                        "type": "object",
                        "properties": {"answer": {"type": "string"}},
                        "required": ["answer"],
                    },
                }
            },
        }

        create_resp = self.make_request("/v1/responses", "POST", data)
        self.assertEqual(create_resp.status_code, 200)

        create_data = create_resp.json()
        self.assertIn("id", create_data)
        self.assertIn("output", create_data)
        self.assertIn("text", create_data)

        # Verify text format was echoed back correctly
        self.assertIn("format", create_data["text"])
        self.assertEqual(create_data["text"]["format"]["type"], "json_schema")
        self.assertEqual(create_data["text"]["format"]["name"], "math_answer")
        self.assertIn("schema", create_data["text"]["format"])

        # Find the message output
        output_text = next(
            (
                content.get("text", "")
                for item in create_data.get("output", [])
                if item.get("type") == "message"
                for content in item.get("content", [])
                if content.get("type") == "output_text"
            ),
            None,
        )

        self.assertIsNotNone(output_text, "No output_text found in response")
        self.assertTrue(output_text.strip(), "output_text is empty")

        # Parse JSON output
        output_json = json.loads(output_text)

        # Verify simple schema structure (just answer field)
        self.assertIn("answer", output_json)
        self.assertIsInstance(output_json["answer"], str)
        self.assertTrue(output_json["answer"], "Answer is empty")

    @unittest.skip("TODO: Temporary skip since deepwiki might hit rate limit")
    def test_mcp_basic_tool_call(self):
        return super().test_mcp_basic_tool_call()

    @unittest.skip("Temporary skip since deepwiki might hit rate limit")
    def test_mcp_basic_tool_call_streaming(self):
        return super().test_mcp_basic_tool_call_streaming()


class TestGrpcHarmonyBackend(
    StateManagementTests,
    MCPTests,
    FunctionCallingBaseTest,
    StreamingEventsTests,
    HarmonyStreamingEventsTests,
    StructuredOutputBaseTest,
    ToolChoiceTests,
):
    """End to end tests for Harmony backend.

    Note: Tool choice tests require --reasoning-parser gpt-oss in the worker.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = "/home/ubuntu/models/openai/gpt-oss-20b"
        cls.base_url_port = "http://127.0.0.1:30030"

        cls.cluster = popen_launch_workers_and_router(
            cls.model,
            cls.base_url_port,
            timeout=90,
            num_workers=1,
            tp_size=2,
            policy="round_robin",
            router_args=["--history-backend", "memory"],
            worker_args=["--reasoning-parser", "gpt-oss"],
        )

        cls.base_url = cls.cluster["base_url"]

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.cluster["router"].pid)
        for worker in cls.cluster.get("workers", []):
            kill_process_tree(worker.pid)

    def test_previous_response_id_chaining(self):
        super().test_previous_response_id_chaining()

    @unittest.skip(
        "TODO: fix requests.exceptions.JSONDecodeError: Expecting value: line 1 column 1 (char 0)"
    )
    def test_mutually_exclusive_parameters(self):
        super().test_mutually_exclusive_parameters()

    @unittest.skip("TODO: 501 Not Implemented")
    def test_conversation_with_multiple_turns(self):
        super().test_conversation_with_multiple_turns()

    # Inherited from MCPTests:
    # - test_mcp_basic_tool_call
    # - test_mcp_basic_tool_call_streaming
    # - test_mixed_mcp_and_function_tools (requires external MCP server)
    # - test_mixed_mcp_and_function_tools_streaming (requires external MCP server)

    # Inherited from StreamingEventsTests:
    # - test_output_index_zero_based
    # - test_output_item_done_event_emitted
    # - test_output_array_in_completed_event

    # Inherited from HarmonyStreamingEventsTests:
    # - test_reasoning_content_output_index
    # - test_reasoning_content_in_output_array

    # Inherited from ToolChoiceTests:
    # - test_tool_choice_auto
    # - test_tool_choice_required
    # - test_tool_choice_specific_function
    # - test_tool_choice_streaming
    # - test_tool_choice_with_mcp_tools
    # - test_tool_choice_mixed_function_and_mcp


if __name__ == "__main__":
    unittest.main()
