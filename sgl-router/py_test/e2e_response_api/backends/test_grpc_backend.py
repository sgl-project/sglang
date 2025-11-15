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

import openai

# Add e2e_response_api directory for imports
_TEST_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_TEST_DIR))

# Import local modules
from mixins.function_call import FunctionCallingBaseTest
from mixins.mcp import MCPTests
from mixins.state_management import StateManagementTests
from mixins.structured_output import StructuredOutputBaseTest
from router_fixtures import popen_launch_workers_and_router
from util import kill_process_tree


class TestGrpcBackend(StateManagementTests, MCPTests, StructuredOutputBaseTest):
    """End to end tests for gRPC backend (Regular backend with Qwen2.5)."""

    @classmethod
    def setUpClass(cls):
        cls.model = "/home/ubuntu/models/Qwen/Qwen2.5-14B-Instruct"
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
                "qwen",
            ],
        )

        cls.base_url = cls.cluster["base_url"]
        cls.client = openai.Client(api_key=cls.api_key, base_url=cls.base_url + "/v1")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.cluster["router"].pid)
        for worker in cls.cluster.get("workers", []):
            kill_process_tree(worker.pid)

    @unittest.skip("TODO: return 501 Not Implemented")
    def test_conversation_with_multiple_turns(self):
        super().test_conversation_with_multiple_turns()

    def test_structured_output_json_schema(self):
        """Override with simpler schema for Llama model (complex schemas not well supported)."""
        params = {
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

        create_resp = self.create_response(**params)
        self.assertIsNone(create_resp.error)
        self.assertIsNotNone(create_resp.id)
        self.assertIsNotNone(create_resp.output)
        self.assertIsNotNone(create_resp.text)

        # Verify text format was echoed back correctly
        self.assertIsNotNone(create_resp.text.format)
        self.assertEqual(create_resp.text.format.type, "json_schema")
        self.assertEqual(create_resp.text.format.name, "math_answer")
        self.assertIsNotNone(create_resp.text.format.schema_)

        # Find the message output
        output_text = next(
            (
                content.text
                for item in create_resp.output
                if item.type == "message"
                for content in item.content
                if content.type == "output_text"
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


class TestGrpcHarmonyBackend(
    StateManagementTests, MCPTests, FunctionCallingBaseTest, StructuredOutputBaseTest
):
    """End to end tests for Harmony backend."""

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
            worker_args=[
                "--reasoning-parser=gpt-oss",
            ],
            router_args=[
                "--history-backend",
                "memory",
            ],
        )

        cls.base_url = cls.cluster["base_url"]
        cls.client = openai.Client(api_key=cls.api_key, base_url=cls.base_url + "/v1")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.cluster["router"].pid)
        for worker in cls.cluster.get("workers", []):
            kill_process_tree(worker.pid)

    @unittest.skip("TODO: 501 Not Implemented")
    def test_conversation_with_multiple_turns(self):
        super().test_conversation_with_multiple_turns()

    # Inherited from MCPTests:
    # - test_mcp_basic_tool_call
    # - test_mcp_basic_tool_call_streaming
    # - test_mixed_mcp_and_function_tools (requires external MCP server)
    # - test_mixed_mcp_and_function_tools_streaming (requires external MCP server)


if __name__ == "__main__":
    unittest.main()
