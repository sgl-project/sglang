"""
gRPC backend tests for Response API (including Harmony).

Run with:
    python3 -m pytest py_test/e2e_response_api/backends/test_grpc_backend.py -v
    python3 -m unittest e2e_response_api.backends.test_grpc_backend.TestGrpcBackend
"""

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
from router_fixtures import popen_launch_workers_and_router
from util import kill_process_tree


class TestGrpcBackend(StateManagementTests, MCPTests):
    """End to end tests for gRPC backend."""

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
            router_args=["--history-backend", "memory"],
        )

        cls.base_url = cls.cluster["base_url"]

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.cluster["router"].pid)
        for worker in cls.cluster.get("workers", []):
            kill_process_tree(worker.pid)

    @unittest.skip(
        "TODO: transport error, details: [], metadata: MetadataMap { headers: {} }"
    )
    def test_previous_response_id_chaining(self):
        super().test_previous_response_id_chaining()

    @unittest.skip("TODO: return 501 Not Implemented")
    def test_conversation_with_multiple_turns(self):
        super().test_conversation_with_multiple_turns()

    @unittest.skip("TODO: decode error message")
    def test_mutually_exclusive_parameters(self):
        super().test_mutually_exclusive_parameters()

    @unittest.skip(
        "TODO: Pipeline execution failed: Pipeline stage WorkerSelection failed"
    )
    def test_mcp_basic_tool_call(self):
        super().test_mcp_basic_tool_call()

    @unittest.skip("TODO: no event fields")
    def test_mcp_basic_tool_call_streaming(self):
        return super().test_mcp_basic_tool_call_streaming()


class TestHarmonyBackend(StateManagementTests, MCPTests, FunctionCallingBaseTest):
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
            router_args=["--history-backend", "memory"],
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

    def test_mcp_basic_tool_call(self):
        """Test basic MCP tool call (non-streaming)."""
        tools = [
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "require_approval": "never",
            }
        ]

        resp = self.create_response(
            "What transport protocols does the 2025-03-26 version of the MCP spec (modelcontextprotocol/modelcontextprotocol) support?",
            tools=tools,
            stream=False,
        )

        # Should successfully make the request
        self.assertEqual(resp.status_code, 200)

        data = resp.json()

        # Basic response structure
        self.assertIn("id", data)
        self.assertIn("status", data)
        self.assertEqual(data["status"], "completed")
        self.assertIn("output", data)
        self.assertIn("model", data)

        # Verify output array is not empty
        output = data["output"]
        self.assertIsInstance(output, list)
        self.assertGreater(len(output), 0)

        # Check for MCP-specific output types
        output_types = [item.get("type") for item in output]

        # Should have mcp_list_tools - tools are listed before calling
        self.assertIn(
            "mcp_list_tools", output_types, "Response should contain mcp_list_tools"
        )

        # Should have at least one mcp_call
        mcp_calls = [item for item in output if item.get("type") == "mcp_call"]
        self.assertGreater(
            len(mcp_calls), 0, "Response should contain at least one mcp_call"
        )

        # Verify mcp_call structure
        for mcp_call in mcp_calls:
            self.assertIn("id", mcp_call)
            self.assertIn("status", mcp_call)
            self.assertEqual(mcp_call["status"], "completed")
            self.assertIn("server_label", mcp_call)
            self.assertEqual(mcp_call["server_label"], "deepwiki")
            self.assertIn("name", mcp_call)
            self.assertIn("arguments", mcp_call)
            self.assertIn("output", mcp_call)

    def test_mcp_basic_tool_call_streaming(self):
        """Test basic MCP tool call (streaming)."""
        tools = [
            {
                "type": "mcp",
                "server_label": "deepwiki",
                "server_url": "https://mcp.deepwiki.com/mcp",
                "require_approval": "never",
            }
        ]

        resp = self.create_response(
            "What transport protocols does the 2025-03-26 version of the MCP spec (modelcontextprotocol/modelcontextprotocol) support?",
            tools=tools,
            stream=True,
        )

        # Should successfully make the request
        self.assertEqual(resp.status_code, 200)

        events = self.parse_sse_events(resp)
        self.assertGreater(len(events), 0)

        event_types = [e.get("event") for e in events]

        # Check for lifecycle events
        self.assertIn(
            "response.created", event_types, "Should have response.created event"
        )
        self.assertIn(
            "response.completed", event_types, "Should have response.completed event"
        )

        # Check for MCP list tools events
        self.assertIn(
            "response.output_item.added",
            event_types,
            "Should have output_item.added events",
        )
        self.assertIn(
            "response.mcp_list_tools.in_progress",
            event_types,
            "Should have mcp_list_tools.in_progress event",
        )
        self.assertIn(
            "response.mcp_list_tools.completed",
            event_types,
            "Should have mcp_list_tools.completed event",
        )

        # Check for MCP call events
        self.assertIn(
            "response.mcp_call.in_progress",
            event_types,
            "Should have mcp_call.in_progress event",
        )
        self.assertIn(
            "response.mcp_call_arguments.delta",
            event_types,
            "Should have mcp_call_arguments.delta event",
        )
        self.assertIn(
            "response.mcp_call_arguments.done",
            event_types,
            "Should have mcp_call_arguments.done event",
        )
        self.assertIn(
            "response.mcp_call.completed",
            event_types,
            "Should have mcp_call.completed event",
        )

        # Verify final completed event has full response
        completed_events = [e for e in events if e.get("event") == "response.completed"]
        self.assertEqual(len(completed_events), 1)

        final_response = completed_events[0].get("data", {}).get("response", {})
        self.assertIn("id", final_response)
        self.assertEqual(final_response.get("status"), "completed")
        self.assertIn("output", final_response)

        # Verify final output contains expected items
        final_output = final_response.get("output", [])
        final_output_types = [item.get("type") for item in final_output]

        self.assertIn("mcp_list_tools", final_output_types)
        self.assertIn("mcp_call", final_output_types)

        # Verify mcp_call items in final output
        mcp_calls = [item for item in final_output if item.get("type") == "mcp_call"]
        self.assertGreater(len(mcp_calls), 0)

        for mcp_call in mcp_calls:
            self.assertEqual(mcp_call.get("status"), "completed")
            self.assertEqual(mcp_call.get("server_label"), "deepwiki")
            self.assertIn("name", mcp_call)
            self.assertIn("arguments", mcp_call)
            self.assertIn("output", mcp_call)

    @unittest.skip("TODO: 501 Not Implemented")
    def test_conversation_with_multiple_turns(self):
        super().test_conversation_with_multiple_turns()


if __name__ == "__main__":
    unittest.main()
