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
            router_args=["--history-backend", "memory", "--tool-call-parser", "llama"],
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

    # Inherited from MCPTests:
    # - test_mcp_basic_tool_call
    # - test_mcp_basic_tool_call_streaming
    # - test_mixed_mcp_and_function_tools (requires external MCP server)
    # - test_mixed_mcp_and_function_tools_streaming (requires external MCP server)


class TestGrpcHarmonyBackend(StateManagementTests, MCPTests, FunctionCallingBaseTest):
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
