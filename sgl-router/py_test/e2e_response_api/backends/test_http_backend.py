"""
HTTP backend tests for Response API (OpenAI and XAI).

Run with:
    export OPENAI_API_KEY=your_key
    export XAI_API_KEY=your_key
    python3 -m pytest py_test/e2e_response_api/backends/test_http_backend.py -v
    python3 -m unittest e2e_response_api.backends.test_http_backend.TestOpenaiBackend
"""

import os
import sys
import unittest
from pathlib import Path

# Add e2e_response_api directory for imports
_TEST_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_TEST_DIR))

# Import local modules
from mixins.basic_crud import ConversationCRUDBaseTest, ResponseCRUDBaseTest
from mixins.function_call import FunctionCallingBaseTest
from mixins.mcp import MCPTests
from mixins.state_management import StateManagementTests
from mixins.structured_output import StructuredOutputBaseTest
from router_fixtures import popen_launch_openai_xai_router
from util import kill_process_tree


class TestOpenaiBackend(
    ResponseCRUDBaseTest,
    ConversationCRUDBaseTest,
    StateManagementTests,
    MCPTests,
    FunctionCallingBaseTest,
    StructuredOutputBaseTest,
):
    """End to end tests for OpenAI backend."""

    api_key = os.environ.get("OPENAI_API_KEY")
    mcp_validation_mode = "strict"  # Enable strict validation for HTTP backend

    @classmethod
    def setUpClass(cls):
        cls.model = "gpt-5-nano"
        cls.base_url_port = "http://127.0.0.1:30010"

        cls.cluster = popen_launch_openai_xai_router(
            backend="openai",
            base_url=cls.base_url_port,
            history_backend="memory",
        )

        cls.base_url = cls.cluster["base_url"]

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.cluster["router"].pid)

    # Inherited from MCPTests:
    # - test_mcp_basic_tool_call (with strict validation)
    # - test_mcp_basic_tool_call_streaming (with strict validation)
    # - test_mixed_mcp_and_function_tools (requires external MCP server)
    # - test_mixed_mcp_and_function_tools_streaming (requires external MCP server)

    @unittest.skip(
        "Requires external MCP server (deepwiki) - may not be accessible in CI"
    )
    def test_mixed_mcp_and_function_tools(self):
        super().test_mixed_mcp_and_function_tools()

    @unittest.skip(
        "Requires external MCP server (deepwiki) - may not be accessible in CI"
    )
    def test_mixed_mcp_and_function_tools_streaming(self):
        super().test_mixed_mcp_and_function_tools_streaming()


class TestXaiBackend(StateManagementTests):
    """End to end tests for XAI backend."""

    api_key = os.environ.get("XAI_API_KEY")

    @classmethod
    def setUpClass(cls):
        cls.model = "grok-4-fast"
        cls.base_url_port = "http://127.0.0.1:30023"

        cls.cluster = popen_launch_openai_xai_router(
            backend="xai",
            base_url=cls.base_url_port,
            history_backend="memory",
        )

        cls.base_url = cls.cluster["base_url"]

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.cluster["router"].pid)


if __name__ == "__main__":
    unittest.main()
