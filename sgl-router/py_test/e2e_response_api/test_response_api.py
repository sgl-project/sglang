"""
OpenAI backend tests for Response API.

Run with:
    export OPENAI_API_KEY=your_key
    python3 -m pytest py_test/e2e_response_api/test_openai_backend.py -v
    python3 -m unittest e2e_response_api.test_openai_backend.TestOpenAIStateManagement
"""

import os
import sys
import unittest
from pathlib import Path

# Add current directory for imports
_TEST_DIR = Path(__file__).parent
sys.path.insert(0, str(_TEST_DIR))

# Import local modules
from base import ConversationCRUDBaseTest, ResponseCRUDBaseTest
from mcp import MCPTests
from router_fixtures import (
    popen_launch_openai_xai_router,
    popen_launch_workers_and_router,
)
from state_management import StateManagementTests
from util import kill_process_tree


class TestOpenaiBackend(
    ResponseCRUDBaseTest, ConversationCRUDBaseTest, StateManagementTests, MCPTests
):
    """End to end tests for OpenAI backend."""

    api_key = os.environ.get("OPENAI_API_KEY")

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


if __name__ == "__main__":
    unittest.main()
