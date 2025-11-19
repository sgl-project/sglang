"""
Oracle database storage backend tests for Response API.

Run with:
    export OPENAI_API_KEY=your_key
    python3 -m pytest py_test/e2e_response_api/persistence/test_oracle_store.py -v
    python3 -m unittest e2e_response_api.persistence.test_oracle_store.TestOracleStore
"""

import os
import sys
import unittest
from pathlib import Path

import openai

# Add e2e_response_api directory for imports
_TEST_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(_TEST_DIR))

# Import local modules
from mixins.basic_crud import ConversationCRUDBaseTest, ResponseCRUDBaseTest
from router_fixtures import popen_launch_openai_xai_router
from util import kill_process_tree


class TestOracleStore(ResponseCRUDBaseTest, ConversationCRUDBaseTest):
    """End to end tests for Oracle database storage backend."""

    api_key = os.environ.get("OPENAI_API_KEY")

    @classmethod
    def setUpClass(cls):
        cls.model = "gpt-5-nano"
        cls.base_url_port = "http://127.0.0.1:30040"

        cls.cluster = popen_launch_openai_xai_router(
            backend="openai",
            base_url=cls.base_url_port,
            history_backend="oracle",
        )

        cls.base_url = cls.cluster["base_url"]
        cls.client = openai.Client(api_key=cls.api_key, base_url=cls.base_url + "/v1")

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.cluster["router"].pid)


if __name__ == "__main__":
    unittest.main()
