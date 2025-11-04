"""
python3 -m unittest openai_server.features.test_json_constrained.TestJSONConstrainedOutlinesBackend.test_json_generate
python3 -m unittest openai_server.features.test_json_constrained.TestJSONConstrainedXGrammarBackend.test_json_generate
python3 -m unittest openai_server.features.test_json_constrained.TestJSONConstrainedLLGuidanceBackend.test_json_generate
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.json_constrained_kit import TestJSONConstrainedMixin
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestJSONConstrained(CustomTestCase, TestJSONConstrainedMixin):
    backend = "xgrammar"

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        launch_args = ["--max-running-requests", "10", "--grammar-backend", cls.backend]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestJSONConstrainedOutlinesBackend(TestJSONConstrained):
    backend = "outlines"


class TestJSONConstrainedLLGuidanceBackend(TestJSONConstrained):
    backend = "llguidance"


if __name__ == "__main__":
    unittest.main()
