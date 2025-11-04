"""
python3 -m unittest test_ebnf_constrained.TestEBNFConstrained.test_ebnf_generate_email
python3 -m unittest test_ebnf_constrained.TestEBNFConstrained.test_ebnf_generate_greeting
python3 -m unittest test_ebnf_constrained.TestEBNFConstrained.test_ebnf_generate_all_optional_function_params
python3 -m unittest test_ebnf_constrained.TestEBNFConstrainedLLGuidance.test_ebnf_generate_email
python3 -m unittest test_ebnf_constrained.TestEBNFConstrainedLLGuidance.test_ebnf_generate_greeting
python3 -m unittest test_ebnf_constrained.TestEBNFConstrainedLLGuidance.test_ebnf_generate_all_optional_function_params
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.ebnf_constrained_kit import TestEBNFConstrainedMinxin
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestEBNFConstrained(CustomTestCase, TestEBNFConstrainedMinxin):
    backend = "xgrammar"
    disable_overlap = False

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.ebnf_grammar = 'root ::= "test"'  # Default grammar

        launch_args = [
            "--max-running-requests",
            "10",
            "--grammar-backend",
            cls.backend,
        ]

        if cls.disable_overlap:
            launch_args += ["--disable-overlap-schedule"]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=launch_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestEBNFConstrainedLLGuidance(TestEBNFConstrained):
    backend = "llguidance"
    disable_overlap = False


if __name__ == "__main__":
    unittest.main()
