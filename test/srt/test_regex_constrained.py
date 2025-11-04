"""
python3 -m unittest test_regex_constrained.TestRegexConstrained.test_regex_generate_email
python3 -m unittest test_regex_constrained.TestRegexConstrained.test_regex_generate_greeting
python3 -m unittest test_regex_constrained.TestRegexConstrainedLLGuidance.test_regex_generate_email
python3 -m unittest test_regex_constrained.TestRegexConstrainedLLGuidance.test_regex_generate_greeting
"""

import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.regex_constrained_kit import TestRegexConstrainedMixin
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestRegexConstrained(CustomTestCase, TestRegexConstrainedMixin):
    backend = "xgrammar"
    disable_overlap = False

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST

        other_args = [
            "--max-running-requests",
            "10",
            "--grammar-backend",
            cls.backend,
        ]

        if cls.disable_overlap:
            other_args += ["--disable-overlap-schedule"]

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


class TestRegexConstrainedLLGuidance(TestRegexConstrained):
    backend = "llguidance"
    disable_overlap = True


if __name__ == "__main__":
    unittest.main()
