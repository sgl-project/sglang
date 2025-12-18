import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.ebnf_constrained_kit import TestEBNFConstrainedMinxin
from sglang.test.kits.json_constrained_kit import TestJSONConstrainedMixin
from sglang.test.kits.regex_constrained_kit import TestRegexConstrainedMixin
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class ServerWithGrammar(CustomTestCase):
    backend = "xgrammar"
    disable_overlap = False

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
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


class TestXGrammarBackend(
    ServerWithGrammar,
    TestJSONConstrainedMixin,
    TestEBNFConstrainedMinxin,
    TestRegexConstrainedMixin,
):
    backend = "xgrammar"


class TestOutlinesBackend(ServerWithGrammar, TestJSONConstrainedMixin):
    backend = "outlines"


class TestLLGuidanceBackend(
    ServerWithGrammar,
    TestJSONConstrainedMixin,
    TestEBNFConstrainedMinxin,
    TestRegexConstrainedMixin,
):
    backend = "llguidance"


if __name__ == "__main__":
    unittest.main()
