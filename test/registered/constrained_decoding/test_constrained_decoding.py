import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci, register_cuda_ci
from sglang.test.kits.ebnf_constrained_kit import EBNFConstrainedMixin
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.regex_constrained_kit import RegexConstrainedMixin
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=131, suite="stage-b-test-1-gpu-small")
register_amd_ci(est_time=179, suite="stage-b-test-1-gpu-small-amd")


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
    JSONConstrainedMixin,
    EBNFConstrainedMixin,
    RegexConstrainedMixin,
):
    backend = "xgrammar"


class TestOutlinesBackend(ServerWithGrammar, JSONConstrainedMixin):
    backend = "outlines"


class TestLLGuidanceBackend(
    ServerWithGrammar,
    JSONConstrainedMixin,
    EBNFConstrainedMixin,
    RegexConstrainedMixin,
):
    backend = "llguidance"


if __name__ == "__main__":
    unittest.main()
