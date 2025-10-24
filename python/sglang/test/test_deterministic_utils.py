import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.test_deterministic import BenchArgs, test_deterministic
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DEFAULT_MODEL = "Qwen/Qwen3-8B"
COMMON_SERVER_ARGS = [
    "--trust-remote-code",
    "--cuda-graph-max-bs",
    "32",
    "--enable-deterministic-inference",
]


class TestDeterministicBase(CustomTestCase):
    @classmethod
    def get_server_args(cls):
        return COMMON_SERVER_ARGS

    @classmethod
    def get_model(cls):
        return DEFAULT_MODEL

    @classmethod
    def setUpClass(cls):
        cls.model = cls.get_model()
        cls.base_url = DEFAULT_URL_FOR_TEST
        if "--attention-backend" not in cls.get_server_args():
            raise unittest.SkipTest("Skip the base test class")

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=cls.get_server_args(),
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _extract_host_and_port(self, url):
        return url.split("://")[-1].split(":")[0], int(url.split(":")[-1])

    def test_single(self):
        args = BenchArgs()
        url = DEFAULT_URL_FOR_TEST
        args.host, args.port = self._extract_host_and_port(url)
        args.test_mode = "single"
        args.n_start = 10
        args.n_trials = 20
        results = test_deterministic(args)
        args.temperature = 0.5  # test for deterministic sampling
        for result in results:
            assert result == 1

    def test_prefix(self):
        args = BenchArgs()
        url = DEFAULT_URL_FOR_TEST
        args.host, args.port = self._extract_host_and_port(url)
        args.test_mode = "prefix"
        args.n_start = 10
        args.n_trials = 10
        args.temperature = 0.5  # test for deterministic sampling
        results = test_deterministic(args)
        for result in results:
            assert result == 1
