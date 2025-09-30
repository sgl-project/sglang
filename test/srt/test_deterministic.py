"""
Usage:
cd test/srt
python3 -m unittest test_deterministic.TestDeterministic.TESTCASE

Note that there is also `python/sglang/test/test_deterministic.py` as an interactive test. We are converting that
test into unit tests so that's easily reproducible in CI.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.test_deterministic import BenchArgs, test_deterministic
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestDeterministicFlashinfer(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "Qwen/Qwen3-8B"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            device="auto",
            other_args=[
                "--enable-deterministic",
                "--disable-radix-cache",  # this should be by default now but still set it for safety
                "--attention-backend",
                "flashinfer",
            ],
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
        args.n_trials = 10
        results = test_deterministic(args)
        assert all(lambda x: x == 1, results)

    def test_mixed(self):
        args = BenchArgs()
        url = DEFAULT_URL_FOR_TEST
        args.host, args.port = self._extract_host_and_port(url)
        args.test_mode = "mixed"
        args.n_start = 10
        args.n_trials = 10
        results = test_deterministic(args)
        assert all(lambda x: x == 1, results)

    def test_prefix(self):
        args = BenchArgs()
        url = DEFAULT_URL_FOR_TEST
        args.host, args.port = self._extract_host_and_port(url)
        args.test_mode = "prefix"
        args.n_start = 10
        args.n_trials = 10
        results = test_deterministic(args)
        assert all(lambda x: x == 1, results)


if __name__ == "__main__":
    unittest.main()
