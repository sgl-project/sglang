import os
import unittest

from sglang.srt.utils import kill_process_tree
from sglang.test.test_deterministic_tp import BenchArgs, test_deterministic
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
    "--rl-on-policy-target",
    "fsdp_tp",
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

        base_ip = DEFAULT_URL_FOR_TEST.split("://")[-1].split(":")[0]
        base_port = int(DEFAULT_URL_FOR_TEST.split(":")[-1])

        cls.port_a = base_port
        cls.port_b = base_port + 1
        cls.url_a = f"http://{base_ip}:{cls.port_a}"
        cls.url_b = f"http://{base_ip}:{cls.port_b}"

        if "--attention-backend" not in cls.get_server_args():
            raise unittest.SkipTest("Skip the base test class")

        base_args = list(cls.get_server_args())

        env_a = os.environ.copy()
        env_a["CUDA_VISIBLE_DEVICES"] = "0"
        args_a = base_args + ["--tensor-parallel-size", "1"]

        cls.process_a = popen_launch_server(
            cls.model,
            cls.url_a,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args_a,
            env=env_a,
        )

        env_b = os.environ.copy()
        env_b["CUDA_VISIBLE_DEVICES"] = "1,2"
        args_b = base_args + ["--tensor-parallel-size", "2"]

        cls.process_b = popen_launch_server(
            cls.model,
            cls.url_b,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=args_b,
            env=env_b,
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process_a"):
            kill_process_tree(cls.process_a.pid)
        if hasattr(cls, "process_b"):
            kill_process_tree(cls.process_b.pid)

    def _extract_host_and_port(self, url):
        return url.split("://")[-1].split(":")[0], int(url.split(":")[-1])

    def test_prefix_consistency_between_hosts(self):
        args = BenchArgs()

        args.host_a, args.port_a = self._extract_host_and_port(self.url_a)
        args.host_b, args.port_b = self._extract_host_and_port(self.url_b)

        args.test_mode = "prefix"
        args.n_start = 10
        args.n_trials = 10
        args.temperature = 0.5
        args.return_logprob = True

        results = test_deterministic(args)
        for result in results:
            assert result == 1, "Host A and Host B produced different prefix results"

    def test_radix_cache_consistency_between_hosts(self):
        args = BenchArgs()

        args.host_a, args.port_a = self._extract_host_and_port(self.url_a)
        args.host_b, args.port_b = self._extract_host_and_port(self.url_b)

        args.test_mode = "radix_cache"
        args.n_start = 10
        args.n_trials = 10
        args.temperature = 0.5
        args.return_logprob = True

        results = test_deterministic(args)
        for result in results:
            assert (
                result == 1
            ), "Host A and Host B produced different radix cache behaviors"
