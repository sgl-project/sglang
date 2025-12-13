import json
import unittest
from types import SimpleNamespace

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MLA_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestPureTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--moe-a2a-backend",
                "deepep",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)


class TestDPAttn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MLA_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENABLE_JIT_DEEPGEMM.override(False):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--tp",
                    "2",
                    "--dp",
                    "2",
                    "--enable-dp-attention",
                    "--moe-a2a-backend",
                    "deepep",
                    "--deepep-mode",
                    "normal",
                    "--disable-cuda-graph",
                    # Test custom config
                    "--deepep-config",
                    json.dumps(
                        {
                            "normal_dispatch": {
                                "num_sms": 20,
                                "num_max_nvl_chunked_send_tokens": 16,
                                "num_max_nvl_chunked_recv_tokens": 256,
                                "num_max_rdma_chunked_send_tokens": 6,
                                "num_max_rdma_chunked_recv_tokens": 128,
                            },
                            "normal_combine": {
                                "num_sms": 20,
                                "num_max_nvl_chunked_send_tokens": 6,
                                "num_max_nvl_chunked_recv_tokens": 256,
                                "num_max_rdma_chunked_send_tokens": 6,
                                "num_max_rdma_chunked_recv_tokens": 128,
                            },
                        }
                    ),
                ],
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_mmlu(self):
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreater(metrics["score"], 0.5)


if __name__ == "__main__":
    unittest.main()
