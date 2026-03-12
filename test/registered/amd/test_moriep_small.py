import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_amd_ci(est_time=1200, suite="stage-c-test-large-8-gpu-amd")


common_args = [
    "--tp-size",
    "8",
    "--ep-size",
    "8",
    "--dp-size",
    "8",
    "--enable-dp-attention",
    "--moe-a2a-backend",
    "mori",
    "--trust-remote-code",
    "--load-balance-method",
    "round_robin",
    "--moe-dense-tp-size",
    "1",
    "--enable-dp-lm-head",
    "--mem-fraction-static",
    "0.6",
    "--chunked-prefill-size",
    "131072",
    "--max-running-requests",
    "128",
    "--context-length",
    "12288",
    "--attention-backend",
    "aiter",
    "--cuda-graph-max-bs",
    "32",
]

mtp_args = [
    "--speculative-algo",
    "EAGLE",
    "--speculative-num-steps",
    "3",
    "--speculative-eagle-topk",
    "1",
    "--speculative-num-draft-tokens",
    "4",
]


class TestPureDP(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = common_args

        env = dict(os.environ)
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_MORI_FP8_DISP"] = "False"
        env["SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "16384"
        env["MORI_SHMEM_MODE"] = "ISOLATION"  # avoid out of symmetric heap memory

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(
        self,
    ):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.935)


class TestMTP(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = common_args + mtp_args

        env = dict(os.environ)
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_MORI_FP8_DISP"] = "False"
        env["SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "16384"
        env["MORI_SHMEM_MODE"] = "ISOLATION"  # avoid out of symmetric heap memory

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(
        self,
    ):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")
        self.assertGreater(metrics["accuracy"], 0.935)

        server_info = requests.get(self.base_url + "/get_server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 2.8)


class TestNormal(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = common_args + [
            "--deepep-mode",
            "normal",
        ]

        env = dict(os.environ)
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_MORI_FP8_DISP"] = "False"
        env["SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "16384"
        env["MORI_SHMEM_MODE"] = "ISOLATION"  # avoid out of symmetric heap memory

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(
        self,
    ):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.935)


class TestLowLatency(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = common_args + [
            "--deepep-mode",
            "low_latency",
        ]

        env = dict(os.environ)
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_MORI_FP8_DISP"] = "False"
        env["SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "16384"
        env["MORI_SHMEM_MODE"] = "ISOLATION"  # avoid out of symmetric heap memory
        env["MORI_DISABLE_P2P"] = "1"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(
        self,
    ):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.935)


class TestTBOwithLowLatency(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = common_args + [
            "--deepep-mode",
            "low_latency",
            "--enable-two-batch-overlap",
        ]

        env = dict(os.environ)
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_MORI_FP8_DISP"] = "False"
        env["SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "16384"
        env["MORI_SHMEM_MODE"] = "ISOLATION"  # avoid out of symmetric heap memory
        env["MORI_DISABLE_P2P"] = "1"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(
        self,
    ):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.935)


class TestMTPwithTBO(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = (
            common_args
            + mtp_args
            + [
                "--deepep-mode",
                "low_latency",
                "--enable-two-batch-overlap",
            ]
        )

        env = dict(os.environ)
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_MORI_FP8_DISP"] = "False"
        env["SGLANG_MORI_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "16384"
        env["MORI_SHMEM_MODE"] = "ISOLATION"  # avoid out of symmetric heap memory
        env["MORI_DISABLE_P2P"] = "1"

        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 5,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_gsm8k(
        self,
    ):
        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            max_new_tokens=512,
            parallel=128,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        self.assertGreater(metrics["accuracy"], 0.935)


if __name__ == "__main__":
    unittest.main()
