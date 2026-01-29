import os
import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.test_utils import (
    DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST,
    DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST_NEXTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)


class TestPureDP(CustomTestCase):

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
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
        ]

        env = dict(os.environ)
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_MORI_FP8_DISP"] = "True"
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
        other_args = [
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
            "--speculative-algo",
            "EAGLE",
            "--speculative-draft-model-path",
            DEFAULT_DEEPEP_MODEL_NAME_FOR_TEST_NEXTN,
            "--speculative-num-steps",
            "1",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "2",
            "--cuda-graph-max-bs",
            "32",
        ]

        env = dict(os.environ)
        env["SGLANG_USE_AITER"] = "1"
        env["SGLANG_MORI_FP8_DISP"] = "True"
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


if __name__ == "__main__":
    unittest.main()
