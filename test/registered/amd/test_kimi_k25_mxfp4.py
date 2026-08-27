"""Kimi-K2.5-MXFP4 aiter MLA backend test (8-GPU, FP8 KV cache)

PR-level test for Kimi-K2.5-MXFP4 with aiter unified attention backend
and fp8_e4m3 KV cache on MI35x.

"""

import os
import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.few_shot_gsm8k import run_eval as run_eval_few_shot_gsm8k
from sglang.test.send_one import BenchArgs, send_one_prompt
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(est_time=3600, suite="stage-c-test-large-8-gpu-amd-mi35x")

KIMI_K25_MXFP4_MODEL_PATH = "amd/Kimi-K2.5-MXFP4"
KIMI_K25_MXFP4_REVISION = "b071bc6f8eb042e093e14f3b8bdbad71c18e09d3"
SERVER_LAUNCH_TIMEOUT = 3600


class TestKimiK25MXFP4(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = KIMI_K25_MXFP4_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--revision",
            KIMI_K25_MXFP4_REVISION,
            "--tp",
            "8",
            "--attention-backend",
            "aiter",
            "--kv-cache-dtype",
            "fp8_e4m3",
            "--chunked-prefill-size",
            "131072",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.8",
            "--max-running-requests",
            "64",
            "--trust-remote-code",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true}',
        ]
        env = os.environ.copy()
        env["SGLANG_AITER_MLA_PERSIST"] = "1"
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=8,
            data_path=None,
            num_questions=1319,
            parallel=1319,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (Kimi-K2.5-MXFP4)\n" f'{metrics["accuracy"]=:.3f}\n'
            )
            self.assertGreater(metrics["accuracy"], 0.92)

    def test_bs_1_speed(self):
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        _, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (Kimi-K2.5-MXFP4)\n" f"{speed=:.2f} token/s\n"
            )
            if is_in_amd_ci():
                self.assertGreater(speed, 30)
            else:
                self.assertGreater(speed, 45)


if __name__ == "__main__":
    unittest.main()
