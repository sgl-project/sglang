"""MI35x Qwen3-Coder-Next Functionality Test (8-GPU)

Tests Qwen3-Coder-Next model with basic configuration
on MI35x. Covers GSM8K accuracy and BS=1 decode speed.

Server args match run_qwen3-coder-next_spec.sh.

Registry: stage-c-test-large-8-gpu-amd-mi35x-qwen3-coder-next suite
"""

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
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_amd_ci(est_time=3600, suite="stage-c-test-large-8-gpu-amd-mi35x")

QWEN3_CODER_NEXT_MODEL_PATH = "Qwen/Qwen3-Coder-Next"
SERVER_LAUNCH_TIMEOUT = 1800


class TestQwen3CoderNext(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_CODER_NEXT_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
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
            "--trust-remote-code",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_a_gsm8k(self):
        """GSM8K few-shot accuracy (runs first to warm up server)."""
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            num_shots=5,
            data_path=None,
            num_questions=200,
            parallel=128,
            max_new_tokens=512,
            host="http://127.0.0.1",
            port=int(self.base_url.split(":")[-1]),
        )
        metrics = run_eval_few_shot_gsm8k(args)
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (qwen3-coder-next)\n" f'{metrics["accuracy"]=:.3f}\n'
            )
            self.assertGreater(metrics["accuracy"], 0.90)

    def test_bs_1_speed(self):
        """Batch-size 1 decode speed."""
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        _, speed = send_one_prompt(args)

        print(f"{speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (qwen3-coder-next)\n" f"{speed=:.2f} token/s\n"
            )
            # self.assertGreater(speed, 50)


@unittest.skip("MTP perf not ready yet — Triton extend_attention fp8 kv cache TODO")
class TestQwen3CoderNextMTP(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_CODER_NEXT_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        # TODO: Support MTP with fp8 kv cache on gfx950.
        # Note: no --kv-cache-dtype fp8_e4m3 because Triton extend_attention
        # used by MTP does not support fp8 kv cache on gfx950.
        other_args = [
            "--tp",
            "8",
            "--attention-backend",
            "aiter",
            "--chunked-prefill-size",
            "131072",
            "--disable-radix-cache",
            "--mem-fraction-static",
            "0.8",
            "--trust-remote-code",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=SERVER_LAUNCH_TIMEOUT,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _test_a_gsm8k(self):
        """GSM8K few-shot accuracy with MTP (runs first to warm up server)."""
        requests.get(self.base_url + "/flush_cache")

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

        server_info = requests.get(self.base_url + "/get_server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (qwen3-coder-next mtp)\n"
                f'{metrics["accuracy"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )
            self.assertGreater(metrics["accuracy"], 0.90)
            self.assertGreater(avg_spec_accept_length, 2.0)

    def _test_bs_1_speed(self):
        """Batch-size 1 decode speed with MTP."""
        args = BenchArgs(port=int(self.base_url.split(":")[-1]), max_new_tokens=2048)
        acc_length, speed = send_one_prompt(args)

        print(f"{acc_length=:.2f} {speed=:.2f}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_bs_1_speed (qwen3-coder-next mtp)\n"
                f"{acc_length=:.2f}\n"
                f"{speed=:.2f} token/s\n"
            )
            # self.assertGreater(acc_length, 2.0)
            # self.assertGreater(speed, 100)


if __name__ == "__main__":
    import unittest

    unittest.main()
