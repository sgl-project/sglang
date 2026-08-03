import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_cuda_ci(est_time=79, stage="base-b", runner_config="2-gpu-large")

QWEN35_MODEL = "Qwen/Qwen3.5-9B"
SERVER_LAUNCH_TIMEOUT = 600


class TestQwen35EagleRS(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = QWEN35_MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "2",
            "--speculative-algorithm",
            "NEXTN",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--speculative-use-rejection-sampling",
            "--mem-fraction-static",
            "0.8",
            "--disable-radix-cache",
        ]
        with envs.SGLANG_ENABLE_ASYNC_ASSERT.override(True):
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
        requests.get(self.base_url + "/flush_cache")

        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="completion",
            max_tokens=512,
            num_examples=200,
            num_threads=128,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")

        server_info = requests.get(self.base_url + "/server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (qwen3.5-9b nextn rs)\n"
                f'{metrics["score"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )

        self.assertGreater(metrics["score"], 0.8)
        self.assertGreater(avg_spec_accept_length, 2.5)


if __name__ == "__main__":
    unittest.main()
