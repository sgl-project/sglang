import unittest
from types import SimpleNamespace

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_ci,
    popen_launch_server,
    write_github_step_summary,
)

register_cuda_ci(est_time=617, suite="stage-c-test-8-gpu-h200")

STEP3P5_FLASH_MODEL_PATH = "stepfun-ai/Step-3.5-Flash"


class TestStep3p5FlashChainMTP(CustomTestCase):
    """Chain-style multi-layer EAGLE speculative decoding on Step-3.5-Flash.

    Step3p5ForCausalLM auto-enables multi-layer EAGLE and spec v2 when
    --speculative-algorithm=EAGLE is set.  The chain MTP propagation
    (each MTP layer feeds its hidden states to the next) is activated
    automatically for the Step3p5MTP draft architecture.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = STEP3P5_FLASH_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--tp",
            "8",
            "--trust-remote-code",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            "--attention-backend",
            "fa3",
            "--enable-multi-layer-eagle",
            "--mem-fraction-static",
            "0.75",
            "--chunked-prefill-size",
            "4096",
            "--model-loader-extra-config",
            '{"enable_multithread_load": true, "num_threads": 64}',
        ]
        with envs.SGLANG_ENABLE_SPEC_V2.override(True):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH * 3,
                other_args=other_args,
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
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

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        print(f"{metrics=}")

        if is_in_ci():
            write_github_step_summary(
                f"### test_gsm8k (step-3.5-flash chain mtp)\n"
                f'{metrics["score"]=:.3f}\n'
                f"{avg_spec_accept_length=:.2f}\n"
            )
            self.assertGreater(metrics["score"], 0.84)
            self.assertGreater(avg_spec_accept_length, 2.6)


if __name__ == "__main__":
    unittest.main()
