"""Archived test classes split out of test/registered/distributed/test_dp_attention.py.

Originally registered with `register_cuda_ci(...)`. Moved here as part of
the per-commit pruning effort to keep the code reachable manually.
Run with `python3 test/manual/distributed/test_dp_attention_archived.py`.
"""

import unittest
from types import SimpleNamespace

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.kits.ebnf_constrained_kit import EBNFConstrainedMixin
from sglang.test.kits.json_constrained_kit import JSONConstrainedMixin
from sglang.test.kits.regex_constrained_kit import RegexConstrainedMixin
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    is_in_amd_ci,
    popen_launch_server,
)


class TestDPAttentionDP2TP2DeepseekV3MTP(
    CustomTestCase,
    JSONConstrainedMixin,
    EBNFConstrainedMixin,
    RegexConstrainedMixin,
):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--disable-radix",
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "2",
            "--speculative-eagle-topk",
            "4",
            "--speculative-num-draft-tokens",
            "4",
            "--speculative-draft-model-path",
            DEFAULT_MODEL_NAME_FOR_TEST_MLA_NEXTN,
            "--tp-size",
            "2",
            "--enable-dp-attention",
            "--dp-size",
            "2",
        ]
        if not is_in_amd_ci():
            other_args += ["--mem-frac", "0.7"]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )

    @classmethod
    def tearDownClass(cls):
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
        print(metrics)

        self.assertGreater(metrics["score"], 0.60)

        server_info = requests.get(self.base_url + "/server_info")
        avg_spec_accept_length = server_info.json()["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(
            f"###test_gsm8k (deepseek-v3 mtp + dp):\n"
            f"accuracy={metrics['score']=:.3f}\n"
            f"{avg_spec_accept_length=:.3f}\n"
        )
        self.assertGreater(avg_spec_accept_length, 2.5)


if __name__ == "__main__":
    unittest.main()
