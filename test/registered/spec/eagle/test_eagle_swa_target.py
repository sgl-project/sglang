import unittest
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.eval_accuracy_kit import GSM8KMixin, _check_accept_length
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.default_fixture import DefaultServerBase

register_cuda_ci(est_time=1200, suite="base-b-test-1-gpu-large")


GPTOSS_TARGET = "openai/gpt-oss-20b"
DOGACEL_DRAFT = "Dogacel/specdrift-gpt-oss-20b-eagle3"


class TestEAGLE3SWATarget(GSM8KMixin, DefaultServerBase):
    """Regression for #22679: EAGLE3 on a SWA target (gpt-oss-20b).

    GSM8K confirms the SWA spec-decoding paths produce correct output, and
    the accept-length check confirms speculation is effective.
    """

    model = GPTOSS_TARGET
    other_args = [
        "--speculative-algorithm",
        "EAGLE3",
        "--speculative-draft-model-path",
        DOGACEL_DRAFT,
        "--speculative-num-steps",
        "5",
        "--speculative-eagle-topk",
        "8",
        "--speculative-num-draft-tokens",
        "16",
        "--mem-fraction-static",
        "0.7",
        "--max-running-requests",
        "32",
        "--cuda-graph-max-bs",
        "4",
        "--dtype",
        "bfloat16",
        "--attention-backend",
        "triton",
    ]

    gsm8k_accuracy_thres = 0.9
    gsm8k_accept_length_thres = 1.0

    def test_gsm8k(self):
        # gpt-oss is a reasoning model; evaluate via its native chat API
        # (the mixin's completion mode underperforms on reasoning models).
        requests.get(self.base_url + "/flush_cache")
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.model,
            eval_name="gsm8k",
            api="chat",
            max_tokens=2048,
            num_examples=self.gsm8k_num_questions,
            num_threads=32,
            num_shots=self.gsm8k_num_shots,
        )
        metrics = run_eval(args)
        print(f"{metrics=}")
        self.assertGreaterEqual(metrics["score"], self.gsm8k_accuracy_thres)
        _check_accept_length(self, self.base_url, self.gsm8k_accept_length_thres)


if __name__ == "__main__":
    unittest.main()
