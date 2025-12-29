import unittest
from types import SimpleNamespace

import requests

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.server_fixtures.eagle_fixture import EagleServerBase
from sglang.test.test_utils import (
    DEFAULT_DRAFT_MODEL_EAGLE3,
    DEFAULT_TARGET_MODEL_EAGLE3,
)

register_cuda_ci(est_time=50, suite="stage-b-test-small-1-gpu")


class TestEagle3Basic(EagleServerBase):
    target_model = DEFAULT_TARGET_MODEL_EAGLE3
    draft_model = DEFAULT_DRAFT_MODEL_EAGLE3

    spec_algo = "EAGLE3"
    spec_steps = 2
    spec_topk = 1
    spec_tokens = 3
    extra_args = ["--dtype=float16", "--chunked-prefill-size", 1024]

    def test_mmlu(self):
        """Override to add EAGLE-specific assertions"""
        args = SimpleNamespace(
            base_url=self.base_url,
            model=self.target_model,
            eval_name="mmlu",
            num_examples=64,
            num_threads=32,
        )

        metrics = run_eval(args)
        self.assertGreaterEqual(metrics["score"], 0.72)

        server_info = requests.get(self.base_url + "/server_info").json()
        avg_spec_accept_length = server_info["internal_states"][0][
            "avg_spec_accept_length"
        ]
        print(f"{avg_spec_accept_length=}")
        self.assertGreater(avg_spec_accept_length, 2.26)


if __name__ == "__main__":

    unittest.main()
