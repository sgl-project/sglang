"""EPLB with redundant experts on the no-a2a MoE path (--moe-a2a-backend none).

There, all EP ranks run the MoE over the same tokens and sum their partial
outputs, so the logical->physical pick has to be identical on every rank -- a
rank-dependent one counts a replicated logical expert several times and silently
degrades output instead of failing.

`test/manual/ep/test_eplb.py` covers EPLB with an a2a backend.
"""

import unittest
from types import SimpleNamespace

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.run_eval import run_eval
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
    try_cached_model,
)

register_cuda_ci(est_time=200, stage="nightly", runner_config="2-gpu-large")

# 72 routed experts + 48 replicas = 120 physical, 60 per rank, so two thirds of
# the routed (token, expert) pairs get double-counted when ranks disagree. At 24
# replicas the score only fell to 0.575 against the 0.60 threshold.
NUM_REDUNDANT_EXPERTS = 48


class TestEPLBNoA2A(CustomTestCase):
    """Initial placement, no rebalance during the eval.

    Guards the candidate-map half of the fix: only the initial placement goes
    through `_compute_logical_to_all_physical_map`, where the rank-local collapse
    lives, so a regression there is invisible once rebalancing starts.
    """

    extra_args = []
    # Never reached by a 200-question eval. Also sizes the expert-distribution
    # recorder buffer, so it cannot be made arbitrarily large.
    rebalance_num_iterations = "20000"

    @classmethod
    def setUpClass(cls):
        cls.model = try_cached_model(DEFAULT_MODEL_NAME_FOR_TEST_MLA)
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "2",
                "--ep-size",
                "2",
                "--enable-eplb",
                "--ep-num-redundant-experts",
                str(NUM_REDUNDANT_EXPERTS),
                "--eplb-rebalance-num-iterations",
                cls.rebalance_num_iterations,
                "--mem-fraction-static",
                "0.5",
                *cls.extra_args,
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_gsm8k(self):
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

        # Measured over six runs: 0.625-0.66 correct, 0.415-0.435 with the
        # rank-dependent pick restored. 0.60 (what the other EP tests use for
        # this model) sits under a sigma of the low end, so leave room.
        self.assertGreater(metrics["score"], 0.55)


class TestEPLBNoA2ADPAttention(TestEPLBNoA2A):
    """DP attention -- the MoE runs over the DP-gathered global token buffer --
    plus ~70 real rebalances, which exercise the post-rebalance placements and
    the expert-weight migration."""

    extra_args = [
        "--enable-dp-attention",
        "--dp",
        "2",
    ]
    rebalance_num_iterations = "50"


if __name__ == "__main__":
    unittest.main()
