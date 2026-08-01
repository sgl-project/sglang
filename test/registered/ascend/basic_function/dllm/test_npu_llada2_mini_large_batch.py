"""Large-batch dLLM coverage on Ascend.

test_npu_llada2_mini.py pins the single-request latency recipe: one running
request and synchronous denoise (`--no-dllm-fdfo`). That leaves the FDFO
scheduler -- the default, and the one a rollout deployment runs -- with no NPU
e2e coverage at all, and several scheduler and denoise paths only exist above a
row-count threshold that bs=1 cannot reach.

This runs the same gsm8k check on the throughput recipe instead: FDFO, a wide
batch, and (in the second case) prefill and decode rows scheduled into one
round. Keep both files: they cover opposite ends of the deployment range.
"""

import unittest

import requests

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.test_ascend_utils import LLaDA2_0_MINI_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="stage-b-test-4-npu-a3", nightly=False)
register_npu_ci(est_time=400, suite="nightly-1-npu-a3", nightly=True)

# 64 concurrent requests x a 32-token denoise block = 2048 rows per forward,
# past every row-count threshold on the dLLM path. Each concurrent request also
# costs roughly 50 MB outside the static pool (its slice of the denoise
# reduction's transients), so raising this means lowering mem-fraction-static.
MAX_RUNNING_REQUESTS = 64


def assert_mixed_rounds(test, expected: bool):
    """The flag is requested by env var but downgraded when FDFO is off, so the
    only honest check is what the scheduler resolved -- otherwise a test that
    fails to pass the flag through looks identical to one that passes it."""
    states = requests.get(test.base_url + "/server_info").json()["internal_states"]
    test.assertTrue(states, "server reported no internal states")
    for state in states:
        test.assertEqual(state["dllm_mixed_batch_enabled"], expected)


_LARGE_BATCH_ARGS = [
    "--trust-remote-code",
    "--mem-fraction-static",
    "0.8",
    "--attention-backend",
    "ascend",
    "--dllm-algorithm",
    "LowConfidence",
    "--dllm-fdfo",
    "--max-running-requests",
    str(MAX_RUNNING_REQUESTS),
    # Capture the decode graph rather than inheriting the mixin's
    # --disable-cuda-graph: graph replay is how this model is served at batch,
    # and the NPU graph runner's own dLLM path is otherwise untested.
    "--cuda-graph-config",
    '{"decode":{"backend":"full","max_bs":%d,"bs":[1,8,16,32,%d]}}'
    % (MAX_RUNNING_REQUESTS, MAX_RUNNING_REQUESTS),
]


class TestLLaDA2MiniLargeBatch(GSM8KAscendMixin, CustomTestCase):
    """FDFO at a batch wide enough to reach the large-batch code paths."""

    model = LLaDA2_0_MINI_WEIGHTS_PATH
    other_args = _LARGE_BATCH_ARGS
    gsm8k_parallel = MAX_RUNNING_REQUESTS
    accuracy = 0.88

    def test_resolved_mixed_round_mode(self):
        assert_mixed_rounds(self, False)


class TestLLaDA2MiniMixedRound(GSM8KAscendMixin, CustomTestCase):
    """Same batch with prefill and decode rows scheduled into one round.

    Accuracy must be unaffected: mixing changes which rows share a forward, not
    what any row denoises.
    """

    model = LLaDA2_0_MINI_WEIGHTS_PATH
    other_args = _LARGE_BATCH_ARGS
    gsm8k_parallel = MAX_RUNNING_REQUESTS
    accuracy = 0.88
    # The mixin snapshots os.environ into `env` at class-definition time and
    # passes it to the server process, so the flag has to go in there rather
    # than through envs.override() in setUpClass.
    env = {**GSM8KAscendMixin.env, "SGLANG_ENABLE_DLLM_MIXED_BATCH": "1"}

    def test_resolved_mixed_round_mode(self):
        assert_mixed_rounds(self, True)


if __name__ == "__main__":
    unittest.main()
