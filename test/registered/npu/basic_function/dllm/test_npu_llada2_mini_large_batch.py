"""Large-batch dLLM coverage on Ascend.

test_npu_llada2_mini.py pins the single-request latency recipe: one running
request and synchronous denoise (`--no-dllm-fdfo`). This file covers the FDFO
scheduler at a batch wide enough to reach the row-count thresholds.

This runs the same gsm8k check on the throughput recipe instead: FDFO, a wide
batch, and (in the second case) prefill and decode rows scheduled into one
round. Keep both files: they cover opposite ends of the deployment range.
"""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.test.ascend.gsm8k_ascend_mixin import GSM8KAscendMixin
from sglang.test.ascend.npu_eval_accuracy_kit import _is_pr_pipeline
from sglang.test.ascend.test_ascend_utils import LLaDA2_0_MINI_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import CustomTestCase

register_npu_ci(est_time=400, suite="base-b-test-4-npu-a3")
register_npu_ci(est_time=400, suite="nightly-4-npu-a3", nightly=True)

# 64 concurrent requests x a 32-token denoise block = 2048 rows per forward,
# past every row-count threshold on the dLLM path. Each concurrent request also
# costs roughly 50 MB outside the static pool (its slice of the denoise
# reduction's transients), so raising this means lowering mem-fraction-static.
MAX_RUNNING_REQUESTS = 64


def assert_mixed_rounds(test, expected: bool):
    """Assert on what the scheduler actually built, not just what it resolved.

    ``dllm_mixed_batch_enabled`` is a startup constant: a server that resolved
    it to True but never put a prefill row and a decode row in the same round
    would satisfy it identically. ``dllm_num_mixed_rounds`` counts the rounds
    whose can_run_list held both phases, so it separates the two. Call this
    after traffic has flowed -- on an idle server every counter is 0.
    """
    states = requests.get(test.base_url + "/server_info").json()["internal_states"]
    test.assertTrue(states, "server reported no internal states")
    for state in states:
        test.assertEqual(state["dllm_mixed_batch_enabled"], expected)
        test.assertGreater(state["dllm_num_rounds"], 0, "no dLLM round was scheduled")
        if expected:
            test.assertGreater(
                state["dllm_num_mixed_rounds"], 0, "no round mixed the two phases"
            )
        else:
            test.assertEqual(state["dllm_num_mixed_rounds"], 0)


def run_staggered_pr_traffic(base_url, num_requests=32):
    """Concurrent, staggered requests so a mixed round can actually form.

    On PR pipelines the mixin's test_gsm8k short-circuits to a single smoke
    request. One request is either prefilling or decoding, never both, so
    ``_should_mix_dllm_batches`` (which needs num_prefill_reqs > 0 AND
    num_decode_reqs > 0 in the same round) can never pass and
    ``dllm_num_mixed_rounds`` stays 0. Staggered arrivals keep later requests
    in prefill while earlier ones are still denoising their 128-token
    completions (>= 4 decode blocks, many rounds), which is exactly the
    overlap the gate waits for. Prompts differ per request so prefix caching
    cannot collapse a prefill into a no-op.
    """

    def one_request(i):
        time.sleep(0.05 * i)
        response = requests.post(
            f"{base_url}/generate",
            json={
                "text": f"Question {i}: compute {i} + {i} and explain. Answer:",
                "sampling_params": {"temperature": 0, "max_new_tokens": 128},
            },
            timeout=300,
        )
        assert response.status_code == 200, response.text

    # max_workers must cover every request: the stagger relies on all threads
    # starting their sleeps together rather than queueing behind the pool.
    with ThreadPoolExecutor(max_workers=num_requests) as pool:
        # list() drains the iterator so a worker's exception re-raises here.
        list(pool.map(one_request, range(num_requests)))


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

    def test_gsm8k(self):
        # Override rather than assert in a sibling test method: the round
        # counters only mean something after traffic, and unittest's
        # alphabetical ordering is not a contract to lean on.
        super().test_gsm8k()
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

    def test_gsm8k(self):
        # See TestLLaDA2MiniLargeBatch.test_gsm8k.
        super().test_gsm8k()
        if _is_pr_pipeline:
            # The PR smoke path sends one request, which can never put a
            # prefill row and a decode row in the same round; drive real
            # overlapping traffic before asserting the counter moved.
            run_staggered_pr_traffic(self.base_url)
        assert_mixed_rounds(self, True)


if __name__ == "__main__":
    unittest.main()
