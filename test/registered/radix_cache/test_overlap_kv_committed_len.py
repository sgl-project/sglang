"""
Test that kv_committed_len is not inflated by 1 when a request finishes
during overlap-scheduled decoding (issue: overlap scheduler off-by-one).

With overlap scheduling enabled (the default), prepare_for_decode() is called
for every request in the running batch before the previous batch's EOS results
are processed.  This means a request that generated EOS in batch N has its
kv_committed_len incremented by 1 during batch N+1's setup, causing
cache_finished_req to insert one extra token into the radix tree.

The observable symptom is that a follow-up request whose input_ids extend the
finished sequence by 1 token reports cached_tokens == seqlen instead of the
correct seqlen - 1.
"""

import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=90, suite="stage-b-test-1-gpu-small")

MODEL = DEFAULT_SMALL_MODEL_NAME_FOR_TEST

# Fixed token sequences so the test is deterministic and model-independent.
INPUT_IDS_A = [1, 2, 3, 4]
MAX_NEW_TOKENS_A = (
    2  # A will finish with seqlen = len(INPUT_IDS_A) + MAX_NEW_TOKENS_A = 6
)


def generate(base_url, input_ids, max_new_tokens):
    resp = requests.post(
        base_url + "/generate",
        json={
            "input_ids": input_ids,
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": max_new_tokens,
            },
            "stream": False,
            "return_logprob": False,
        },
    )
    resp.raise_for_status()
    return resp.json()


class TestOverlapKvCommittedLen(CustomTestCase):
    """
    Regression test for the overlap-scheduler kv_committed_len off-by-one.

    Reference: https://github.com/sgl-project/sglang/issues/24590
    """

    @classmethod
    def setUpClass(cls):
        cls.model = MODEL
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Overlap scheduling is enabled by default; we also need cache reporting.
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=["--enable-cache-report"],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_cached_tokens_after_finish(self):
        """
        After request A finishes, a request B that extends A's output by one
        token should report cached_tokens == seqlen_A - 1, not seqlen_A.
        """
        # Step 1: run request A and get its output token ids.
        result_a = generate(self.base_url, INPUT_IDS_A, MAX_NEW_TOKENS_A)
        output_ids_a = result_a["output_ids"]  # top-level field in /generate response
        seqlen_a = len(INPUT_IDS_A) + len(output_ids_a)

        # Step 2: build request B = A's full sequence + 1 extra token.
        input_ids_b = INPUT_IDS_A + output_ids_a + [1]
        result_b = generate(self.base_url, input_ids_b, max_new_tokens=1)
        cached_tokens_b = result_b["meta_info"].get("cached_tokens", 0)

        # Without the fix, cached_tokens_b == seqlen_a (off by one).
        # With the fix, cached_tokens_b == seqlen_a - 1 (correct).
        self.assertLessEqual(
            cached_tokens_b,
            seqlen_a - 1,
            msg=(
                f"cached_tokens={cached_tokens_b} should be at most seqlen_a-1="
                f"{seqlen_a - 1}; got seqlen_a={seqlen_a}.  "
                "This suggests the overlap scheduler kv_committed_len off-by-one is back."
            ),
        )


if __name__ == "__main__":
    unittest.main()
