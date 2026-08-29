"""Tensor-parallel startup for the capture-derived DSpark SPS table.

The cost table is measured from captured CUDA graphs, and captured graphs carry the model's
collectives. Three failure modes live only at tp>1, and all of them are fatal at startup rather than
wrong at runtime, so no single-GPU test can see them:

  * ranks that derive their own curves choose different graph tiers for the same batch, replay
    different shapes, and take an illegal memory access on a non-zero rank;
  * ranks that decide locally how many times to replay issue mismatched collective counts and hang
    in the sweep;
  * a rank whose local measurement raises skips the broadcast and strands its peers in the
    collective until the distributed timeout.

All three are prevented by construction -- rank 0's measurement is broadcast, the sweep performs the
same replays on every rank, and the broadcast sits outside the failure guard with a shape that does
not depend on what any rank measured. This test is the regression guard for that construction: it
asserts the server starts and serves — all three failure modes are fatal before that point.
"""

import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=300, stage="base-b", runner_config="2-gpu-large")

TARGET_MODEL = "Qwen/Qwen3-14B"
DRAFT_MODEL = "deepseek-ai/dspark_qwen3_14b_block7"


class TestDSparkCaptureDerivedSpsTp2(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            TARGET_MODEL,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp",
                "2",
                "--trust-remote-code",
                "--attention-backend",
                "fa3",
                "--speculative-algorithm",
                "DSPARK",
                "--speculative-draft-model-path",
                DRAFT_MODEL,
                "--speculative-draft-attention-backend",
                "fa3",
                "--mem-fraction-static",
                "0.7",
                "--page-size",
                "1",
                "--cuda-graph-max-bs-decode",
                "32",
            ],
            env={
                # Compact is the only mode that captures per-token-tier verify graphs, so it is the
                # only mode the derivation runs in, and the derivation is opt-in. Without BOTH the
                # server would start with the uninitialized table and this test would pass while
                # exercising none of the code it exists to guard.
                "SGLANG_RAGGED_VERIFY_MODE": "compact",
                "SGLANG_DSPARK_ENABLE_CAPTURE_DERIVED_SPS": "1",
            },
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def test_server_serves_after_deriving_the_cost_table(self):
        # Reaching this point already covers the hang: setUpClass would have timed out.
        response = requests.post(
            self.base_url + "/generate",
            json={
                "text": "Speculative decoding is",
                "sampling_params": {"max_new_tokens": 32, "temperature": 0},
            },
            timeout=120,
        )
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["text"])

    def test_both_ranks_kept_planning_after_the_first_steps(self):
        """A cross-rank disagreement shows up as a dead scheduler on the next batch, not at launch,
        so drive several requests that are genuinely in flight together — serial requests would only
        ever schedule single-request batches and never exercise a multi-request tier choice.
        """

        def _generate(i):
            return requests.post(
                self.base_url + "/generate",
                json={
                    "text": f"Count to ten starting from {i}:",
                    "sampling_params": {"max_new_tokens": 48, "temperature": 0},
                },
                timeout=120,
            )

        with ThreadPoolExecutor(max_workers=8) as pool:
            for _ in range(2):
                responses = list(pool.map(_generate, range(8)))
                for response in responses:
                    self.assertEqual(response.status_code, 200)


if __name__ == "__main__":
    unittest.main(verbosity=3)
