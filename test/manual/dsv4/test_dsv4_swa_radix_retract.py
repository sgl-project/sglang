"""DSV4 stress test for SWA radix cache + tombstone + retract interaction.

Reproduces the assert in `swa_radix_cache.cache_unfinished_req`:
    assert old_prefix_len <= len(new_indices)

Trip conditions (all required):
  1. Fork-only SWA leaf early-release on (`SGLANG_OPT_SWA_RELEASE_LEAF_LOCK_AFTER_WINDOW=1`)
  2. Multiple requests share a long prefix (so one req's tombstoned leaf
     poisons match_prefix for others walking the same radix path).
  3. Memory pressure forces retract while at least one req has tombstoned
     its leaf (decode_batch_idx >= sliding_window_size at retract time).

After main #19427 changed `old_prefix_len = req.cache_protected_len`
(stable), tombstone-induced shrinks in match's `best_value_len` across
chunked-prefill rounds can make stale `cache_protected_len` exceed
current matchable length -> assert trips.

Test passes iff the scheduler does not crash under this stress workload.
"""

import random
import threading
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

DSV4_FLASH_MODEL_PATH = "sgl-project/DeepSeek-V4-Flash-FP8"

# Long shared prefix forces multi-chunk prefill and ensures cross-request
# prefix-cache hits so one req's tombstone affects later reqs.
SHARED_PREFIX_BLOCK = (
    "You are a careful, expert assistant. Answer concisely.\n"
    "Context: " + ("the quick brown fox jumps over the lazy dog. " * 600)
)

QUESTION_TAILS = [
    " Q: What is 17*23?\n",
    " Q: List three primary colors.\n",
    " Q: Where is Mount Everest?\n",
    " Q: Summarize gradient descent in two sentences.\n",
    " Q: Name two bodies of water in Africa.\n",
    " Q: What language is spoken in Brazil?\n",
]


class TestDSV4FlashSWARadixRetract(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DSV4_FLASH_MODEL_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--trust-remote-code",
            "--tp",
            "4",
            "--dp",
            "4",
            "--enable-dp-attention",
            "--moe-a2a-backend",
            "deepep",
            "--cuda-graph-max-bs",
            "128",
            "--max-running-requests",
            "256",
            "--deepep-config",
            '{"normal_dispatch":{"num_sms":96},"normal_combine":{"num_sms":96}}',
            "--speculative-algorithm",
            "EAGLE",
            "--speculative-num-steps",
            "3",
            "--speculative-eagle-topk",
            "1",
            "--speculative-num-draft-tokens",
            "4",
            # Tight static memory so SWA pool fills up under load and
            # retract is forced.
            "--mem-fraction-static",
            "0.7",
        ]
        env = {
            "SGLANG_DSV4_FP4_EXPERTS": "0",
            "SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK": "1024",
            "SGLANG_OPT_SWA_RADIX_CACHE_COMPACT": "0",
            "SGLANG_TEST_RETRACT": "1",
            "SGLANG_TEST_RETRACT_INTERVAL": "3",
        }
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
            env=env,
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _send_req(self, prompt: str, max_new_tokens: int):
        try:
            resp = requests.post(
                self.base_url + "/generate",
                json={
                    "text": prompt,
                    "sampling_params": {
                        # Vary outputs slightly so reqs don't share decode
                        # paths perfectly; we want some to finish, some to
                        # be retracted under pressure.
                        "temperature": 0.7,
                        "max_new_tokens": max_new_tokens,
                    },
                },
                timeout=600,
            )
            # Per-request success is not the gate; some requests are
            # expected to be retracted/aborted under heavy pressure.
            return resp.status_code == 200
        except Exception:
            return False

    def test_swa_tombstone_retract_does_not_crash(self):
        """Stress: 64 concurrent long-prompt reqs with long generation force
        retract under SWA pool pressure. Reqs share a 30k+ token prefix so
        tombstoned leaves from retracted reqs are on the radix path of new
        reqs. Scheduler must not crash on the swa_radix_cache assert."""

        random.seed(0)
        concurrency = 64
        # Long enough generation to push past sliding_window_size -> fires
        # `dec_swa_lock_only` -> tombstones leaves. Combined with SWA pool
        # pressure this guarantees retract while tombstones are live.
        max_new_tokens = 1024

        threads = []
        for i in range(concurrency):
            tail = QUESTION_TAILS[i % len(QUESTION_TAILS)]
            # Add a small per-req suffix so reqs don't dedup at radix root
            # but still share the bulk of the prefix.
            prompt = SHARED_PREFIX_BLOCK + tail + f"(seed={i})"
            t = threading.Thread(target=self._send_req, args=(prompt, max_new_tokens))
            threads.append(t)
            t.start()
            # Stagger so requests enter prefill in waves; some are still in
            # decode (and have tombstoned leaves) when later waves of
            # chunked-prefill reqs walk the same radix path.
            time.sleep(0.05)

        for t in threads:
            t.join(timeout=600)

        # The only invariant: scheduler survived. Per-request completion is
        # best-effort under retract pressure.
        self.assertIsNone(self.process.poll())


if __name__ == "__main__":
    unittest.main()
