import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_8B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=400, suite="full-1-npu-a3", nightly=True)


class TestNPURadixEvictionPolicyConfig(CustomTestCase):
    """Testcase: --radix-eviction-policy-config on Ascend NPU.

    Validates that ``slru`` accepts its tuning parameter
    ``--radix-eviction-policy-config '{"protected_threshold": 4}'`` and that
    the eviction path actually runs: the KV pool is capped small enough to be
    exhausted, so the eviction strategy's ``get_priority`` is exercised. A
    prefix requested 4 times reaches ``hit_count == 4`` and survives, while a
    prefix requested twice stays at ``hit_count == 2`` and is evicted; this
    proves the configured threshold (4) -- not the SLRU default (2) -- drives
    eviction.

    [Test Category] HiCache
    [Test Target] --radix-eviction-policy-config
    """

    # Cap the KV pool so a handful of long prompts exhaust it and force
    # eviction (the eviction policy only runs when the pool is full).
    _MAX_TOTAL_TOKENS = 2048
    _PROTECTED_THRESHOLD = 4
    _SHARED_PROMPT = "What is the capital of France? " * 36
    # Distinct first token from _SHARED_PROMPT so the two prefixes never share
    # radix-cache nodes, keeping their hit counts independent. Repeats are
    # chosen long enough to survive page-aligned prefix matching (a too-short
    # prompt is floored to a 0-length key and never cached).
    _VICTIM_PROMPT = "Explain the theory of general relativity in detail " * 36

    @classmethod
    def setUpClass(cls):
        cls.model = QWEN3_8B_WEIGHTS_PATH
        cls.base_url = DEFAULT_URL_FOR_TEST
        other_args = [
            "--attention-backend",
            "ascend",
            "--disable-cuda-graph",
            "--enable-metrics",
            "--tp-size",
            "1",
            "--max-total-tokens",
            str(cls._MAX_TOTAL_TOKENS),
            "--radix-eviction-policy",
            "slru",
            "--radix-eviction-policy-config",
            '{"protected_threshold": 4}',
        ]
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=other_args,
        )
        cls.base_url += "/v1"

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def tearDown(self):
        try:
            response = requests.post(f"{DEFAULT_URL_FOR_TEST}/flush_cache")
            self.assertEqual(response.status_code, 200, "Failed to flush cache")
        except Exception as e:
            self.fail(f"Flush cache failed with error: {str(e)}")

    def _generate(self, text, max_new_tokens):
        response = requests.post(
            f"{DEFAULT_URL_FOR_TEST}/generate",
            json={
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
        )
        self.assertEqual(response.status_code, 200)
        return response.json()

    def _read_metric_total(self, name):
        """Sum every sample of a prometheus counter (labels aside)."""
        response = requests.get(f"{DEFAULT_URL_FOR_TEST}/metrics", timeout=30)
        self.assertEqual(response.status_code, 200)
        total = 0.0
        for line in response.text.splitlines():
            if line.startswith("#"):
                continue
            # Match the exact metric name, skipping *_created companions.
            if line.startswith(name) and not line.startswith(name + "_"):
                total += float(line.rsplit(" ", 1)[-1])
        return total

    def test_eviction_triggers_and_protected_prefix_survives(self):
        # Phase 1: warm the shared prefix enough that its hit_count reaches the
        # protected segment (hit_count >= protected_threshold). The first
        # request misses; every later one reuses the cached prefix.
        for i in range(self._PROTECTED_THRESHOLD):
            resp = self._generate(self._SHARED_PROMPT, 8)
            cached = int(resp["meta_info"]["cached_tokens"])
            if i == 0:
                self.assertEqual(cached, 0, "First request should not be cached")
            else:
                self.assertGreater(cached, 0, "Shared prefix should be reused")

        # Phase 2: request the victim prefix twice. A hit is counted only when
        # a request finishes, so each finished request bumps hit_count by one:
        # two requests leave hit_count == 2, exactly at the SLRU default
        # threshold but still below the configured threshold of 4. Under
        # threshold 4 the victim must be evicted; under the default 2 it would
        # be protected and survive, so its eviction proves the config is honored.
        for i in range(2):
            resp = self._generate(self._VICTIM_PROMPT, 8)
            cached = int(resp["meta_info"]["cached_tokens"])
            if i == 0:
                self.assertEqual(cached, 0, "First victim request should not be cached")
            else:
                self.assertGreater(cached, 0, "Victim prefix should be reused")

        # Phase 3: flood the small KV pool with distinct long prompts until
        # eviction is forced (get_priority is consulted).
        for i in range(32):
            self._generate(f"unique filler sequence number {i} " * 40, 4)

        evicted = self._read_metric_total("sglang:evicted_tokens_total")
        self.assertGreater(
            evicted,
            0,
            "Expected radix eviction when the KV pool is exhausted",
        )

        # The victim (2 hits < threshold) must be evicted while the protected
        # prefix (>= 4 hits) survives; this proves the configured threshold (4)
        # drives eviction, not the SLRU default of 2.
        victim = self._generate(self._VICTIM_PROMPT, 8)
        self.assertEqual(
            int(victim["meta_info"]["cached_tokens"]),
            0,
            "Victim prefix just below the threshold should be evicted",
        )
        resp = self._generate(self._SHARED_PROMPT, 8)
        self.assertGreater(
            int(resp["meta_info"]["cached_tokens"]),
            0,
            "Protected prefix should survive eviction",
        )


if __name__ == "__main__":
    unittest.main()
