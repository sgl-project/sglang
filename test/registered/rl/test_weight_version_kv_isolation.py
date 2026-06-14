"""E2E for --enable-weight-version-kv-isolation: requests admitted under
different weight versions never share radix-cache entries, an in-flight
request survives a pause(in_place) + update_weights_from_disk(flush_cache=
false) + continue commit and reports (weight_version_start, weight_version_end)
across it, and a paused engine auto-continues after SGLANG_ENGINE_PAUSE_TIMEOUT.
"""

import threading
import time
import unittest

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=240, suite="base-b-test-1-gpu-small")

# Long prompt so cached_tokens hits are unambiguous.
PROMPT = "You are a careful assistant. " + " ".join(
    f"fact number {i} is interesting;" for i in range(200)
)
# Generous watchdog: the commit test's pause window (~seconds for a 1B
# reload) must stay well below it.
PAUSE_WATCHDOG_SECONDS = 20


class TestWeightVersionKVIsolation(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_SMALL_MODEL_NAME_FOR_TEST
        cls.base_url = DEFAULT_URL_FOR_TEST
        with envs.SGLANG_ENGINE_PAUSE_TIMEOUT.override(PAUSE_WATCHDOG_SECONDS):
            cls.process = popen_launch_server(
                cls.model,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=["--enable-weight-version-kv-isolation"],
            )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _generate(self, prompt, max_new_tokens=8, ignore_eos=False):
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "text": prompt,
                "sampling_params": {
                    "max_new_tokens": max_new_tokens,
                    "temperature": 0,
                    "ignore_eos": ignore_eos,
                },
            },
            timeout=300,
        )
        self.assertEqual(resp.status_code, 200)
        return resp.json()["meta_info"]

    def _commit(self, weight_version):
        resp = requests.post(
            self.base_url + "/pause_generation", json={"mode": "in_place"}, timeout=60
        )
        self.assertEqual(resp.status_code, 200)
        try:
            resp = requests.post(
                self.base_url + "/update_weights_from_disk",
                json={
                    "model_path": self.model,
                    "flush_cache": False,
                    "weight_version": weight_version,
                },
                timeout=300,
            )
            self.assertEqual(resp.status_code, 200)
            self.assertTrue(resp.json()["success"], resp.text)
        finally:
            resp = requests.post(
                self.base_url + "/continue_generation", json={}, timeout=60
            )
            self.assertEqual(resp.status_code, 200)

    def test_a_isolation_across_inplace_commit(self):
        # Unique per attempt: CustomTestCase retries re-run the whole method
        # against the same server, so versions must move forward each time.
        new_version = f"t{time.time_ns()}"

        # Warm the current namespace. "Cold" tolerates incidental short
        # prefixes (e.g. the BOS token cached by the launch warmup request).
        meta_cold = self._generate(PROMPT)
        base_version = meta_cold["weight_version_start"]
        self.assertLess(meta_cold["cached_tokens"], 50)
        self.assertEqual(meta_cold["weight_version_end"], base_version)
        meta_hit = self._generate(PROMPT)
        self.assertGreater(meta_hit["cached_tokens"], 1000)

        # Commit while a long generation is in flight; it must finish and
        # report crossing the version boundary.
        inflight = {}

        def long_gen():
            # Long enough to still be decoding when the commit lands even on
            # fast hardware (a 1B decodes ~700 tok/s on H100-class GPUs).
            inflight.update(
                self._generate(PROMPT + " tell me a story.", 4000, ignore_eos=True)
            )

        t = threading.Thread(target=long_gen, daemon=True)
        t.start()
        time.sleep(0.5)  # let it get into decode
        self._commit(new_version)
        t.join(timeout=300)
        self.assertEqual(inflight.get("weight_version_start"), base_version)
        self.assertEqual(inflight.get("weight_version_end"), new_version)
        # Completed normally (EOS or token budget), i.e. not aborted/retracted.
        self.assertIn(inflight["finish_reason"]["type"], ("length", "stop"))

        # New-era requests must miss the old namespace, then warm their own.
        meta_post = self._generate(PROMPT)
        self.assertLess(meta_post["cached_tokens"], 50)
        self.assertEqual(meta_post["weight_version_start"], new_version)
        meta_post_hit = self._generate(PROMPT)
        self.assertGreater(meta_post_hit["cached_tokens"], 1000)

    def test_z_pause_watchdog_auto_continues(self):
        resp = requests.post(
            self.base_url + "/pause_generation", json={"mode": "in_place"}, timeout=60
        )
        self.assertEqual(resp.status_code, 200)
        # No continue_generation: the watchdog must un-pause the engine.
        time.sleep(PAUSE_WATCHDOG_SECONDS + 5)
        meta = self._generate("ping", max_new_tokens=4)
        self.assertEqual(meta["finish_reason"]["type"], "length")


if __name__ == "__main__":
    unittest.main(verbosity=3)
