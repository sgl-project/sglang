"""Unified memory under memory pressure, end to end.

One small hybrid model (full-attention KV + Mamba state in one arena) is served
with a tiny KV pool and SGLANG_TEST_RETRACT, so the scheduler retracts decode
requests every few steps. On top of that churn the tests run concurrent decodes,
mid-stream aborts, and a second wave that reuses the aborted prefixes.

Leaks are caught by the server itself: the busy and idle pool checks raise on a
leak, which takes the server down, so every test ends by requiring the server to
still be alive and healthy after it has gone idle.
"""

import time
import unittest
import uuid
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.environ import envs
from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.abort_timeout_kit import AbortAllMixin
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=240, stage="extra-a", runner_config="1-gpu-small")

MODEL = "yujiepan/kimi-linear-tiny-random"
_REQUEST_TIMEOUT = 300
# Seconds to let the scheduler go idle so the idle pool checks run.
_IDLE_SETTLE_SECS = 3


def _prompt(seed: int, num_words: int = 200) -> str:
    return " ".join(f"w{(seed * 7919 + i * 104729) % 5000}" for i in range(num_words))


class TestUnifiedMemoryPressure(AbortAllMixin, CustomTestCase):
    # Small enough that a few concurrent decodes fill the pool; the scheduler
    # caps running requests further by the Mamba slots that fit.
    max_total_tokens = 4096
    max_running_requests = 32
    extra_args: list = []

    abort_all_num_requests = 16
    abort_all_max_new_tokens = 2048

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        with (
            envs.SGLANG_TEST_RETRACT.override(True),
            envs.SGLANG_ENABLE_STRICT_MEM_CHECK_DURING_BUSY.override(1),
        ):
            cls.process = popen_launch_server(
                MODEL,
                cls.base_url,
                timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
                other_args=[
                    "--trust-remote-code",
                    "--enable-unified-memory",
                    "--max-total-tokens",
                    str(cls.max_total_tokens),
                    "--max-running-requests",
                    str(cls.max_running_requests),
                    *cls.extra_args,
                ],
            )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate(self, rid: str, text: str, max_new_tokens: int) -> dict:
        resp = requests.post(
            self.base_url + "/generate",
            json={
                "rid": rid,
                "text": text,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                    "ignore_eos": True,
                },
            },
            timeout=_REQUEST_TIMEOUT,
        )
        self.assertEqual(resp.status_code, 200, resp.text)
        return resp.json()["meta_info"]

    def _abort(self, rid: str):
        # /abort_request matches rids by prefix; uuid4 hex rids share a length,
        # so a prefix match is an exact match.
        requests.post(
            self.base_url + "/abort_request", json={"rid": rid}, timeout=10
        ).raise_for_status()

    def _assert_alive_after_idle(self):
        time.sleep(_IDLE_SETTLE_SECS)
        self.assertIsNone(self.process.poll(), "server exited; see its log for a leak")
        resp = requests.get(self.base_url + "/health_generate", timeout=60)
        self.assertEqual(resp.status_code, 200)

    def test_concurrent_decode_with_retraction(self):
        num_requests, max_new_tokens = 24, 256
        with ThreadPoolExecutor(num_requests) as ex:
            metas = list(
                ex.map(
                    lambda i: self._generate(
                        uuid.uuid4().hex, _prompt(i), max_new_tokens
                    ),
                    range(num_requests),
                )
            )

        for meta in metas:
            self.assertEqual(meta["finish_reason"]["type"], "length")
            self.assertEqual(meta["completion_tokens"], max_new_tokens)
        # Without this the test could pass on a pool that never filled.
        self.assertGreater(sum(m["num_retractions"] for m in metas), 0)
        self._assert_alive_after_idle()

    def test_abort_subset_then_reuse_prefixes(self):
        num_abort, num_keep = 8, 16
        abort_rids = [uuid.uuid4().hex for _ in range(num_abort)]
        with ThreadPoolExecutor(num_abort + num_keep) as ex:
            # Long enough that none can finish before the abort lands.
            aborted = [
                ex.submit(self._generate, rid, _prompt(100 + i), 2048)
                for i, rid in enumerate(abort_rids)
            ]
            kept = [
                ex.submit(self._generate, uuid.uuid4().hex, _prompt(200 + i), 128)
                for i in range(num_keep)
            ]
            time.sleep(2)
            for rid in abort_rids:
                self._abort(rid)

            for future in aborted:
                self.assertEqual(future.result()["finish_reason"]["type"], "abort")
            for future in kept:
                meta = future.result()
                self.assertEqual(meta["finish_reason"]["type"], "length")
                self.assertEqual(meta["completion_tokens"], 128)

        # Pages the aborts returned, and any prefix they left in the radix
        # cache, must be safe to reuse.
        with ThreadPoolExecutor(num_abort) as ex:
            second_wave = list(
                ex.map(
                    lambda i: self._generate(uuid.uuid4().hex, _prompt(100 + i), 128),
                    range(num_abort),
                )
            )
        for meta in second_wave:
            self.assertEqual(meta["finish_reason"]["type"], "length")
            self.assertEqual(meta["completion_tokens"], 128)
        self._assert_alive_after_idle()

    def test_abort_all(self):
        super().test_abort_all()
        self._assert_alive_after_idle()


class TestUnifiedMemoryPressureNoOverlap(TestUnifiedMemoryPressure):
    extra_args = ["--disable-overlap-schedule"]


if __name__ == "__main__":
    unittest.main()
