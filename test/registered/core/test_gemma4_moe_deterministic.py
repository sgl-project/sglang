"""Regression test for issue #24394.

`--enable-deterministic-inference` with `--attention-backend triton` on a
hybrid `SWAKVPool` model (Gemma4 family) used to crash with
`CUDA error: an illegal memory access` inside `_fwd_kernel_unified`: the
unified extend kernel read the new tokens at `out_cache_loc` (full-pool
index space) while `SWAKVPool.set_kv_buffer` had written them at the
SWA-translated indices. With diverse prompts the OOB never materialises;
the repro is same-prompt × high-concurrency, which is what this test fires.
"""

import concurrent.futures
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=129, suite="stage-b-test-2-gpu-large")


PROMPT = (
    "Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast "
    "every morning and bakes muffins for her friends every day with four. She "
    "sells the remainder at the farmers' market daily for $2 per fresh duck "
    "egg. How much in dollars does she make every day at the farmers' market?\n"
    "Answer:"
)
NUM_REQUESTS = 180
CONCURRENCY = 128
MAX_TOKENS = 256


class TestGemma4MoeDeterministic(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = "google/gemma-4-26B-A4B-it"
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--tp-size",
                "2",
                "--attention-backend",
                "triton",
                "--enable-deterministic-inference",
                "--dtype",
                "bfloat16",
                "--mem-fraction-static",
                "0.55",
                "--max-running-requests",
                "16",
                "--context-length",
                "2048",
                "--max-total-tokens",
                "32768",
                "--skip-server-warmup",
                "--random-seed",
                "0",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        if hasattr(cls, "process") and cls.process:
            kill_process_tree(cls.process.pid)

    def _fire_one(self):
        try:
            r = requests.post(
                self.base_url + "/v1/completions",
                json={
                    "model": self.model,
                    "prompt": PROMPT,
                    "max_tokens": MAX_TOKENS,
                    "temperature": 0.0,
                    "top_k": 1,
                },
                timeout=300,
            )
            r.raise_for_status()
            return True, ""
        except Exception as e:
            return False, repr(e)

    def test_no_ima_under_concurrent_load(self):
        try:
            requests.get(self.base_url + "/flush_cache", timeout=30)
        except Exception:
            pass

        n_ok = n_fail = 0
        first_fail = ""
        with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
            futs = [ex.submit(self._fire_one) for _ in range(NUM_REQUESTS)]
            for f in concurrent.futures.as_completed(futs):
                ok, msg = f.result()
                if ok:
                    n_ok += 1
                else:
                    if n_fail == 0:
                        first_fail = msg
                    n_fail += 1

        print(f"n_ok={n_ok} n_fail={n_fail} first_fail={first_fail!r}")
        self.assertEqual(
            n_fail,
            0,
            f"{n_fail}/{NUM_REQUESTS} requests failed; first error: {first_fail}",
        )


if __name__ == "__main__":
    unittest.main()
