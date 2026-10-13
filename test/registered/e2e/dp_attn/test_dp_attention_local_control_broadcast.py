"""Control-message routing under DP attention with local control broadcast.

With --enable-dp-attention-local-control-broadcast the tokenizer's control
messages (flush_cache, pause_generation, continue_generation) go to every DP
group leader over ZMQ and are broadcast within attn_tp_group instead of the
full tp_group. A rank that misses one of them either keeps generating while
the others are paused or hangs the next gloo broadcast, so this test drives
the whole tokenizer -> DP controller -> scheduler path on a tp4/dp2 server
(attn_tp_size=2, two DP groups of two ranks each) and checks that greedy
outputs survive flush / pause / continue round-trips.
"""

import time
import unittest
from concurrent.futures import ThreadPoolExecutor

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kits.pause_generation_kit import PauseResumeInPlaceMixin
from sglang.test.test_utils import (
    DEFAULT_MODEL_NAME_FOR_TEST_MLA,
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_cuda_ci(est_time=400, stage="base-c", runner_config="4-gpu-h100")

_INPUT_IDS = list(range(10, 30))
_REQUEST_TIMEOUT = 120


class TestDPAttentionLocalControlBroadcast(PauseResumeInPlaceMixin, CustomTestCase):
    # This checkpoint can emit EOS on the first token.
    pause_ignore_eos = True

    @classmethod
    def setUpClass(cls):
        cls.model = DEFAULT_MODEL_NAME_FOR_TEST_MLA
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            cls.model,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--tp",
                "4",
                "--dp",
                "2",
                "--enable-dp-attention",
                "--enable-dp-attention-local-control-broadcast",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def _generate_greedy(self, n: int, max_new_tokens: int = 16) -> list[list[int]]:
        resp = requests.post(
            f"{self.base_url}/generate",
            json={
                "input_ids": [_INPUT_IDS] * n,
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": max_new_tokens,
                },
            },
            timeout=_REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        outputs = resp.json()
        if isinstance(outputs, dict):
            outputs = [outputs]
        return [o["output_ids"] for o in outputs]

    def _post_control(self, route: str, payload: dict):
        requests.post(
            f"{self.base_url}/{route}", json=payload, timeout=60
        ).raise_for_status()

    def test_flush_and_pause_round_trips_keep_greedy_outputs(self):
        baseline = self._generate_greedy(n=4)
        self.assertEqual(len(baseline), 4)
        for ids in baseline:
            self.assertGreater(len(ids), 0)

        self._post_control("flush_cache", {})
        self.assertEqual(
            self._generate_greedy(n=4),
            baseline,
            "greedy outputs changed after flush_cache",
        )

        self._post_control("pause_generation", {"mode": "in_place"})
        with ThreadPoolExecutor(max_workers=1) as pool:
            blocked = pool.submit(self._generate_greedy, 1, 5)
            time.sleep(2.0)
            self.assertFalse(
                blocked.done(),
                "generation completed while paused: at least one rank did not "
                "receive the pause control message",
            )
            self._post_control("continue_generation", {})
            resumed = blocked.result(timeout=60)
        self.assertEqual(len(resumed), 1)
        self.assertGreater(len(resumed[0]), 0)

        self.assertEqual(
            self._generate_greedy(n=4),
            baseline,
            "greedy outputs changed after pause/continue",
        )

        self._post_control("flush_cache", {})
        self.assertEqual(
            self._generate_greedy(n=2),
            baseline[:2],
            "greedy outputs changed after the second flush_cache",
        )


if __name__ == "__main__":
    unittest.main()
