"""NPU CI test for the topologies where the ``comm`` tag must refuse.

The comm cleanup brackets its device work with world-group collectives, so the
control request has to reach every rank of the process set. That does not hold
with pipeline parallelism, where each stage is its own process set, so ``comm``
is rejected there: an explicit request must fail with 400 *before* anything is
released, and a tag-less full sleep must still work by excluding it.

Usage:
python3 -m unittest test_sleep_comm_cleanup_topology.TestSleepCommCleanupTopology
"""

import os
import unittest

import requests
import torch

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import QWEN3_0_6B_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(
    # One server launch (tp=2 x pp=2 = 4 ranks).
    est_time=900,
    suite="full-2-npu-a3",
    nightly=True,
)

if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3"

_MIB = 1 << 20
FULL_RELEASE_MIN_DELTA_MB = 500
WAKE_BASELINE_TOL_MB = 400
# A refused request must not move device memory at all; allow allocator noise.
NO_CHANGE_TOL_MB = 50
_GREEDY_PROMPT = "Where is the capital of France?"


def _device_used_mb():
    free, total = torch.npu.mem_get_info()
    return (total - free) / _MIB


def _post(base_url, endpoint, payload=None):
    return requests.post(f"{base_url}{endpoint}", json=payload or {}, timeout=300)


def _generate_greedy(test_case, base_url):
    response = _post(
        base_url,
        "/generate",
        {
            "text": _GREEDY_PROMPT,
            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        },
    )
    test_case.assertEqual(response.status_code, 200, response.text)
    return response.json()["text"]


class TestSleepCommCleanupTopology(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            QWEN3_0_6B_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                "--trust-remote-code",
                "--device",
                "npu",
                "--attention-backend",
                "ascend",
                "--tp-size",
                "2",
                "--pp-size",
                "2",
                "--mem-fraction-static",
                "0.8",
                "--enable-memory-saver",
                "--enable-weights-cpu-backup",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_1_explicit_comm_is_rejected_before_any_release(self):
        """An explicit `comm` is a 400 that names pipeline parallelism, and the
        rejection moves no memory and leaves the server serving."""
        text_before = _generate_greedy(self, self.base_url)
        used_before_mb = _device_used_mb()

        response = _post(
            self.base_url, "/release_memory_occupation", {"tags": ["comm"]}
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("pipeline parallelism", response.text)

        used_after_mb = _device_used_mb()
        self.assertLessEqual(
            abs(used_after_mb - used_before_mb),
            WAKE_BASELINE_TOL_MB,
            f"a rejected release must not move memory "
            f"({used_before_mb:.0f}MB -> {used_after_mb:.0f}MB)",
        )
        self.assertEqual(text_before, _generate_greedy(self, self.base_url))

    def test_2_mixed_tags_are_rejected_without_a_partial_release(self):
        """A request that mixes `comm` with releaseable tags is rejected as a
        whole, so the weights/kv are not left offloaded by the rejected call."""
        used_before_mb = _device_used_mb()
        response = _post(
            self.base_url,
            "/release_memory_occupation",
            {"tags": ["comm", "weights"]},
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("pipeline parallelism", response.text)
        self.assertLessEqual(
            abs(_device_used_mb() - used_before_mb), WAKE_BASELINE_TOL_MB
        )

        # Resuming weights is refused in the scheduler (log-only, HTTP 200)
        # because nothing was released; it must not move memory either.
        used_before_mb = _device_used_mb()
        response = _post(
            self.base_url,
            "/resume_memory_occupation",
            {"tags": ["weights"]},
        )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertLessEqual(
            abs(_device_used_mb() - used_before_mb),
            NO_CHANGE_TOL_MB,
            "the refused resume must change no state",
        )

    def test_3_tagless_full_sleep_still_works(self):
        """`comm` is excluded from the default set on this topology instead of
        failing the request, and the rest of the sleep/wake cycle works."""
        text_before = _generate_greedy(self, self.base_url)
        used_before_mb = _device_used_mb()

        response = _post(self.base_url, "/release_memory_occupation")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreater(
            used_before_mb - _device_used_mb(),
            FULL_RELEASE_MIN_DELTA_MB,
            "a tag-less release must still free weights + kv_cache",
        )

        response = _post(self.base_url, "/resume_memory_occupation")
        self.assertEqual(response.status_code, 200, response.text)
        self.assertLessEqual(
            abs(_device_used_mb() - used_before_mb),
            WAKE_BASELINE_TOL_MB,
            "wake must return to the pre-sleep baseline",
        )
        self.assertEqual(text_before, _generate_greedy(self, self.base_url))

    def test_4_resume_comm_is_refused_under_pp(self):
        """Naming `comm` on resume is also refused under pp - logged, HTTP 200,
        no hang, and the server keeps serving."""
        text_before = _generate_greedy(self, self.base_url)
        response = _post(self.base_url, "/resume_memory_occupation", {"tags": ["comm"]})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(text_before, _generate_greedy(self, self.base_url))


if __name__ == "__main__":
    unittest.main()
