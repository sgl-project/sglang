"""NPU CI test for the sleep-mode ``comm`` memory tag (HCCL comm cleanup).

Drives /release_memory_occupation and /resume_memory_occupation on a tp=2 NPU
server:

* ``comm`` alone destroys the HCCL communication domains and frees their device
  buffers; resume rebuilds them and recaptures the CUDA graphs.
* A tag-less request (full sleep) on NPU includes ``comm``, so it must free
  weights + kv_cache + graphs + comms and come back bit-identical.
* A legacy client resuming the historical triple ``[weights, kv_cache,
  cuda_graph]`` still gets a complete wake.
* Invalid resumes are refused in the scheduler (log-only, the HTTP call still
  returns 200) and change no state; the server keeps serving.
* Releasing twice is a no-op; three full cycles return to the memory baseline.

Usage:
python3 -m unittest test_sleep_comm_cleanup.TestSleepCommCleanup
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
    # Two server launches (graph-enabled + graph-disabled classes).
    est_time=1400,
    suite="full-2-npu-a3",
    nightly=True,
)

if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1"

_MIB = 1 << 20
# Each HCCL communication domain holds hundreds of MiB of device buffers, so a
# comm-only sleep must free well above this floor.
COMM_RELEASE_MIN_DELTA_MB = 100
# A tag-less sleep also pauses weights + kv_cache; keep a conservative floor.
FULL_RELEASE_MIN_DELTA_MB = 500
# After wake, usage must return near the pre-sleep baseline: far below means a
# silent eager fallback, far above means a leaked comm or graph pool.
WAKE_BASELINE_TOL_MB = 400
# A refused request must not move device memory at all; allow allocator noise.
NO_CHANGE_TOL_MB = 50

_GREEDY_PROMPT = "Where is the capital of France?"


def _device_used_mb():
    # Device-wide usage (like npu-smi), so it covers the server processes too.
    free, total = torch.npu.mem_get_info()
    return (total - free) / _MIB


def _post(base_url, endpoint, payload=None):
    # Timeout so a wedged scheduler fails fast instead of hanging CI.
    return requests.post(f"{base_url}{endpoint}", json=payload or {}, timeout=300)


def _generate_greedy(test_case, base_url, prompt=_GREEDY_PROMPT):
    response = _post(
        base_url,
        "/generate",
        {"text": prompt, "sampling_params": {"temperature": 0, "max_new_tokens": 32}},
    )
    test_case.assertEqual(response.status_code, 200, response.text)
    return response.json()["text"]


def _assert_generate_works(test_case, base_url):
    test_case.assertIn("Paris", _generate_greedy(test_case, base_url))


def _assert_mem_near_baseline(test_case, baseline_mb, what):
    used_mb = _device_used_mb()
    test_case.assertLessEqual(
        abs(used_mb - baseline_mb),
        WAKE_BASELINE_TOL_MB,
        f"{what}: device usage {used_mb:.0f}MB drifted from baseline "
        f"{baseline_mb:.0f}MB by more than {WAKE_BASELINE_TOL_MB}MB",
    )


def _release(base_url, tags=None):
    payload = {} if tags is None else {"tags": tags}
    return _post(base_url, "/release_memory_occupation", payload)


def _resume(base_url, tags=None):
    payload = {} if tags is None else {"tags": tags}
    return _post(base_url, "/resume_memory_occupation", payload)


class TestSleepCommCleanup(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Graph capture stays enabled: the comm tag owns graph drop + recapture.
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
                "--mem-fraction-static",
                "0.8",
                "--enable-memory-saver",
                # Required: without the CPU backup a full sleep discards the
                # weights and wake cannot produce correct output.
                "--enable-weights-cpu-backup",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_1_comm_only_sleep_wake(self):
        """comm tag: comms are destroyed on sleep and rebuilt on wake.

        The delta also includes the dropped graphs; the pure HCCL share is
        isolated in TestSleepCommCleanupNoGraph.
        """
        _assert_generate_works(self, self.base_url)

        used_before_mb = _device_used_mb()
        response = _release(self.base_url, ["comm"])
        self.assertEqual(response.status_code, 200, response.text)
        used_released_mb = _device_used_mb()
        self.assertGreater(
            used_before_mb - used_released_mb,
            COMM_RELEASE_MIN_DELTA_MB,
            f"comm sleep should free HCCL buffers "
            f"(before={used_before_mb:.0f}MB, after={used_released_mb:.0f}MB)",
        )

        response = _resume(self.base_url, ["comm"])
        self.assertEqual(response.status_code, 200, response.text)

        # Rebuilt comms + recaptured graphs must serve real traffic.
        _assert_generate_works(self, self.base_url)

    def test_2_full_sleep_wake_includes_comm(self):
        """A tag-less sleep defaults to all tags, which on NPU includes comm.

        Weights come back from the CPU backup and greedy decoding must be
        identical before and after the cycle.
        """
        text_before = _generate_greedy(self, self.base_url)

        used_before_mb = _device_used_mb()
        response = _release(self.base_url)
        self.assertEqual(response.status_code, 200, response.text)
        used_released_mb = _device_used_mb()
        self.assertGreater(
            used_before_mb - used_released_mb,
            FULL_RELEASE_MIN_DELTA_MB,
            f"full sleep should free weights+kv+graphs+comms "
            f"(before={used_before_mb:.0f}MB, after={used_released_mb:.0f}MB)",
        )

        response = _resume(self.base_url)
        self.assertEqual(response.status_code, 200, response.text)
        _assert_mem_near_baseline(self, used_before_mb, "full sleep/wake")

        text_after = _generate_greedy(self, self.base_url)
        self.assertEqual(text_before, text_after)

    def test_3_resume_of_unreleased_tag_is_refused(self):
        """A resume naming a tag that was never released is refused in the
        scheduler (the refusal is logged; resume has no error channel, so the
        HTTP call still returns 200), changes nothing, and the server keeps
        serving afterwards."""
        response = _release(self.base_url, ["comm"])
        self.assertEqual(response.status_code, 200, response.text)
        try:
            used_before_mb = _device_used_mb()
            response = _resume(self.base_url, ["weights"])
            self.assertEqual(response.status_code, 200, response.text)
            self.assertLessEqual(
                abs(_device_used_mb() - used_before_mb),
                NO_CHANGE_TOL_MB,
                "a refused resume must change no state",
            )
        finally:
            # Restore the server to a fully awake state for later tests.
            response = _resume(self.base_url, ["comm"])
            self.assertEqual(response.status_code, 200, response.text)

        _assert_generate_works(self, self.base_url)

    def test_4_legacy_triple_resume_auto_includes_comm(self):
        """Backward compatibility: after a full sleep, a legacy client resuming
        with the pre-comm triple [weights, kv_cache, cuda_graph] still gets a
        complete wake - comm is auto-included instead of staying destroyed."""
        baseline_mb = _device_used_mb()
        response = _release(self.base_url)
        self.assertEqual(response.status_code, 200, response.text)

        response = _resume(self.base_url, ["weights", "kv_cache", "cuda_graph"])
        self.assertEqual(response.status_code, 200, response.text)

        # Proves comm really was auto-resumed: the device is back at the
        # pre-sleep baseline (the HCCL buffers are allocated again) and
        # generation works, which needs the rebuilt comms.
        _assert_mem_near_baseline(self, baseline_mb, "legacy wake")
        self.assertIn("Paris", _generate_greedy(self, self.base_url))

    def test_4b_staged_legacy_wake_completes_on_the_last_stage(self):
        """A client that wakes in stages ([weights] then [kv_cache]) must not be
        rejected: comm is rebuilt on the stage that completes the dependency
        set, because recapturing graphs needs weights and the KV cache mapped."""
        baseline_mb = _device_used_mb()
        response = _release(self.base_url)
        self.assertEqual(response.status_code, 200, response.text)

        response = _resume(self.base_url, ["weights"])
        self.assertEqual(response.status_code, 200, response.text)
        # Still sleeping: comm must not be rebuilt yet, so the HCCL buffers
        # stay freed. Naming comm now is refused in the scheduler (log-only,
        # HTTP 200) and changes nothing.
        response = _resume(self.base_url, ["comm"])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertGreater(
            baseline_mb - _device_used_mb(),
            COMM_RELEASE_MIN_DELTA_MB,
            "comm buffers must stay freed until the dependency set completes",
        )

        response = _resume(self.base_url, ["kv_cache"])
        self.assertEqual(response.status_code, 200, response.text)
        _assert_mem_near_baseline(self, baseline_mb, "staged wake")
        _assert_generate_works(self, self.base_url)

    def test_5_resume_comm_missing_dependency_is_refused(self):
        """Resuming comm while kv_cache is still offloaded would recapture
        graphs on unmapped memory; the request must be refused in the scheduler
        (log-only, HTTP 200), changing nothing, so a proper resume still works."""
        response = _release(self.base_url)
        self.assertEqual(response.status_code, 200, response.text)
        try:
            used_before_mb = _device_used_mb()
            response = _resume(self.base_url, ["comm", "weights"])
            self.assertEqual(response.status_code, 200, response.text)
            self.assertLessEqual(
                abs(_device_used_mb() - used_before_mb),
                NO_CHANGE_TOL_MB,
                "the refused resume must leave every tag offloaded",
            )
        finally:
            response = _resume(self.base_url)
            self.assertEqual(response.status_code, 200, response.text)

        _assert_generate_works(self, self.base_url)

    def test_6_double_release_is_idempotent(self):
        """Releasing the same tag twice is a safe no-op (the second release
        finds the comms already gone), not a double destroy."""
        for _ in range(2):
            response = _release(self.base_url, ["comm"])
            self.assertEqual(response.status_code, 200, response.text)

        response = _resume(self.base_url, ["comm"])
        self.assertEqual(response.status_code, 200, response.text)

        _assert_generate_works(self, self.base_url)

    def test_7_repeated_cycles_return_to_baseline(self):
        """Three full sleep/wake cycles: memory returns to the baseline band
        every time (no HCCL handle or graph pool leak, graphs really
        recaptured), and inference stays correct."""
        _assert_generate_works(self, self.base_url)
        baseline_mb = _device_used_mb()

        for cycle in range(3):
            response = _release(self.base_url)
            self.assertEqual(response.status_code, 200, response.text)
            response = _resume(self.base_url)
            self.assertEqual(response.status_code, 200, response.text)
            _assert_mem_near_baseline(self, baseline_mb, f"cycle {cycle + 1}")
            _assert_generate_works(self, self.base_url)

    def test_8_cuda_graph_tag_is_cleared_after_comm_resume(self):
        """cuda_graph is destroyed + recaptured (never memory-saver paused) when
        comm is involved, so after a comm resume it must not linger in the
        offloaded state: the graphs are already live, hence a later
        cuda_graph-only resume is a refused no-op that changes nothing."""
        response = _release(self.base_url)
        self.assertEqual(response.status_code, 200, response.text)
        response = _resume(self.base_url, ["weights", "kv_cache", "comm"])
        self.assertEqual(response.status_code, 200, response.text)

        used_before_mb = _device_used_mb()
        response = _resume(self.base_url, ["cuda_graph"])
        self.assertEqual(response.status_code, 200, response.text)
        self.assertLessEqual(
            abs(_device_used_mb() - used_before_mb),
            NO_CHANGE_TOL_MB,
            "resuming an already-live cuda_graph tag must change no state",
        )

        _assert_generate_works(self, self.base_url)


class TestSleepCommCleanupNoGraph(CustomTestCase):
    """comm-only sleep/wake with graphs disabled: the memory released by the
    comm tag alone (no graph memory to drop)."""

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
                "--mem-fraction-static",
                "0.8",
                "--enable-memory-saver",
                "--disable-cuda-graph",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_comm_only_sleep_frees_hccl_buffers(self):
        _assert_generate_works(self, self.base_url)
        used_before_mb = _device_used_mb()
        response = _release(self.base_url, ["comm"])
        self.assertEqual(response.status_code, 200, response.text)
        used_released_mb = _device_used_mb()
        self.assertGreater(
            used_before_mb - used_released_mb,
            COMM_RELEASE_MIN_DELTA_MB,
            f"comm sleep without graphs must still free HCCL buffers "
            f"(before={used_before_mb:.0f}MB, after={used_released_mb:.0f}MB)",
        )

        response = _resume(self.base_url, ["comm"])
        self.assertEqual(response.status_code, 200, response.text)
        _assert_generate_works(self, self.base_url)


if __name__ == "__main__":
    unittest.main()
