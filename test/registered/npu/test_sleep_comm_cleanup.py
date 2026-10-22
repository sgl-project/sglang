"""NPU CI test for sleep-mode comm cleanup (the ``comm`` tag).

Drives /release_memory_occupation and /resume_memory_occupation on tp=2
servers: comm-only and full sleep/wake, legacy resume, invalid resumes (400)
and repeated cycles (no leak, graphs really recaptured).

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
# Each HCCL communicator holds ~200MB of device buffers, so a comm-only sleep
# must free well above this floor.
COMM_RELEASE_MIN_DELTA_MB = 100
# A tag-less sleep also pauses weights + kv_cache; keep a conservative floor.
FULL_RELEASE_MIN_DELTA_MB = 500
# After wake, usage should return near the pre-sleep baseline: far below means
# recapture silently fell back to eager, far above means a leak.
WAKE_BASELINE_TOL_MB = 400

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
    text = _generate_greedy(test_case, base_url)
    test_case.assertIn("Paris", text)


def _assert_mem_near_baseline(test_case, baseline_mb, what):
    used_mb = _device_used_mb()
    test_case.assertLessEqual(
        abs(used_mb - baseline_mb),
        WAKE_BASELINE_TOL_MB,
        f"{what}: device usage {used_mb:.0f}MB drifted from baseline "
        f"{baseline_mb:.0f}MB by more than {WAKE_BASELINE_TOL_MB}MB",
    )


class TestSleepCommCleanup(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        # Graph capture stays enabled: the comm tag implies drop + recapture.
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
        """comm tag: comms are destroyed on sleep, rebuilt on wake, and
        inference is correct again. The delta also includes dropped graphs; the
        pure HCCL share is isolated in TestSleepCommCleanupNoGraph."""
        _assert_generate_works(self, self.base_url)

        used_before_mb = _device_used_mb()
        response = _post(
            self.base_url, "/release_memory_occupation", {"tags": ["comm"]}
        )
        self.assertEqual(response.status_code, 200, response.text)
        used_released_mb = _device_used_mb()
        self.assertGreater(
            used_before_mb - used_released_mb,
            COMM_RELEASE_MIN_DELTA_MB,
            f"comm sleep should free HCCL buffers "
            f"(before={used_before_mb:.0f}MB, after={used_released_mb:.0f}MB)",
        )

        response = _post(self.base_url, "/resume_memory_occupation", {"tags": ["comm"]})
        self.assertEqual(response.status_code, 200, response.text)

        # Rebuilt communicators + recaptured graphs must serve real traffic.
        _assert_generate_works(self, self.base_url)

    def test_2_full_sleep_wake_includes_comm(self):
        """A tag-less sleep defaults to all tags, which on NPU includes comm.
        Weights come back from the CPU backup and greedy decoding must be
        bit-identical before and after the cycle."""
        text_before = _generate_greedy(self, self.base_url)

        used_before_mb = _device_used_mb()
        response = _post(self.base_url, "/release_memory_occupation")
        self.assertEqual(response.status_code, 200, response.text)
        used_released_mb = _device_used_mb()
        self.assertGreater(
            used_before_mb - used_released_mb,
            FULL_RELEASE_MIN_DELTA_MB,
            f"full sleep should free weights+kv+graphs+comms "
            f"(before={used_before_mb:.0f}MB, after={used_released_mb:.0f}MB)",
        )

        response = _post(self.base_url, "/resume_memory_occupation")
        self.assertEqual(response.status_code, 200, response.text)
        _assert_mem_near_baseline(self, used_before_mb, "full sleep/wake")

        text_after = _generate_greedy(self, self.base_url)
        self.assertEqual(text_before, text_after)

    def test_3_resume_of_unreleased_tag_fails(self):
        """A resume naming a tag that was never released is rejected with a
        400 (not a dead server), and the server keeps serving afterwards."""
        response = _post(
            self.base_url, "/release_memory_occupation", {"tags": ["comm"]}
        )
        self.assertEqual(response.status_code, 200, response.text)
        try:
            response = _post(
                self.base_url,
                "/resume_memory_occupation",
                {"tags": ["weights"]},
            )
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn("never released", response.text)
        finally:
            # Restore the server to a fully awake state for later tests.
            response = _post(
                self.base_url, "/resume_memory_occupation", {"tags": ["comm"]}
            )
            self.assertEqual(response.status_code, 200, response.text)

        _assert_generate_works(self, self.base_url)

    def test_4_legacy_triple_resume_auto_includes_comm(self):
        """Backward compatibility: after a full sleep, a legacy client
        resuming with the pre-comm triple [weights, kv_cache, cuda_graph]
        still gets a complete wake — comm is auto-included rather than
        leaving the communicators destroyed."""
        response = _post(self.base_url, "/release_memory_occupation")
        self.assertEqual(response.status_code, 200, response.text)

        response = _post(
            self.base_url,
            "/resume_memory_occupation",
            {"tags": ["weights", "kv_cache", "cuda_graph"]},
        )
        self.assertEqual(response.status_code, 200, response.text)

        text_after = _generate_greedy(self, self.base_url)
        self.assertIn("Paris", text_after)

    def test_5_resume_missing_dependency_fails(self):
        """Resuming comm while kv_cache (or weights) is still offloaded would
        recapture graphs on unmapped memory; it must be rejected with a 400
        that names the missing dependency, leaving state intact for a proper
        full resume afterwards."""
        response = _post(self.base_url, "/release_memory_occupation")
        self.assertEqual(response.status_code, 200, response.text)
        try:
            response = _post(
                self.base_url,
                "/resume_memory_occupation",
                {"tags": ["comm", "weights"]},
            )
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn("kv_cache", response.text)
        finally:
            response = _post(self.base_url, "/resume_memory_occupation")
            self.assertEqual(response.status_code, 200, response.text)

        _assert_generate_works(self, self.base_url)

    def test_6_double_release_is_idempotent(self):
        """Releasing the same tag twice is a safe no-op (second release finds
        the communicators already gone), not a double-destroy."""
        for _ in range(2):
            response = _post(
                self.base_url, "/release_memory_occupation", {"tags": ["comm"]}
            )
            self.assertEqual(response.status_code, 200, response.text)

        response = _post(self.base_url, "/resume_memory_occupation", {"tags": ["comm"]})
        self.assertEqual(response.status_code, 200, response.text)

        _assert_generate_works(self, self.base_url)

    def test_7_repeated_cycles_return_to_baseline(self):
        """Three full sleep/wake cycles: memory returns to the baseline band
        every time (no HCCL handle/graph pool leak, graphs really
        recaptured), and inference stays correct."""
        _assert_generate_works(self, self.base_url)
        baseline_mb = _device_used_mb()

        for cycle in range(3):
            response = _post(self.base_url, "/release_memory_occupation")
            self.assertEqual(response.status_code, 200, response.text)
            response = _post(self.base_url, "/resume_memory_occupation")
            self.assertEqual(response.status_code, 200, response.text)
            _assert_mem_near_baseline(self, baseline_mb, f"cycle {cycle + 1}")
            _assert_generate_works(self, self.base_url)

    def test_8_cuda_graph_tag_is_cleared_after_comm_resume(self):
        """cuda_graph is destroyed+recaptured (never TMS-paused) when comm is
        involved, so after a comm resume it must not linger in offload state:
        a later cuda_graph-only resume is a 400 'never released', and the
        server keeps serving."""
        response = _post(self.base_url, "/release_memory_occupation")
        self.assertEqual(response.status_code, 200, response.text)
        response = _post(
            self.base_url,
            "/resume_memory_occupation",
            {"tags": ["weights", "kv_cache", "comm"]},
        )
        self.assertEqual(response.status_code, 200, response.text)

        response = _post(
            self.base_url,
            "/resume_memory_occupation",
            {"tags": ["cuda_graph"]},
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("never released", response.text)

        _assert_generate_works(self, self.base_url)


class TestSleepCommCleanupNoGraph(CustomTestCase):
    """comm-only sleep/wake with graphs disabled: the released memory can
    only come from HCCL communicator buffers, isolating the comm effect from
    graph memory (which test_1 intentionally includes)."""

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
                "--disable-radix-cache",
                "--disable-cuda-graph",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_comm_only_sleep_frees_hccl_buffers(self):
        _assert_generate_works(self, self.base_url)

        used_before_mb = _device_used_mb()
        response = _post(
            self.base_url, "/release_memory_occupation", {"tags": ["comm"]}
        )
        self.assertEqual(response.status_code, 200, response.text)
        used_released_mb = _device_used_mb()
        self.assertGreater(
            used_before_mb - used_released_mb,
            COMM_RELEASE_MIN_DELTA_MB,
            f"with graphs disabled, comm sleep must free HCCL buffers "
            f"(before={used_before_mb:.0f}MB, after={used_released_mb:.0f}MB)",
        )

        response = _post(self.base_url, "/resume_memory_occupation", {"tags": ["comm"]})
        self.assertEqual(response.status_code, 200, response.text)

        _assert_generate_works(self, self.base_url)


if __name__ == "__main__":
    unittest.main()
