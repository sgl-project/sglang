"""NPU CI test for sleep-mode comm cleanup (the ``comm`` memory-occupation tag).

Launches tp=2 servers with the memory saver enabled and drives the
/release_memory_occupation and /resume_memory_occupation endpoints:

  * ``comm``-only sleep must free HCCL communicator device buffers
    (destroy_process_group -> hcclCommDestroy) and the matching wake must
    rebuild the communicators and recapture the graphs so inference keeps
    working;
  * a full (tag-less) sleep includes ``comm`` on NPU; greedy output must be
    bit-identical before and after the cycle (weights CPU backup + recaptured
    graphs + rebuilt comms);
  * legacy clients resuming with the pre-comm triple
    [weights, kv_cache, cuda_graph] still get a complete wake (comm is
    auto-included);
  * invalid resumes (unreleased tag, missing still-offloaded dependencies)
    are rejected with HTTP 400 while the server stays alive;
  * repeated sleep/wake cycles return device memory to the baseline band
    (no leak, graphs really recaptured — an eager fallback would leave
    memory well below baseline);
  * with graphs disabled, the comm-only delta isolates pure HCCL buffers.

Not covered here (see PR description for the manual checklist): DeepEP/MoE
dispatcher buffer rebuild, MindSpore gate, adaptive-spec and dflash/dspark
hook idempotency, cross-rank recapture-failure vote.

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
# A comm-only sleep destroys every device-side HCCL communicator of the
# server. Each communicator holds HCCL_BUFFSIZE (default ~200MB) of device
# buffers and even a dense tp=2 server owns several of them (tp, world, ...),
# so per-card usage must drop well above this floor.
COMM_RELEASE_MIN_DELTA_MB = 100
# A tag-less sleep additionally pauses weights + kv_cache, so the drop is far
# larger; keep a conservative floor.
FULL_RELEASE_MIN_DELTA_MB = 500
# After a successful wake the device usage must return near the pre-sleep
# baseline: recaptured graphs + restored weights re-occupy their memory. If
# recapture silently failed (eager fallback) usage would stay well below the
# band; a leak would push it above.
WAKE_BASELINE_TOL_MB = 400

_GREEDY_PROMPT = "Where is the capital of France?"


def _device_used_mb():
    # Device-wide usage (like npu-smi), so it covers the server processes too.
    free, total = torch.npu.mem_get_info()
    return (total - free) / _MIB


def _post(base_url, endpoint, payload=None):
    # Timeout matters: a regression that kills or wedges the scheduler must
    # fail the test fast instead of hanging the CI job forever.
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
        # Graph capture stays enabled on purpose: the comm tag implies graph
        # destruction on sleep and recapture on wake, which this test
        # exercises end to end.
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
                # Required for test_2: without the CPU backup a full sleep
                # DISCARDS the weights and a wake cannot produce correct
                # output (that is the documented RL flow: reload weights
                # after wake). With the backup, weights survive the cycle.
                "--enable-weights-cpu-backup",
                "--disable-radix-cache",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_1_comm_only_sleep_wake(self):
        """comm tag: communicators are destroyed on sleep and rebuilt on wake
        (with graph recapture), and inference is correct again. The memory
        delta here intentionally includes dropped graph memory; the pure
        HCCL share is isolated in TestSleepCommCleanupNoGraph."""
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
