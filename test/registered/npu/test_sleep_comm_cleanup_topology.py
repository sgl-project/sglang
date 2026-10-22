"""NPU CI test for sleep-mode comm cleanup on multi-subgroup topologies.

The destroy/rebuild of device groups must replay the FULL group_ranks list on
every rank in the original order (torch's new_group is an all-rank contract
and group names come from a process-level counter). These topologies have
disjoint non-trivial subgroups — exactly where a naive "rebuild only my own
subgroup" implementation deadlocks or cross-wires HCCL root info:

  * tp=2 + pp=2: tp subgroups [0,1]/[2,3] and pp subgroups [0,2]/[1,3];
  * dp-attention (dp=2, tp=2): attn_tp / attn_dp subgroups, and the sampler's
    attn-tp sync path.

Each class does a full sleep/wake and then requires serving to work again;
a broken rebuild hangs inside the wake or the first post-wake collective,
which the request timeout turns into a fast failure.

Pipeline parallelism cannot run the comm cleanup itself (its world-group
collectives are unreachable from a PP scheduler while the request is still
being relayed), so the PP class asserts the explicit rejection instead of a
comm-only sleep/wake cycle.

Usage:
python3 -m unittest test_sleep_comm_cleanup_topology
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
    # Two server launches (pp class + dp-attention class).
    est_time=1600,
    suite="full-4-npu-a3",
    nightly=True,
)

if "ASCEND_RT_VISIBLE_DEVICES" not in os.environ:
    os.environ["ASCEND_RT_VISIBLE_DEVICES"] = "0,1,2,3"

_MIB = 1 << 20
COMM_RELEASE_MIN_DELTA_MB = 100
FULL_RELEASE_MIN_DELTA_MB = 500
WAKE_BASELINE_TOL_MB = 400

_COMMON_ARGS = [
    "--trust-remote-code",
    "--device",
    "npu",
    "--attention-backend",
    "ascend",
    "--mem-fraction-static",
    "0.8",
    "--enable-memory-saver",
    "--enable-weights-cpu-backup",
    "--disable-radix-cache",
]


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
            "text": "Where is the capital of France?",
            "sampling_params": {"temperature": 0, "max_new_tokens": 32},
        },
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


class _SleepWakeTopologyMixin:
    """Shared full-cycle checks for one server configuration."""

    # Whether this deployment accepts the "comm" tag. PP deployments reject an
    # explicit "comm" (world-group collectives are unreachable from a PP
    # scheduler) and drop it from a tag-less request.
    comm_supported = True

    def test_full_sleep_wake_and_repeat(self):
        _assert_generate_works(self, self.base_url)
        baseline_mb = _device_used_mb()

        for cycle in range(2):
            used_before_mb = _device_used_mb()
            response = _post(self.base_url, "/release_memory_occupation")
            self.assertEqual(response.status_code, 200, response.text)
            used_released_mb = _device_used_mb()
            self.assertGreater(
                used_before_mb - used_released_mb,
                FULL_RELEASE_MIN_DELTA_MB,
                f"cycle {cycle + 1}: full sleep should free memory "
                f"(before={used_before_mb:.0f}MB, after={used_released_mb:.0f}MB)",
            )

            response = _post(self.base_url, "/resume_memory_occupation")
            self.assertEqual(response.status_code, 200, response.text)
            _assert_mem_near_baseline(self, baseline_mb, f"cycle {cycle + 1}")
            _assert_generate_works(self, self.base_url)

    def test_comm_only_sleep_wake(self):
        text_before = _generate_greedy(self, self.base_url)

        used_before_mb = _device_used_mb()
        response = _post(
            self.base_url, "/release_memory_occupation", {"tags": ["comm"]}
        )
        if not self.comm_supported:
            # Must be a clean rejection (400), not a hang or a dead server:
            # nothing is released, so memory and greedy output stay untouched.
            self.assertEqual(response.status_code, 400, response.text)
            self.assertIn("pipeline parallelism", response.text)
            _assert_mem_near_baseline(self, used_before_mb, "rejected comm release")
            text_after = _generate_greedy(self, self.base_url)
            self.assertEqual(text_before, text_after)
            return

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

        # Rebuilt subgroups must be functionally identical (greedy decode).
        text_after = _generate_greedy(self, self.base_url)
        self.assertEqual(text_before, text_after)


class TestSleepCommCleanupPP(_SleepWakeTopologyMixin, CustomTestCase):
    """tp=2 pp=2: disjoint TP subgroups per stage and disjoint PP subgroups —
    the topology that exercises the full group_ranks replay most directly."""

    # Pipeline parallelism does not support the comm tag (see the module
    # docstring), so the comm case asserts the rejection instead.
    comm_supported = False

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            QWEN3_0_6B_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *_COMMON_ARGS,
                "--tp-size",
                "2",
                "--pp-size",
                "2",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)

    def test_explicit_comm_with_other_tags_rejects_before_any_release(self):
        """An explicit "comm" rejects the WHOLE request on PP, and the rejection
        happens before any tag is acted on: weights/kv stay mapped, so memory
        does not move and the server keeps serving."""
        text_before = _generate_greedy(self, self.base_url)
        used_before_mb = _device_used_mb()

        response = _post(
            self.base_url,
            "/release_memory_occupation",
            {"tags": ["weights", "kv_cache", "comm"]},
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("pipeline parallelism", response.text)

        _assert_mem_near_baseline(self, used_before_mb, "rejected mixed-tag release")
        self.assertEqual(text_before, _generate_greedy(self, self.base_url))

    def test_resume_comm_is_rejected_under_pp(self):
        """The resume side applies the same gate: naming "comm" (which PP never
        released) is a 400 with the PP reason, not a hang or a state change."""
        text_before = _generate_greedy(self, self.base_url)

        response = _post(
            self.base_url, "/resume_memory_occupation", {"tags": ["comm"]}
        )
        self.assertEqual(response.status_code, 400, response.text)
        self.assertIn("pipeline parallelism", response.text)

        self.assertEqual(text_before, _generate_greedy(self, self.base_url))


class TestSleepCommCleanupDPAttention(_SleepWakeTopologyMixin, CustomTestCase):
    """dp=2 tp=2 with dp-attention: attn_tp/attn_dp subgroups plus the
    sampler's attn-tp sync group after a comm rebuild."""

    @classmethod
    def setUpClass(cls):
        cls.base_url = DEFAULT_URL_FOR_TEST
        cls.process = popen_launch_server(
            QWEN3_0_6B_WEIGHTS_PATH,
            cls.base_url,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=[
                *_COMMON_ARGS,
                "--tp-size",
                "2",
                "--dp-size",
                "2",
                "--enable-dp-attention",
            ],
        )

    @classmethod
    def tearDownClass(cls):
        kill_process_tree(cls.process.pid)


if __name__ == "__main__":
    unittest.main()
