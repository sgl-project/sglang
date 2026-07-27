import threading
import time
import unittest

import requests

from sglang.srt.utils import kill_process_tree
from sglang.test.ascend.test_ascend_utils import LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH
from sglang.test.ci.ci_register import register_npu_ci
from sglang.test.test_utils import (
    DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
    DEFAULT_URL_FOR_TEST,
    CustomTestCase,
    popen_launch_server,
)

register_npu_ci(est_time=200, suite="full-1-npu-a3", nightly=True)


class TestDefaultPriorityValue(CustomTestCase):
    """--default-priority-value A/B comparison — high-default vs low-default.

    [Test Category] Parameter
    [Test Target] --default-priority-value
    [Scenario] D1 (default priority value affects scheduling behaviour)
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    _BASE_ARGS = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--enable-priority-scheduling",
        "--enable-metrics",
        "--max-running-requests",
        "1",
        "--priority-scheduling-preemption-threshold",
        "0",
    ]

    @staticmethod
    def _run_scenario(base_url, no_priority_first=True):
        """Run: no-priority request vs explicit-priority request.

        Returns (no_priority_finished_at, explicit_finished_at).
        """
        no_pri_result = {}
        expl_result = {}

        def _send_no_priority():
            resp = requests.post(
                f"{base_url}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 256},
                },
                timeout=120,
            )
            no_pri_result["finished_at"] = time.time()
            no_pri_result["status"] = resp.status_code

        def _send_explicit():
            resp = requests.post(
                f"{base_url}/generate",
                json={
                    "text": "What is 1+1? Answer:",
                    "sampling_params": {"temperature": 0, "max_new_tokens": 16},
                    "priority": 5,
                },
                timeout=120,
            )
            expl_result["finished_at"] = time.time()
            expl_result["status"] = resp.status_code

        if no_priority_first:
            first_thread = threading.Thread(target=_send_no_priority, daemon=True)
            first_thread.start()
            time.sleep(3)
            _send_explicit()
            first_thread.join(timeout=120)
            assert not first_thread.is_alive(), "No-priority request timed out"
        else:
            first_thread = threading.Thread(target=_send_explicit, daemon=True)
            first_thread.start()
            time.sleep(3)
            _send_no_priority()
            first_thread.join(timeout=120)
            assert not first_thread.is_alive(), "Explicit request timed out"

        assert (
            no_pri_result.get("status") == 200
        ), f"No-priority failed: {no_pri_result.get('status')}"
        assert (
            expl_result.get("status") == 200
        ), f"Explicit failed: {expl_result.get('status')}"

        return no_pri_result["finished_at"], expl_result["finished_at"]

    def test_default_priority_value_affects_ordering(self):
        """Requests without explicit priority are scheduled at the default value.

        Two servers, everything identical except --default-priority-value:

          1. --default-priority-value 10 → no-priority request gets default=10,
             priority=5 request has lower priority → no-priority finishes first.
          2. --default-priority-value 0  → no-priority request gets default=0,
             priority=5 request has higher priority → explicit finishes first.

        The ordering flips solely because of --default-priority-value.
        """
        # ── Round 1: default=10 → no-priority (10) > explicit (5) ──
        process1 = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=list(self._BASE_ARGS) + ["--default-priority-value", "10"],
        )
        try:
            no_pri_high, expl_low = self._run_scenario(
                DEFAULT_URL_FOR_TEST, no_priority_first=True
            )
        finally:
            kill_process_tree(process1.pid)

        self.assertLess(
            no_pri_high,
            expl_low,
            f"default=10 (no-priority) should finish before priority=5, "
            f"but no-priority={no_pri_high:.1f} explicit={expl_low:.1f}",
        )

        # ── Round 2: default=0 → no-priority (0) < explicit (5) ──
        process2 = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=list(self._BASE_ARGS) + ["--default-priority-value", "0"],
        )
        try:
            no_pri_low, expl_high = self._run_scenario(
                DEFAULT_URL_FOR_TEST, no_priority_first=True
            )
        finally:
            kill_process_tree(process2.pid)

        self.assertLess(
            expl_high,
            no_pri_low,
            f"priority=5 should finish before default=0 (no-priority), "
            f"but explicit={expl_high:.1f} no-priority={no_pri_low:.1f}",
        )

        print(
            f"  [default=10] no-pri={no_pri_high:.2f} explicit(5)={expl_low:.2f} "
            f"→ no_pri_first={no_pri_high < expl_low}"
        )
        print(
            f"  [default=0]  no-pri={no_pri_low:.2f} explicit(5)={expl_high:.2f} "
            f"→ explicit_first={expl_high < no_pri_low}"
        )


class TestDisablePriorityPreemption(CustomTestCase):
    """--disable-priority-preemption A/B comparison — with vs without.

    [Test Category] Parameter
    [Test Target] --disable-priority-preemption
    [Scenario] P1 (preemption enabled → high first) vs P2 (preemption disabled → low first)
    """

    model = LLAMA_3_2_1B_INSTRUCT_WEIGHTS_PATH

    _BASE_ARGS = [
        "--attention-backend",
        "ascend",
        "--disable-cuda-graph",
        "--enable-priority-scheduling",
        "--default-priority-value",
        "0",
        "--enable-metrics",
        "--max-running-requests",
        "1",
        "--priority-scheduling-preemption-threshold",
        "0",
    ]

    @staticmethod
    def _run_preemption_scenario(base_url):
        """Run the high-vs-low priority scenario on *base_url*, return timings.

        Returns (low_finished_at, high_finished_at).
        """
        low_result = {}

        def _send_low_priority():
            resp = requests.post(
                f"{base_url}/generate",
                json={
                    "text": "The capital of France is",
                    "sampling_params": {
                        "temperature": 0,
                        "max_new_tokens": 256,
                    },
                    "priority": 0,
                },
                timeout=120,
            )
            low_result["status"] = resp.status_code
            low_result["finished_at"] = time.time()

        t = threading.Thread(target=_send_low_priority, daemon=True)
        t.start()
        time.sleep(3)

        resp = requests.post(
            f"{base_url}/generate",
            json={
                "text": "What is 1+1? Answer:",
                "sampling_params": {
                    "temperature": 0,
                    "max_new_tokens": 16,
                },
                "priority": 10,
            },
            timeout=120,
        )
        high_finished_at = time.time()
        assert (
            resp.status_code == 200
        ), f"High-priority request failed: {resp.status_code}"

        t.join(timeout=120)
        assert not t.is_alive(), "Low-priority request timed out"
        assert (
            low_result.get("status") == 200
        ), f"Low-priority request failed: {low_result.get('status')}"

        return low_result["finished_at"], high_finished_at

    def test_disable_preemption_changes_ordering(self):
        """With --disable-priority-preemption, low finishes first; without, high finishes first.

        Runs the same scenario twice, only changing the flag:

          1. With the flag → low-priority finishes BEFORE high-priority.
          2. Without the flag → high-priority finishes BEFORE low-priority.
        """
        # ── Round 1: preemption ENABLED ──
        process1 = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=list(self._BASE_ARGS),
        )
        try:
            low_enabled, high_enabled = self._run_preemption_scenario(
                DEFAULT_URL_FOR_TEST
            )
        finally:
            kill_process_tree(process1.pid)

        # ── Round 2: preemption DISABLED ──
        process2 = popen_launch_server(
            self.model,
            DEFAULT_URL_FOR_TEST,
            timeout=DEFAULT_TIMEOUT_FOR_SERVER_LAUNCH,
            other_args=list(self._BASE_ARGS) + ["--disable-priority-preemption"],
        )
        try:
            low_disabled, high_disabled = self._run_preemption_scenario(
                DEFAULT_URL_FOR_TEST
            )
        finally:
            kill_process_tree(process2.pid)

        print(
            f"  [preemption enabled]  high={high_enabled:.2f} low={low_enabled:.2f} "
            f"→ high_first={high_enabled < low_enabled}"
        )
        print(
            f"  [preemption disabled] low={low_disabled:.2f} high={high_disabled:.2f} "
            f"→ low_first={low_disabled < high_disabled}"
        )
        self.assertLess(
            high_enabled,
            low_enabled,
            f"Preemption ENABLED: expected high-priority to finish before low, "
            f"but high={high_enabled:.1f} low={low_enabled:.1f}",
        )
        self.assertLess(
            low_disabled,
            high_disabled,
            f"Preemption DISABLED: expected low-priority to finish before high, "
            f"but low={low_disabled:.1f} high={high_disabled:.1f}",
        )


if __name__ == "__main__":
    unittest.main()
