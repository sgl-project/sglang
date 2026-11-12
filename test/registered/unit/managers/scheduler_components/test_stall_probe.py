# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
import gc
import logging
import time
import unittest

from sglang.srt.environ import envs
from sglang.srt.managers.scheduler_components import stall_probe
from sglang.srt.managers.scheduler_components.stall_probe import StallProbe
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


class _Node:
    def __init__(self):
        self.peer = self


def _make_cycles(n):
    for _ in range(n):
        _Node()


class TestStallProbe(unittest.TestCase):
    """The probe must report GC passes and main-thread stalls at least as long as its threshold, with their
    measured length, once per stall, and nothing below the threshold."""

    def _probe(self, **kw):
        probe = StallProbe(**kw)
        self.addCleanup(probe.close)
        return probe

    def test_gc_pass_reported_with_its_duration(self):
        self._probe(gc_threshold_s=0.001, stack_threshold_s=0)
        gc.collect()
        gc.disable()  # one collection, the explicit one below
        self.addCleanup(gc.enable)
        _make_cycles(300_000)
        with self.assertLogs(stall_probe.logger, logging.WARNING) as logs:
            start = time.perf_counter()
            gc.collect(2)
            outer_ms = (time.perf_counter() - start) * 1e3
        lines = [l for l in logs.output if "GC stall: generation 2 took" in l]
        self.assertEqual(len(lines), 1, logs.output)
        took_ms = float(lines[0].split(" took ")[1].split(" ms")[0])
        self.assertGreaterEqual(took_ms, 1.0)
        self.assertLessEqual(took_ms, outer_ms + 0.05)  # the pass sits inside the outer window (log rounds to 0.1)

    def test_gc_pass_below_threshold_not_reported(self):
        self._probe(gc_threshold_s=10.0, stack_threshold_s=0)
        _make_cycles(10_000)
        with self.assertNoLogs(stall_probe.logger, logging.WARNING):
            gc.collect(2)

    def test_main_thread_stall_reported_once_per_stall(self):
        self._probe(gc_threshold_s=0, stack_threshold_s=0.05)
        with self.assertLogs(stall_probe.logger, logging.WARNING) as logs:
            time.sleep(0.4)  # first stall
            marker = 1  # the position changes between the stalls
            time.sleep(0.4)  # second stall
            time.sleep(0.02)  # let the sampler finish its last check
        stalls = [l for l in logs.output if "Main-thread stall" in l]
        self.assertEqual(len(stalls), 2, logs.output)
        self.assertTrue(all("time.sleep(0.4)" in l for l in stalls), stalls)
        self.assertEqual(marker, 1)

    def test_short_blocking_not_reported(self):
        # Several samples see the thread at one position, for less than the threshold.
        self._probe(gc_threshold_s=0, stack_threshold_s=0.4)
        with self.assertNoLogs(stall_probe.logger, logging.WARNING):
            time.sleep(0.25)

    def test_env_selects_the_probes(self):
        with envs.SGLANG_LOG_GC_STALL_MS.override(0), envs.SGLANG_LOG_STALL_STACK_MS.override(0):
            self.assertIsNone(StallProbe.maybe_create())
        before = list(gc.callbacks)
        with envs.SGLANG_LOG_GC_STALL_MS.override(7), envs.SGLANG_LOG_STALL_STACK_MS.override(0):
            probe = StallProbe.maybe_create()
        self.addCleanup(probe.close)
        self.assertEqual(probe._gc_threshold_s, 0.007)
        self.assertIsNone(probe._thread)
        self.assertEqual(len(gc.callbacks), len(before) + 1)

    def test_close_removes_the_gc_callback(self):
        before = list(gc.callbacks)
        probe = StallProbe(gc_threshold_s=0.001, stack_threshold_s=0.05)
        probe.close()
        self.assertEqual(gc.callbacks, before)
        self.assertFalse(probe._thread.is_alive())


if __name__ == "__main__":
    unittest.main()
