"""
Unit tests for Engine.get_all_child_pids().

Verifies that launching an Engine exposes the PIDs of all child processes
(schedulers, detokenizer) and that those PIDs correspond to live processes.

Usage:
    python -m unittest test_engine_child_pids -v
"""

import os
import unittest

import psutil

import sglang as sgl
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import (
    DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
    CustomTestCase,
)

register_cuda_ci(est_time=84, suite="stage-b-test-1-gpu-small")


class TestEngineChildPids(CustomTestCase):

    def test_get_all_child_pids_returns_live_pids(self):
        engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            random_seed=42,
        )
        try:
            pids = engine.get_all_child_pids()

            self.assertIsInstance(pids, list)
            self.assertGreater(len(pids), 0, "Expected at least one child PID")

            for pid in pids:
                self.assertIsInstance(pid, int)
                self.assertTrue(
                    psutil.pid_exists(pid),
                    f"PID {pid} does not correspond to a running process",
                )

            current_proc = psutil.Process(os.getpid())
            child_pids = {c.pid for c in current_proc.children(recursive=True)}
            for pid in pids:
                self.assertIn(
                    pid,
                    child_pids,
                    f"PID {pid} is not a child of the current process",
                )
        finally:
            engine.shutdown()

    def test_child_pids_include_scheduler_and_detokenizer(self):
        engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            random_seed=42,
        )
        try:
            pids = engine.get_all_child_pids()
            # dp_size=1 gives one scheduler + one detokenizer = at least 2 PIDs
            self.assertGreaterEqual(
                len(pids),
                2,
                "Expected at least 2 child PIDs (scheduler + detokenizer)",
            )
        finally:
            engine.shutdown()

    def test_child_pids_no_duplicates(self):
        engine = sgl.Engine(
            model_path=DEFAULT_SMALL_MODEL_NAME_FOR_TEST,
            random_seed=42,
        )
        try:
            pids = engine.get_all_child_pids()
            self.assertEqual(
                len(pids),
                len(set(pids)),
                f"Duplicate PIDs found: {pids}",
            )
        finally:
            engine.shutdown()


if __name__ == "__main__":
    unittest.main()
