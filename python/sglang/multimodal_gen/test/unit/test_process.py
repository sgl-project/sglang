# SPDX-License-Identifier: Apache-2.0

import contextlib
import gc
import importlib.util
import os
import subprocess
import sys
import textwrap
import unittest
from pathlib import Path
from unittest.mock import patch

import psutil

PROCESS_PATH = Path(__file__).parents[2] / "runtime" / "utils" / "process.py"
SPEC = importlib.util.spec_from_file_location("multimodal_process", PROCESS_PATH)
assert SPEC is not None and SPEC.loader is not None
PROCESS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PROCESS)
kill_process_tree = PROCESS.kill_process_tree
defer_full_gc_for_imports = PROCESS.defer_full_gc_for_imports


@unittest.skipUnless(os.name == "posix", "requires POSIX process semantics")
class TestKillProcessTree(unittest.TestCase):
    def setUp(self):
        self.children = []

    def tearDown(self):
        for process in self.children:
            # let the tree root reap its children before terminating the root
            kill_process_tree(process.pid, include_parent=False)
            kill_process_tree(process.pid)
            process.wait(timeout=10)

    def spawn_child(self):
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(600)"]
        )
        self.children.append(process)
        return process.pid

    def spawn_tree(self):
        script = textwrap.dedent(
            """
            import subprocess
            import sys
            import time

            child = subprocess.Popen(
                [sys.executable, "-c", "import time; time.sleep(600)"]
            )
            print(child.pid, flush=True)
            child.wait()
            time.sleep(600)
            """
        )
        process = subprocess.Popen(
            [sys.executable, "-c", script], stdout=subprocess.PIPE, text=True
        )
        self.children.append(process)
        with process.stdout:
            child_pid = int(process.stdout.readline())
        return process.pid, child_pid

    def test_waits_for_killed_children(self):
        parent_pid, child_pid = self.spawn_tree()
        sibling_pid = self.spawn_child()

        kill_process_tree(parent_pid, include_parent=False)

        self.assertFalse(psutil.pid_exists(child_pid))
        self.assertTrue(psutil.pid_exists(parent_pid))
        self.assertTrue(psutil.pid_exists(sibling_pid))

    def test_skip_pid_keeps_child_alive(self):
        parent_pid, child_pid = self.spawn_tree()
        sibling_pid = self.spawn_child()

        kill_process_tree(parent_pid, include_parent=False, skip_pid=child_pid)

        self.assertTrue(psutil.pid_exists(child_pid))
        self.assertTrue(psutil.pid_exists(parent_pid))
        self.assertTrue(psutil.pid_exists(sibling_pid))

    def test_waits_for_included_parent(self):
        child_pid = self.spawn_child()

        kill_process_tree(child_pid)

        self.assertFalse(psutil.pid_exists(child_pid))

    def test_missing_parent_is_noop(self):
        kill_process_tree(0x7FFFFFFF)


class TestDeferFullGcForImports(unittest.TestCase):
    def full_collections_while_allocating(self, block):
        gc.collect()
        threshold0, threshold1, threshold2 = gc.get_threshold()
        # Enough survivors that the collector's own rule makes a full one due.
        count = len(gc.get_objects()) + 2 * threshold0 * threshold1 * threshold2
        before = gc.get_stats()[2]["collections"]
        with block():
            kept = [[] for _ in range(count)]
            full = gc.get_stats()[2]["collections"] - before
        del kept
        return full

    @unittest.skipIf(
        (3, 14) <= sys.version_info < (3, 14, 5),
        "incremental GC does not use threshold2",
    )
    def test_no_full_collection_inside_and_thresholds_restored(self):
        thresholds = gc.get_threshold()

        self.assertGreater(
            self.full_collections_while_allocating(contextlib.nullcontext), 0
        )
        self.assertEqual(
            self.full_collections_while_allocating(defer_full_gc_for_imports), 0
        )
        with self.assertRaises(ImportError):
            with defer_full_gc_for_imports():
                raise ImportError
        self.assertEqual(gc.get_threshold(), thresholds)

    def test_incremental_gc_is_unchanged(self):
        thresholds = gc.get_threshold()
        for version in ((3, 14, 0), (3, 14, 4)):
            with (
                self.subTest(version=version),
                patch.object(PROCESS.sys, "version_info", version),
            ):
                with defer_full_gc_for_imports():
                    self.assertEqual(gc.get_threshold(), thresholds)
                self.assertEqual(gc.get_threshold(), thresholds)

    def test_nested_exception_preserves_disabled_collector(self):
        thresholds = gc.get_threshold()
        self.addCleanup(gc.enable if gc.isenabled() else gc.disable)
        self.addCleanup(gc.set_threshold, *thresholds)
        gc.disable()
        with defer_full_gc_for_imports():
            deferred = gc.get_threshold()
            with self.assertRaises(ImportError):
                with defer_full_gc_for_imports():
                    raise ImportError
            self.assertEqual(gc.get_threshold(), deferred)
            self.assertFalse(gc.isenabled())
        self.assertEqual(gc.get_threshold(), thresholds)
        self.assertFalse(gc.isenabled())

    @unittest.skipUnless(os.name == "posix", "requires fork")
    def test_fork_restores_child_without_changing_parent(self):
        thresholds = gc.get_threshold()
        with defer_full_gc_for_imports():
            deferred = gc.get_threshold()
            pid = os.fork()
            if pid == 0:
                restored = gc.get_threshold() == thresholds
        if pid == 0:
            with defer_full_gc_for_imports():
                restored &= gc.get_threshold() == deferred
            os._exit(int(not (restored and gc.get_threshold() == thresholds)))
        self.assertEqual(os.waitstatus_to_exitcode(os.waitpid(pid, 0)[1]), 0)
        self.assertEqual(gc.get_threshold(), thresholds)


if __name__ == "__main__":
    unittest.main()
