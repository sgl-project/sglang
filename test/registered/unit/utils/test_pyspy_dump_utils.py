"""Unit tests for srt/utils/cudacore_pyspy_dump_utils.py: which processes a
py-spy dump reaches, how a failed `py-spy dump` is retried, and the
server-wide dump slot."""

import json
import logging
import os
import subprocess
import sys
import time
import unittest
import uuid
from unittest.mock import patch

import psutil

from sglang.srt.utils import cudacore_pyspy_dump_utils as dump_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

_TITLED_CHILD = (
    "import setproctitle, sys, time; "
    "setproctitle.setproctitle(sys.argv[1]); time.sleep(600)"
)


def _spawn_titled(title: str) -> subprocess.Popen:
    """A child whose cmdline starts with `title`, the way a scheduler's does
    after setproctitle."""
    proc = subprocess.Popen([sys.executable, "-c", _TITLED_CHILD, title])
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            if psutil.Process(proc.pid).cmdline()[0] == title:
                return proc
        except (psutil.Error, IndexError):
            pass
        time.sleep(0.05)
    proc.kill()
    proc.wait()
    raise RuntimeError(f"child {proc.pid} never became {title!r}")


# Runs as a child of the test process, titled like a tokenizer worker, so the
# titled children are its siblings and it matches the sglang:: prefix itself.
_WORKER_CODE = """
import json, os, psutil, setproctitle
setproctitle.setproctitle("sglang::tokenizer_worker:test")
from sglang.srt.utils import cudacore_pyspy_dump_utils as d
print(json.dumps({
    "pid": os.getpid(),
    "children_only": [p.pid for p in d.collect_scheduler_processes()],
    "with_parent": sorted(
        p.pid
        for p, _ in d._collect_sglang_processes(
            also_under=psutil.Process(os.getppid())
        )
    ),
}))
"""


class TestCollectSglangProcesses(CustomTestCase):
    @classmethod
    def setUpClass(cls):
        cls.children = []
        for title in (
            "sglang::scheduler_TP0",
            "sglang::scheduler_TP1",
            "sglang::detokenizer",
        ):
            proc = _spawn_titled(title)
            cls.children.append(proc)
            cls.addClassCleanup(proc.wait)
            cls.addClassCleanup(proc.kill)
        cls.scheduler_pids = {p.pid for p in cls.children[:2]}
        cls.detok_pid = cls.children[2].pid
        cls.all_pids = cls.scheduler_pids | {cls.detok_pid}

    def test_scheduler_filter_excludes_other_sglang_processes(self):
        """A non-scheduler sglang:: process in the subtree must not be treated
        as a scheduler."""
        schedulers = {p.pid for p in dump_utils.collect_scheduler_processes()}
        self.assertTrue(self.scheduler_pids <= schedulers)
        self.assertNotIn(self.detok_pid, schedulers)

    def test_parent_subtree_is_opt_in(self):
        """From a tokenizer worker, sibling sglang:: processes are reachable
        only through the explicit extra root, never from the default walk."""
        result = subprocess.run(
            [sys.executable, "-c", _WORKER_CODE],
            capture_output=True,
            text=True,
            check=True,
            timeout=120,
        )
        found = json.loads(result.stdout.strip().splitlines()[-1])
        self.assertEqual(found["children_only"], [])
        self.assertTrue(self.all_pids <= set(found["with_parent"]))
        self.assertNotIn(found["pid"], found["with_parent"])


class TestPyspyDumpPid(CustomTestCase):
    def _run_with(self, fake_run, **kwargs):
        with patch.object(dump_utils.subprocess, "run", side_effect=fake_run):
            with self.assertLogs(dump_utils.logger, level=logging.ERROR) as logs:
                dump_utils._pyspy_dump_pid(
                    pid=4242, title="sglang::scheduler_TP0", **kwargs
                )
        return logs.output

    def test_native_failure_falls_back_to_plain_dump(self):
        """A failed `--native` attempt must still yield the plain stack."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if "--native" in cmd:
                raise subprocess.CalledProcessError(1, cmd, stderr="no native")
            return subprocess.CompletedProcess(cmd, 0, stdout="Thread 1 (idle)")

        output = self._run_with(fake_run)
        self.assertEqual(
            calls,
            [
                ["py-spy", "dump", "--native", "--pid", "4242"],
                ["py-spy", "dump", "--pid", "4242"],
            ],
        )
        self.assertTrue(any("Thread 1 (idle)" in line for line in output))

    def test_native_timeout_falls_back_to_plain_dump(self):
        """A `--native` timeout must still yield the plain stack."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if "--native" in cmd:
                raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])
            return subprocess.CompletedProcess(cmd, 0, stdout="Thread 1 (idle)")

        output = self._run_with(fake_run, deadline=time.monotonic() + 30)
        self.assertEqual(len(calls), 2)
        self.assertTrue(any("timed out after 15s" in line for line in output))
        self.assertTrue(any("Thread 1 (idle)" in line for line in output))

    def test_stopped_process_falls_back_to_nonblocking(self):
        """Both suspend modes fail on an already-stopped process; --nonblocking still yields the stack."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            if "--nonblocking" not in cmd:
                raise subprocess.CalledProcessError(
                    1, cmd, stderr="Error: Failed to get stack traces"
                )
            return subprocess.CompletedProcess(cmd, 0, stdout="Thread 1 (idle)")

        output = self._run_with(fake_run)
        self.assertEqual(
            calls,
            [
                ["py-spy", "dump", "--native", "--pid", "4242"],
                ["py-spy", "dump", "--pid", "4242"],
                ["py-spy", "dump", "--nonblocking", "--pid", "4242"],
            ],
        )
        self.assertTrue(any("Thread 1 (idle)" in line for line in output))
        self.assertTrue(any("retrying with --nonblocking" in line for line in output))

    def test_ptrace_denial_is_not_retried(self):
        """A ptrace denial is one spawn and one line with the privilege hint."""
        calls = []

        def fake_run(cmd, **kwargs):
            calls.append(cmd)
            raise subprocess.CalledProcessError(
                1, cmd, stderr="Error: Operation not permitted (os error 1)"
            )

        output = self._run_with(fake_run)
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(output), 1)
        self.assertIn("CAP_SYS_PTRACE", output[0])

    def test_missing_py_spy_does_not_raise(self):
        """A missing py-spy binary is one error line, never an exception."""
        output = self._run_with(FileNotFoundError("py-spy"))
        self.assertEqual(len(output), 1)
        self.assertIn("not on PATH", output[0])


class TestExclusiveDumpSlot(CustomTestCase):
    def test_slot_is_exclusive_and_rate_limited_per_key(self):
        """A second holder is refused, a new holder inside the interval is
        refused, and a different owner of the same key starts fresh."""
        key = f"test-{os.getpid()}-{uuid.uuid4().hex}"
        path = dump_utils._dump_slot_path(key)
        self.addCleanup(lambda: path.unlink(missing_ok=True))

        def slot(owner="a", min_interval_secs=3600):
            return dump_utils._exclusive_dump_slot(
                key=key, owner=owner, min_interval_secs=min_interval_secs
            )

        with slot() as first:
            self.assertIs(first, dump_utils.DumpSlot.WON)
            with slot() as second:
                self.assertIs(second, dump_utils.DumpSlot.BUSY)
        with slot() as within_interval:
            self.assertIs(within_interval, dump_utils.DumpSlot.RECENT)
        with slot(min_interval_secs=0) as no_interval:
            self.assertIs(no_interval, dump_utils.DumpSlot.WON)
        with slot(owner="b") as new_owner:
            self.assertIs(new_owner, dump_utils.DumpSlot.WON)


if __name__ == "__main__":
    unittest.main()
