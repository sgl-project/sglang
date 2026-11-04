"""--disaggregation-decode-late-cuda-context: one helper per process, never in the
decode loop's way, holding its context until the worker exits."""

import os
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.disaggregation import late_cuda_context as L
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=20, suite="base-a-test-cpu")


def _fake_torch_dir(*, fail: bool) -> Path:
    """A torch stand-in for the helper process: CUDA calls succeed or raise."""
    d = Path(tempfile.mkdtemp(prefix="late_ctx_torch_"))
    (d / "torch").mkdir()
    body = "raise RuntimeError('no device')" if fail else "return None"
    (d / "torch" / "__init__.py").write_text(
        textwrap.dedent(
            f"""
            class cuda:
                @staticmethod
                def synchronize():
                    return None
            def ones(*a, **k):
                {body}
            """
        )
    )
    return d


def _option(enabled):
    return mock.patch.object(
        L,
        "get_disagg",
        lambda: SimpleNamespace(disaggregation_decode_late_cuda_context=enabled),
    )


class TestLateCudaContextGate(CustomTestCase):
    def setUp(self):
        L._started = False

    def test_off_never_spawns_and_on_spawns_once(self):
        for enabled, want in ((False, 0), (True, 1)):
            L._started = False
            with _option(enabled), mock.patch.object(L, "_start_helper") as start:
                for _ in range(5):
                    L.maybe_open_late_cuda_context()
            self.assertEqual(start.call_count, want, enabled)

    def test_a_failed_spawn_is_logged_once_and_never_raised(self):
        with (
            _option(True),
            mock.patch.object(
                L, "_start_helper", side_effect=OSError(11, "EAGAIN")
            ) as start,
            self.assertLogs(L.logger, level="WARNING") as logs,
        ):
            L.maybe_open_late_cuda_context()
            L.maybe_open_late_cuda_context()
        self.assertEqual(start.call_count, 1)
        self.assertEqual(len(logs.records), 1)
        self.assertIn("helper not started", logs.output[0])

    def test_physical_device_follows_cuda_visible_devices(self):
        for visible, device, want in (
            (None, 2, "2"),
            ("", 1, "1"),
            ("4,5,6,7", 0, "4"),
            ("4, 5,6", 1, "5"),
            ("GPU-aa,GPU-bb", 1, "GPU-bb"),
        ):
            env = {} if visible is None else {"CUDA_VISIBLE_DEVICES": visible}
            with mock.patch.dict(os.environ, env, clear=False):
                if visible is None:
                    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
                self.assertEqual(L._physical_device(device), want)


class TestLateCudaContextHelper(CustomTestCase):
    def _start(self, *, fail):
        env = {"PYTHONPATH": str(_fake_torch_dir(fail=fail))}
        with (
            mock.patch.dict(os.environ, env),
            mock.patch.object(torch.cuda, "current_device", lambda: 0),
            mock.patch.object(L, "_helper_cpus", lambda: ""),
            mock.patch.object(L.threading, "Thread") as thread,
        ):
            L._start_helper()
        _, kwargs = thread.call_args
        return kwargs["args"]

    def test_holds_line_only_after_ready_and_warns_when_the_helper_exits(self):
        helper, gpu, worker = self._start(fail=False)
        try:
            with self.assertLogs(L.logger, level="INFO") as logs:
                watcher = threading.Thread(
                    target=L._watch, args=(helper, gpu, worker), daemon=True
                )
                watcher.start()
                deadline = time.time() + 30
                while time.time() < deadline and not any(
                    "holds a context" in line for line in logs.output
                ):
                    time.sleep(0.05)
                self.assertTrue(helper.poll() is None, "helper must outlive this call")
                helper.kill()
                watcher.join(timeout=10)
        finally:
            helper.kill()
        self.assertIn("holds a context", logs.output[0])
        self.assertIn("exited with code", logs.output[-1])

    def test_a_helper_that_fails_before_ready_logs_no_holds_line(self):
        helper, gpu, worker = self._start(fail=True)
        with self.assertLogs(L.logger, level="INFO") as logs:
            L._watch(helper, gpu, worker)
        self.assertFalse(any("holds a context" in line for line in logs.output))
        self.assertIn("exited with code 1", logs.output[-1])

    def test_the_helper_exits_after_its_worker(self):
        """The helper polls its parent; a worker that exits must not leave it holding the GPU."""
        fake = _fake_torch_dir(fail=False)
        worker = subprocess.run(
            [
                sys.executable,
                "-c",
                "import subprocess, sys\n"
                "h = subprocess.Popen([sys.executable, '-c', sys.argv[1], str(__import__('os').getpid()), ''],"
                " stdout=subprocess.PIPE, text=True, start_new_session=True)\n"
                "assert h.stdout.readline().strip() == 'ready'\n"
                "print(h.pid)\n",
                L._HELPER,
            ],
            env=dict(os.environ, PYTHONPATH=str(fake)),
            capture_output=True,
            text=True,
            timeout=60,
        )
        self.assertEqual(worker.returncode, 0, worker.stderr)
        pid = int(worker.stdout.strip())
        deadline = time.time() + 20
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                break
            time.sleep(0.2)
        else:
            os.kill(pid, 9)
            self.fail("the helper outlived its worker")


if __name__ == "__main__":
    unittest.main()
