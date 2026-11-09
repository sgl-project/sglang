# SPDX-License-Identifier: Apache-2.0
"""Real Linux process-lifecycle tests with a fake torch_npu parser (no NPU)."""

import json
import multiprocessing
import os
import signal
import subprocess
import sys
import tempfile
import textwrap
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from sglang.srt.observability.profile_analysis_npu import analyze_npu_profile
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def daemon_worker(directory):
    assert multiprocessing.current_process().daemon
    analyze_npu_profile(directory, timeout=20)


def alive(pid):
    try:
        # Zombies have exited, but a container's PID 1 may not reap them promptly.
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except FileNotFoundError:
        return False


@unittest.skipUnless(
    sys.platform == "linux", "Ascend parser process groups require Linux"
)
class TestNPUProfileAnalysis(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(dir=Path.cwd())))
        package = self.root / "modules/torch_npu/profiler"
        package.mkdir(parents=True)
        (package.parent / "__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        (package / "profiler.py").write_text(
            textwrap.dedent(
                """\
                import json
                import multiprocessing as mp
                import os
                import time
                from pathlib import Path

                def render(directory):
                    root = Path(directory)
                    output = root / "ASCEND_PROFILER_OUTPUT"
                    output.mkdir()
                    (output / "trace_view.json").write_text(json.dumps({
                        "daemon": mp.current_process().daemon,
                        "directory": str(root),
                    }))

                def analyse(directory, max_process_number):
                    assert not mp.current_process().daemon
                    assert max_process_number == 1
                    root = Path(directory)
                    mode = (root / "mode").read_text()
                    if mode == "error":
                        raise RuntimeError("native parser failed")
                    if mode == "empty":
                        print("native parser swallowed an error", flush=True)
                        return
                    if mode == "hang":
                        process = mp.get_context("fork").Process(target=time.sleep, args=(30,))
                        process.start()
                        (root / "ready").write_text(f"{os.getpid()} {process.pid}")
                        time.sleep(30)
                        process.join()
                        return
                    process = mp.get_context("fork").Process(target=render, args=(directory,))
                    process.start()
                    process.join()
                    assert process.exitcode == 0
                """
            )
        )
        self.enterContext(
            patch.dict(
                os.environ,
                PYTHONPATH=str(package.parents[1]),
                PYTHONDONTWRITEBYTECODE="1",
            )
        )

    def capture(self, name="rank0_ascend_pt", mode="ok"):
        directory = self.root / name
        directory.mkdir()
        (directory / "mode").write_text(mode)
        (directory / "raw.data").write_text("original capture")
        return directory

    def wait_ready(self, directory):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                pids = [int(pid) for pid in (directory / "ready").read_text().split()]
                if len(pids) == 2:
                    return pids
            except FileNotFoundError:
                pass
            time.sleep(0.02)
        self.fail(f"Parser did not start its pool: {directory}")

    def assert_processes_exited(self, pids):
        deadline = time.monotonic() + 5
        while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
            time.sleep(0.02)
        self.assertFalse(any(alive(pid) for pid in pids), pids)

    def cleanup_worker(self, worker, directory):
        if worker.is_alive():
            worker.kill()
        worker.join(5)
        ready = directory / "ready"
        if ready.exists():
            parser_pid = int(ready.read_text().split()[0])
            try:
                os.killpg(parser_pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

    def test_real_daemon_worker_gets_a_parsed_trace_automatically(self):
        directory = self.capture(name="capture with spaces_ascend_pt")
        worker = multiprocessing.get_context("fork").Process(
            target=daemon_worker, args=(str(directory),), daemon=True
        )
        worker.start()
        self.addCleanup(self.cleanup_worker, worker, directory)
        worker.join(15)
        self.assertEqual(worker.exitcode, 0)
        trace = directory / "ASCEND_PROFILER_OUTPUT/trace_view.json"
        self.assertFalse(json.loads(trace.read_text())["daemon"])
        self.assertEqual((directory / "raw.data").read_text(), "original capture")

    def test_each_rank_parses_only_its_own_directory(self):
        directories = [self.capture(f"rank{rank}_ascend_pt") for rank in (0, 1)]
        for directory in directories:
            trace = analyze_npu_profile(str(directory), timeout=10)
            self.assertEqual(json.loads(trace.read_text())["directory"], str(directory))
        self.assertEqual(len(list(self.root.rglob("trace_view.json"))), 2)

    def test_parser_failure_and_silent_missing_output_are_errors(self):
        for mode in ("error", "empty"):
            with self.subTest(mode=mode):
                directory = self.capture(mode, mode=mode)
                with self.assertRaisesRegex(RuntimeError, "Raw data is retained"):
                    analyze_npu_profile(str(directory), timeout=10)
                log = (directory / "sglang_analysis.log").read_text()
                self.assertIn("native parser", log)
                self.assertTrue((directory / "raw.data").is_file())

    def test_timeout_terminates_parser_and_its_pool(self):
        directory = self.capture(mode="hang")
        with self.assertRaisesRegex(RuntimeError, "timed out"):
            analyze_npu_profile(str(directory), timeout=2)
        self.assert_processes_exited(self.wait_ready(directory))
        self.assertTrue((directory / "raw.data").is_file())

    def test_interrupt_terminates_parser_and_its_pool(self):
        directory = self.capture(mode="hang")
        wait = subprocess.Popen.wait
        interrupted = False

        def interrupt_once(process, *args, **kwargs):
            nonlocal interrupted
            if not interrupted:
                self.wait_ready(directory)
                interrupted = True
                raise KeyboardInterrupt()
            return wait(process, *args, **kwargs)

        with patch.object(subprocess.Popen, "wait", interrupt_once):
            with self.assertRaises(KeyboardInterrupt):
                analyze_npu_profile(str(directory), timeout=10)
        self.assert_processes_exited(self.wait_ready(directory))

    def test_worker_sigkill_terminates_parser_and_its_pool(self):
        directory = self.capture(mode="hang")
        worker = multiprocessing.get_context("fork").Process(
            target=daemon_worker, args=(str(directory),), daemon=True
        )
        worker.start()
        self.addCleanup(self.cleanup_worker, worker, directory)
        pids = self.wait_ready(directory)
        worker.kill()
        worker.join(5)
        self.assert_processes_exited(pids)

    def test_missing_capture_does_not_launch_a_parser(self):
        with patch("subprocess.Popen") as process:
            for path in ("", str(self.root / "missing")):
                with self.assertRaisesRegex(RuntimeError, "capture directory"):
                    analyze_npu_profile(path)
            process.assert_not_called()


if __name__ == "__main__":
    unittest.main()
