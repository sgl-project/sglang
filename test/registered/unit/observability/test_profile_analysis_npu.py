# SPDX-License-Identifier: Apache-2.0
"""Real detached parser lifecycles with a fake torch_npu analysis pool (no NPU)."""

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

from sglang.srt.observability import profile_analysis_npu as analysis
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def daemon_worker(directory, linger):
    assert multiprocessing.current_process().daemon
    analysis.start_npu_profile_analysis(directory)
    (Path(directory) / "worker-returned").touch()
    if linger:
        time.sleep(30)


def alive(pid):
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
        return state != "Z"
    except FileNotFoundError:
        return False


@unittest.skipUnless(sys.platform == "linux", "Ascend analysis requires Linux")
class TestNPUProfileAnalysis(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory(dir=Path.cwd())))
        package = self.root / "modules/torch_npu/profiler"
        package.mkdir(parents=True)
        (package.parent / "__init__.py").write_text("")
        (package / "__init__.py").write_text("")
        (package / "profiler.py").write_text(
            textwrap.dedent("""\
            import json
            import multiprocessing as mp
            import os
            import sys
            import time
            from pathlib import Path

            def render(directory, hold):
                root = Path(directory)
                if hold:
                    while not (root / "release").exists():
                        time.sleep(0.01)
                output = root / "ASCEND_PROFILER_OUTPUT"
                output.mkdir()
                (output / "trace_view.json").write_text(json.dumps({
                    "daemon": mp.current_process().daemon,
                    "directory": str(root),
                }))

            def analyse(directory, max_process_number):
                assert not mp.current_process().daemon
                assert not any(name == "sglang" or name.startswith("sglang.") for name in sys.modules)
                assert max_process_number == 1
                root = Path(directory)
                mode = (root / "mode").read_text()
                if mode == "error":
                    raise RuntimeError("native parser failed")
                if mode == "empty":
                    print("native parser swallowed an error", flush=True)
                    return
                if mode == "empty_trace":
                    output = root / "ASCEND_PROFILER_OUTPUT"
                    output.mkdir()
                    (output / "trace_view.json").touch()
                    return
                process = mp.get_context("fork").Process(
                    target=render, args=(directory, mode == "hold"))
                process.start()
                (root / "ready").write_text(f"{os.getpid()} {process.pid}")
                process.join()
                assert process.exitcode == 0
            """)
        )
        self.enterContext(
            patch.dict(
                os.environ,
                PYTHONPATH=str(package.parents[1]),
                PYTHONDONTWRITEBYTECODE="1",
            )
        )
        self.captures = []
        self.addCleanup(self.cleanup_parsers)

    def capture(self, name="rank0_ascend_pt", mode="ok"):
        directory = self.root / name
        directory.mkdir()
        (directory / "mode").write_text(mode)
        (directory / "raw.data").write_text("original capture")
        self.captures.append(directory)
        return directory

    def wait_for(self, predicate):
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            result = predicate()
            if result:
                return result
            time.sleep(0.02)
        self.fail("Background parser did not reach the expected state")

    def status(self, directory):
        try:
            return json.loads((directory / "sglang_analysis.json").read_text())
        except FileNotFoundError:
            return {}

    def wait_status(self, directory, state):
        return self.wait_for(
            lambda: (
                status
                if (status := self.status(directory)).get("state") == state
                else None
            )
        )

    def wait_ready(self, directory):
        def read_pids():
            try:
                pids = [int(pid) for pid in (directory / "ready").read_text().split()]
                return pids if len(pids) == 2 else None
            except FileNotFoundError:
                return None

        return self.wait_for(read_pids)

    def cleanup_parsers(self):
        for directory in self.captures:
            (directory / "release").touch()
        for directory in self.captures:
            if (directory / "ready").exists():
                pids = self.wait_ready(directory)
                deadline = time.monotonic() + 3
                while any(alive(pid) for pid in pids) and time.monotonic() < deadline:
                    time.sleep(0.02)
                if any(alive(pid) for pid in pids):
                    try:
                        os.killpg(pids[0], signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_submit_returns_while_parser_is_still_running(self):
        directory = self.capture(mode="hold")
        status_path = analysis.start_npu_profile_analysis(str(directory))
        self.assertEqual(status_path, directory / "sglang_analysis.json")
        self.wait_status(directory, "running")
        self.wait_ready(directory)
        self.assertFalse(
            (directory / "ASCEND_PROFILER_OUTPUT/trace_view.json").exists()
        )
        (directory / "release").touch()
        status = self.wait_status(directory, "succeeded")
        self.assertTrue(Path(status["trace_path"]).is_file())
        self.assertGreaterEqual(status["finished_at"], status["started_at"])
        self.assertIsNone(status["error"])

    def test_native_errors_and_missing_or_empty_trace_report_failure(self):
        for mode in ("error", "empty", "empty_trace"):
            with self.subTest(mode=mode):
                directory = self.capture(mode, mode=mode)
                analysis.start_npu_profile_analysis(str(directory))
                status = self.wait_status(directory, "failed")
                self.assertIsNone(status["trace_path"])
                self.assertIn(
                    "native parser failed" if mode == "error" else "no nonempty trace",
                    status["error"],
                )
                self.assertIn(
                    "Traceback", (directory / "sglang_analysis.log").read_text()
                )
                self.assertTrue((directory / "raw.data").is_file())

    def test_parser_survives_normal_and_killed_daemon_worker_exit(self):
        for killed in (False, True):
            with self.subTest(killed=killed):
                directory = self.capture(
                    f"worker-{killed} with spaces_ascend_pt", mode="hold"
                )
                worker = multiprocessing.get_context("fork").Process(
                    target=daemon_worker, args=(str(directory), killed), daemon=True
                )
                worker.start()
                try:
                    self.wait_for(lambda: (directory / "worker-returned").exists())
                    pids = self.wait_ready(directory)
                    if killed:
                        worker.kill()
                    worker.join(5)
                    self.assertEqual(worker.exitcode, -signal.SIGKILL if killed else 0)
                    self.assertTrue(all(alive(pid) for pid in pids))
                    (directory / "release").touch()
                    self.wait_status(directory, "succeeded")
                    self.wait_for(lambda: not any(alive(pid) for pid in pids))
                finally:
                    if worker.is_alive():
                        worker.kill()
                    worker.join(5)

    def test_launch_failure_writes_failed_status(self):
        directory = self.capture()
        with patch.object(subprocess, "Popen", side_effect=OSError("cannot exec")):
            with self.assertRaisesRegex(OSError, "cannot exec"):
                analysis.start_npu_profile_analysis(str(directory))
        self.assertIn("cannot exec", self.status(directory)["error"])
        self.assertEqual(self.status(directory)["state"], "failed")


if __name__ == "__main__":
    unittest.main()
