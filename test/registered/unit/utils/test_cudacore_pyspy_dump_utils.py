"""CPU-only tests for process-level diagnostic dump helpers."""

import os
import subprocess
import unittest
from errno import ENXIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import psutil

from sglang.srt.utils import cudacore_pyspy_dump_utils as dump_utils
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _process(*, pid=123, cwd="/work", cmdline=None):
    process = MagicMock(spec=psutil.Process)
    process.pid = pid
    process.cwd.return_value = cwd
    process.cmdline.return_value = cmdline or []
    return process


class TestCudaCorePipePath(CustomTestCase):
    def test_default_relative_path_uses_process_working_directory(self):
        process = _process(pid=42, cwd="/srv/sglang")

        with patch.dict(os.environ, {}, clear=True), patch.object(
            dump_utils.platform, "node", return_value="worker-a"
        ):
            path = dump_utils._resolve_cuda_coredump_pipe_path(process)

        self.assertEqual(path, Path("/srv/sglang/corepipe.cuda.worker-a.42"))

    def test_template_expands_placeholders_and_preserves_absolute_paths(self):
        process = _process(pid=42)

        with patch.dict(
            os.environ, {"CUDA_COREDUMP_PIPE": "/dumps/%h-%p-%t"}, clear=True
        ), patch.object(dump_utils.platform, "node", return_value="worker-a"), patch.object(
            dump_utils.time, "time", return_value=123.9
        ):
            path = dump_utils._resolve_cuda_coredump_pipe_path(process)

        self.assertEqual(path, Path("/dumps/worker-a-42-123"))

    def test_relative_path_falls_back_to_current_directory_when_cwd_is_unavailable(self):
        process = _process()
        process.cwd.side_effect = psutil.AccessDenied(pid=process.pid)

        with patch.dict(
            os.environ, {"CUDA_COREDUMP_PIPE": "relative.pipe"}, clear=True
        ), patch.object(dump_utils.Path, "cwd", return_value=Path("/fallback")):
            path = dump_utils._resolve_cuda_coredump_pipe_path(process)

        self.assertEqual(path, Path("/fallback/relative.pipe"))


class TestSchedulerProcessSelection(CustomTestCase):
    def test_identifies_scheduler_process_and_rejects_other_processes(self):
        scheduler = _process(cmdline=["sglang::scheduler", "--tp", "2"])
        worker = _process(cmdline=["sglang::detokenizer"])
        unreadable = _process()
        unreadable.cmdline.side_effect = psutil.AccessDenied(pid=unreadable.pid)

        self.assertTrue(dump_utils._is_sglang_scheduler_process(scheduler))
        self.assertFalse(dump_utils._is_sglang_scheduler_process(worker))
        self.assertFalse(dump_utils._is_sglang_scheduler_process(unreadable))

    def test_collects_only_scheduler_children(self):
        scheduler = _process(pid=1, cmdline=["sglang::scheduler"])
        worker = _process(pid=2, cmdline=["sglang::detokenizer"])
        current = _process()
        current.children.return_value = [scheduler, worker]

        with patch.object(dump_utils.psutil, "Process", return_value=current):
            processes = dump_utils.collect_scheduler_processes()

        self.assertEqual(processes, [scheduler])
        current.children.assert_called_once_with(recursive=True)


class TestPySpyDump(CustomTestCase):
    def test_retries_without_native_flag_after_native_dump_failure(self):
        error = subprocess.CalledProcessError(1, "py-spy", stderr="no native")
        completed = subprocess.CompletedProcess("py-spy", 0, stdout="stack")
        process = _process(pid=123)

        with patch.object(
            dump_utils.subprocess, "run", side_effect=[error, completed]
        ) as run, patch.object(
            dump_utils.psutil, "Process", return_value=process
        ), patch.object(dump_utils.logger, "error") as log_error:
            dump_utils.pyspy_dump_schedulers()

        self.assertEqual(run.call_count, 2)
        self.assertIn("--native", run.call_args_list[0].args[0])
        self.assertNotIn("--native", run.call_args_list[1].args[0])
        self.assertTrue(
            any(
                len(call.args) > 0 and "Pyspy dump" in call.args[0]
                for call in log_error.call_args_list
            )
        )

    def test_scheduler_only_mode_returns_when_no_scheduler_exists(self):
        with patch.object(
            dump_utils, "collect_scheduler_processes", return_value=[]
        ), patch.object(dump_utils.subprocess, "run") as run, patch.object(
            dump_utils.logger, "error"
        ) as log_error:
            dump_utils.pyspy_dump_schedulers(scheduler_only=True)

        run.assert_not_called()
        log_error.assert_called_once()
        self.assertIn("No sglang scheduler", log_error.call_args[0][0])


class TestCudaCoreDumpTrigger(CustomTestCase):
    def test_writes_trigger_byte_and_closes_pipe(self):
        process = _process(pid=9)
        pipe_path = Path("/tmp/corepipe")

        with patch.dict(
            os.environ, {"CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "1"}, clear=True
        ), patch.object(dump_utils.psutil, "Process", return_value=process), patch.object(
            dump_utils, "_resolve_cuda_coredump_pipe_path", return_value=pipe_path
        ), patch.object(dump_utils.os, "open", return_value=17) as open_pipe, patch.object(
            dump_utils.os, "write"
        ) as write_pipe, patch.object(dump_utils.os, "close") as close_pipe:
            dump_utils.trigger_cuda_user_coredump()

        open_pipe.assert_called_once_with(pipe_path, os.O_WRONLY | os.O_NONBLOCK)
        write_pipe.assert_called_once_with(17, b"1")
        close_pipe.assert_called_once_with(17)

    def test_reports_pipe_without_reader(self):
        process = _process(pid=9)
        no_reader = OSError(ENXIO, "no reader")

        with patch.dict(
            os.environ, {"CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "1"}, clear=True
        ), patch.object(dump_utils.psutil, "Process", return_value=process), patch.object(
            dump_utils.os, "open", side_effect=no_reader
        ), patch.object(dump_utils.logger, "error") as log_error:
            dump_utils.trigger_cuda_user_coredump()

        log_error.assert_called_once()
        self.assertIn("has no reader", log_error.call_args[0][0])


if __name__ == "__main__":
    unittest.main()
