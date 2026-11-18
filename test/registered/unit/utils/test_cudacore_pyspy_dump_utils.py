# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Regression tests for the default CUDA user-coredump pipe.

When ``CUDA_COREDUMP_PIPE`` is unset, libcuda creates ``corepipe_<host>_<pid>``.
The CUDA-GDB guide documents ``corepipe.cuda.<host>.<pid>``. Opening only the
documented path drops the dump (sgl-project/sglang#42907).
"""

import os
import platform
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from sglang.srt.utils.cudacore_pyspy_dump_utils import trigger_cuda_user_coredump
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

_LOGGER = "sglang.srt.utils.cudacore_pyspy_dump_utils"


class _Proc:
    def __init__(self, pid: int, cwd: str):
        self.pid = pid
        self._cwd = cwd

    def cwd(self) -> str:
        return self._cwd


def _pipe_names(pid: int) -> tuple[str, str]:
    host = platform.node()
    return (f"corepipe_{host}_{pid}", f"corepipe.cuda.{host}.{pid}")


@contextmanager
def _coredump_env(pipe: str | None):
    updates = {"CUDA_ENABLE_USER_TRIGGERED_COREDUMP": "1"}
    if pipe is not None:
        updates["CUDA_COREDUMP_PIPE"] = pipe
    with mock.patch.dict(os.environ, updates, clear=False):
        if pipe is None:
            os.environ.pop("CUDA_COREDUMP_PIPE", None)
        yield


@contextmanager
def _held_fifo(path: Path):
    os.mkfifo(path)
    fd = os.open(path, os.O_RDWR | os.O_NONBLOCK)
    try:
        yield fd
    finally:
        os.close(fd)


def _trigger(proc: _Proc) -> None:
    with mock.patch(f"{_LOGGER}.psutil.Process", return_value=proc):
        trigger_cuda_user_coredump(scheduler_only=False)


def _read_pipe(fd: int) -> bytes:
    try:
        return os.read(fd, 8)
    except BlockingIOError:
        return b""


class TestCudaCoredumpPipe(CustomTestCase):
    def test_trigger_writes_the_driver_fifo(self):
        """The byte reaches ``corepipe_<host>_<pid>``, not the documented name."""
        pid = 424242
        driver_name, documented_name = _pipe_names(pid)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            documented = directory / documented_name
            documented.write_bytes(b"")
            with _held_fifo(directory / driver_name) as fd, _coredump_env(None):
                _trigger(_Proc(pid, tmp))
                self.assertEqual(_read_pipe(fd), b"1")
            self.assertEqual(documented.read_bytes(), b"")

    def test_trigger_writes_the_documented_fifo_when_it_is_the_only_one(self):
        """A driver that follows the CUDA-GDB guide still receives the trigger."""
        pid = 424243
        _, documented_name = _pipe_names(pid)
        with tempfile.TemporaryDirectory() as tmp:
            with (
                _held_fifo(Path(tmp) / documented_name) as fd,
                _coredump_env(None),
            ):
                _trigger(_Proc(pid, tmp))
                self.assertEqual(_read_pipe(fd), b"1")

    def test_explicit_pipe_wins_over_the_driver_fifo(self):
        """``CUDA_COREDUMP_PIPE`` is the pipe the driver was told to open."""
        pid = 424244
        driver_name, _ = _pipe_names(pid)
        with tempfile.TemporaryDirectory() as tmp:
            directory = Path(tmp)
            explicit = directory / "explicit-pipe"
            with (
                _held_fifo(directory / driver_name) as driver_fd,
                _held_fifo(explicit) as explicit_fd,
                _coredump_env(str(explicit)),
            ):
                _trigger(_Proc(pid, tmp))
                self.assertEqual(_read_pipe(explicit_fd), b"1")
                self.assertEqual(_read_pipe(driver_fd), b"")

    def test_missing_pipe_log_names_the_driver_fifo(self):
        """The not-found log names the FIFO libcuda creates, not the guide."""
        pid = 424245
        driver_name, documented_name = _pipe_names(pid)
        with tempfile.TemporaryDirectory() as tmp:
            with _coredump_env(None), self.assertLogs(_LOGGER, level="ERROR") as logs:
                _trigger(_Proc(pid, tmp))
        text = "\n".join(logs.output)
        self.assertIn(str(Path(tmp) / driver_name), text)
        self.assertNotIn(documented_name, text)


if __name__ == "__main__":
    unittest.main()
