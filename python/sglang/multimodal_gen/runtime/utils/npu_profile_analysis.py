# SPDX-License-Identifier: Apache-2.0
"""Run NPU trace analysis outside daemon workers, without importing the runtime.

The worker imports ``analyze_npu_profile``. The parser is exec'ed by file path
so its fresh interpreter does not initialize SGLang or inherit an NPU context.
Only the parser imports torch_npu; all output stays in the current capture.
"""

import argparse
import contextlib
import ctypes
import os
import signal
import subprocess
import sys
import traceback
from pathlib import Path

_ANALYSIS_TIMEOUT_S = 600


def _trace_path(directory: Path) -> Path:
    trace = directory / "ASCEND_PROFILER_OUTPUT" / "trace_view.json"
    if not trace.is_file() or trace.stat().st_size == 0:
        raise RuntimeError(
            f"NPU analysis produced no nonempty trace_view.json in {directory}"
        )
    return trace


def analyze_npu_profile(
    profiler_path: str, *, timeout: float = _ANALYSIS_TIMEOUT_S
) -> Path:
    """Wait for one capture to be parsed; keep raw data and the log on failure."""
    if not profiler_path or not Path(profiler_path).is_dir():
        raise RuntimeError(
            f"NPU profiler did not produce a capture directory: {profiler_path!r}"
        )
    directory = Path(profiler_path).resolve()
    log_path = directory / "sglang_analysis.log"
    command = [
        sys.executable,
        "-c",
        # run_path avoids putting utils/ on sys.path (it contains argparse.py).
        "import runpy, sys; sys.argv = sys.argv[1:]; "
        "runpy.run_path(sys.argv[0], run_name='__main__')",
        str(Path(__file__).resolve()),
        "--parent-pid",
        str(os.getpid()),
        str(directory),
    ]
    with log_path.open("w") as log:
        with subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        ) as process:
            try:
                process.wait(timeout=timeout)
                if process.returncode:
                    raise RuntimeError(
                        f"parser exited with status {process.returncode}"
                    )
                return _trace_path(directory)
            except BaseException as error:
                # Kill the parser AND its native analysis pool on timeout/cancel.
                # The parser is the leader of a new session, never the worker's group.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(process.pid, signal.SIGKILL)
                process.wait()
                if not isinstance(error, Exception):
                    raise
                raise RuntimeError(
                    f"NPU profile analysis failed for {directory}: {error}. "
                    f"Raw data is retained; see {log_path}"
                ) from error


def _terminate_parser_group(_signum=None, _frame=None):
    os.killpg(os.getpgrp(), signal.SIGKILL)


def _arm_parent_death(parent_pid: int) -> None:
    # This entrypoint is private and must be exec'ed in its own session.
    if os.getpgrp() != os.getpid():
        raise RuntimeError("The NPU parser must run in its own process group")
    signal.signal(signal.SIGTERM, _terminate_parser_group)
    signal.signal(signal.SIGINT, _terminate_parser_group)
    # Ascend is Linux-only. SIGTERM lets the handler terminate the whole process
    # group, unlike killing just the parser and leaving its pool orphaned.
    libc = ctypes.CDLL("libc.so.6", use_errno=True)
    if libc.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    # Cover parent exit between Popen and arming the death signal.
    if os.getppid() != parent_pid:
        _terminate_parser_group()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-pid", type=int, required=True)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    _arm_parent_death(args.parent_pid)
    try:
        from torch_npu.profiler.profiler import analyse

        analyse(str(args.directory), max_process_number=1)
        # torch_npu can log an error and return normally: exit status alone
        # does not prove that the user has a usable trace.
        _trace_path(args.directory)
    except BaseException:
        traceback.print_exc()
        sys.stderr.flush()
        _terminate_parser_group()


if __name__ == "__main__":
    main()
