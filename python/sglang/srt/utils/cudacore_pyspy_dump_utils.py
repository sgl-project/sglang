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
"""CUDA core dump and py-spy dump utilities."""

from __future__ import annotations

import logging
import os
import platform
import stat
import subprocess
import time
from errno import ENXIO
from pathlib import Path
from typing import List

import psutil

logger = logging.getLogger(__name__)


def _resolve_cuda_coredump_pipe_path(proc: psutil.Process) -> Path:
    pipe_template = os.environ.get("CUDA_COREDUMP_PIPE")
    if pipe_template is None:
        pipe_path = f"corepipe.cuda.{platform.node()}.{proc.pid}"
    else:
        pipe_path = (
            pipe_template.replace("%h", platform.node())
            .replace("%p", str(proc.pid))
            .replace("%t", str(int(time.time())))
        )

    path = Path(pipe_path)
    if path.is_absolute():
        return path

    try:
        return Path(proc.cwd()) / path
    except (psutil.Error, OSError):
        return Path.cwd() / path


def _get_cuda_coredump_pipe_candidates(proc: psutil.Process) -> List[Path]:
    pipe_path = _resolve_cuda_coredump_pipe_path(proc)
    if os.environ.get("CUDA_COREDUMP_PIPE") is not None:
        return [pipe_path]

    # Some CUDA driver builds create the default FIFO without the documented
    # ".cuda." component. Keep the documented name first, then try the observed
    # driver name in the same working directory.
    driver_default_path = pipe_path.with_name(f"corepipe_{platform.node()}_{proc.pid}")
    return [pipe_path, driver_default_path]


def _is_sglang_scheduler_process(proc: psutil.Process) -> bool:
    try:
        proc_title = " ".join(proc.cmdline())
    except (psutil.Error, OSError):
        return False
    return proc_title.startswith("sglang::scheduler")


def collect_scheduler_processes() -> List[psutil.Process]:
    current = psutil.Process()
    return [
        proc
        for proc in current.children(recursive=True)
        if _is_sglang_scheduler_process(proc)
    ]


def pyspy_dump_schedulers(scheduler_only=False):
    """py-spy dump on all scheduler in a local node."""
    if scheduler_only:
        procs = collect_scheduler_processes()
        if not procs:
            logger.error("No sglang scheduler processes found for py-spy dump.")
            return
        pids = [proc.pid for proc in procs]
    else:
        pids = [psutil.Process().pid]
    for pid in pids:
        for attempt, native_flag in enumerate(["--native", ""]):
            try:
                cmd = f"py-spy dump {native_flag} --pid {pid}".strip()
                result = subprocess.run(
                    cmd, shell=True, capture_output=True, text=True, check=True
                )
                logger.error(f"Pyspy dump for PID {pid} ({cmd}):\n{result.stdout}")
                break
            except subprocess.CalledProcessError as e:
                logger.error(f"Pyspy failed ({cmd}). Error: {e.stderr}")
                if attempt == 1:
                    logger.error(f"All pyspy dump attempts failed for PID {pid}.")


def trigger_cuda_user_coredump(scheduler_only=False):
    """Trigger CUDA user-induced GPU core dumps by writing to coredump pipes."""
    if os.environ.get("CUDA_ENABLE_USER_TRIGGERED_COREDUMP") != "1":
        logger.error(
            "CUDA user-triggered coredump is not enabled. Set "
            "CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 before CUDA initialization."
        )

    if scheduler_only:
        procs = collect_scheduler_processes()
        if not procs:
            logger.error("No sglang scheduler processes found for CUDA coredump.")
            return
    else:
        procs = [psutil.Process()]

    for proc in procs:
        pipe_paths = _get_cuda_coredump_pipe_candidates(proc)
        for index, pipe_path in enumerate(pipe_paths):
            try:
                flags = os.O_WRONLY | os.O_NONBLOCK
                if index > 0:
                    flags |= getattr(os, "O_NOFOLLOW", 0)
                fd = os.open(pipe_path, flags)
                if index > 0:
                    try:
                        is_fifo = stat.S_ISFIFO(os.fstat(fd).st_mode)
                    except OSError:
                        os.close(fd)
                        raise
                    if not is_fifo:
                        os.close(fd)
                        logger.error(
                            "CUDA coredump fallback path is not a FIFO for PID %s: %s",
                            proc.pid,
                            pipe_path,
                        )
                        break
                try:
                    os.write(fd, b"1")
                finally:
                    os.close(fd)
            except FileNotFoundError:
                if index + 1 < len(pipe_paths):
                    continue
                logger.error(
                    "CUDA coredump pipe not found for PID %s: tried %s. Ensure "
                    "CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 was set before this "
                    "process initialized CUDA.",
                    proc.pid,
                    ", ".join(str(path) for path in pipe_paths),
                )
                break
            except OSError as e:
                if e.errno == ENXIO:
                    logger.error(
                        "CUDA coredump pipe has no reader for PID %s: %s",
                        proc.pid,
                        pipe_path,
                    )
                else:
                    logger.exception(
                        "Failed to trigger CUDA user coredump for PID %s via %s",
                        proc.pid,
                        pipe_path,
                    )
                break
            else:
                logger.error(
                    "Triggered CUDA user coredump for PID %s via %s",
                    proc.pid,
                    pipe_path,
                )
                break
