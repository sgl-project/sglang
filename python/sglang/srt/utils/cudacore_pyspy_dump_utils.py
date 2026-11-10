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

import errno
import fcntl
import logging
import os
import platform
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from enum import Enum
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

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


def _proc_cmdline(proc: psutil.Process) -> str:
    try:
        return " ".join(proc.cmdline()).strip()
    except (psutil.Error, OSError):
        return ""


def _proc_title(proc: psutil.Process) -> str:
    return _proc_cmdline(proc)[:80]


def _is_sglang_launcher(proc: psutil.Process) -> bool:
    # PID 1 is init only when it is not our own entrypoint; a worker whose
    # launcher died was reparented to init and must not walk the container.
    return proc.pid != 1 or "sglang" in _proc_cmdline(proc)


def _collect_sglang_processes(
    prefix: str = "sglang::", also_under: Optional[psutil.Process] = None
) -> List[Tuple[psutil.Process, str]]:
    current = psutil.Process()
    candidates: List[psutil.Process] = []
    for root in (current, also_under):
        if root is None:
            continue
        try:
            candidates += root.children(recursive=True)
        except (psutil.Error, OSError):
            pass
    seen = {current.pid}
    out = []
    for proc in candidates:
        if proc.pid in seen:
            continue
        seen.add(proc.pid)
        title = _proc_title(proc)
        if title.startswith(prefix):
            out.append((proc, title))
    return out


def collect_scheduler_processes() -> List[psutil.Process]:
    # Descendants only: a crash dump, coredump trigger or shutdown wait must
    # never touch another server's schedulers.
    return [proc for proc, _ in _collect_sglang_processes(prefix="sglang::scheduler")]


def _tokenizer_worker_launcher(current: psutil.Process) -> Optional[psutil.Process]:
    # --tokenizer-worker-num > 1 runs HTTP handlers in spawned workers titled
    # sglang::tokenizer_worker:<pid>; their schedulers are siblings, not children.
    if not _proc_title(current).startswith("sglang::tokenizer_worker"):
        return None
    try:
        parent = current.parent()
    except (psutil.Error, OSError):
        return None
    if parent is None or not _is_sglang_launcher(parent):
        return None
    return parent


_PTRACE_DENIED = "Operation not permitted"
_PTRACE_HINT = (
    " (py-spy needs root or CAP_SYS_PTRACE, or kernel.yama.ptrace_scope=0 "
    "with the same uid)"
)
_MIN_ATTEMPT_SECS = 1.0


def _pyspy_dump_pid(pid: int, title: str, deadline: Optional[float] = None) -> None:
    # --native, then plain, then --nonblocking. The first two modes suspend the
    # target's threads themselves, which is impossible for an already stopped
    # (e.g. SIGSTOPped) process; --nonblocking reads memory without suspending
    # and is the only mode that captures one. Non-final attempts get half of
    # the deadline left, so a slow native symbolization cannot eat it all.
    attempts = (["--native"], [], ["--nonblocking"])
    retry_modes = ("without --native", "with --nonblocking")
    last_error = None
    for i, extra_flags in enumerate(attempts):
        timeout = None
        if deadline is not None:
            remaining = deadline - time.monotonic()
            timeout = remaining / 2 if i < len(attempts) - 1 else remaining
            if timeout < _MIN_ATTEMPT_SECS:
                last_error = "deadline exhausted"
                break
        cmd = ["py-spy", "dump"] + extra_flags + ["--pid", str(pid)]
        cmd_str = subprocess.list2cmdline(cmd)
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                errors="replace",
                check=True,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            error = f"{cmd_str}: timed out after {timeout:.0f}s"
        except subprocess.CalledProcessError as e:
            error = f"{cmd_str}: {e.stderr.strip()}"
        except FileNotFoundError:
            logger.error(f"py-spy is not on PATH; cannot dump PID {pid} [{title}].")
            return
        except OSError as e:
            error = f"{cmd_str}: {e}"
        else:
            logger.error(
                f"Pyspy dump for PID {pid} [{title}] ({cmd_str}):\n{result.stdout}"
            )
            return
        last_error = error
        if _PTRACE_DENIED in error or "No such process" in error:
            break  # ptrace denial and a dead pid fail identically in every mode
        if i < len(attempts) - 1:
            logger.error(
                f"Pyspy failed for PID {pid} ({error}); retrying {retry_modes[i]}"
            )
    denied = any(s in (last_error or "") for s in (_PTRACE_DENIED, "Permission denied"))
    logger.error(
        f"All pyspy dump attempts failed for PID {pid} "
        f"[{title or '<unknown>'}]: {last_error}{_PTRACE_HINT if denied else ''}"
    )


def _pyspy_dump_procs(
    dumps: List[Tuple[int, str]], deadline: Optional[float], max_parallel: int
) -> None:
    if not dumps:
        return
    if max_parallel <= 1:
        # Inline: the crash path runs from a signal handler, where a pool's
        # submit() can self-deadlock on the executor module lock.
        for pid, title in dumps:
            _pyspy_dump_pid(pid=pid, title=title, deadline=deadline)
        return
    with ThreadPoolExecutor(max_workers=min(len(dumps), max_parallel)) as pool:
        futures = {
            pool.submit(_pyspy_dump_pid, pid=pid, title=title, deadline=deadline): (
                pid,
                title,
            )
            for pid, title in dumps
        }
        for future, (pid, title) in futures.items():
            try:
                future.result()
            except Exception:
                logger.exception(f"py-spy dump of PID {pid} [{title}] raised")


def pyspy_dump_schedulers(scheduler_only: bool = False) -> None:
    """py-spy dump on all scheduler in a local node."""
    if scheduler_only:
        procs = _collect_sglang_processes(prefix="sglang::scheduler")
        if not procs:
            logger.error("No sglang scheduler processes found for py-spy dump.")
            return
    else:
        current = psutil.Process()
        procs = [(current, _proc_title(current))]
    dumps = [(proc.pid, title) for proc, title in procs]
    _pyspy_dump_procs(dumps=dumps, deadline=None, max_parallel=1)


class DumpSlot(Enum):
    WON = "won"
    BUSY = "busy"  # another process of this server is dumping
    RECENT = "recent"  # the last dump is younger than the interval


def _dump_slot_path(key: str) -> Path:
    return Path(tempfile.gettempdir()) / f"sglang-pyspy-dump-{os.getuid()}-{key}"


@contextmanager
def _exclusive_dump_slot(
    key: str, owner: str, min_interval_secs: float, wait_secs: float = 0.0
) -> Iterator[DumpSlot]:
    # Best effort: the file holds "<owner> <monotonic time of the last dump>"
    # and a file error must never veto the diagnostic. A busy slot is waited
    # for up to wait_secs so the caller's state flip stays behind the dump.
    path = _dump_slot_path(key)
    try:
        fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
    except OSError as e:
        logger.error(f"Cannot open {path} ({e}); dumping without the exclusive slot.")
        yield DumpSlot.WON
        return
    try:
        give_up = time.monotonic() + wait_secs
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError as e:
                if e.errno not in (errno.EWOULDBLOCK, errno.EAGAIN):
                    logger.error(f"flock({path}) failed ({e}); dumping without it.")
                    yield DumpSlot.WON
                    return
                if time.monotonic() >= give_up:
                    yield DumpSlot.BUSY
                    return
                time.sleep(0.5)
        last = float("nan")
        try:
            stored_owner, stored_time = os.read(fd, 128).decode().split()
            if stored_owner == owner:  # else: a previous server with this key
                last = float(stored_time)
        except (OSError, ValueError, UnicodeDecodeError):
            pass
        now = time.monotonic()
        # A negative age is a stale file from before a reboot.
        if 0 <= now - last < min_interval_secs:
            yield DumpSlot.RECENT
            return
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            os.ftruncate(fd, 0)
            os.write(fd, f"{owner} {now!r}".encode())
        except OSError as e:
            logger.error(f"Cannot record the dump time in {path} ({e}).")
        yield DumpSlot.WON
    finally:
        os.close(fd)


def pyspy_dump_all_sglang_processes(
    min_interval_secs: float = 0.0,
    budget_secs: float = 30.0,
    max_parallel: int = 8,
) -> bool:
    """py-spy dump the current process, the launcher, and every sglang::
    process of this server (scheduler ranks, detokenizer, tokenizer workers)
    into the log.

    Returns False when the dump was skipped: another tokenizer worker of the
    same server is dumping, or the last dump is less than min_interval_secs
    old. All processes share one budget_secs deadline.
    """
    current = psutil.Process()
    launcher = _tokenizer_worker_launcher(current)
    server = launcher or current
    try:
        owner = str(int(server.create_time()))
    except (psutil.Error, OSError):
        owner = "unknown"
    with _exclusive_dump_slot(
        key=str(server.pid),
        owner=owner,
        min_interval_secs=min_interval_secs,
        wait_secs=budget_secs,
    ) as slot:
        if slot is not DumpSlot.WON:
            logger.error(f"py-spy dump skipped ({slot.value}).")
            return False
        if shutil.which("py-spy") is None:
            logger.error("py-spy is not on PATH; skipping the health-failure dump.")
            return False
        dumps = [(current.pid, _proc_title(current))]
        if server.pid != current.pid:
            # In the multi-worker mode `current` is a tokenizer worker; the
            # launcher process is not sglang::-titled, so add it explicitly.
            dumps.append((server.pid, _proc_title(server)))
        dumps += [
            (proc.pid, title)
            for proc, title in _collect_sglang_processes(also_under=launcher)
        ]
        logger.error(
            f"py-spy dump of {len(dumps)} process(es): {[title for _, title in dumps]}"
        )
        _pyspy_dump_procs(
            dumps=dumps,
            deadline=time.monotonic() + budget_secs,
            max_parallel=max_parallel,
        )
    return True


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
        pipe_path = _resolve_cuda_coredump_pipe_path(proc)
        try:
            fd = os.open(pipe_path, os.O_WRONLY | os.O_NONBLOCK)
            try:
                os.write(fd, b"1")
            finally:
                os.close(fd)
            logger.error(
                "Triggered CUDA user coredump for PID %s via %s",
                proc.pid,
                pipe_path,
            )
        except FileNotFoundError:
            logger.error(
                "CUDA coredump pipe not found for PID %s: %s. Ensure "
                "CUDA_ENABLE_USER_TRIGGERED_COREDUMP=1 was set before this "
                "process initialized CUDA.",
                proc.pid,
                pipe_path,
            )
        except OSError as e:
            if e.errno == errno.ENXIO:
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
