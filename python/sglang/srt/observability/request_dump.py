"""The request dump file: requests the Python ``TokenizerManager`` served,
pickled with the launch state. ``--crash-dump-folder`` writes one before a
crash (the Python ``TokenizerManager`` and the scheduler that embeds the Rust
server alike), and the ``dump_requests_folder`` logging setting writes one
every ``dump_requests_threshold`` requests; the dump readers such as
``scripts/playground/replay_request_dump.py`` load either."""

from __future__ import annotations

import logging
import os
import pickle
import secrets
import signal
import socket
import sys
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Callable, List, Optional, Sequence, Tuple

from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.watermarking.core import (
    redact_watermark_command_line,
    redact_watermark_secrets,
)
from sglang.srt.server_args import redact_secret_cli_args, redact_server_args_secrets

logger = logging.getLogger(__name__)

# How long the Rust-mode main process waits, after a child reports a failure,
# for the schedulers that embed the Rust server to finish their dumps, and how
# long it looks for one to start.
CRASH_DUMP_GRACE_SECS = 30.0
CRASH_DUMP_START_SECS = 1.0


def crash_dump_hostname() -> str:
    """The folder under ``--crash-dump-folder`` this host's dumps go to."""
    return os.getenv("HOSTNAME", socket.gethostname())


def _redacted_config(config: Any) -> Optional[dict]:
    """``config`` -- a ``ServerArgs`` or a dict of its fields -- as a dict
    without the secrets, or None when it cannot be serialized."""
    if config is None:
        return None
    try:
        if not isinstance(config, dict):
            config = config.resolved_dict()
        return redact_server_args_secrets(config)
    except Exception as e:
        logger.error(f"Failed to snapshot the config for the dump: {e!r}")
        return None


def build_request_dump(
    requests: List[Tuple],
    server_args: Any,
    resolved_config: Optional[dict],
    launch_command: Optional[str] = None,
) -> dict:
    """``requests`` are ``(request object, output dict, created time, finished
    time)`` tuples."""
    data = {
        "server_args": _redacted_config(server_args),
        "config_updates": redact_watermark_secrets(get_context().overrides_log()),
        "resolved_config": _redacted_config(resolved_config),
        "requests": redact_watermark_secrets(requests),
    }
    if launch_command is not None:
        data["launch_command"] = launch_command
    return data


def save_request_dump(filename: str, data: dict) -> None:
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "wb") as f:
        try:
            pickle.dump(data, f)
        except Exception as e:
            # When the server is launched with --trust-remote-code,
            # server_args sometimes fails to pickle. Retry without
            # server_args so the request data still gets persisted.
            logger.error(
                f"Failed to pickle dump with server_args: {e!r}; "
                "retrying without server_args"
            )
            f.seek(0)
            f.truncate()
            data["server_args"] = None
            # The snapshot copies the same object field by field.
            data["resolved_config"] = None
            pickle.dump(data, f)


def write_crash_dump(
    folder: str,
    hostname: str,
    requests: List[Tuple],
    server_args: Any,
    resolved_config: Optional[dict],
    watermark_config: Any,
    filename_suffix: str = "",
) -> str:
    """Write the crash dump to
    ``<folder>/<hostname>/crash_dump_<time><filename_suffix>.pkl`` and return
    the path. The file appears whole or not at all: a process killed while
    writing leaves no partial dump behind.

    Secrets are redacted here, for every writer: ``server_args`` (a
    ``ServerArgs`` or a dict of its fields) and ``resolved_config`` are stored
    as dicts without them, and the launch command without their values."""
    filename = os.path.join(
        folder,
        hostname,
        f"crash_dump_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        f"{filename_suffix}.pkl",
    )
    data = build_request_dump(
        requests,
        server_args,
        resolved_config,
        launch_command=redact_watermark_command_line(
            redact_secret_cli_args(sys.argv), watermark_config
        ),
    )
    temp_path = os.path.join(
        os.path.dirname(filename), f".crash_dump_{secrets.token_hex(8)}.tmp"
    )
    try:
        save_request_dump(temp_path, data)
        os.replace(temp_path, filename)
    except BaseException:
        if os.path.exists(temp_path):
            os.unlink(temp_path)
        raise
    return filename


def dumping_marker(folder: str, hostname: str, pid: int) -> str:
    """The file a process holds while it writes its crash dump."""
    return os.path.join(folder, hostname, f".dumping.{pid}")


@contextmanager
def dumping(folder: str, hostname: str):
    """Mark this process as writing its crash dump, for the main process to
    wait on (see ``wait_for_crash_dumps``)."""
    marker = dumping_marker(folder, hostname, os.getpid())
    os.makedirs(os.path.dirname(marker), exist_ok=True)
    open(marker, "wb").close()
    try:
        yield
    finally:
        os.unlink(marker)


def wait_for_crash_dumps(
    folder: str,
    hostname: str,
    writer_pids: Sequence[int],
    start_timeout: float = CRASH_DUMP_START_SECS,
    timeout: float = CRASH_DUMP_GRACE_SECS,
    poll_interval: float = 0.05,
) -> str:
    """Wait for the processes in ``writer_pids`` that are writing a crash dump
    (they hold ``dumping_marker`` while they do). Returns at once with
    ``"none"`` when no writer is known, ``"none"`` after ``start_timeout`` if
    none has started, else ``"written"`` once every marker is gone or
    ``"timeout"`` after ``timeout`` seconds."""
    markers = [dumping_marker(folder, hostname, pid) for pid in writer_pids]
    if not markers:
        return "none"
    start = time.monotonic()
    while not any(os.path.exists(m) for m in markers):
        if time.monotonic() - start >= start_timeout:
            return "none"
        time.sleep(poll_interval)
    while any(os.path.exists(m) for m in markers):
        if time.monotonic() - start >= timeout:
            return "timeout"
        time.sleep(poll_interval)
    return "written"


class CrashDumpGrace:
    """The Rust-mode main process's ``SIGQUIT`` handler with
    ``--crash-dump-folder`` set. A child reports a failure with ``SIGQUIT``;
    the requests live in the schedulers that embed the Rust server
    (``writer_pids``), which dump on their own exception or watchdog timeout.
    Killing the process tree at once would cut a dump in progress short, so
    the first ``SIGQUIT`` waits for the dumps under way (see
    ``wait_for_crash_dumps``) off the main thread, then signals the process
    again; that ``SIGQUIT`` runs the handler this one replaced
    (``previous``), which kills the tree on the main thread as before. Other
    ``SIGQUIT``s while waiting belong to the same failure.

    With no known writer (a DP-controller layout reports none) it does not
    wait. TODO: take the writers from ``SchedulerInitResult.rust_listener_procs``
    once the graceful-shutdown work that adds it lands."""

    def __init__(
        self,
        folder: str,
        writer_pids: Sequence[int],
        previous: Callable[[int, Any], None],
        before_wait: Callable[[], None] = lambda: None,
        timeout: float = CRASH_DUMP_GRACE_SECS,
    ):
        self.folder = folder
        self.writer_pids = list(writer_pids)
        self.previous = previous
        self.before_wait = before_wait
        self.timeout = timeout
        self._started = threading.Event()
        self._waited = threading.Event()

    def handle_sigquit(self, signum=None, frame=None) -> None:
        if self._waited.is_set():
            self.previous(signum, frame)
            return
        logger.error(
            "Received sigquit from a child process. It usually means the child failed."
        )
        if self._started.is_set():
            return
        self._started.set()
        threading.Thread(
            target=self._wait_then_signal, name="crash-dump-grace", daemon=True
        ).start()

    def _wait_then_signal(self) -> None:
        try:
            self.before_wait()
            reason = wait_for_crash_dumps(
                self.folder,
                crash_dump_hostname(),
                self.writer_pids,
                timeout=self.timeout,
            )
            logger.error(f"Crash dump wait ended ({reason}); killing the process tree.")
        finally:
            self._waited.set()
            os.kill(os.getpid(), signal.SIGQUIT)
