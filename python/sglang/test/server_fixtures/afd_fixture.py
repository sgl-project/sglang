"""Local A/F service ownership for public SGLang serving tests.

The commands run concurrently and are never retried individually. F readiness
requires every expected lane's post-handshake log from this launch; A readiness
also requires /health_generate. This fixture does not allocate devices or claim
GPU memory has been released merely because its processes have exited.
"""

from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence
from urllib.parse import urlparse

import psutil

from sglang.test.test_utils import collect_process_tree_pids

# A process, rather than a daemon thread, gives even DNS / slow-header reads an
# absolute bound. No response body is consumed and every probe is reaped.
_HEALTH_PROBE = """
import os, sys
import requests
headers = {"Authorization": os.environ["AFD_HEALTH_AUTH"]} if "AFD_HEALTH_AUTH" in os.environ else {}
try:
    with requests.get(sys.argv[1], headers=headers, stream=True, timeout=1) as response:
        code = 0 if response.status_code == 200 else 1
except requests.RequestException:
    code = 1
sys.exit(code)
"""


@dataclass(frozen=True)
class _OwnedProcess:
    process: psutil.Process
    created: float

    @classmethod
    def capture(cls, pid):
        process = psutil.Process(pid)
        return cls(process, process.create_time())

    def is_live(self):
        try:
            return (
                self.process.is_running()
                and psutil.Process(self.process.pid).create_time() == self.created
                and self.process.status() != psutil.STATUS_ZOMBIE
            )
        except psutil.NoSuchProcess:
            return False


_FFN_READY = re.compile(r"AFD FFN lane (\d+) ready(?:\s|$)")


@dataclass(frozen=True)
class AFDProcessSpec:
    name: str
    role: str
    command: Sequence[str]
    env: Mapping[str, str] = field(default_factory=dict)
    ffn_lanes: tuple[int, ...] = ()

    @classmethod
    def server(
        cls,
        *,
        role: str,
        model: str,
        afd_config: Mapping,
        other_args: Sequence[str] = (),
        env: Mapping[str, str] | None = None,
        base_url: str | None = None,
    ) -> AFDProcessSpec:
        """Build one local A or F launcher using the public serving CLI.

        Device placement, TP/EP, backend, and any model-cache/offline settings
        remain explicit caller arguments. All owned processes are local; this
        fixture does not manage remote SSH sessions or remote process lifetimes.
        """
        if role not in ("attention", "ffn"):
            raise ValueError("role must be attention or ffn")
        if any(arg == "--nnodes" or arg.startswith("--nnodes=") for arg in other_args):
            raise ValueError(
                "server() owns a local role; --nnodes overrides are unsupported"
            )
        command = [
            sys.executable,
            "-m",
            "sglang.launch_server",
            "--model-path",
            model,
            *other_args,
            "--afd-execution-mode",
            role,
            "--afd-config",
            json.dumps(dict(afd_config)),
            "--log-level",
            "info",
        ]
        if role == "attention":
            parsed = urlparse(base_url or "")
            if parsed.scheme != "http" or not parsed.hostname or not parsed.port:
                raise ValueError("attention requires an explicit http://host:port")
            command += ["--host", parsed.hostname, "--port", str(parsed.port)]
        return cls(
            name=role,
            role=role,
            command=command,
            env=dict(env or {}),
            ffn_lanes=tuple(range(afd_config.get("lanes", 1))) if role == "ffn" else (),
        )


class AFDServerGroup:
    """Own all A/F roots and their descendants for one bounded test launch.

    Use as a context manager and call ``check_healthy`` between requests. A
    background monitor stops the entire group on any unexpected exit (even 0),
    and the context exit surfaces that failure. No watcher exits the test runner.
    Logs live in caller-owned ``log_dir`` and are truncated for each new group.
    The default shutdown budget covers the default AFD close deadline (95s).
    For custom configs use at least ``3 * close_timeout_seconds + 5`` seconds.
    """

    def __init__(
        self,
        specs: Sequence[AFDProcessSpec],
        *,
        base_url: str,
        log_dir: str | Path,
        startup_timeout: float = 600,
        shutdown_timeout: float = 100,
        poll_interval: float = 0.1,
        api_key: str | None = None,
    ):
        self.specs = tuple(specs)
        if (
            sum(spec.role == "attention" for spec in self.specs) != 1
            or not any(spec.role == "ffn" for spec in self.specs)
            or len({spec.name for spec in self.specs}) != len(self.specs)
            or any(spec.role not in ("attention", "ffn") for spec in self.specs)
            or any(spec.role == "ffn" and not spec.ffn_lanes for spec in self.specs)
            or any(not re.fullmatch(r"[\w-]+", spec.name) for spec in self.specs)
        ):
            raise ValueError("require one A, at least one F, unique names and F lanes")
        if os.name != "posix":
            raise ValueError("AFDServerGroup requires POSIX process sessions")
        if min(startup_timeout, shutdown_timeout, poll_interval) <= 0:
            raise ValueError("timeouts and polling interval must be positive")
        self.base_url = base_url.rstrip("/")
        self.log_dir = Path(log_dir)
        self.startup_timeout = startup_timeout
        self.shutdown_timeout = shutdown_timeout
        self.poll_interval = poll_interval
        self.api_key = api_key
        self.processes: dict[str, subprocess.Popen] = {}
        self._roots: dict[str, _OwnedProcess] = {}
        self._owned: dict[tuple[int, float], _OwnedProcess] = {}
        self.log_paths = {
            spec.name: self.log_dir / f"{spec.name}.log" for spec in specs
        }
        self.ready_lanes = {spec.name: set() for spec in specs if spec.role == "ffn"}
        self._logs = []
        self._stop = threading.Event()
        self._close_lock = threading.Lock()
        self._monitor = None
        self._failure: str | None = None
        self._started = False
        self._closed = False
        self.forced_kill_pids: set[int] = set()

    def start(self) -> AFDServerGroup:
        if self._started or self._closed:
            raise RuntimeError("AFDServerGroup is single-use; restart the whole group")
        self._started = True
        self.log_dir.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.startup_timeout
        try:
            # Launch every side before waiting for either side's readiness.
            for spec in self.specs:
                log = self.log_paths[spec.name].open("w")
                self._logs.append(log)
                self.processes[spec.name] = subprocess.Popen(
                    list(spec.command),
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    env={**os.environ, **spec.env, "PYTHONUNBUFFERED": "1"},
                    start_new_session=True,
                )
                root = _OwnedProcess.capture(self.processes[spec.name].pid)
                self._roots[spec.name] = root
                self._owned[(root.process.pid, root.created)] = root
            self._monitor = threading.Thread(target=self._watch, daemon=True)
            self._monitor.start()
            while time.monotonic() < deadline:
                self.check_healthy()
                ffn_ready = True
                for spec in self.specs:
                    if spec.role != "ffn":
                        continue
                    text = self.log_paths[spec.name].read_text(errors="replace")
                    self.ready_lanes[spec.name].update(
                        map(int, _FFN_READY.findall(text))
                    )
                    ffn_ready &= set(spec.ffn_lanes) <= self.ready_lanes[spec.name]
                if ffn_ready and self._http_ready(deadline):
                    self.check_healthy()
                    if time.monotonic() < deadline:
                        return self
                self._stop.wait(
                    min(self.poll_interval, max(0, deadline - time.monotonic()))
                )
            raise TimeoutError(
                f"AFD group readiness timed out; F lanes={self.ready_lanes}; logs={self.log_dir}"
            )
        except BaseException:
            self.close()
            raise

    def _http_ready(self, deadline):
        env = os.environ.copy()
        if self.api_key:
            env["AFD_HEALTH_AUTH"] = f"Bearer {self.api_key}"
        else:
            env.pop("AFD_HEALTH_AUTH", None)
        probe = subprocess.Popen(
            [sys.executable, "-c", _HEALTH_PROBE, self.base_url + "/health_generate"],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        try:
            while time.monotonic() < deadline:
                self.check_healthy()
                code = probe.poll()
                if code is not None:
                    return code == 0 and time.monotonic() < deadline
                self._stop.wait(
                    min(self.poll_interval, max(0, deadline - time.monotonic()))
                )
            return False
        finally:
            if probe.poll() is None:
                probe.kill()
            probe.wait(timeout=2)

    def _remember_children(self):
        for root in self._roots.values():
            if not root.is_live():
                continue
            children = []
            for pid in collect_process_tree_pids(root.process.pid):
                try:
                    child = _OwnedProcess.capture(pid)
                    # A PID from the earlier tree scan may already have been
                    # reused. Adopt only a member of this owned session/tree.
                    if child.is_live() and (
                        os.getsid(pid) == root.process.pid
                        or root.process in child.process.parents()
                    ):
                        children.append(child)
                except (psutil.NoSuchProcess, ProcessLookupError):
                    pass
            # Never adopt children of a PID which changed owner during discovery.
            if root.is_live():
                for child in children:
                    self._owned[(child.process.pid, child.created)] = child

    def _watch(self):
        while not self._stop.wait(self.poll_interval):
            self._remember_children()
            for name, process in self.processes.items():
                code = process.poll()
                if code is not None and not self._stop.is_set():
                    self._failure = f"AFD {name} exited unexpectedly: rc={code}; log={self.log_paths[name]}"
                    self.close()
                    return

    def check_healthy(self):
        if not self._started:
            raise RuntimeError("AFD group has not started")
        if self._failure is not None:
            raise RuntimeError(self._failure)
        if self._closed:
            raise RuntimeError("AFD group is closed")
        for name, process in self.processes.items():
            code = process.poll()
            if code is not None:
                self._failure = f"AFD {name} exited unexpectedly: rc={code}; log={self.log_paths[name]}"
                raise RuntimeError(self._failure)

    def close(self):
        self._stop.set()
        with self._close_lock:
            if self._closed:
                return
            errors = []

            def attempt(action):
                try:
                    action()
                except (OSError, psutil.Error, subprocess.TimeoutExpired) as exc:
                    errors.append(str(exc))

            attempt(self._remember_children)
            owned = tuple(self._owned.values())

            def find_live():
                live = []
                for member in owned:
                    try:
                        if member.is_live():
                            live.append(member)
                    except (OSError, psutil.Error) as exc:
                        errors.append(str(exc))
                        live.append(member)  # Unknown ownership is not a clean exit.
                return live

            deadline = time.monotonic() + self.shutdown_timeout
            try:
                for spec in self.specs:
                    root = self._roots.get(spec.name)
                    if spec.role == "attention" and root:
                        attempt(
                            lambda root=root: (
                                root.process.terminate() if root.is_live() else None
                            )
                        )
                # Let A issue CLOSE and F acknowledge it before forcing F to stop.
                while time.monotonic() < deadline:
                    if all(
                        process.poll() is not None
                        for process in self.processes.values()
                    ):
                        break
                    time.sleep(
                        min(self.poll_interval, max(0, deadline - time.monotonic()))
                    )
            finally:
                # A surviving, identity-checked session member authorizes group
                # signaling even if the original launcher has already exited.
                for root in self._roots.values():
                    for member in owned:
                        try:
                            if (
                                member.is_live()
                                and os.getsid(member.process.pid) == root.process.pid
                                and os.getpgid(member.process.pid) == root.process.pid
                            ):
                                os.killpg(root.process.pid, signal.SIGKILL)
                                self.forced_kill_pids.add(member.process.pid)
                                break
                        except (OSError, psutil.Error):
                            # Individual identity-aware signals below are the
                            # fallback on hosts which restrict group signals.
                            pass

                # psutil keeps PID creation identity and checks reuse before
                # signaling; never rebuild an unrelated process from a stale PID.
                def kill_owned(member):
                    if member.is_live():
                        member.process.kill()
                        self.forced_kill_pids.add(member.process.pid)

                for member in owned:
                    attempt(lambda member=member: kill_owned(member))
                reap_deadline = time.monotonic() + 2
                try:
                    for process in self.processes.values():
                        attempt(
                            lambda process=process: process.wait(
                                timeout=max(0.001, reap_deadline - time.monotonic())
                            )
                        )
                    live = find_live()
                    while live and time.monotonic() < reap_deadline:
                        time.sleep(
                            min(
                                self.poll_interval,
                                max(0, reap_deadline - time.monotonic()),
                            )
                        )
                        live = find_live()
                    self._closed = not live
                    if live:
                        errors.append(
                            f"survived teardown: {[member.process.pid for member in live]}"
                        )
                finally:
                    for log in self._logs:
                        attempt(log.close)
                if errors:
                    raise RuntimeError("AFD teardown failed: " + "; ".join(errors))

    def __enter__(self):
        return self.start()

    def __exit__(self, exc_type, exc, tb):
        self.close()
        if exc_type is None and self._failure is not None:
            raise RuntimeError(self._failure)
