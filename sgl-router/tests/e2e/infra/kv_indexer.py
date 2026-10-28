# SPDX-FileCopyrightText: Copyright (c) 2026 The SGLang Authors
# SPDX-License-Identifier: Apache-2.0

"""The standalone KV Indexer stack, wired the way a deployment wires it.

    engine --ZMQ kv_events--> bridge --gRPC--> indexer --> Tair KVCache Manager
    router --gRPC prefix query--> indexer

Started with ``--kv-indexer-endpoint`` the router keeps no KV tree of its own
(``KvEventIndex::new_metadata_only_*``), and the Tair KVCache backend keeps no
placements of its own either — every prefix answer is a ``GetHostCacheState``
RPC to the manager. Both ends are therefore stateless for placements, which is
what lets a test treat "the repeat came back" as "the manager knew".

The manager is provisioned by the crate's own harness script (event-report
storages, instance group, instance) — the same script the ``tier-kv-indexer``
job runs, driven programmatically here so a test owns a manager of its own.
The three processes are thin: this module is process plumbing plus the
environment each of them needs, and nothing else.

Process binaries come from the router workspace's release build (the same
``CARGO_TARGET_DIR`` convention the router itself uses). The manager is not
built here: ``TAIR_KVCACHE_MANAGER_BINARY`` names it, and CI fills that from
the artifact ``tier-kv-indexer`` uploads, so the GPU runner only runs the
stack.
"""

from __future__ import annotations

import os
import shlex
import signal
import socket
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Protocol

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_HARNESS = (
    _ROOT / "sgl-kv-indexer/tests/test_tair_kvcache_live/setup_tair_kvcache_manager.py"
)

# ``infra/model_pool.py`` starts the workers' publisher with this topic, so the
# bridge's ZMQ subscription prefix has to match it or every message is
# filtered out before the bridge ever sees one.
KV_EVENT_TOPIC = "kv"

# The harness runs the manager with deliberately tight liveness knobs (2s
# timeout, 2s grace) so its reclamation tests finish in seconds, while the
# backend's production cadence is 30s. At that cadence the manager reclaims the
# reporter between two beats and drops its placements, so the beat has to lead
# the timeout; 300ms is the value the crate's own live suite beats at.
HEARTBEAT_INTERVAL_MS = 300


def release_binary(name: str) -> Path:
    """``name`` from the router workspace's release build."""
    target_dir = os.environ.get("CARGO_TARGET_DIR", str(_ROOT / "target"))
    return Path(target_dir) / "release" / name


def kvcm_binary() -> Path | None:
    """The manager binary named by ``TAIR_KVCACHE_MANAGER_BINARY``, if any.

    Named after the harness's own ``TAIR_KVCACHE_MANAGER_REPO``: both point at
    the same checkout, one at the tree, one at the built binary.
    """
    raw = os.environ.get("TAIR_KVCACHE_MANAGER_BINARY", "")
    return Path(raw) if raw else None


def missing_stack_binaries() -> list[str]:
    """Binaries the stack needs that are not on disk, for a skip/fail message."""
    missing = [
        str(path)
        for path in (
            release_binary("kv-indexer-server"),
            release_binary("kv-indexer-bridge"),
        )
        if not path.exists()
    ]
    manager = kvcm_binary()
    if manager is None:
        missing.append("TAIR_KVCACHE_MANAGER_BINARY (unset)")
    elif not manager.exists():
        missing.append(str(manager))
    return missing


def _free_port() -> int:
    """An ephemeral TCP port.

    Unlike the worker helpers this does not keep the port inside a fixed
    range: that range exists for SGLang's ``port + 10000`` gRPC derivation,
    which none of these processes use.
    """
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_tcp(port: int, *, timeout: float, process: subprocess.Popen) -> None:
    """Wait until ``port`` accepts connections, or the process dies.

    Neither the indexer's gRPC listener nor the manager's admin port has a
    cheap health endpoint, so a connect is the readiness signal; the process
    check keeps a crash from looking like a timeout.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1.0)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        if process.poll() is not None:
            raise RuntimeError(
                f"process exited with {process.returncode} before 127.0.0.1:{port} was up"
            )
        time.sleep(0.2)
    raise RuntimeError(
        f"nothing accepted a connection on 127.0.0.1:{port} in {timeout}s"
    )


def _spawn(
    cmd: list[str], env: dict[str, str], log_path: Path
) -> tuple[subprocess.Popen, IO[bytes]]:
    """Run ``cmd`` in its own process group, appending both streams to a log.

    Its own group so a stop takes the whole tree with it, and a log file
    because a stack this thin is diagnosed from what its processes said.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("ab")
    process = subprocess.Popen(
        cmd, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    return process, log


def _stop(process: subprocess.Popen | None, log: IO[bytes] | None) -> None:
    """SIGTERM the process group, escalating to SIGKILL, and close its log."""
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        else:
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    pass
    if log is not None and not log.closed:
        log.close()


def _log_tail(*candidates: Path) -> str:
    """The tail of the first log that exists.

    The manager keeps its own log directory under the workdir while the
    harness redirects its stdout, so both are candidates; a missing one is
    then a blank section rather than a silently empty diagnostic.
    """
    for path in candidates:
        if path.exists():
            return "\n".join(path.read_text(errors="replace").splitlines()[-40:])
    return "(no log)"


@dataclass
class KvcmManager:
    """A provisioned Tair KVCache Manager, run by the crate's harness script.

    The harness owns provisioning (storages, instance group, instance) and is
    idempotent per workdir; the environment it writes is what tells the
    indexer which instance to report into and how its blocks are keyed.
    """

    workdir: Path
    env: dict[str, str] = field(default_factory=dict)
    # Whether this object started the manager, so teardown does not depend on
    # the environment file having been read successfully.
    _started: bool = field(default=False, init=False, repr=False)

    def _harness(self, *args: str) -> None:
        result = subprocess.run(
            [sys.executable, str(_HARNESS), *args], capture_output=True, text=True
        )
        if result.returncode != 0:
            # The harness prints why it gave up; keeping that out of the
            # message would turn every provisioning failure into a bare code.
            raise RuntimeError(
                f"manager harness {' '.join(args)} failed ({result.returncode}):\n"
                f"{result.stdout}\n{result.stderr}"
            )

    def start(self) -> None:
        binary = kvcm_binary()
        if binary is None:
            raise RuntimeError(
                "TAIR_KVCACHE_MANAGER_BINARY must name the manager binary"
            )
        self.workdir.mkdir(parents=True, exist_ok=True)
        self._harness("start", "--binary", str(binary), "--workdir", str(self.workdir))
        self._started = True
        try:
            self.env = _read_env_file(self.workdir / "tair_kvcache_live.env")
        except Exception:
            # A half-started manager would otherwise outlive the failure and
            # collide with the next attempt on the same workdir.
            self.stop()
            raise

    def stop(self) -> None:
        if not self._started:
            return
        self._started = False
        # Teardown must not replace whatever the test was already failing on;
        # the manager is per-workdir and the harness reclaims it either way.
        with suppress(RuntimeError):
            self._harness("stop", "--workdir", str(self.workdir))
        self.env = {}

    def log_tail(self) -> str:
        return _log_tail(
            self.workdir / "logs" / "kv_cache_manager.log", self.workdir / "stderr.log"
        )

    def __enter__(self) -> "KvcmManager":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


def _read_env_file(path: Path) -> dict[str, str]:
    """The ``export K=V`` lines the harness writes, as a dict."""
    env: dict[str, str] = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, _, value = line.removeprefix("export ").partition("=")
        env[key] = shlex.split(value)[0] if value else ""
    return env


@dataclass
class IndexerService:
    """``kv-indexer-server``: the gRPC prefix index the router queries.

    Configured with the manager's endpoint and instance, so every fact it
    answers with has been through Tair KVCache.
    """

    kvcm_env: dict[str, str]
    log_dir: Path
    port: int = field(default_factory=_free_port)
    process: subprocess.Popen | None = None
    log: IO[bytes] | None = None

    @property
    def endpoint(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def log_path(self) -> Path:
        return self.log_dir / f"kv-indexer-server-{self.port}.log"

    def start(self) -> None:
        env = {
            **os.environ,
            "KV_INDEXER_LISTEN_ADDR": f"127.0.0.1:{self.port}",
            "KV_INDEXER_BACKEND": "tair_kvcache_kvcm",
            "KV_INDEXER_TAIR_KVCACHE_KVCM_ENDPOINT": self.kvcm_env[
                "TAIR_KVCACHE_LIVE_ENDPOINT"
            ],
            "KV_INDEXER_TAIR_KVCACHE_KVCM_INSTANCE_ID": self.kvcm_env[
                "TAIR_KVCACHE_LIVE_INSTANCE_ID"
            ],
            "KV_INDEXER_TAIR_KVCACHE_KVCM_HEARTBEAT_INTERVAL_MS": str(
                HEARTBEAT_INTERVAL_MS
            ),
        }
        self.process, self.log = _spawn(
            [str(release_binary("kv-indexer-server"))], env, self.log_path
        )
        try:
            _wait_tcp(self.port, timeout=60.0, process=self.process)
        except Exception:
            self.stop()
            raise

    def stop(self) -> None:
        _stop(self.process, self.log)
        self.process = None
        self.log = None

    def log_tail(self) -> str:
        return _log_tail(self.log_path)

    def __enter__(self) -> "IndexerService":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


@dataclass
class Bridge:
    """``kv-indexer-bridge``: one worker's ZMQ KV events -> indexer gRPC.

    One process per worker, because the worker's KV-transfer address is what
    the indexer hands back to the router as the owner of a prefix.

    No cache spec is declared. SGLang's ZMQ stream carries whole-block events
    with no component detail, and the backend fails closed on a worker that
    declares components without reporting them: it would exclude the worker
    from every prefix answer. Undeclared, the worker is a plain FULL-only
    reporter and stays answerable.
    """

    worker_url: str
    kv_events_endpoint: str
    indexer_endpoint: str
    log_dir: Path
    process: subprocess.Popen | None = None
    log: IO[bytes] | None = None

    @property
    def log_path(self) -> Path:
        return self.log_dir / f"kv-indexer-bridge-{Path(self.worker_url).name}.log"

    def start(self) -> None:
        env = {
            **os.environ,
            "KV_INDEXER_WORKER_ID": self.worker_url,
            "KV_INDEXER_WORKER_ADDRESS": self.worker_url,
            # The worker advertises a bind address (`tcp://*:<port>`); a
            # subscriber has to connect to a concrete host.
            "SGLANG_KV_EVENT_ENDPOINT": self.kv_events_endpoint.replace(
                "tcp://*:", "tcp://127.0.0.1:"
            ),
            "SGLANG_KV_EVENT_TOPIC": KV_EVENT_TOPIC,
            "KV_INDEXER_ENDPOINT": self.indexer_endpoint,
        }
        self.process, self.log = _spawn(
            [str(release_binary("kv-indexer-bridge"))], env, self.log_path
        )

    def stop(self) -> None:
        _stop(self.process, self.log)
        self.process = None
        self.log = None

    def log_tail(self) -> str:
        return _log_tail(self.log_path)

    def __enter__(self) -> "Bridge":
        self.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop()


class _HasLog(Protocol):
    def log_tail(self) -> str: ...


def stack_logs(*parts: _HasLog) -> str:
    """Every process's log tail, for a failure message."""
    return "\n".join(
        f"--- {type(part).__name__} ---\n{part.log_tail()}" for part in parts
    )
