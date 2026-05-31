"""Caller-side resource: long-lived HTTP server + per-test script dispatch.

Owned by :class:`ScriptedTestCase`. Spawns a dedicated subprocess
that runs :func:`launch_scripted_http_server`, which launches a real sglang
HTTP server; the caller communicates with the scheduler-side dispatch loop
(owned by :class:`ScriptedSchedulerHook`) over a ZMQ ``PAIR`` socket bound
to a loopback ``tcp://`` endpoint — the same transport sglang uses for its
own inter-process ZMQ channels.

One :class:`ScriptedHttpServer` instance per test class — every
``test_*`` method shares the same HTTP server / engine and submits a fresh
``_script_*`` over the socket via :meth:`ScriptedHttpServer.run`.

The HTTP server runs in its own process (not a background thread of the
test process) so uvicorn can install its signal handlers in that process's
main thread, and so the topology matches a real deployment.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import zmq

from sglang.srt.environ import envs
from sglang.srt.utils.network import get_free_port, get_zmq_socket_on_host
from sglang.test.scripted_runtime.http_server_subprocess import (
    launch_scripted_http_server,
)
from sglang.test.scripted_runtime.io_struct import (
    HookReady,
    RunScript,
    ScriptFailed,
    ScriptSucceeded,
    Shutdown,
)

DEFAULT_RUN_TIMEOUT_S: float = 120.0
SHUTDOWN_JOIN_TIMEOUT_S: float = 60.0
LISTENER_ACCEPT_TIMEOUT_S: float = 300.0
SERVER_HOST: str = "127.0.0.1"


class ScriptedHttpServer:
    """Long-lived HTTP-server process wrapper. One instance per test class."""

    def __init__(
        self,
        *,
        ctx: zmq.Context,
        socket: zmq.Socket,
        server_process: mp.process.BaseProcess,
        traceback_path: Path,
    ) -> None:
        self._ctx = ctx
        self._socket = socket
        self._server_process = server_process
        self._traceback_path = traceback_path
        self._shutdown_done = False
        self._dirty: Optional[str] = None

    @classmethod
    def start(cls, **engine_kwargs: Any) -> "ScriptedHttpServer":
        """Spawn the HTTP-server process and await the dispatch loop's handshake.

        The PAIR socket binds to a random loopback ``tcp://`` port *before* the
        server process starts so the dispatch loop's connect call never
        races. If the server fails to start (slow load, model error, CUDA
        failure) the first-message poll times out after
        :data:`LISTENER_ACCEPT_TIMEOUT_S` instead of hanging the test process
        indefinitely. The dispatch loop sends a :class:`HookReady` as its
        first message once it connects, which the startup handshake confirms.

        The socket binds to ``127.0.0.1`` only (no authkey), so it is reachable
        from the spawned server process but not from the network — matching
        sglang's no-auth, loopback-only ZMQ posture (CVE-2026-3060).

        ``engine_kwargs`` (notably ``model_path`` plus any ServerArgs
        overrides) are forwarded to :func:`launch_scripted_http_server`.
        """
        tb_fd, tb_path = tempfile.mkstemp(prefix="scripted_runtime_tb_", suffix=".txt")
        os.close(tb_fd)
        traceback_path = Path(tb_path)

        ctx = zmq.Context()
        dispatch_port, socket = get_zmq_socket_on_host(ctx, zmq.PAIR, host=SERVER_HOST)
        endpoint = f"tcp://{SERVER_HOST}:{dispatch_port}"

        host = SERVER_HOST
        port = get_free_port()

        mp_ctx = mp.get_context("spawn")
        server_process = mp_ctx.Process(
            target=launch_scripted_http_server,
            args=(),
            kwargs=dict(
                host=host,
                port=port,
                traceback_path=str(traceback_path),
                **engine_kwargs,
            ),
            name="scripted-runtime-http-server",
            daemon=False,
        )
        # Spawn-mode children snapshot os.environ at start(); seed the IPC
        # endpoint via override so the spawned process (and the scheduler it
        # launches) inherit it, then restore the caller's env.
        with envs.SGLANG_TEST_SCRIPTED_RUNTIME_IPC_ADDR.override(endpoint):
            server_process.start()

        # Poll for the dispatch loop's HookReady; a stuck server startup
        # surfaces as TimeoutError instead of blocking forever (poll takes
        # milliseconds).
        if not socket.poll(int(LISTENER_ACCEPT_TIMEOUT_S * 1000)):
            socket.setsockopt(zmq.LINGER, 0)
            socket.close()
            ctx.term()
            cls._terminate_process(server_process)
            raise TimeoutError(
                f"ScriptedHttpServer: HTTP server did not connect within "
                f"{LISTENER_ACCEPT_TIMEOUT_S}s"
            )

        ready = socket.recv_pyobj()
        assert isinstance(
            ready, HookReady
        ), f"ScriptedHttpServer: expected HookReady handshake, got {ready!r}"

        return cls(
            ctx=ctx,
            socket=socket,
            server_process=server_process,
            traceback_path=traceback_path,
        )

    def execute_script(
        self,
        script_fn: Callable,
        *,
        args: Tuple[Any, ...] = (),
        timeout_s: float = DEFAULT_RUN_TIMEOUT_S,
    ) -> None:
        """Dispatch ``script_fn`` to the scheduler hook and block on its result.

        ``args`` are forwarded positionally to the script after the
        ``ScriptedContext`` handle (``script_fn(t, *args)``), letting one
        parameterized script back many ``subTest`` combos. They cross the
        ipc socket as a pickled :class:`RunScript`, so every element must be
        picklable.

        Re-raises the captured traceback as :class:`AssertionError` if the
        sub-script failed; raises :class:`TimeoutError` (and marks the
        session dirty) if the server did not respond within ``timeout_s``;
        raises :class:`RuntimeError` if the server process died before
        responding.
        """
        if self._dirty:
            raise RuntimeError(f"ScriptedHttpServer is dirty: {self._dirty}")

        fn_path = f"{script_fn.__module__}:{script_fn.__qualname__}"
        self._socket.send_pyobj(RunScript(fn_path=fn_path, args=args))

        if not self._socket.poll(int(timeout_s * 1000)):
            if not self._server_process.is_alive():
                self._dirty = f"server process died before responding to {fn_path!r}"
                raise RuntimeError(self._dirty)
            self._dirty = f"script {fn_path!r} timed out after {timeout_s}s"
            raise TimeoutError(self._dirty)

        reply = self._socket.recv_pyobj()
        match reply:
            case ScriptFailed(traceback=tb):
                raise AssertionError(f"scripted-runtime script failed:\n{tb}")
            case ScriptSucceeded():
                return
            case _:
                raise RuntimeError(
                    f"scheduler hook replied with unexpected message {reply!r}"
                )

    def shutdown(self) -> None:
        """Tell the dispatch loop to return, join the server process, and clean up.

        Idempotent — calling twice is safe. Surfaces a fatal scheduler-side
        traceback (written to the traceback file) as an ``AssertionError``.
        """
        if self._shutdown_done:
            return

        fatal_tb: Optional[str] = None
        try:
            try:
                self._socket.send_pyobj(Shutdown())
            except zmq.ZMQError:
                pass

            self._server_process.join(timeout=SHUTDOWN_JOIN_TIMEOUT_S)
            self._terminate_process(self._server_process)

            fatal_tb = self._read_traceback()
        finally:
            try:
                self._socket.setsockopt(zmq.LINGER, 0)
                self._socket.close()
                self._ctx.term()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
            self._cleanup_files()
            self._shutdown_done = True

        if fatal_tb:
            raise AssertionError(f"scripted-runtime server failed:\n{fatal_tb}")

    def __enter__(self) -> "ScriptedHttpServer":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()

    def _read_traceback(self) -> Optional[str]:
        try:
            text = self._traceback_path.read_text()
        except OSError:
            return None
        text = text.strip()
        return text or None

    def _cleanup_files(self) -> None:
        try:
            self._traceback_path.unlink()
        except OSError:
            pass

    @staticmethod
    def _terminate_process(process: mp.process.BaseProcess) -> None:
        """Best-effort terminate + join; escalate to kill if it lingers."""
        if not process.is_alive():
            return
        process.terminate()
        process.join(timeout=10.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=10.0)
