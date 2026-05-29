"""Caller-side resource: long-lived HTTP server + per-test script dispatch.

Owned by :class:`ScriptedTestCase`. Spawns a dedicated subprocess
that runs :func:`execute_scripted_http_server`, which launches a real sglang
HTTP server; the caller communicates with the scheduler-side
``router_script`` over a unix-socket :class:`multiprocessing.connection.Listener`.

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
import secrets
import socket
import tempfile
from multiprocessing.connection import Connection, Listener
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

from sglang.srt.utils.network import get_free_port
from sglang.test.scripted_runtime.entrypoint import execute_scripted_http_server
from sglang.test.scripted_runtime.router_script import router_script

DEFAULT_RUN_TIMEOUT_S: float = 120.0
SHUTDOWN_JOIN_TIMEOUT_S: float = 60.0
LISTENER_ACCEPT_TIMEOUT_S: float = 300.0
SERVER_HOST: str = "127.0.0.1"


class ScriptedHttpServer:
    """Long-lived HTTP-server process wrapper. One instance per test class."""

    def __init__(
        self,
        *,
        listener: Listener,
        conn: Connection,
        server_process: mp.process.BaseProcess,
        socket_dir: Path,
        traceback_path: Path,
    ) -> None:
        self._listener = listener
        self._conn = conn
        self._server_process = server_process
        self._socket_dir = socket_dir
        self._traceback_path = traceback_path
        self._shutdown_done = False
        self._dirty: Optional[str] = None

    @classmethod
    def start(cls, **engine_kwargs: Any) -> "ScriptedHttpServer":
        """Spawn the HTTP-server process and accept the router connection.

        The listener is bound *before* the server process starts so the
        router's :class:`Client` connect call never races. If the server
        fails to start (slow load, model error, CUDA failure) the listener
        accept times out after :data:`LISTENER_ACCEPT_TIMEOUT_S` instead of
        hanging the test process indefinitely.

        ``engine_kwargs`` (notably ``model_path`` plus any ServerArgs
        overrides) are forwarded to :func:`execute_scripted_http_server`.
        """
        socket_dir = Path(tempfile.mkdtemp(prefix="sglang_scripted_ipc_"))
        addr = str(socket_dir / "ipc.sock")
        authkey = secrets.token_bytes(16)

        tb_fd, tb_path = tempfile.mkstemp(prefix="scripted_runtime_tb_", suffix=".txt")
        os.close(tb_fd)
        traceback_path = Path(tb_path)

        listener = Listener(addr, authkey=authkey)
        # Bound socket timeout so a stuck server startup surfaces as
        # TimeoutError instead of blocking forever on accept().
        listener_socket: socket.socket = listener._listener._socket
        listener_socket.settimeout(LISTENER_ACCEPT_TIMEOUT_S)

        # Spawn-mode children snapshot os.environ at start(); set the IPC
        # coordinates before launching so the router can connect.
        os.environ["SGLANG_SCRIPTED_RUNTIME_IPC_ADDR"] = addr
        os.environ["SGLANG_SCRIPTED_RUNTIME_AUTHKEY"] = authkey.hex()

        host = SERVER_HOST
        port = get_free_port()

        ctx = mp.get_context("spawn")
        server_process = ctx.Process(
            target=execute_scripted_http_server,
            args=(router_script,),
            kwargs=dict(
                host=host,
                port=port,
                traceback_path=str(traceback_path),
                **engine_kwargs,
            ),
            name="scripted-runtime-http-server",
            daemon=False,
        )
        server_process.start()

        try:
            conn = listener.accept()
        except (socket.timeout, TimeoutError) as exc:
            listener.close()
            cls._terminate_process(server_process)
            raise TimeoutError(
                f"ScriptedHttpServer: HTTP server did not connect within "
                f"{LISTENER_ACCEPT_TIMEOUT_S}s"
            ) from exc

        return cls(
            listener=listener,
            conn=conn,
            server_process=server_process,
            socket_dir=socket_dir,
            traceback_path=traceback_path,
        )

    def execute_script(
        self,
        script_fn: Callable,
        *,
        args: Tuple[Any, ...] = (),
        timeout_s: float = DEFAULT_RUN_TIMEOUT_S,
    ) -> None:
        """Dispatch ``script_fn`` to the router and block on its result.

        ``args`` are forwarded positionally to the script after the
        ``ScriptedContext`` handle (``script_fn(t, *args)``), letting one
        parameterized script back many ``subTest`` combos. They cross the
        IPC pipe, so every element must be picklable.

        Re-raises the captured traceback as :class:`AssertionError` if the
        sub-script failed; raises :class:`TimeoutError` (and marks the
        session dirty) if the server did not respond within ``timeout_s``;
        raises :class:`RuntimeError` if the server process died before
        responding.
        """
        if self._dirty:
            raise RuntimeError(f"ScriptedHttpServer is dirty: {self._dirty}")

        fn_path = f"{script_fn.__module__}:{script_fn.__qualname__}"
        self._conn.send({"kind": "run", "fn_path": fn_path, "args": args})

        if not self._conn.poll(timeout_s):
            if not self._server_process.is_alive():
                self._dirty = f"server process died before responding to {fn_path!r}"
                raise RuntimeError(self._dirty)
            self._dirty = f"script {fn_path!r} timed out after {timeout_s}s"
            raise TimeoutError(self._dirty)

        result = self._conn.recv()
        if result["kind"] == "error":
            raise AssertionError(f"scripted-runtime script failed:\n{result['tb']}")
        if result["kind"] != "ok":
            raise RuntimeError(
                f"router replied with unexpected kind {result['kind']!r}"
            )

    def shutdown(self) -> None:
        """Tell the router to return, join the server process, and clean up.

        Idempotent — calling twice is safe. Surfaces a fatal scheduler-side
        traceback (written to the traceback file) as an ``AssertionError``.
        """
        if self._shutdown_done:
            return

        fatal_tb: Optional[str] = None
        try:
            try:
                self._conn.send({"kind": "shutdown"})
            except (BrokenPipeError, EOFError, OSError):
                pass

            self._server_process.join(timeout=SHUTDOWN_JOIN_TIMEOUT_S)
            self._terminate_process(self._server_process)

            fatal_tb = self._read_traceback()
        finally:
            for closer in (self._conn.close, self._listener.close):
                try:
                    closer()
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
        try:
            for p in self._socket_dir.iterdir():
                p.unlink()
            self._socket_dir.rmdir()
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
