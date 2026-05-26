"""Caller-side resource: long-lived engine + per-test script dispatch.

Owned by :class:`ScriptedRuntimeTestCase`. Internally delegates engine
startup to :func:`execute_scripted_runtime` in a background thread; the
caller communicates with the scheduler-side ``router_script`` over a
unix-socket :class:`multiprocessing.connection.Listener`.

One :class:`ScriptedRuntimeSession` instance per test class — every
``test_*`` method shares the same engine and submits a fresh
``_script_*`` over the socket via :meth:`ScriptedRuntimeSession.run`.
"""

from __future__ import annotations

import os
import secrets
import socket
import tempfile
import threading
from multiprocessing.connection import Connection, Listener
from pathlib import Path
from typing import Any, Callable, Optional

from sglang.test.scripted_runtime.entrypoint import execute_scripted_runtime
from sglang.test.scripted_runtime.router_script import router_script

DEFAULT_RUN_TIMEOUT_S: float = 120.0
SHUTDOWN_JOIN_TIMEOUT_S: float = 60.0
LISTENER_ACCEPT_TIMEOUT_S: float = 300.0


class ScriptedRuntimeSession:
    """Long-lived Engine wrapper. One instance per test class."""

    def __init__(
        self,
        *,
        listener: Listener,
        conn: Connection,
        engine_thread: threading.Thread,
        socket_dir: Path,
    ) -> None:
        self._listener = listener
        self._conn = conn
        self._engine_thread = engine_thread
        self._socket_dir = socket_dir
        self._shutdown_done = False
        self._dirty: Optional[str] = None

    @classmethod
    def start(cls, **engine_kwargs: Any) -> "ScriptedRuntimeSession":
        """Start the engine in a background thread and accept the router connection.

        The listener is bound *before* the engine thread starts so the
        router's :class:`Client` connect call never races. If the engine
        fails to start (slow load, model error, CUDA failure) the listener
        accept times out after :data:`LISTENER_ACCEPT_TIMEOUT_S` instead
        of hanging the test process indefinitely.
        """
        socket_dir = Path(tempfile.mkdtemp(prefix="sglang_scripted_ipc_"))
        addr = str(socket_dir / "ipc.sock")
        authkey = secrets.token_bytes(16)

        listener = Listener(addr, authkey=authkey)
        # Bound socket timeout so a stuck engine startup surfaces as
        # TimeoutError instead of blocking forever on accept().
        listener_socket: socket.socket = listener._listener._socket
        listener_socket.settimeout(LISTENER_ACCEPT_TIMEOUT_S)

        os.environ["SGLANG_SCRIPTED_RUNTIME_IPC_ADDR"] = addr
        os.environ["SGLANG_SCRIPTED_RUNTIME_AUTHKEY"] = authkey.hex()

        engine_thread = threading.Thread(
            target=execute_scripted_runtime,
            args=(router_script,),
            kwargs=engine_kwargs,
            name="scripted-runtime-engine",
            daemon=False,
        )
        engine_thread.start()

        try:
            conn = listener.accept()
        except (socket.timeout, TimeoutError) as exc:
            listener.close()
            raise TimeoutError(
                f"ScriptedRuntimeSession: engine did not connect within "
                f"{LISTENER_ACCEPT_TIMEOUT_S}s"
            ) from exc

        return cls(
            listener=listener,
            conn=conn,
            engine_thread=engine_thread,
            socket_dir=socket_dir,
        )

    def run(
        self,
        script_fn: Callable,
        *,
        timeout_s: float = DEFAULT_RUN_TIMEOUT_S,
    ) -> None:
        """Dispatch ``script_fn`` to the router and block on its result.

        Re-raises the captured traceback as :class:`AssertionError` if
        the sub-script failed; raises :class:`TimeoutError` (and marks
        the session dirty) if the engine did not respond within
        ``timeout_s``; raises :class:`RuntimeError` if the engine thread
        died before responding.
        """
        if self._dirty:
            raise RuntimeError(f"ScriptedRuntimeSession is dirty: {self._dirty}")
        fn_path = f"{script_fn.__module__}:{script_fn.__qualname__}"
        self._conn.send({"kind": "run", "fn_path": fn_path})
        if not self._conn.poll(timeout_s):
            if not self._engine_thread.is_alive():
                self._dirty = f"engine thread died before responding to {fn_path!r}"
                raise RuntimeError(self._dirty)
            self._dirty = f"script {fn_path!r} timed out after {timeout_s}s"
            raise TimeoutError(self._dirty)
        result = self._conn.recv()
        if result["kind"] == "error":
            raise AssertionError(f"ScriptedRuntime script failed:\n{result['tb']}")
        if result["kind"] != "ok":
            raise RuntimeError(
                f"router replied with unexpected kind {result['kind']!r}"
            )

    def shutdown(self) -> None:
        """Tell the router to return, join the engine thread, and clean up.

        Idempotent — calling twice is safe.
        """
        if self._shutdown_done:
            return
        try:
            try:
                self._conn.send({"kind": "shutdown"})
            except (BrokenPipeError, EOFError, OSError):
                pass
            self._engine_thread.join(timeout=SHUTDOWN_JOIN_TIMEOUT_S)
        finally:
            for closer in (self._conn.close, self._listener.close):
                try:
                    closer()
                except Exception:  # noqa: BLE001 — best-effort cleanup
                    pass
            try:
                for p in self._socket_dir.iterdir():
                    p.unlink()
                self._socket_dir.rmdir()
            except OSError:
                pass
            self._shutdown_done = True

    def __enter__(self) -> "ScriptedRuntimeSession":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.shutdown()
