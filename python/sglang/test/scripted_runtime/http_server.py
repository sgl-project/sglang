from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import zmq

from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.environ import envs
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.network import get_free_port, get_zmq_socket_on_host
from sglang.test.scripted_runtime.io_struct import (
    HookReady,
    OutOfBandError,
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

    def __init__(
        self,
        *,
        ctx: zmq.Context,
        socket: zmq.Socket,
        server_process: mp.process.BaseProcess,
        out_of_band_error_path: Path,
    ) -> None:
        self._ctx = ctx
        self._socket = socket
        self._server_process = server_process
        self._out_of_band_error_path = out_of_band_error_path
        self._shutdown_done = False
        self._dirty: Optional[str] = None

    @classmethod
    def start(cls, **engine_kwargs: Any) -> "ScriptedHttpServer":
        err_fd, err_path = tempfile.mkstemp(
            prefix="scripted_runtime_oob_error_", suffix=".json"
        )
        os.close(err_fd)
        out_of_band_error_path = Path(err_path)

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
                **engine_kwargs,
            ),
            name="scripted-runtime-http-server",
            daemon=False,
        )
        sys_path_entry = os.path.dirname(os.path.abspath(__file__))
        with (
            envs.SGLANG_TEST_SCRIPTED_RUNTIME.override(True),
            envs.SGLANG_TEST_SCRIPTED_RUNTIME_IPC_ADDR.override(endpoint),
            envs.SGLANG_TEST_SCRIPTED_RUNTIME_OUT_OF_BAND_ERROR_PATH.override(
                str(out_of_band_error_path)
            ),
            envs.SGLANG_TEST_SCRIPTED_RUNTIME_SYS_PATH_ENTRY.override(sys_path_entry),
        ):
            server_process.start()

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
            out_of_band_error_path=out_of_band_error_path,
        )

    def execute_script(
        self,
        script_fn: Callable,
        *,
        args: Tuple[Any, ...] = (),
        timeout_s: float = DEFAULT_RUN_TIMEOUT_S,
    ) -> None:
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
        if self._shutdown_done:
            return

        fatal_error: Optional[OutOfBandError] = None
        try:
            try:
                self._socket.send_pyobj(Shutdown())
            except zmq.ZMQError:
                pass

            self._server_process.join(timeout=SHUTDOWN_JOIN_TIMEOUT_S)
            self._terminate_process(self._server_process)

            fatal_error = self._read_out_of_band_error()
        finally:
            try:
                self._socket.setsockopt(zmq.LINGER, 0)
                self._socket.close()
                self._ctx.term()
            except Exception:  # noqa: BLE001 — best-effort cleanup
                pass
            self._cleanup_files()
            self._shutdown_done = True

        if fatal_error:
            raise AssertionError(
                f"scripted-runtime server failed:\n{fatal_error.traceback}"
            )

    def _read_out_of_band_error(self) -> Optional[OutOfBandError]:
        try:
            text = self._out_of_band_error_path.read_text()
        except OSError:
            return None
        text = text.strip()
        return OutOfBandError.from_json(text) if text else None

    def _cleanup_files(self) -> None:
        try:
            self._out_of_band_error_path.unlink()
        except OSError:
            pass

    @staticmethod
    def _terminate_process(process: mp.process.BaseProcess) -> None:
        if not process.is_alive():
            return
        process.terminate()
        process.join(timeout=10.0)
        if process.is_alive():
            process.kill()
            process.join(timeout=10.0)


def launch_scripted_http_server(**engine_kwargs: Any) -> None:
    launch_server(ServerArgs(**engine_kwargs))
