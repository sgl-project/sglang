from __future__ import annotations

import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import requests
import torch
import zmq

from sglang.kernels.ops.kv_canary._dispatch import use_torch_reference
from sglang.srt.entrypoints.http_server import launch_server
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import sock_recv, sock_send, wrap_as_pickle
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils.common import get_device
from sglang.srt.utils.network import get_free_port, get_zmq_socket_on_host
from sglang.test.scripted_runtime.io_struct import (
    HookReady,
    OutOfBandError,
    RunScript,
    ScriptFailed,
    ScriptSucceeded,
    Shutdown,
)
from sglang.test.scripted_runtime.utils import close_zmq_socket

DEFAULT_RUN_TIMEOUT_S: float = 120.0

CANARY_REFERENCE_TIMEOUT_SCALE: float = 4.0
SHUTDOWN_JOIN_TIMEOUT_S: float = 60.0
LISTENER_ACCEPT_TIMEOUT_S: float = 300.0
HTTP_READY_TIMEOUT_S: float = 300.0
HTTP_READY_POLL_INTERVAL_S: float = 0.5
SERVER_HOST: str = "127.0.0.1"

CANARY_LAUNCH_DEFAULTS: Dict[str, Any] = dict(
    kv_canary="raise",
    kv_canary_real_data="partial",
    kv_canary_sweep_interval=100,
)


class ScriptedHttpServer:
    def __init__(
        self,
        *,
        ctx: zmq.Context,
        socket: zmq.Socket,
        server_process: mp.process.BaseProcess,
        out_of_band_error_path: Path,
        http_port: int,
        run_timeout_s: float = DEFAULT_RUN_TIMEOUT_S,
    ) -> None:
        self._ctx = ctx
        self._socket = socket
        self._server_process = server_process
        self._out_of_band_error_path = out_of_band_error_path
        self._base_url = f"http://{SERVER_HOST}:{http_port}"
        self._shutdown_done = False
        self._dirty: Optional[str] = None
        self._run_timeout_s = run_timeout_s

    @classmethod
    def start(cls, **engine_kwargs: Any) -> ScriptedHttpServer:
        out_of_band_error_path = _create_oob_error_file()

        ctx = zmq.Context()
        dispatch_port, socket = get_zmq_socket_on_host(ctx, zmq.PAIR, host=SERVER_HOST)
        server_process, http_port, run_timeout_s = _spawn_server_process(
            endpoint=f"tcp://{SERVER_HOST}:{dispatch_port}",
            out_of_band_error_path=out_of_band_error_path,
            engine_kwargs=engine_kwargs,
        )

        self = cls(
            ctx=ctx,
            socket=socket,
            server_process=server_process,
            out_of_band_error_path=out_of_band_error_path,
            http_port=http_port,
            run_timeout_s=run_timeout_s,
        )
        try:
            self._await_handshake()
            self._await_http_ready()
        except BaseException:
            self._teardown()
            raise
        return self

    def execute_script(
        self,
        script_fn: Callable,
        *,
        args: Tuple[Any, ...] = (),
        timeout_s: Optional[float] = None,
    ) -> None:
        if timeout_s is None:
            timeout_s = self._run_timeout_s
        if self._dirty:
            raise RuntimeError(f"ScriptedHttpServer is dirty: {self._dirty}")

        fn_path = f"{script_fn.__module__}:{script_fn.__qualname__}"
        sock_send(self._socket, wrap_as_pickle(RunScript(fn_path=fn_path, args=args)))

        if not self._socket.poll(int(timeout_s * 1000)):
            if not self._server_process.is_alive():
                self._dirty = f"server process died before responding to {fn_path!r}"
                raise RuntimeError(self._dirty)
            self._dirty = f"script {fn_path!r} timed out after {timeout_s}s"
            raise TimeoutError(self._dirty)

        reply = sock_recv(self._socket)
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
                sock_send(self._socket, wrap_as_pickle(Shutdown()))
            except zmq.ZMQError:
                pass

            self._server_process.join(timeout=SHUTDOWN_JOIN_TIMEOUT_S)
            self._terminate_process(self._server_process)
            fatal_error = self._read_out_of_band_error()
        finally:
            self._teardown()
            self._shutdown_done = True

        if fatal_error:
            raise AssertionError(
                f"scripted-runtime server failed:\n{fatal_error.traceback}"
            )

    def _await_handshake(self) -> None:
        if not self._socket.poll(int(LISTENER_ACCEPT_TIMEOUT_S * 1000)):
            raise TimeoutError(
                f"ScriptedHttpServer: HTTP server did not connect within "
                f"{LISTENER_ACCEPT_TIMEOUT_S}s"
            )

        ready = sock_recv(self._socket)
        if not isinstance(ready, HookReady):
            raise RuntimeError(
                f"ScriptedHttpServer: expected HookReady handshake, got {ready!r}"
            )

    def _await_http_ready(self) -> None:
        # HookReady only means the scheduler dispatch loop started; the uvicorn
        # entrypoint may not be bound yet. The first script runs
        # _reset_engine_state, which POSTs to this server's own HTTP port, so
        # block until the port is bound and routing before any script can run.
        #
        # Poll /model_info rather than /health: once the server reports Up,
        # /health runs the generation-based health check (a real probe request
        # through the scheduler), which cannot complete while the scheduler
        # waits for scripts between RunScript commands. Any /model_info
        # response proves the socket is bound and routes are registered, which
        # is all the control POSTs need.
        url = f"{self._base_url}/model_info"
        deadline = time.monotonic() + HTTP_READY_TIMEOUT_S
        while time.monotonic() < deadline:
            if not self._server_process.is_alive():
                raise RuntimeError(
                    "ScriptedHttpServer: server process died during HTTP startup"
                )
            try:
                requests.get(url, timeout=2.0)
                return
            except requests.RequestException:
                pass
            time.sleep(HTTP_READY_POLL_INTERVAL_S)

        raise TimeoutError(
            f"ScriptedHttpServer: HTTP endpoint {url} not bound within "
            f"{HTTP_READY_TIMEOUT_S}s"
        )

    def _teardown(self) -> None:
        try:
            close_zmq_socket(self._socket, self._ctx)
        except Exception:  # noqa: BLE001 — best-effort cleanup
            pass
        self._terminate_process(self._server_process)
        self._cleanup_files()

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


def _create_oob_error_file() -> Path:
    err_fd, err_path = tempfile.mkstemp(
        prefix="scripted_runtime_oob_error_", suffix=".json"
    )
    os.close(err_fd)
    return Path(err_path)


def _default_run_timeout_s(*, kv_canary: str, device: str) -> float:
    if _canary_decode_graph_override(kv_canary=kv_canary, device=device):
        return DEFAULT_RUN_TIMEOUT_S * CANARY_REFERENCE_TIMEOUT_SCALE
    return DEFAULT_RUN_TIMEOUT_S


def _canary_decode_graph_override(*, kv_canary: str, device: str) -> Dict[str, Any]:
    if kv_canary != "none" and use_torch_reference(torch.device(device)):
        # install_canary refuses a captured decode over the torch reference.
        return {"cuda_graph_backend_decode": "disabled"}
    return {}


def _spawn_server_process(
    *,
    endpoint: str,
    out_of_band_error_path: Path,
    engine_kwargs: Dict[str, Any],
) -> Tuple[mp.process.BaseProcess, int, float]:
    mp_ctx = mp.get_context("spawn")
    device = engine_kwargs.get("device") or get_device()
    launch_kwargs: Dict[str, Any] = dict(
        host=SERVER_HOST,
        port=get_free_port(),
        disable_prefill_cuda_graph=True,
        **CANARY_LAUNCH_DEFAULTS,
    )
    launch_kwargs.update(engine_kwargs)
    for key, value in _canary_decode_graph_override(
        kv_canary=launch_kwargs["kv_canary"], device=device.split(":")[0]
    ).items():
        launch_kwargs.setdefault(key, value)
    http_port = launch_kwargs["port"]
    run_timeout_s = _default_run_timeout_s(
        kv_canary=launch_kwargs["kv_canary"], device=device.split(":")[0]
    )
    server_process = mp_ctx.Process(
        target=_launch_scripted_http_server,
        kwargs=launch_kwargs,
        name="scripted-runtime-http-server",
        daemon=False,
    )

    sys_path_entry = str(Path(__file__).resolve().parent)
    with (
        envs.SGLANG_TEST_SCRIPTED_RUNTIME.override(True),
        envs.SGLANG_TEST_SCRIPTED_RUNTIME_IPC_ADDR.override(endpoint),
        envs.SGLANG_TEST_SCRIPTED_RUNTIME_OUT_OF_BAND_ERROR_PATH.override(
            str(out_of_band_error_path)
        ),
        envs.SGLANG_TEST_SCRIPTED_RUNTIME_SYS_PATH_ENTRY.override(sys_path_entry),
    ):
        server_process.start()

    return server_process, http_port, run_timeout_s


def _launch_scripted_http_server(**engine_kwargs: Any) -> None:
    launch_server(ServerArgs(**engine_kwargs))
