"""Standalone-process host for the Rust TM's Python MM fallback.

Under ``SGLANG_ENABLE_STANDALONE_MM``, the Python mm-processor stack
(:class:`~sglang.srt.managers.rust_server.MmProcessorHost`) runs in a spawned
child process instead of in-scheduler threads, so its heavy Python work (media
decode, HF processor) contends with its *own* GIL, not the scheduler loop's —
protecting inter-token latency at the cost of one IPC hop per mm request
(TTFT).

Topology::

    scheduler process                          mm host process
    Rust mm-worker threads ──ipc://(REQ/ROUTER)──▶ zmq.asyncio ROUTER
      StandaloneMmClient.handle_sync              _serve → MmProcessorHost._process

Wire protocol (pooled REQ sockets, one owned exclusively per in-flight call,
strict send/recv lockstep):

* request: ``[rid, payload]`` — the msgpack payload from Rust, forwarded
  untouched.
* reply ok: ``[b"ok", input_ids_int64_le, pickle((mm_inputs, token_type_ids))]``
  — feature tensors shm-wrapped by the child (``wrap_mm_inputs_shm``), so only
  small pointer pickles ride the socket; the client materializes them on the
  calling Rust worker thread, never on the scheduler loop.
* reply err: ``[b"err", message]`` — the client raises, and the Rust worker
  rejects the request as a 400 (parity with the in-process path).

Failure handling mirrors the detokenizer process: a fatal child exception
SIGQUITs the parent; a hard child crash (segfault/OOM-kill) is detected by the
client's liveness poll; the child dies with the scheduler via PDEATHSIG.
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import os
import pickle
import queue
import signal
import tempfile
import threading
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import psutil
import setproctitle
import zmq
import zmq.asyncio

from sglang.srt.managers.mm_request_processing import StageTimes
from sglang.srt.utils import configure_logger, kill_itself_when_parent_died
from sglang.srt.utils.network import get_zmq_socket
from sglang.utils import get_exception_traceback

if TYPE_CHECKING:
    from sglang.srt.managers.rust_server import MmProcessorHost
    from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)

# Generous cap on child startup (spawn + imports + HF processor load).
_HANDSHAKE_TIMEOUT_S = 300


def launch_standalone_mm_host(
    *,
    server_args: ServerArgs,
    max_req_input_len: int,
    cores: Optional[List[int]],
) -> StandaloneMmClient:
    """Spawn the mm host process, wait for its handshake, return the client.

    ``spawn`` (not fork): the scheduler has a live CUDA context by launch time
    and the child's fast image processor may create its own. The child is
    non-daemonic so the mm processor's internal fork pool can create children.
    """
    ipc_name = f"ipc://{tempfile.NamedTemporaryFile(delete=False).name}"
    ctx = mp.get_context("spawn")
    reader, writer = ctx.Pipe(duplex=False)
    started = time.monotonic()
    proc = ctx.Process(
        target=run_standalone_mm_process,
        kwargs=dict(
            server_args=server_args,
            ipc_name=ipc_name,
            max_req_input_len=max_req_input_len,
            cores=cores,
            pipe_writer=writer,
        ),
        name="sglang::mm_host",
    )
    proc.start()
    writer.close()  # child's copy stays open; EOF on our end tracks child death
    try:
        if not reader.poll(_HANDSHAKE_TIMEOUT_S):
            raise EOFError
        handshake = reader.recv()
    except EOFError:
        proc.kill()
        raise RuntimeError("standalone MM host process failed to start") from None
    logger.info(
        "standalone MM host ready in %.1fs (pid=%d, native=%s)",
        time.monotonic() - started,
        proc.pid,
        handshake["spec"] is not None,
    )
    return StandaloneMmClient(
        ipc_name=ipc_name,
        proc=proc,
        spec=handshake["spec"],
        native=handshake["native"],
    )


class StandaloneMmClient:
    """Scheduler-process proxy for the mm host process.

    Duck-types the ``MmProcessorHost`` surface that ``RustServer.launch`` /
    ``drain`` consume: ``handle_sync``, ``results``, ``native_spec()``,
    ``native_enabled``, ``build_native_mm``, ``MM_WORKERS``.
    """

    def __init__(
        self,
        *,
        ipc_name: str,
        proc: Any,
        spec: Optional[str],
        native: Optional[Dict[str, Any]],
    ):
        from sglang.srt.managers.rust_server import MmProcessorHost

        self.MM_WORKERS = MmProcessorHost.MM_WORKERS
        # rid -> (mm_inputs, token_type_ids); popped by RustServer.drain.
        self.results: Dict[str, Tuple[Any, Optional[List[int]]]] = {}
        self._ipc_name = ipc_name
        self._proc = proc
        self._spec = spec
        self._native = native
        self._ctx = zmq.Context()
        # Checkout/checkin pool of REQ sockets (REQ is lockstep and not
        # thread-safe, so each in-flight call owns one exclusively; the pool
        # size is bounded by the Rust worker count). NOT thread-local: the
        # callers are pyo3-attached foreign threads whose Python thread state
        # (and thus `threading.local` slot) is NOT sticky across calls — a
        # thread-local here mints a new socket per request and leaks fds.
        self._pool: queue.SimpleQueue[zmq.Socket] = queue.SimpleQueue()
        self._times = StageTimes("mm client")

    def native_spec(self) -> Optional[str]:
        """Computed by the child at startup (it owns the processor stack)."""
        return self._spec

    @property
    def native_enabled(self) -> bool:
        return self._native is not None

    def build_native_mm(self, entry: tuple):
        from sglang.srt.managers.rust_server import _build_native_mm

        return _build_native_mm(native=self._native, entry=entry)

    def handle_sync(self, rid: str, payload: bytes) -> bytes:
        """Same contract as ``MmProcessorHost.handle_sync``, one IPC hop away.

        The blocking poll releases the GIL; the only GIL-held work on this
        (Rust worker) thread is the small pointer-pickle load — feature
        materialization is a GIL-releasing ``torch.clone``.
        """
        from sglang.srt.managers.mm_utils import unwrap_mm_inputs_shm

        t0 = time.monotonic()
        sock = self._acquire_socket()
        try:
            sock.send_multipart([rid.encode(), payload])
            # 1s liveness poll instead of RCVTIMEO: long requests (video, many
            # images) must not be capped, but a hard child crash (segfault /
            # OOM-kill, no SIGQUIT) must not hang the Rust worker pool.
            while not sock.poll(1000):
                if not self._proc.is_alive():
                    raise ValueError("standalone MM host process died")
            status, *frames = sock.recv_multipart()
        except BaseException:
            # A REQ socket abandoned mid send→recv cycle can't be reused.
            sock.close(linger=0)
            raise
        self._pool.put(sock)
        if status == b"err":
            raise ValueError(frames[0].decode())
        input_ids_bytes, result = frames
        t1 = time.monotonic()
        mm_inputs, token_type_ids = pickle.loads(result)
        t2 = time.monotonic()
        if mm_inputs is not None:
            unwrap_mm_inputs_shm(mm_inputs)
        t3 = time.monotonic()
        self._times.add(ipc_wait=t1 - t0, unpickle=t2 - t1, unwrap=t3 - t2)
        # Store BEFORE returning — same ordering contract as MmProcessorHost.
        self.results[rid] = (mm_inputs, token_type_ids)
        return input_ids_bytes

    def _acquire_socket(self) -> zmq.Socket:
        try:
            return self._pool.get_nowait()
        except queue.Empty:
            sock = get_zmq_socket(self._ctx, zmq.REQ, self._ipc_name, bind=False)
            sock.setsockopt(zmq.LINGER, 0)
            return sock

    def close(self) -> None:
        """Release the sockets and context. Only call once no thread is inside
        :meth:`handle_sync` (the Rust worker pool has been joined)."""
        while True:
            try:
                self._pool.get_nowait().close(linger=0)
            except queue.Empty:
                break
        self._ctx.term()


def run_standalone_mm_process(
    *,
    server_args: ServerArgs,
    ipc_name: str,
    max_req_input_len: int,
    cores: Optional[List[int]],
    pipe_writer,
) -> None:
    """Child entrypoint (spawn target). Builds the same ``MmProcessorHost`` the
    in-process mode uses and serves it over ``ipc_name``."""
    kill_itself_when_parent_died()
    setproctitle.setproctitle("sglang::mm_host")
    configure_logger(server_args, prefix=" MM")
    if cores is not None:
        # Mirror the in-process mode: mm work stays off the scheduler's
        # reserved launch cores.
        try:
            os.sched_setaffinity(0, set(cores))
        except OSError as e:
            logger.warning("mm host: cannot confine to server cores: %s", e)
    parent_process = psutil.Process().parent()

    try:
        from sglang.srt.configs.model_config import ModelConfig
        from sglang.srt.managers.rust_server import MmProcessorHost
        from sglang.srt.runtime_context import get_context

        # base_processor and the shm transport gate read the process-global
        # server_args.
        get_context().set_server_args(server_args)
        host = MmProcessorHost(
            server_args=server_args,
            model_config=ModelConfig.from_server_args(server_args),
            max_req_input_len=max_req_input_len,
        )
        asyncio.run(_serve(host=host, ipc_name=ipc_name, pipe_writer=pipe_writer))
    except Exception:
        logger.error(
            "standalone MM host hit an exception: %s", get_exception_traceback()
        )
        parent_process.send_signal(signal.SIGQUIT)


async def _serve(*, host: MmProcessorHost, ipc_name: str, pipe_writer) -> None:
    """Accept loop: one asyncio task per in-flight request (bounded by the
    parent's lockstep REQ sockets — at most one outstanding request per Rust
    mm-worker thread)."""
    sock = zmq.asyncio.Context(1).socket(zmq.ROUTER)
    sock.setsockopt(zmq.LINGER, 0)
    sock.bind(ipc_name)
    # Handshake strictly after bind, so the parent can connect immediately.
    # native_spec() also sets host._native, consumed by the parent's
    # drain-time build_native_mm.
    pipe_writer.send({"spec": host.native_spec(), "native": host._native})
    times = StageTimes("mm host")
    while True:
        identity, _, rid, payload = await sock.recv_multipart()
        asyncio.create_task(
            _handle(
                host=host,
                sock=sock,
                identity=identity,
                rid=rid.decode(),
                payload=payload,
                times=times,
            )
        )


async def _handle(
    *,
    host: MmProcessorHost,
    sock: zmq.asyncio.Socket,
    identity: bytes,
    rid: str,
    payload: bytes,
    times: StageTimes,
) -> None:
    from sglang.srt.managers.mm_utils import wrap_mm_inputs_shm
    from sglang.srt.managers.rust_server import _encode_input_ids

    try:
        t0 = time.monotonic()
        input_ids, mm_inputs, token_type_ids = await host._process(rid, payload)
        t1 = time.monotonic()
        if mm_inputs is not None:
            wrap_mm_inputs_shm(mm_inputs)
        reply = [
            b"ok",
            _encode_input_ids(input_ids),
            pickle.dumps((mm_inputs, token_type_ids), pickle.HIGHEST_PROTOCOL),
        ]
        times.add(process=t1 - t0, wrap_pickle=time.monotonic() - t1)
    except Exception as e:  # per-request failure → 400, parity with in-process
        logger.warning("mm host: request %s failed: %s", rid, e)
        reply = [b"err", str(e).encode()]
    await sock.send_multipart([identity, b"", *reply])
