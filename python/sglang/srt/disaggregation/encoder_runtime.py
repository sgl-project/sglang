"""Protocol-neutral runtime for the EPD encoder server.

The current runtime keeps :class:`EncoderScheduler` and the rank-0
:class:`encode_server.MMEncoder` in the same process.  It also owns HTTP's
existing DP replica processes and dispatch plumbing so another transport can
reuse that backend topology without importing the HTTP server.
"""

import asyncio
import atexit
import contextlib
import copy
import logging
import multiprocessing as mp
import os
import time
import traceback
import uuid
from collections import defaultdict
from dataclasses import dataclass
from http import HTTPStatus
from typing import Dict, List, Optional, Set, Tuple

import zmq
import zmq.asyncio

import sglang.srt.disaggregation.encode_server as encode_server_module
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.encode_server import (
    ENCODER_MAX_BATCH_SIZE,
    EncoderProfiler,
    MMEncoder,
    MMError,
    launch_encoder,
)
from sglang.srt.environ import envs
from sglang.srt.managers.io_struct import (
    ProfileReq,
    ProfileReqType,
    async_sock_recv,
    async_sock_send,
    sock_send,
    wrap_as_pickle,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.observability.metrics_collector import EncoderMetricsCollector
from sglang.srt.observability.req_time_stats import EncoderReqTimeStats
from sglang.srt.observability.trace import (
    process_tracing_init,
    trace_set_thread_info,
)
from sglang.srt.server_args import PortArgs, ServerArgs
from sglang.srt.utils import configure_logger, random_uuid, set_prometheus_multiproc_dir
from sglang.srt.utils.common import maybe_reindex_device_id
from sglang.srt.utils.network import NetworkAddress, get_free_port, get_zmq_socket

logger = logging.getLogger(__name__)


class PendingRequest:
    __slots__ = ("request", "future", "submit_time")

    def __init__(self, request: dict, loop: asyncio.AbstractEventLoop):
        self.request = request
        self.future: asyncio.Future = loop.create_future()
        self.submit_time = time.time()


# VIDEO excluded: per-video preprocess kwargs (do_sample_frames, video_metadata)
# vary per request and can't merge into one HF processor call.
_BATCHABLE_MODALITIES = {Modality.IMAGE, Modality.AUDIO}

# Minimal 32x32 black PNG for health check dummy encode
MINIMUM_PNG_PICTURE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="

# Minimal WAV: 16kHz mono 16-bit PCM, 160 samples (0.01s) of silence
MINIMUM_WAV_SILENCE_BASE64 = "UklGRmQBAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YUABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="


class EncoderScheduler:
    """Aggregate concurrent /encode requests into bounded image/audio batches."""

    def __init__(
        self,
        encoder: "MMEncoder",
        send_sockets: List[zmq.Socket],
        max_batch_size: int,
        request_timeout: float = encode_server_module.ENCODER_REQ_TIMEOUT,
    ):
        self.encoder = encoder
        self.send_sockets = send_sockets
        self.max_batch_size = max(1, int(max_batch_size))
        self.request_timeout = max(1.0, float(request_timeout))
        self.pending_queue: asyncio.Queue[PendingRequest] = asyncio.Queue()
        self._worker_task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if self._worker_task is None:
            self._worker_task = asyncio.create_task(self._batch_worker())
            logger.info(
                f"EncoderScheduler started with max_batch_size={self.max_batch_size}"
            )

    async def stop(self) -> None:
        if self._worker_task is not None:
            self._worker_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._worker_task
            self._worker_task = None
        # Reject any requests still queued so their HTTP handlers don't hang.
        while True:
            try:
                pending = self.pending_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if not pending.future.done():
                pending.future.set_exception(RuntimeError("EncoderScheduler stopped"))

    async def submit(self, request: dict) -> Tuple:
        pending = PendingRequest(request, asyncio.get_running_loop())
        await self.pending_queue.put(pending)
        try:
            return await asyncio.wait_for(pending.future, timeout=self.request_timeout)
        except asyncio.TimeoutError:
            if not pending.future.done():
                pending.future.cancel()
            req_id = request.get("req_id")
            logger.error(
                f"EncoderScheduler.submit timed out after {self.request_timeout}s "
                f"for req_id={req_id}"
            )
            raise

    async def _collect_batch(self) -> List[PendingRequest]:
        batch = [await self.pending_queue.get()]
        while len(batch) < self.max_batch_size:
            try:
                batch.append(self.pending_queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        return batch

    async def _batch_worker(self) -> None:
        while True:
            batch: List[PendingRequest] = []
            try:
                batch = await self._collect_batch()
                groups: Dict[Modality, List[PendingRequest]] = defaultdict(list)
                for p in batch:
                    groups[
                        Modality.from_str(p.request.get("modality", "image"))
                    ].append(p)
                for modality, group in groups.items():
                    await self._dispatch_group(group, modality)
            except asyncio.CancelledError:
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(RuntimeError("EncoderScheduler stopped"))
                raise
            except Exception as e:
                logger.error(
                    f"Error in EncoderScheduler batch worker: {e}", exc_info=True
                )
                for p in batch:
                    if not p.future.done():
                        p.future.set_exception(e)

    @staticmethod
    def _validate_request_shape(req: dict) -> Optional[str]:
        # Cheap pre-broadcast checks: shape errors that don't require running
        # the HF processor. Once a request reaches TP workers they enter
        # batch_encode and expect to join its collectives — a malformed batch
        # that makes rank-0 bail mid-flight would deadlock the workers.
        if not isinstance(req, dict):
            return f"request is not a dict: {type(req).__name__}"
        if not req.get("req_id"):
            return "missing req_id"
        if not req.get("mm_items"):
            return "missing or empty mm_items"
        if "num_parts" not in req or "part_idx" not in req:
            return "missing num_parts / part_idx"
        h = req.get("hashes")
        if h is not None and not isinstance(h, (list, tuple, str, int, bytes)):
            return f"hashes must be list/scalar, got {type(h).__name__}"
        return None

    async def _dispatch_group(
        self, group: List[PendingRequest], modality: Modality
    ) -> None:
        # Video can't fuse (per-video preprocess kwargs vary).
        if modality not in _BATCHABLE_MODALITIES:
            await self._dispatch_per_request(group, modality)
            return

        # Drop structurally-bad requests before broadcasting; otherwise TP
        # workers would join batch_encode collectives that rank-0 has already
        # abandoned.
        valid: List[PendingRequest] = []
        for p in group:
            err = self._validate_request_shape(p.request)
            if err is None:
                valid.append(p)
                continue
            logger.error(f"Dropping req_id={p.request.get('req_id')} from batch: {err}")
            if not p.future.done():
                p.future.set_exception(encode_server_module.BadRequestError(err))
        if not valid:
            return
        group = valid

        requests = [p.request for p in group]
        start = time.time()
        modality_str = modality.name.lower()
        if encode_server_module.encoder_metrics_collector is not None:
            for p in group:
                encode_server_module.encoder_metrics_collector.observe_queue_wait(
                    max(0.0, start - p.submit_time), modality=modality_str
                )
        for sock in self.send_sockets:
            sock_send(
                sock,
                wrap_as_pickle(
                    {
                        "type": "batch_encode",
                        "modality": modality.name,
                        "requests": requests,
                        "enter_time": start,
                    }
                ),
            )

        logger.info(f"Dispatching batch of {len(group)} {modality.name} requests")

        try:
            results = await self.encoder.batch_encode(requests, modality)
            if len(group) > 1:
                logger.info(
                    f"Batch of {len(group)} {modality.name} requests completed in "
                    f"{(time.time() - start) * 1000:.1f}ms"
                )
        except Exception as e:
            # batch_encode normally catches and returns errors via _batch_set_error.
            # If it raised, rank-0 may have skipped a collective broadcast, leaving
            # TP workers stuck. Don't try to recover — fail every pending future
            # and let the client retry. Re-broadcasting would risk a deadlock.
            logger.error(f"batch_encode raised: {e}", exc_info=True)
            for p in group:
                if not p.future.done():
                    p.future.set_exception(e)
            return

        if len(results) != len(group):
            err = RuntimeError(
                f"batch_encode returned {len(results)} results for {len(group)} requests"
            )
            logger.error(str(err))
            for p in group:
                if not p.future.done():
                    p.future.set_exception(err)
            return

        for p, result in zip(group, results):
            if not p.future.done():
                p.future.set_result(result)

    async def _dispatch_per_request(
        self,
        group: List[PendingRequest],
        modality: Modality,
    ) -> None:
        modality_str = modality.name.lower()
        for p in group:
            req = p.request
            try:
                start = time.time()
                if encode_server_module.encoder_metrics_collector is not None:
                    encode_server_module.encoder_metrics_collector.observe_queue_wait(
                        max(0.0, start - p.submit_time), modality=modality_str
                    )
                for sock in self.send_sockets:
                    sock_send(sock, wrap_as_pickle(req))
                result = await self.encoder.encode_request(req, modality)
                if not p.future.done():
                    p.future.set_result(result)
            except Exception as e:
                logger.error(
                    f"Per-request encode failed for req_id={req.get('req_id')}: {e}"
                )
                if not p.future.done():
                    p.future.set_exception(e)


@dataclass
class EncoderRuntime:
    """Current non-DP backend runtime.

    The Scheduler and rank-0 MMEncoder remain colocated.  TP followers use the
    existing ZMQ control path and are intentionally not split behind a new
    Scheduler/Worker IPC contract in this phase.
    """

    encoder: MMEncoder
    scheduler: EncoderScheduler
    send_sockets: List[zmq.Socket]
    zmq_context: zmq.Context
    tp_processes: List[mp.Process]

    def start(self) -> None:
        self.scheduler.start()

    async def stop(self) -> None:
        # Preserve the existing lifecycle: Uvicorn stops the Scheduler, while
        # daemon TP followers exit with their parent process.
        await self.scheduler.stop()


class DPDispatcher:
    """Routes encode requests across DP ranks by least-pending count."""

    def __init__(
        self,
        dp_size: int,
        dispatch_sockets: List,
        result_socket,
        worker_processes: List[mp.Process],
        enable_metrics: bool = False,
        labels: Optional[Dict[str, str]] = None,
    ):
        self.dp_size = dp_size
        self.dispatch_sockets = dispatch_sockets
        self.result_socket = result_socket
        self.worker_processes = worker_processes
        # Key = req_id for encode/broadcast, req_id + "_send" for mooncake /send.
        self.pending_futures: List[Dict[str, asyncio.Future]] = [
            {} for _ in range(dp_size)
        ]
        self.req_id_to_rank: Dict[str, int] = {}
        self._rr_counter = 0
        self._broadcast_counter = 0
        self._dead_ranks: Set[int] = set()
        # req_id -> monotonic ts a mooncake mapping has waited for its /send.
        self._pending_send_at: Dict[str, float] = {}
        # Set when _result_listener gives up; makes alive_ranks report empty.
        self._listener_failed = False

        # Prometheus gauge: pending requests per DP rank. Lives in the main
        # process (the dispatcher), unlike the per-worker EncoderMetricsCollector.
        self.labels = dict(labels or {})
        self.pending_gauge = None
        if enable_metrics:
            from prometheus_client import Gauge

            self.pending_gauge = Gauge(
                name="sglang:encoder_dp_pending_requests",
                documentation="Number of pending requests per encoder DP rank.",
                labelnames=list(self.labels.keys()) + ["dp_rank"],
                multiprocess_mode="mostrecent",
            )

    @property
    def pending_counts(self) -> List[int]:
        return [len(d) for d in self.pending_futures]

    def _update_pending_gauge(self) -> None:
        """Push current pending counts to the Prometheus gauge (absolute set)."""
        if self.pending_gauge is not None:
            for i, c in enumerate(self.pending_counts):
                self.pending_gauge.labels(**self.labels, dp_rank=str(i)).set(c)

    @property
    def alive_ranks(self) -> List[int]:
        # Empty if the result listener died; else ranks not marked dead.
        if self._listener_failed:
            return []
        return [r for r in range(self.dp_size) if r not in self._dead_ranks]

    @property
    def all_ranks_alive(self) -> bool:
        # Strict health (only /health uses this); routing still degrades.
        return len(self.alive_ranks) == self.dp_size

    def start(self) -> None:
        logger.info(f"DP dispatcher started: {self.dp_size} ranks (all remote)")
        asyncio.create_task(self._result_listener())
        asyncio.create_task(self._worker_watchdog())
        asyncio.create_task(self._cleanup_stale_mappings())

    def _drop_pending_and_mapping(self, rank: int, req_id: str) -> None:
        # dispatch / broadcast failure: no follow-up /send expected.
        self.pending_futures[rank].pop(req_id, None)
        self.req_id_to_rank.pop(req_id, None)
        self._update_pending_gauge()

    def _fail_pending_for_rank(self, rank: int, reason: str, error_type: str) -> None:
        # Resolve a rank's outstanding futures with 503 so awaiters don't hang.
        pending = self.pending_futures[rank]
        for key, future in list(pending.items()):
            if not future.done():
                future.set_result(
                    {
                        "req_id": key.removesuffix("_send"),
                        "_dp_type": "send" if key.endswith("_send") else "encode",
                        "content": None,
                        "_error": reason,
                        "_error_type": error_type,
                        "_error_code": int(HTTPStatus.SERVICE_UNAVAILABLE),
                    }
                )
            pending.pop(key, None)
        self._update_pending_gauge()

    def _fail_all_pending(self, reason: str, error_type: str) -> None:
        for rank in range(self.dp_size):
            self._fail_pending_for_rank(rank, reason, error_type)
        self.req_id_to_rank.clear()
        self._pending_send_at.clear()

    @staticmethod
    def _timeout_envelope(req_id: str, dp_type: str, reason: str) -> dict:
        return {
            "req_id": req_id,
            "_dp_type": dp_type,
            "content": None,
            "_error": reason,
            "_error_type": "TimeoutError",
            "_error_code": int(HTTPStatus.GATEWAY_TIMEOUT),
        }

    async def dispatch(self, request: dict) -> dict:
        counts = self.pending_counts
        # Skip ranks whose worker process has died.
        alive_ranks = self.alive_ranks
        if not alive_ranks:
            raise encode_server_module.MMError(
                "All encoder DP workers are dead.",
                code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        min_p = min(counts[r] for r in alive_ranks)
        candidates = [r for r in alive_ranks if counts[r] == min_p]
        rank = candidates[self._rr_counter % len(candidates)]
        self._rr_counter += 1
        req_id = request["req_id"]
        self.req_id_to_rank[req_id] = rank
        future = asyncio.get_running_loop().create_future()
        self.pending_futures[rank][req_id] = future
        self._update_pending_gauge()
        logger.info(
            f"MM-Encoder DP dispatch: req_id={req_id}, "
            f"modality={request.get('modality', 'image')}, "
            f"dp_rank={rank}, pending={self.pending_counts}"
        )

        try:
            await async_sock_send(self.dispatch_sockets[rank], wrap_as_pickle(request))
            # An alive-but-stuck worker (NCCL deadlock etc.) wouldn't trip
            # the watchdog, so bound the wait explicitly.
            return await asyncio.wait_for(
                future, timeout=encode_server_module.ENCODER_REQ_TIMEOUT
            )
        except asyncio.TimeoutError:
            self._drop_pending_and_mapping(rank, req_id)
            return self._timeout_envelope(
                req_id,
                "encode",
                f"Encoder DP rank={rank} timed out after {encode_server_module.ENCODER_REQ_TIMEOUT}s",
            )
        except BaseException:
            self._drop_pending_and_mapping(rank, req_id)
            raise

    async def dispatch_send(self, request: dict) -> dict:
        req_id = request["req_id"]
        # /send arrived → stop tracking it for stale-mapping GC.
        self._pending_send_at.pop(req_id, None)
        if self._listener_failed:
            return {
                "req_id": req_id,
                "_error": "encoder DP result listener stopped; cannot route /send",
                "_error_code": int(HTTPStatus.SERVICE_UNAVAILABLE),
            }
        rank = self.req_id_to_rank.get(req_id)
        if rank is None:
            logger.warning(
                f"MM-Encoder dispatch_send: unknown req_id={req_id}, "
                f"cannot route to worker"
            )
            return {"req_id": req_id, "_error": f"Unknown req_id: {req_id}"}
        if rank in self._dead_ranks:
            # Worker died between encode and /send; embedding is gone.
            self.req_id_to_rank.pop(req_id, None)
            return {
                "req_id": req_id,
                "_error": f"DP worker rank={rank} died before /send for req_id={req_id}",
                "_error_code": int(HTTPStatus.SERVICE_UNAVAILABLE),
            }
        key = req_id + "_send"
        future = asyncio.get_running_loop().create_future()
        self.pending_futures[rank][key] = future
        request["_dp_type"] = "send"
        logger.info(
            f"MM-Encoder DP dispatch_send: req_id={req_id}, "
            f"dp_rank={rank}, pending={self.pending_counts}"
        )
        try:
            await async_sock_send(self.dispatch_sockets[rank], wrap_as_pickle(request))
            return await asyncio.wait_for(
                future, timeout=encode_server_module.ENCODER_REQ_TIMEOUT
            )
        except asyncio.TimeoutError:
            self.pending_futures[rank].pop(key, None)
            self.req_id_to_rank.pop(req_id, None)
            return self._timeout_envelope(
                req_id,
                "send",
                f"Encoder DP rank={rank} /send timed out after {encode_server_module.ENCODER_REQ_TIMEOUT}s",
            )
        except BaseException:
            self.pending_futures[rank].pop(key, None)
            self.req_id_to_rank.pop(req_id, None)
            raise

    async def broadcast(
        self, request: dict, timeout: Optional[float] = None
    ) -> List[dict]:
        # Skip dead ranks: a PUSH to a gone worker would just buffer and then
        # surface as a spurious per-rank timeout. All dead → 503 (same as
        # dispatch), which the profile endpoints turn into an HTTP error.
        eff_timeout = (
            timeout if timeout is not None else encode_server_module.ENCODER_REQ_TIMEOUT
        )
        alive_ranks = self.alive_ranks
        if not alive_ranks:
            raise encode_server_module.MMError(
                "All encoder DP workers are dead.",
                code=HTTPStatus.SERVICE_UNAVAILABLE,
            )
        batch_id = self._broadcast_counter
        self._broadcast_counter += 1
        rank_keys: List[Tuple[int, str]] = []
        futures: List[asyncio.Future] = []
        dp_type = request.get("_dp_type", "unknown")
        try:
            for rank in alive_ranks:
                req_id = f"_broadcast_{batch_id}_{rank}"
                future = asyncio.get_running_loop().create_future()
                self.pending_futures[rank][req_id] = future
                self.req_id_to_rank[req_id] = rank
                rank_keys.append((rank, req_id))
                request_copy = {**request, "req_id": req_id}
                await async_sock_send(
                    self.dispatch_sockets[rank], wrap_as_pickle(request_copy)
                )
                futures.append(future)
            # Concurrent wait → total bounded by eff_timeout, not
            # dp_size × eff_timeout.
            outcomes = await asyncio.gather(
                *(asyncio.wait_for(fut, timeout=eff_timeout) for fut in futures),
                return_exceptions=True,
            )
            results: List[dict] = []
            for (rank, req_id), outcome in zip(rank_keys, outcomes):
                if isinstance(outcome, asyncio.TimeoutError):
                    self._drop_pending_and_mapping(rank, req_id)
                    results.append(
                        self._timeout_envelope(
                            req_id,
                            dp_type,
                            f"Encoder DP rank={rank} broadcast timed out "
                            f"after {eff_timeout}s",
                        )
                    )
                elif isinstance(outcome, BaseException):
                    self._drop_pending_and_mapping(rank, req_id)
                    raise outcome
                else:
                    results.append(outcome)
            return results
        except BaseException:
            for rank, req_id in rank_keys:
                self._drop_pending_and_mapping(rank, req_id)
            raise

    async def _worker_watchdog(self) -> None:
        # proc.sentinel becomes readable on process exit; fail this rank's
        # pending futures so awaiters don't hang on a dead worker.
        loop = asyncio.get_running_loop()
        watch: Dict[int, asyncio.Future] = {}
        for rank, proc in enumerate(self.worker_processes):
            fut: asyncio.Future = loop.create_future()

            # add_reader is level-triggered, so remove_reader inside the
            # callback to avoid spinning every loop iteration.
            def _on_exit(r=rank, f=fut, p=proc, lp=loop):
                try:
                    lp.remove_reader(p.sentinel)
                except (ValueError, OSError):
                    pass
                if not f.done():
                    f.set_result(r)

            try:
                loop.add_reader(proc.sentinel, _on_exit)
            except (ValueError, OSError):
                continue
            watch[rank] = fut

        while watch:
            done, _ = await asyncio.wait(
                watch.values(), return_when=asyncio.FIRST_COMPLETED
            )
            for fut in done:
                rank = fut.result()
                proc = self.worker_processes[rank]
                logger.error(
                    f"DP worker rank={rank} (pid={proc.pid}) exited "
                    f"with code={proc.exitcode}; failing pending requests"
                )
                self._dead_ranks.add(rank)
                reason = f"DP worker rank={rank} died (exitcode={proc.exitcode})"
                self._fail_pending_for_rank(rank, reason, "WorkerDied")
                self.req_id_to_rank = {
                    r: rk for r, rk in self.req_id_to_rank.items() if rk != rank
                }
                watch.pop(rank, None)

    async def _result_listener(self) -> None:
        # Bounded back-off + give-up so a torn-down context exits in ~3s
        # rather than spinning forever on recv errors.
        consecutive_errors = 0
        while True:
            try:
                msg = await async_sock_recv(self.result_socket)
                consecutive_errors = 0
            except asyncio.CancelledError:
                raise
            except Exception:
                consecutive_errors += 1
                logger.error("_result_listener recv error", exc_info=True)
                if consecutive_errors >= 30:
                    logger.error(
                        "_result_listener giving up after 30 consecutive errors"
                    )
                    self._listener_failed = True
                    self._fail_all_pending(
                        "encoder DP result listener stopped after repeated "
                        "recv errors",
                        "ResultListenerStopped",
                    )
                    return
                await asyncio.sleep(min(0.1 * consecutive_errors, 1.0))
                continue
            req_id = msg.get("req_id", "")
            dp_type = msg.get("_dp_type", "encode")
            key = (req_id + "_send") if dp_type == "send" else req_id
            rank = self.req_id_to_rank.get(req_id)
            if rank is None or key not in self.pending_futures[rank]:
                logger.warning(
                    f"_result_listener: no pending future for "
                    f"req_id={req_id}, dp_type={dp_type}, dropping"
                )
                continue
            future = self.pending_futures[rank].pop(key)
            self._update_pending_gauge()
            # Only mooncake encode (content=request dict) needs the mapping
            # kept for the follow-up /send.
            keep_mapping = dp_type == "encode" and msg.get("content") is not None
            if keep_mapping:
                self._pending_send_at[req_id] = time.monotonic()
            else:
                self.req_id_to_rank.pop(req_id, None)
            try:
                future.set_result(msg)

            except asyncio.InvalidStateError:
                logger.warning(
                    f"_result_listener: future already done for "
                    f"req_id={req_id}, dp_type={dp_type}"
                )

    async def _cleanup_stale_mappings(self) -> None:
        # Evict req_id->rank mappings whose /send never came. The worker frees
        # its own embedding via the send_timeout cleanup scheduled at encode,
        # so both sides key off the same timeout.
        ttl = envs.SGLANG_ENCODER_SEND_TIMEOUT.get()
        interval = max(ttl / 4, 30)
        while True:
            await asyncio.sleep(interval)
            now = time.monotonic()
            stale = [rid for rid, ts in self._pending_send_at.items() if now - ts > ttl]
            for rid in stale:
                self._pending_send_at.pop(rid, None)
                self.req_id_to_rank.pop(rid, None)
            if stale:
                logger.warning(
                    f"Evicted {len(stale)} stale encoder DP /send mapping(s) "
                    f"with no /send within {ttl}s"
                )


async def _push_embedding_to_prefill(enc: MMEncoder, request: dict) -> None:
    # No-op for mooncake (its /send is separate). embedding_port=None is
    # rejected upfront, so ports is always a concrete list here.
    req_id = request["req_id"]
    backend = enc.server_args.encoder_transfer_backend

    if backend == "zmq_to_tokenizer":
        await enc.send(
            req_id=req_id,
            prefill_host=request["prefill_host"],
            embedding_port=request["embedding_port"],
        )
        enc.discard_embedding(req_id)
        return

    if backend == "zmq_to_scheduler":
        ports = request["embedding_port"]
        assert isinstance(ports, list)
        await asyncio.gather(
            *(
                enc.send(
                    req_id=req_id,
                    prefill_host=request["prefill_host"],
                    embedding_port=p,
                )
                for p in ports
            )
        )
        enc.discard_embedding(req_id)


async def _dp_worker_encode_and_send(
    enc: MMEncoder,
    sched: Optional[EncoderScheduler],
    request: dict,
) -> Optional[dict]:
    # Mooncake returns metadata for main to forward; zmq inlines the send.
    # Soft errors raise MMError so the dispatcher route maps them to HTTP.
    req_id = request["req_id"]
    time_stats_json = request.pop("time_stats_json", None)
    time_stats = EncoderReqTimeStats()
    if time_stats_json:
        time_stats.decode_json(time_stats_json)
    request["enter_time"] = time.time()
    modality = Modality.from_str(request["modality"])
    time_stats.modality = modality.name.lower()
    time_stats.set_metrics_collector(encode_server_module.encoder_metrics_collector)
    backend = enc.server_args.encoder_transfer_backend

    # URL state lives in main process module globals; workers don't see it.
    if backend == "zmq_to_scheduler" and request.get("embedding_port") is None:
        raise MMError(
            "Encoder DP mode does not support zmq_to_scheduler with "
            "embedding_port=None (URL state isn't synchronised to workers). "
            "Provide an explicit embedding_port list, switch to mooncake / "
            "zmq_to_tokenizer, or run without --dp-size.",
            code=HTTPStatus.BAD_REQUEST,
        )

    time_stats.set_mm_encode_start_time()
    encode_coro = (
        sched.submit(request)
        if sched is not None and modality in _BATCHABLE_MODALITIES
        else enc.encode_request(request, modality)
    )
    try:
        nbytes, embedding_len, embedding_dim, error_msg, error_code = await encode_coro
    except asyncio.TimeoutError:
        time_stats.trace_ctx.abort(abort_info={"reason": "encoder batch timed out"})
        raise

    if error_msg:
        time_stats.trace_ctx.abort(abort_info={"reason": error_msg})
        # zmq backends still forward an error EmbeddingData to P so it
        # doesn't block; send failures here are swallowed.
        try:
            await _push_embedding_to_prefill(enc, request)
        except Exception as e:
            logger.error(
                f"DP error-send failed for req_id={req_id}: {e}", exc_info=True
            )
        # Free the error EmbeddingData stored during encode, or it leaks in
        # pending embedding state and pins /health into "busy". Neither path
        # guarantees cleanup on its own: mooncake's _push_embedding_to_prefill
        # is a no-op, and a swallowed zmq send failure above skips its own pop.
        # zmq lacks the Mooncake inflight state, so discard its embedding
        # directly. Mirrors the non-DP error path.
        if backend == "mooncake":
            await enc.complete_inflight_encode(req_id, None)
        else:
            enc.discard_embedding(req_id)
        raise MMError(error_msg, code=error_code or HTTPStatus.INTERNAL_SERVER_ERROR)

    time_stats.set_mm_encode_end_time()

    if backend == "mooncake":
        request.pop("mm_items", None)
        request.update(
            embedding_size=nbytes,
            embedding_len=embedding_len,
            embedding_dim=embedding_dim,
        )
        # Free the held embedding if the follow-up /send never arrives (same
        # send_timeout cleanup the non-DP path uses).
        await enc.complete_inflight_encode(
            req_id, (nbytes, embedding_len, embedding_dim)
        )
        return request

    await _push_embedding_to_prefill(enc, request)
    return None


async def _dp_worker_health_encode(enc: MMEncoder) -> None:
    """Functional health probe run on a DP worker.

    Process-liveness (proc.sentinel) can't see a worker that's alive but
    wedged — hung GPU, NCCL deadlock, stalled ZMQ, or a blocked event loop.
    When idle, run a tiny dummy encode to exercise the VIT forward and surface
    those stalls. No prefill destination: the embedding is discarded, mirroring
    the non-DP /health path. Raises on encode failure so the worker envelope
    carries ``_error`` back to the dispatcher.
    """
    # Busy worker: in-flight traffic already proves liveness, so skip the probe
    # and report healthy — the same pending-embedding signal the non-DP
    # /health path uses. A wedged-but-busy worker never reaches here because
    # it can't service the recv, so the dispatcher's broadcast still times out → 503.
    if enc.has_pending_embeddings():
        return None

    if enc.supports_modality(Modality.IMAGE):
        mm_items = [f"data:image/png;base64,{MINIMUM_PNG_PICTURE_BASE64}"]
        modality = Modality.IMAGE
    elif enc.supports_modality(Modality.AUDIO):
        mm_items = [f"data:audio/wav;base64,{MINIMUM_WAV_SILENCE_BASE64}"]
        modality = Modality.AUDIO
    else:
        # No processor → can't functionally probe; liveness alone is healthy.
        return None

    # uuid keeps rids unique across workers; a bare time.time() can collide.
    req_id = f"{HEALTH_CHECK_RID_PREFIX}_{uuid.uuid4().hex}"
    try:
        _, _, _, error_msg, error_code = await enc.encode(
            mm_items=mm_items,
            modality=modality,
            req_id=req_id,
            num_parts=1,
            part_idx=0,
        )
    finally:
        # Never leave the dummy embedding sitting in the send map.
        enc.discard_embedding(req_id)

    if error_msg:
        raise MMError(error_msg, code=error_code or HTTPStatus.INTERNAL_SERVER_ERROR)


async def _dp_worker_handle_profile(
    enc: MMEncoder, dp_rank: int, dp_type: str, request: dict
) -> dict:
    prefix = f"dp_rank={dp_rank}: "
    if dp_type == "start_profile":
        req = request.get("profile_req") or ProfileReq()
        req.req_type = ProfileReqType.START_PROFILE
        if enc.profiler is None:
            enc.profiler = EncoderProfiler(dp_rank)
        ok, msg = enc.profiler.start(req)
        detail = (
            f"started profiling, output_dir={enc.profiler.output_dir}" if ok else msg
        )
    else:  # stop_profile
        if enc.profiler is None:
            return {"ok": False, "msg": prefix + "profiling not initialized"}
        ok, msg = enc.profiler.stop()
        detail = "stopped profiling" if ok else msg
    return {"ok": ok, "msg": prefix + detail}


async def _dp_worker_handle_request(
    enc: MMEncoder,
    sched: EncoderScheduler,
    send_sock,
    send_lock: asyncio.Lock,
    dp_rank: int,
    request: dict,
    dp_type: str,
) -> None:
    t0 = time.time()
    modality_str = str(request.get("modality", "image")).lower()
    is_encode = dp_type not in (
        "start_profile",
        "stop_profile",
        "health_encode",
        "send",
    )
    if is_encode and encode_server_module.encoder_metrics_collector is not None:
        encode_server_module.encoder_metrics_collector.inc_requests_received(
            modality=modality_str
        )
    try:
        if dp_type in ("start_profile", "stop_profile"):
            content = await _dp_worker_handle_profile(enc, dp_rank, dp_type, request)
        elif dp_type == "health_encode":
            content = await _dp_worker_health_encode(enc)
        elif dp_type == "send":
            req_id = request["req_id"]
            await enc.send(
                req_id=req_id,
                prefill_host=request["prefill_host"],
                embedding_port=request["embedding_port"],
                session_id=request["session_id"],
                buffer_address=request["buffer_address"],
            )
            await enc.release_inflight_encode(req_id)
            content = None
        else:
            content = await _dp_worker_encode_and_send(enc, sched, request)

        logger.info(
            f"MM-Encoder [dp_rank={dp_rank}] {dp_type} done: "
            f"req_id={request.get('req_id', '?')}, "
            f"modality={request.get('modality', 'image')}, "
            f"cost={(time.time() - t0) * 1000:.1f}ms"
        )
        if is_encode and encode_server_module.encoder_metrics_collector is not None:
            encode_server_module.encoder_metrics_collector.inc_requests_total(
                modality=modality_str, status="success"
            )
        envelope = {
            "req_id": request.get("req_id", ""),
            "_dp_type": dp_type,
            "content": content,
        }
    except Exception as e:
        logger.error(
            f"DP worker {dp_rank} error on {dp_type} "
            f"req_id={request.get('req_id', '?')}: {e}",
            exc_info=True,
        )
        if is_encode and encode_server_module.encoder_metrics_collector is not None:
            encode_server_module.encoder_metrics_collector.inc_requests_total(
                modality=modality_str, status="error"
            )
        err_code = int(getattr(e, "code", None) or HTTPStatus.INTERNAL_SERVER_ERROR)
        envelope = {
            "req_id": request.get("req_id", ""),
            "_dp_type": dp_type,
            "content": None,
            "_error": str(e),
            "_error_type": type(e).__name__,
            "_error_code": err_code,
        }

    # pyzmq async send isn't safe for concurrent senders.
    try:
        async with send_lock:
            await async_sock_send(send_sock, wrap_as_pickle(envelope))
    except Exception:
        logger.error(
            f"DP worker {dp_rank} failed to send envelope for "
            f"req_id={request.get('req_id', '?')}",
            exc_info=True,
        )


async def run_dp_worker(
    server_args: ServerArgs,
    dp_rank: int,
    gpu_id: int,
    dispatch_path: str,
    result_path: str,
):
    logger.info(
        f"DP worker {dp_rank} starting on gpu_id={gpu_id} "
        f"(CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'unset')})"
    )

    # gpu_id is the device chosen by maybe_reindex_device_id in the parent:
    # 0 when CVD is pinned to one GPU, else the absolute id. rank=0, so
    # MMEncoder runs set_device(base_gpu_id).
    args = copy.deepcopy(server_args)
    args.base_gpu_id = gpu_id
    args.tp_size = 1
    enc = MMEncoder(args, dist_init_method=f"tcp://127.0.0.1:{get_free_port()}", rank=0)

    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()
        labels = {
            "model_name": server_args.served_model_name,
            "dp_rank": str(dp_rank),
        }
        if server_args.extra_metric_labels:
            labels.update(server_args.extra_metric_labels)
        encode_server_module.encoder_metrics_collector = EncoderMetricsCollector(labels)
        enc.dp_rank = dp_rank

    sched = EncoderScheduler(
        encoder=enc, send_sockets=[], max_batch_size=ENCODER_MAX_BATCH_SIZE
    )

    ctx = zmq.asyncio.Context(2)
    recv_sock = get_zmq_socket(ctx, zmq.PULL, dispatch_path, False)
    send_sock = get_zmq_socket(ctx, zmq.PUSH, result_path, False)
    send_lock = asyncio.Lock()
    inflight: Set[asyncio.Task] = set()
    max_inflight = envs.SGLANG_ENCODER_DP_WORKER_MAX_INFLIGHT.get()
    if max_inflight < ENCODER_MAX_BATCH_SIZE:
        logger.warning(
            f"SGLANG_ENCODER_DP_WORKER_MAX_INFLIGHT={max_inflight} is below "
            f"ENCODER_MAX_BATCH_SIZE={ENCODER_MAX_BATCH_SIZE}; the encoder "
            f"will never assemble a full batch."
        )
    inflight_sem = asyncio.Semaphore(max_inflight)
    sched.start()
    logger.info(f"DP worker {dp_rank} ready")

    try:
        while True:
            await inflight_sem.acquire()
            spawned = False
            try:
                try:
                    request = await async_sock_recv(recv_sock)
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.error(f"DP worker {dp_rank} recv error", exc_info=True)
                    continue
                if not isinstance(request, dict):
                    logger.error(
                        f"DP worker {dp_rank} received non-dict request "
                        f"({type(request).__name__}); dropping"
                    )
                    continue
                dp_type = request.pop("_dp_type", "encode")

                async def _run(req=request, t=dp_type):
                    try:
                        await _dp_worker_handle_request(
                            enc, sched, send_sock, send_lock, dp_rank, req, t
                        )
                    finally:
                        inflight_sem.release()

                task = asyncio.create_task(_run())
                spawned = True
                inflight.add(task)
                task.add_done_callback(inflight.discard)
            finally:
                if not spawned:
                    inflight_sem.release()
    finally:
        for task in inflight:
            task.cancel()
        ctx.destroy(linger=0)


def launch_dp_worker(
    server_args: ServerArgs,
    dp_rank: int,
    gpu_id: int,
    dispatch_path: str,
    result_path: str,
):
    try:
        configure_logger(server_args, prefix=f" encode_dp_worker[{dp_rank}]")
        asyncio.run(
            run_dp_worker(server_args, dp_rank, gpu_id, dispatch_path, result_path)
        )
    except KeyboardInterrupt:
        logger.info(f"DP worker {dp_rank} exiting")
    except Exception:
        traceback.print_exc()


def launch_local_runtime(server_args: ServerArgs) -> EncoderRuntime:
    """Launch the current non-DP Scheduler and TP Encoder group.

    This function owns backend construction only.  HTTP/gRPC middleware,
    service registration, and network serving remain Transport concerns.
    """
    if server_args.dp_size > 1:
        raise ValueError(
            "launch_local_runtime requires --dp-size 1; got "
            f"dp_size={server_args.dp_size}."
        )

    # Set up prometheus metrics.
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()
        labels = {
            "model_name": server_args.served_model_name,
            "dp_rank": "0",
        }
        if server_args.extra_metric_labels:
            labels.update(server_args.extra_metric_labels)
        encode_server_module.encoder_metrics_collector = EncoderMetricsCollector(labels)

    process_context = mp.get_context("spawn")
    zmq_context = zmq.Context(10)
    ipc_path_prefix = random_uuid()
    port_args = PortArgs.init_new(server_args)
    if server_args.dist_init_addr:
        dist_init_method = NetworkAddress.parse(server_args.dist_init_addr).to_tcp()
    else:
        dist_init_method = NetworkAddress(
            server_args.host or "127.0.0.1", port_args.nccl_port
        ).to_tcp()

    if server_args.enable_trace:
        process_tracing_init(
            server_args.otlp_traces_endpoint,
            "sglang",
            trace_modules=server_args.trace_modules,
        )
        trace_set_thread_info("Encoder")

    send_sockets: List[zmq.Socket] = []
    tp_processes: List[mp.Process] = []
    for rank in range(1, server_args.tp_size):
        schedule_path = f"ipc:///tmp/{ipc_path_prefix}_schedule_{rank}"
        send_sockets.append(
            get_zmq_socket(zmq_context, zmq.PUSH, schedule_path, bind=False)
        )
        process = process_context.Process(
            target=launch_encoder,
            args=(server_args, schedule_path, dist_init_method, rank),
            daemon=True,
        )
        process.start()
        tp_processes.append(process)

    encoder = MMEncoder(server_args, dist_init_method=dist_init_method)
    scheduler = EncoderScheduler(
        encoder,
        send_sockets,
        max_batch_size=ENCODER_MAX_BATCH_SIZE,
    )
    return EncoderRuntime(
        encoder=encoder,
        scheduler=scheduler,
        send_sockets=send_sockets,
        zmq_context=zmq_context,
        tp_processes=tp_processes,
    )


def launch_dp_runtime(server_args: ServerArgs) -> DPDispatcher:
    """Launch the protocol-neutral DP backend and return its dispatcher.

    HTTP uses this entry point today.  gRPC can reuse it later without
    importing HTTP application state or Uvicorn.
    """
    if server_args.dp_size <= 1 or server_args.tp_size != 1:
        raise ValueError(
            "Encoder DP mode requires --dp-size > 1 and --tp-size 1; got "
            f"dp_size={server_args.dp_size}, tp_size={server_args.tp_size}."
        )
    dp_size = server_args.dp_size
    logger.info(f"Launching encoder in DP mode: dp_size={dp_size}")

    # DP mode: workers (subprocesses) write metrics to the shared multiproc dir;
    # the main process exposes the aggregated /metrics endpoint.
    if server_args.enable_metrics:
        set_prometheus_multiproc_dir()

    ctx = mp.get_context("spawn")
    ipc_prefix = random_uuid()
    async_zmq_ctx = zmq.asyncio.Context(dp_size + 1)

    result_path = f"ipc:///tmp/{ipc_prefix}_dp_result"
    result_socket = get_zmq_socket(async_zmq_ctx, zmq.PULL, result_path, True)
    dispatch_sockets: List[zmq.asyncio.Socket] = [
        get_zmq_socket(
            async_zmq_ctx, zmq.PUSH, f"ipc:///tmp/{ipc_prefix}_dp_dispatch_{r}", True
        )
        for r in range(dp_size)
    ]

    worker_processes: List[mp.Process] = []

    def _kill_workers():
        for process in worker_processes:
            if process.is_alive():
                process.kill()
        for process in worker_processes:
            process.join(timeout=5)

    # Register atexit BEFORE spawn loop so partial spawns get reaped on
    # exception (atexit holds the list ref and reads it at exit time).
    atexit.register(_kill_workers)

    for dp_rank in range(dp_size):
        gpu_id = server_args.base_gpu_id + dp_rank
        # Pin the device parent-side around spawn (same convention as the
        # scheduler launcher and DP controller) so the child inherits
        # CUDA_VISIBLE_DEVICES from its first instruction, before any import
        # can enumerate CUDA. No-op unless SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS
        # is set, in which case gpu_id is reindexed to 0 and CVD is pinned.
        with maybe_reindex_device_id(gpu_id) as gpu_id:
            process = ctx.Process(
                target=launch_dp_worker,
                args=(
                    server_args,
                    dp_rank,
                    gpu_id,
                    f"ipc:///tmp/{ipc_prefix}_dp_dispatch_{dp_rank}",
                    result_path,
                ),
                daemon=False,
            )
            process.start()
        worker_processes.append(process)

    labels = {"model_name": server_args.served_model_name}
    if server_args.extra_metric_labels:
        labels.update(server_args.extra_metric_labels)
    return DPDispatcher(
        dp_size,
        dispatch_sockets,
        result_socket,
        worker_processes,
        enable_metrics=server_args.enable_metrics,
        labels=labels,
    )
