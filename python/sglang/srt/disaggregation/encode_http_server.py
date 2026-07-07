"""HTTP API layer for the EPD encoder server.

This module is designed to be replaceable by a Rust implementation.
It contains the FastAPI application, HTTP route handlers, server lifecycle
management, DP dispatch infrastructure, and TP/DP worker process management.

GPU tensor operations remain in :mod:`encode_server.MMEncoder`.
"""

import asyncio
import contextlib
import copy
import logging
import multiprocessing as mp
import os
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from typing import Annotated, Dict, List, Optional, Set, Tuple

import requests as http_requests
import uvicorn
import zmq
import zmq.asyncio
from fastapi import Body, FastAPI
from fastapi.responses import ORJSONResponse, Response

import sglang.srt.disaggregation.encode_server as encode_server_module
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.encode_server import (
    _BATCHABLE_MODALITIES,
    ENCODER_MAX_BATCH_SIZE,
    ENCODER_REQ_TIMEOUT,
    EncoderProfiler,
    EncoderScheduler,
    MMEncoder,
    MMError,
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
from sglang.srt.server_args import (
    PortArgs,
    ServerArgs,
)
from sglang.srt.utils import (
    add_prometheus_middleware,
    configure_logger,
    random_uuid,
    set_prometheus_multiproc_dir,
)
from sglang.srt.utils.common import configure_logger, maybe_reindex_device_id
from sglang.srt.utils.network import (
    NetworkAddress,
    get_free_port,
    get_local_ip_auto,
    get_zmq_socket,
)

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT = 30

# Minimal 32x32 black PNG for health check dummy encode
MINIMUM_PNG_PICTURE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="

# Minimal WAV: 16kHz mono 16-bit PCM, 160 samples (0.01s) of silence
MINIMUM_WAV_SILENCE_BASE64 = "UklGRmQBAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YUABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="

encoder: Optional[MMEncoder] = None
send_sockets: List[zmq.Socket] = []
encoder_scheduler: Optional[EncoderScheduler] = None

# DP mode (--dp-size > 1): each rank runs as a subprocess with its own
# MMEncoder on its own GPU; the main process only routes via ZMQ so the
# asyncio event loop is never blocked by GPU work.
dp_dispatcher: Optional["DPDispatcher"] = None


def is_health_check_request(rid: Optional[str]) -> bool:
    return isinstance(rid, str) and rid.startswith(HEALTH_CHECK_RID_PREFIX)


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
        enc.embedding_to_send.pop(req_id, None)
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
        enc.embedding_to_send.pop(req_id, None)


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
        # embedding_to_send and pins /health into "busy" (a non-empty
        # embedding_to_send reads as busy, skipping the probe). Neither path
        # guarantees cleanup on its own: mooncake's _push_embedding_to_prefill
        # is a no-op, and a swallowed zmq send failure above skips its own pop.
        # zmq lacks the inflight attrs so _cleanup_inflight_encode_state would
        # early-return on it — pop directly. Mirrors the non-DP error path.
        if backend == "mooncake":
            await enc._cleanup_inflight_encode_state(req_id)
        else:
            enc.embedding_to_send.pop(req_id, None)
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
        enc._schedule_inflight_encode_cleanup(req_id)
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
    # and report healthy — same `embedding_to_send` signal the non-DP /health
    # path uses. A wedged-but-busy worker never reaches here (it can't service
    # the recv), so the dispatcher's broadcast still times out → 503.
    if enc.embedding_to_send:
        return None

    if enc.image_processor is not None:
        mm_items = [f"data:image/png;base64,{MINIMUM_PNG_PICTURE_BASE64}"]
        modality = Modality.IMAGE
    elif enc.audio_processor is not None:
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
        enc.embedding_to_send.pop(req_id, None)

    if error_msg:
        raise MMError(error_msg, code=error_code or HTTPStatus.INTERNAL_SERVER_ERROR)


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
            raise MMError(
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
            return await asyncio.wait_for(future, timeout=ENCODER_REQ_TIMEOUT)
        except asyncio.TimeoutError:
            self._drop_pending_and_mapping(rank, req_id)
            return self._timeout_envelope(
                req_id,
                "encode",
                f"Encoder DP rank={rank} timed out after {ENCODER_REQ_TIMEOUT}s",
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
            return await asyncio.wait_for(future, timeout=ENCODER_REQ_TIMEOUT)
        except asyncio.TimeoutError:
            self.pending_futures[rank].pop(key, None)
            self.req_id_to_rank.pop(req_id, None)
            return self._timeout_envelope(
                req_id,
                "send",
                f"Encoder DP rank={rank} /send timed out after {ENCODER_REQ_TIMEOUT}s",
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
        eff_timeout = timeout if timeout is not None else ENCODER_REQ_TIMEOUT
        alive_ranks = self.alive_ranks
        if not alive_ranks:
            raise MMError(
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
            # cancels the scheduled cleanup + frees embedding/forward state
            await enc._cleanup_inflight_encode_state(req_id)
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
    # Acquire-before-recv → back-pressure propagates to the dispatcher
    # PUSH buffer. Must be ≥ ENCODER_MAX_BATCH_SIZE or batching degrades.
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

    # Task-per-request so EncoderScheduler.pending_queue accumulates and
    # actual cross-request batching can happen.
    try:
        while True:
            await inflight_sem.acquire()
            # Released by _run on success or the outer finally if not spawned.
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
                # Ownership transferred to _run; mark before any op that could
                # raise (theoretical: set.add / add_done_callback) and cause a
                # double-release.
                spawned = True
                inflight.add(task)
                task.add_done_callback(inflight.discard)
            finally:
                if not spawned:
                    inflight_sem.release()
    finally:
        # Close zmq on exception/cancellation (normal stop is parent SIGKILL).
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


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    global encoder_scheduler
    if dp_dispatcher is not None:
        dp_dispatcher.start()
        yield
        return
    if encoder is not None:
        encoder_scheduler = EncoderScheduler(
            encoder, send_sockets, max_batch_size=ENCODER_MAX_BATCH_SIZE
        )
        encoder_scheduler.start()
    try:
        yield
    finally:
        if encoder_scheduler is not None:
            await encoder_scheduler.stop()


app = FastAPI(lifespan=_lifespan)


async def run_encoder(
    server_args: ServerArgs, schedule_path, dist_init_method, rank: int
):
    encoder = MMEncoder(server_args, schedule_path, dist_init_method, rank)
    while True:
        request = await async_sock_recv(encoder.schedule_socket)
        await _handle_encoder_worker_request(encoder, request)


async def _handle_encoder_worker_request(encoder: MMEncoder, request):
    if isinstance(request, ProfileReq):
        if request.req_type == ProfileReqType.START_PROFILE:
            if encoder.profiler is None:
                encoder.profiler = EncoderProfiler(encoder.rank)
            encoder.profiler.start(request)
        else:
            encoder.profiler.stop()
    elif isinstance(request, dict) and request.get("type") == "batch_encode":
        await encoder.batch_encode(
            request["requests"],
            Modality.from_str(request["modality"]),
        )
    elif (
        isinstance(request, dict)
        and isinstance(request.get("req_id"), str)
        and request["req_id"].startswith(HEALTH_CHECK_RID_PREFIX)
    ):
        await encoder.encode(
            mm_items=request["mm_items"],
            modality=Modality.from_str(request["modality"]),
            req_id=request["req_id"],
            num_parts=request["num_parts"],
            part_idx=request["part_idx"],
            hashes=request.get("hashes"),
        )
    else:
        await encoder.encode_request(request, Modality.from_str(request["modality"]))


def launch_encoder(server_args, schedule_path, dist_init_method, rank):
    try:
        asyncio.run(run_encoder(server_args, schedule_path, dist_init_method, rank))
    except KeyboardInterrupt:
        logger.info(f"Exit rank {rank}")
    except Exception:
        traceback.print_exc()


def _register_encoder_url_with_bootstrap(server_args: ServerArgs):
    """Asynchronously register this encoder with each bootstrap URL.

    Spawns a daemon thread that retries each URL independently with bounded
    backoff.  The encoder's own startup is not blocked: if some bootstrap
    server is slow or unreachable, only the background worker waits.

    Inspired by ``_ensure_prefill_info`` in disaggregation/decode.py: each
    target keeps its own retry count and is retried at a fixed interval
    instead of serialising sleeps in a single thread.
    """

    host = server_args.host
    if not host or host in ("0.0.0.0", "::"):
        host = get_local_ip_auto(server_args.host)
    scheme = "https" if server_args.ssl_certfile else "http"
    encoder_url = NetworkAddress(host, server_args.port).to_url(scheme)
    payload = {"url": encoder_url}
    bootstrap_urls = list(server_args.encoder_register_urls)
    if not bootstrap_urls:
        return

    max_retries = 30
    retry_interval = 5.0
    request_timeout = 5.0

    def _try_register_once(bootstrap_url: str) -> bool:
        try:
            resp = http_requests.post(
                f"{bootstrap_url}/register_encoder_url",
                json=payload,
                timeout=request_timeout,
            )
            if resp.status_code == 200:
                logger.info(
                    f"Registered encoder URL '{encoder_url}' with bootstrap "
                    f"at {bootstrap_url}"
                )
                return True
            logger.warning(
                f"Bootstrap {bootstrap_url} returned {resp.status_code}: {resp.text}"
            )
        except Exception as e:
            logger.debug(f"Register attempt to {bootstrap_url} failed: {e}")
        return False

    def _worker():
        pending = list(bootstrap_urls)
        retry_count = {url: 0 for url in pending}
        while pending:
            still_pending = []
            for bootstrap_url in pending:
                if _try_register_once(bootstrap_url):
                    continue
                retry_count[bootstrap_url] += 1
                if retry_count[bootstrap_url] >= max_retries:
                    logger.error(
                        f"Giving up on bootstrap {bootstrap_url} after "
                        f"{max_retries} attempts. Encoder discovery via this "
                        f"bootstrap will be incomplete."
                    )
                    continue
                still_pending.append(bootstrap_url)
            pending = still_pending
            if pending:
                time.sleep(retry_interval)

    threading.Thread(
        target=_worker, daemon=True, name="encoder-bootstrap-register"
    ).start()


def _unregister_encoder_url_from_bootstrap(server_args: ServerArgs):
    host = server_args.host
    if not host or host in ("0.0.0.0", "::"):
        host = get_local_ip_auto(server_args.host)
    scheme = "https" if server_args.ssl_certfile else "http"
    encoder_url = NetworkAddress(host, server_args.port).to_url(scheme)
    payload = {"url": encoder_url}

    for bootstrap_url in server_args.encoder_register_urls:
        try:
            resp = http_requests.delete(
                f"{bootstrap_url}/unregister_encoder_url",
                json=payload,
                timeout=2.0,
            )
            if resp.status_code == 200:
                logger.info(
                    f"Unregistered encoder URL '{encoder_url}' from "
                    f"bootstrap at {bootstrap_url}"
                )
            else:
                logger.warning(
                    f"Bootstrap {bootstrap_url} returned "
                    f"{resp.status_code} on unregister: {resp.text}"
                )
        except Exception as e:
            logger.debug(f"Unregister from {bootstrap_url} failed: {e}")


def launch_server(server_args: ServerArgs):
    configure_logger(server_args, prefix=" encode_server")
    if server_args.dp_size > 1:
        _launch_server_dp(server_args)
        return

    global encoder

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
        add_prometheus_middleware(app)

    ctx = mp.get_context("spawn")
    zmq_ctx = zmq.Context(10)
    ipc_path_prefix = random_uuid()
    port_args = PortArgs.init_new(server_args)
    if server_args.dist_init_addr:
        na = NetworkAddress.parse(server_args.dist_init_addr)
        dist_init_method = na.to_tcp()
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
    for rank in range(1, server_args.tp_size):
        schedule_path = f"ipc:///tmp/{ipc_path_prefix}_schedule_{rank}"
        send_sockets.append(
            get_zmq_socket(zmq_ctx, zmq.PUSH, schedule_path, bind=False)
        )
        ctx.Process(
            target=launch_encoder,
            args=(server_args, schedule_path, dist_init_method, rank),
            daemon=True,
        ).start()
    encoder = MMEncoder(server_args, dist_init_method=dist_init_method)

    # Register this encoder's URL with prefill server(s) if configured.
    if server_args.encoder_register_urls:
        import atexit

        _register_encoder_url_with_bootstrap(server_args)
        atexit.register(_unregister_encoder_url_from_bootstrap, server_args)

    uvicorn.run(app, host=server_args.host, port=server_args.port)


def _launch_server_dp(server_args: ServerArgs):
    global dp_dispatcher

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
        add_prometheus_middleware(app)

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

    # Register atexit BEFORE spawn loop so partial spawns get reaped on
    # exception (atexit holds the list ref and reads it at exit time).
    import atexit

    worker_processes: List[mp.Process] = []

    def _kill_workers():
        for p in worker_processes:
            if p.is_alive():
                p.kill()
        for p in worker_processes:
            p.join(timeout=5)

    atexit.register(_kill_workers)

    for dp_rank in range(dp_size):
        gpu_id = server_args.base_gpu_id + dp_rank
        # Pin the device parent-side around spawn (same convention as the
        # scheduler launcher and DP controller) so the child inherits
        # CUDA_VISIBLE_DEVICES from its first instruction, before any import
        # can enumerate CUDA. No-op unless SGLANG_ONE_VISIBLE_DEVICE_PER_PROCESS
        # is set, in which case gpu_id is reindexed to 0 and CVD is pinned.
        with maybe_reindex_device_id(gpu_id) as gpu_id:
            proc = ctx.Process(
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
            proc.start()
        worker_processes.append(proc)

    labels = {"model_name": server_args.served_model_name}
    if server_args.extra_metric_labels:
        labels.update(server_args.extra_metric_labels)
    dp_dispatcher = DPDispatcher(
        dp_size,
        dispatch_sockets,
        result_socket,
        worker_processes,
        enable_metrics=server_args.enable_metrics,
        labels=labels,
    )

    # Register this encoder's URL with prefill server(s) if configured.
    if server_args.encoder_register_urls:
        import atexit

        _register_encoder_url_with_bootstrap(server_args)
        atexit.register(_unregister_encoder_url_from_bootstrap, server_args)

    uvicorn.run(app, host=server_args.host, port=server_args.port)


def _summarise_dp_broadcast(results: List[dict]) -> Response:
    # Treat missing/None content as failure so a stuck rank doesn't hide
    # behind the others' "ok". Status = the most severe per-rank error code
    # (5xx beats 4xx) rather than a blanket 400, so a worker's 500/503/504
    # isn't misreported as a client error.
    msgs: List[str] = []
    error_codes: List[int] = []
    for r in results:
        content = r.get("content")
        if isinstance(content, dict):
            msgs.append(content.get("msg", ""))
            if not content.get("ok"):
                # Worker ran but reported a logical failure; no transport code,
                # so treat as a bad request (matches the non-DP profile path).
                error_codes.append(int(r.get("_error_code") or HTTPStatus.BAD_REQUEST))
        else:
            msgs.append(r.get("_error", "unknown error"))
            error_codes.append(
                int(r.get("_error_code") or HTTPStatus.INTERNAL_SERVER_ERROR)
            )
    status_code = 200 if not error_codes else max(error_codes)
    return Response(
        content="\n".join(msgs) + "\n",
        status_code=status_code,
    )


async def get_condition(rid):
    async with encode_server_module.cond_dict_lock:
        if rid not in encode_server_module.rid_to_cond:
            encode_server_module.rid_to_cond[rid] = asyncio.Condition()
        return encode_server_module.rid_to_cond[rid]


@app.post("/encode")
async def handle_encode_request(request: dict):
    req_id = request["req_id"]
    start_time = time.monotonic()
    time_stats_json = request.pop("time_stats_json", None)
    time_stats = EncoderReqTimeStats()
    if dp_dispatcher is not None:
        if time_stats_json:
            request = dict(request)
            request["time_stats_json"] = time_stats_json
        try:
            result = await dp_dispatcher.dispatch(request)
        except MMError as e:
            # Surface MMError.code (503 when all workers dead) instead of
            # FastAPI's default 500.
            logger.error(f"DP dispatch refused req_id={req_id}: {e}")
            return ORJSONResponse(
                status_code=int(e.code),
                content={"status": "error", "message": str(e), "req_id": req_id},
            )
        if result.get("_error"):
            error_type = result.get("_error_type", "")
            # `or` (not `dict.get(key, default)`) so explicit None falls back too.
            status_code = result.get("_error_code") or (
                HTTPStatus.BAD_REQUEST
                if error_type == "ValueError"
                else HTTPStatus.INTERNAL_SERVER_ERROR
            )
            logger.error(f"DP worker error for req_id={req_id}: {result['_error']}")
            return ORJSONResponse(
                status_code=status_code,
                content={
                    "status": "error",
                    "message": result["_error"],
                    "req_id": req_id,
                },
            )
        elapsed = time.monotonic() - start_time
        logger.info(
            f"[{req_id}] /encode completed in {elapsed:.3f}s, "
            f"modality={request.get('modality', 'image')}"
        )
        return ORJSONResponse(content=result.get("content"))

    modality_str = str(request.get("modality", "image")).lower()
    try:
        # when multiple decoder TP ranks POST /encode
        # with the same req_id, only the first triggers the VIT forward;
        # subsequent callers wait and return the same metadata.
        if encoder.server_args.encoder_transfer_backend == "mooncake":
            async with encoder._inflight_encode_lock:
                if req_id in encoder._inflight_encode_events:
                    event = encoder._inflight_encode_events[req_id]
                    is_duplicate = True
                else:
                    event = asyncio.Event()
                    encoder._inflight_encode_events[req_id] = event
                    is_duplicate = False

            if is_duplicate:
                await event.wait()
                meta = encoder._inflight_encode_meta.get(req_id)
                if meta is None:
                    return ORJSONResponse(
                        status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
                        content={
                            "status": "error",
                            "message": "Encode failed on the first request",
                            "req_id": req_id,
                        },
                    )
                nbytes, embedding_len, embedding_dim = meta
                # Build the same metadata response as the first request
                resp = dict(request)
                del resp["mm_items"]
                resp.update(
                    {
                        "embedding_size": nbytes,
                        "embedding_len": embedding_len,
                        "embedding_dim": embedding_dim,
                    }
                )
                return ORJSONResponse(content=resp)

        def start_background_send(req_id):
            task = asyncio.create_task(encoder.send_with_url(req_id=req_id))
            encoder.background_tasks.add(task)
            task.add_done_callback(encoder.background_tasks.discard)

        # broadcast request, lock together with rank0 await so NCCL
        # launch order matches the ZMQ dispatch order rank>0 sees.
        async with encoder.encode_dispatch_lock:
            request.update({"enter_time": time.time()})
            modality = Modality.from_str(request["modality"])
            if time_stats_json:
                time_stats.decode_json(time_stats_json)

            modality_str = modality.name.lower()
            time_stats.modality = modality_str
            time_stats.set_metrics_collector(
                encode_server_module.encoder_metrics_collector
            )
            time_stats.set_mm_encode_start_time()
            if encode_server_module.encoder_metrics_collector is not None:
                encode_server_module.encoder_metrics_collector.inc_requests_received(
                    modality=modality_str
                )
            if encoder_scheduler is not None and modality in _BATCHABLE_MODALITIES:
                try:
                    nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                        await encoder_scheduler.submit(request)
                    )
                except asyncio.TimeoutError:
                    time_stats.trace_ctx.abort(
                        abort_info={"reason": "encoder batch timed out"}
                    )
                    return ORJSONResponse(
                        status_code=HTTPStatus.GATEWAY_TIMEOUT,
                        content={
                            "status": "error",
                            "message": "encoder batch timed out",
                            "req_id": req_id,
                        },
                    )
            else:
                for socket in send_sockets:
                    sock_send(socket, wrap_as_pickle(request))
                nbytes, embedding_len, embedding_dim, error_msg, error_code = (
                    await encoder.encode_request(request, modality)
                )

        if error_msg:
            time_stats.trace_ctx.abort(abort_info={"reason": error_msg})
        else:
            time_stats.set_mm_encode_end_time()

        if error_msg:
            if encoder.server_args.encoder_transfer_backend == "zmq_to_scheduler":
                if request["embedding_port"] is None:
                    start_background_send(req_id)
                else:
                    for port in request["embedding_port"]:
                        await encoder.send(
                            req_id=req_id,
                            prefill_host=request["prefill_host"],
                            embedding_port=port,
                        )
            # Signal waiters on failure for mooncake
            if encoder.server_args.encoder_transfer_backend == "mooncake":
                encoder._inflight_encode_meta.pop(req_id, None)
                evt = encoder._inflight_encode_events.pop(req_id, None)
                if evt:
                    evt.set()
                await encoder._cleanup_inflight_encode_state(req_id)
            if encode_server_module.encoder_metrics_collector is not None:
                encode_server_module.encoder_metrics_collector.inc_requests_total(
                    modality=modality_str, status="error"
                )
            return ORJSONResponse(
                status_code=error_code,
                content={"status": "error", "message": error_msg, "req_id": req_id},
            )
        if encoder.server_args.encoder_transfer_backend == "mooncake":
            # Store metadata for duplicate callers and signal them
            encoder._inflight_encode_meta[req_id] = (
                nbytes,
                embedding_len,
                embedding_dim,
            )
            evt = encoder._inflight_encode_events.get(req_id)
            if evt:
                evt.set()
            encoder._schedule_inflight_encode_cleanup(req_id)
            del request["mm_items"]
            request.update(
                {
                    "embedding_size": nbytes,
                    "embedding_len": embedding_len,
                    "embedding_dim": embedding_dim,
                }
            )
            if encode_server_module.encoder_metrics_collector is not None:
                encode_server_module.encoder_metrics_collector.inc_requests_total(
                    modality=modality_str, status="success"
                )
            return ORJSONResponse(content=request)
        elif encoder.server_args.encoder_transfer_backend == "zmq_to_scheduler":
            logger.info(f"{request['embedding_port'] = }")
            if request["embedding_port"] is None:
                await encoder.send_with_url(
                    req_id=request["req_id"],
                )
            else:
                assert type(request["embedding_port"]) == list
                tasks = []
                for embedding_port in request["embedding_port"]:
                    tasks.append(
                        encoder.send(
                            req_id=request["req_id"],
                            prefill_host=request["prefill_host"],
                            embedding_port=embedding_port,
                        )
                    )
                await asyncio.gather(*tasks)
                encoder.embedding_to_send.pop(request["req_id"], None)
            if encode_server_module.encoder_metrics_collector is not None:
                encode_server_module.encoder_metrics_collector.inc_requests_total(
                    modality=modality_str, status="success"
                )
            return ORJSONResponse(content=None)
        elif encoder.server_args.encoder_transfer_backend == "zmq_to_tokenizer":
            await encoder.send(
                req_id=request["req_id"],
                prefill_host=request["prefill_host"],
                embedding_port=request["embedding_port"],
            )
            encoder.embedding_to_send.pop(request["req_id"], None)
            elapsed = time.monotonic() - start_time
            logger.info(
                f"[{req_id}] /encode completed in {elapsed:.3f}s, "
                f"modality={request['modality']}, tokens={embedding_len}"
            )
            if encode_server_module.encoder_metrics_collector is not None:
                encode_server_module.encoder_metrics_collector.inc_requests_total(
                    modality=modality_str, status="success"
                )
            return ORJSONResponse(content=None)
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error in encoder logic for {req_id}: {error_msg}")
        encode_server_module.rid_to_err_msg[req_id] = error_msg
        # Ensure inflight waiters are unblocked on unexpected errors
        if encoder.server_args.encoder_transfer_backend == "mooncake":
            encoder._inflight_encode_meta.pop(req_id, None)
            evt = encoder._inflight_encode_events.pop(req_id, None)
            if evt:
                evt.set()
            await encoder._cleanup_inflight_encode_state(req_id)
        if encode_server_module.encoder_metrics_collector is not None:
            encode_server_module.encoder_metrics_collector.inc_requests_total(
                modality=modality_str, status="error"
            )
        return ORJSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": error_msg,
                "req_id": req_id,
            },
        )


@app.post("/send")
async def handle_send_request(request: dict):
    # mooncake backend
    if dp_dispatcher is not None:
        try:
            result = await dp_dispatcher.dispatch_send(request)
        except MMError as e:
            req_id = request.get("req_id", "?")
            logger.error(f"DP dispatch_send refused req_id={req_id}: {e}")
            return Response(
                content=f"Encoder DP worker send error: {e}",
                status_code=int(e.code),
            )
        if result.get("_error"):
            req_id = request.get("req_id", "?")
            status_code = result.get("_error_code") or int(
                HTTPStatus.INTERNAL_SERVER_ERROR
            )
            logger.error(
                f"DP worker send error for req_id={req_id}: {result['_error']}"
            )
            return Response(
                content=f"Encoder DP worker send error: {result['_error']}",
                status_code=status_code,
            )
        return ORJSONResponse(content=result.get("content"))
    await encoder.send(
        req_id=request["req_id"],
        prefill_host=request["prefill_host"],
        embedding_port=request["embedding_port"],
        session_id=request["session_id"],
        buffer_address=request["buffer_address"],
    )
    req_id = request["req_id"]
    # Don't pop embedding_to_send here — other decoder TP ranks may still
    # need it for their /send calls. Cleanup is handled by the scheduled
    # timeout task or _cleanup_inflight_encode_state.
    return ORJSONResponse(content=None)


@app.post("/scheduler_receive_url")
async def handle_scheduler_receive_url_request(request: dict):
    rid = request["req_id"]
    async with encode_server_module.rid_lock:
        if rid not in encode_server_module.rid_to_receive_endpoint:
            encode_server_module.rid_to_receive_endpoint[rid] = set()
            encode_server_module.rid_to_receive_count[rid] = request["receive_count"]
        assert (
            encode_server_module.rid_to_receive_count[rid] == request["receive_count"]
        )
        encode_server_module.rid_to_receive_endpoint[rid].add(request["receive_url"])
    cond = await get_condition(rid)
    async with cond:
        cond.notify_all()


@app.get("/health")
@app.get("/health_generate")
async def health_generate():
    """
    Health check endpoint for the encoder server.
    Performs a dummy encode to verify the encoder is functional.
    Returns 200 if the encoder is healthy, 503 otherwise.
    """
    if dp_dispatcher is not None:
        # Strict: any dead (exited) rank fails health → orchestrator restarts.
        if not dp_dispatcher.all_ranks_alive:
            return Response(status_code=503)
        # Process-liveness (proc.sentinel) can't see a worker that's alive but
        # wedged (hung GPU / NCCL deadlock / stalled ZMQ). Probe every rank with
        # a tiny dummy encode; each worker runs it only when idle and otherwise
        # reports healthy at once, keeping the probe off the GPU under load.
        try:
            results = await dp_dispatcher.broadcast(
                {"_dp_type": "health_encode"},
                timeout=HEALTH_CHECK_TIMEOUT,
            )
        except MMError:
            return Response(status_code=503)
        if any(r.get("_error") for r in results):
            return Response(status_code=503)
        return Response(status_code=200)
    if encoder is None:
        return Response(status_code=503)

    # Skip the dummy encode when real requests are already in flight — the
    # ongoing traffic already proves liveness, matching the scheduler's
    # `is_fully_idle`-based health-check skip pattern.
    if encoder.embedding_to_send:
        return Response(status_code=200)

    # Pick the first available modality for the dummy encode
    if encoder.image_processor is not None:
        mm_items = [f"data:image/png;base64,{MINIMUM_PNG_PICTURE_BASE64}"]
        modality = Modality.IMAGE
    elif encoder.audio_processor is not None:
        mm_items = [f"data:audio/wav;base64,{MINIMUM_WAV_SILENCE_BASE64}"]
        modality = Modality.AUDIO
    else:
        # No processor available, fall back to liveness check only
        return Response(status_code=200)

    try:
        # uuid keeps rids unique across workers; a bare time.time() can collide.
        req_id = f"{HEALTH_CHECK_RID_PREFIX}_{uuid.uuid4().hex}"

        dummy_request = {
            "mm_items": mm_items,
            "modality": modality.name,
            "req_id": req_id,
            "num_parts": 1,
            "part_idx": 0,
        }

        # Broadcast to other TP ranks so distributed ops stay in sync
        for socket in send_sockets:
            sock_send(socket, wrap_as_pickle(dummy_request))

        # Run encode on rank 0 with timeout
        _, _, _, error_msg, _ = await asyncio.wait_for(
            encoder.encode(
                mm_items=mm_items,
                modality=modality,
                req_id=req_id,
                num_parts=1,
                part_idx=0,
            ),
            timeout=HEALTH_CHECK_TIMEOUT,
        )

        # Clean up stored embedding
        encoder.embedding_to_send.pop(req_id, None)

        if error_msg:
            logger.error(f"Encoder health check failed: {error_msg}")
            return Response(status_code=503)

        return Response(status_code=200)

    except asyncio.TimeoutError:
        logger.error(f"Encoder health check timed out after {HEALTH_CHECK_TIMEOUT}s")
        return Response(status_code=503)
    except Exception as e:
        logger.error(f"Encoder health check failed: {e}")
        return Response(status_code=503)


@app.api_route("/start_profile", methods=["GET", "POST"])
async def start_profile_async(obj: Annotated[Optional[ProfileReq], Body()] = None):
    if dp_dispatcher is not None:
        if obj is not None:
            obj.req_type = ProfileReqType.START_PROFILE
        try:
            results = await dp_dispatcher.broadcast(
                {"_dp_type": "start_profile", "profile_req": obj}
            )
        except MMError as e:
            return Response(content=f"{e}\n", status_code=int(e.code))
        return _summarise_dp_broadcast(results)
    if encoder is None:
        return Response(content="encoder not ready\n", status_code=503)
    req = obj or ProfileReq()
    req.req_type = ProfileReqType.START_PROFILE
    for socket in send_sockets:
        sock_send(socket, req)
    if encoder.profiler is None:
        encoder.profiler = EncoderProfiler(encoder.rank)
    ok, msg = encoder.profiler.start(req)
    if ok:
        detail = (
            f"Start profiling. output_dir={encoder.profiler.output_dir} "
            f"profile_id={encoder.profiler.profile_id}\n"
        )
        return Response(content=detail, status_code=200)
    return Response(
        content=(msg or "Start profiling failed.\n"), status_code=HTTPStatus.BAD_REQUEST
    )


@app.api_route("/stop_profile", methods=["GET", "POST"])
async def stop_profile_async():
    if dp_dispatcher is not None:
        try:
            results = await dp_dispatcher.broadcast({"_dp_type": "stop_profile"})
        except MMError as e:
            return Response(content=f"{e}\n", status_code=int(e.code))
        return _summarise_dp_broadcast(results)
    if encoder is None:
        return Response(content="encoder not ready\n", status_code=503)
    if encoder.profiler is None:
        return Response(
            content="profiling not initialized\n", status_code=HTTPStatus.BAD_REQUEST
        )
    req = ProfileReq(req_type=ProfileReqType.STOP_PROFILE)
    for socket in send_sockets:
        sock_send(socket, req)
    ok, msg = encoder.profiler.stop()
    if ok:
        return Response(content="Stop profiling.\n", status_code=200)
    return Response(
        content=(msg or "Stop profiling failed.\n"), status_code=HTTPStatus.BAD_REQUEST
    )
