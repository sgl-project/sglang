"""HTTP API layer for the EPD encoder server.

This module is designed to be replaceable by a Rust implementation.
It contains the FastAPI application, HTTP route handlers, HTTP lifecycle, and
response conversion. Backend scheduling and process management are provided by
the protocol-neutral :mod:`encoder_runtime` module.

GPU tensor operations remain in :mod:`encode_server.MMEncoder`.
"""

import asyncio
import contextlib
import logging
import threading
import time
import uuid
from http import HTTPStatus
from typing import Annotated, List, Optional

import requests as http_requests
import uvicorn
import zmq
from fastapi import Body, FastAPI
from fastapi.responses import ORJSONResponse, Response

import sglang.srt.disaggregation.encode_server as encode_server_module
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.encode_server import (
    EncoderProfiler,
    MMEncoder,
    MMError,
)
from sglang.srt.disaggregation.encoder_runtime import (
    _BATCHABLE_MODALITIES,
    DPDispatcher,
    EncoderRuntime,
    EncoderScheduler,
    launch_dp_runtime,
    launch_local_runtime,
)
from sglang.srt.managers.io_struct import (
    ProfileReq,
    ProfileReqType,
    sock_send,
    wrap_as_pickle,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.observability.req_time_stats import EncoderReqTimeStats
from sglang.srt.server_args import ServerArgs
from sglang.srt.utils import (
    add_prometheus_middleware,
    configure_logger,
)
from sglang.srt.utils.network import NetworkAddress, get_local_ip_auto

logger = logging.getLogger(__name__)

HEALTH_CHECK_TIMEOUT = 30

# Minimal 32x32 black PNG for health check dummy encode
MINIMUM_PNG_PICTURE_BASE64 = "iVBORw0KGgoAAAANSUhEUgAAACAAAAAgCAYAAABzenr0AAAACXBIWXMAAA7EAAAOxAGVKw4bAAAAbUlEQVRYhe3VsQ2AMAxE0Y/lIgNQULD/OqyCMgCihCKSG4yRuKuiNH6JLsoEbMACOGBcua9HOR7Y6w6swBwMy0qLTpkeI77qdEBpBFAHBBDAGH8WrwJKI4AAegUCfAKgEgpQDvh3CR3oQCuav58qlAw73kKCSgAAAABJRU5ErkJggg=="

# Minimal WAV: 16kHz mono 16-bit PCM, 160 samples (0.01s) of silence
MINIMUM_WAV_SILENCE_BASE64 = "UklGRmQBAABXQVZFZm10IBAAAAABAAEAgD4AAAB9AAACABAAZGF0YUABAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA=="

encoder: Optional[MMEncoder] = None
send_sockets: List[zmq.Socket] = []
encoder_scheduler: Optional[EncoderScheduler] = None
local_runtime: Optional[EncoderRuntime] = None

# DP mode (--dp-size > 1): the protocol-neutral runtime owns worker processes
# and ZMQ; HTTP only keeps the dispatcher handle used by route handlers.
dp_dispatcher: Optional["DPDispatcher"] = None


def is_health_check_request(rid: Optional[str]) -> bool:
    return isinstance(rid, str) and rid.startswith(HEALTH_CHECK_RID_PREFIX)


@contextlib.asynccontextmanager
async def _lifespan(app: FastAPI):
    if dp_dispatcher is not None:
        dp_dispatcher.start()
        yield
        return
    if local_runtime is not None:
        local_runtime.start()
    try:
        yield
    finally:
        if local_runtime is not None:
            await local_runtime.stop()


app = FastAPI(lifespan=_lifespan)


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
    global dp_dispatcher, encoder, encoder_scheduler, local_runtime, send_sockets

    configure_logger(server_args, prefix=" encode_server")
    if server_args.dp_size > 1:
        dp_dispatcher = launch_dp_runtime(server_args)
        # encoder_runtime initializes multiprocess metrics before spawning;
        # HTTP only exposes their endpoint.
        if server_args.enable_metrics:
            add_prometheus_middleware(app)
    else:
        local_runtime = launch_local_runtime(server_args)
        # Compatibility aliases for the existing HTTP request path. Runtime is
        # now the sole constructor and lifecycle owner of these objects.
        encoder = local_runtime.encoder
        encoder_scheduler = local_runtime.scheduler
        send_sockets = local_runtime.send_sockets
        if server_args.enable_metrics:
            add_prometheus_middleware(app)

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
            is_owner, meta = await encoder.begin_or_wait_inflight_encode(req_id)
            if not is_owner:
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
                    if encoder.server_args.encoder_transfer_backend == "mooncake":
                        await encoder.complete_inflight_encode(req_id, None)
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
                await encoder.complete_inflight_encode(req_id, None)
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
            await encoder.complete_inflight_encode(
                req_id, (nbytes, embedding_len, embedding_dim)
            )
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
                encoder.discard_embedding(request["req_id"])
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
            encoder.discard_embedding(request["req_id"])
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
            await encoder.complete_inflight_encode(req_id, None)
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
    # Keep the pending embedding here because other decoder TP ranks may still
    # need it for their /send calls. Cleanup is handled by the scheduled
    # timeout task or release_inflight_encode.
    return ORJSONResponse(content=None)


@app.post("/scheduler_receive_url")
async def handle_scheduler_receive_url_request(request: dict):
    await encoder.register_embedding_destinations(
        request["req_id"],
        request["receive_count"],
        [request["receive_url"]],
    )


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
    if encoder.has_pending_embeddings():
        return Response(status_code=200)

    # Pick the first available modality for the dummy encode
    if encoder.supports_modality(Modality.IMAGE):
        mm_items = [f"data:image/png;base64,{MINIMUM_PNG_PICTURE_BASE64}"]
        modality = Modality.IMAGE
    elif encoder.supports_modality(Modality.AUDIO):
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
        encoder.discard_embedding(req_id)

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
