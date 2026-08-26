"""HTTP API layer for the EPD encoder server.

This module is designed to be replaceable by a Rust implementation.
It contains the FastAPI application, HTTP route handlers, HTTP lifecycle, and
response conversion. Backend scheduling and process management are provided by
the protocol-neutral :mod:`runtime` module.

GPU tensor operations remain in :mod:`server.MMEncoder`.
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

import sglang.srt.disaggregation.encoder.server as server_module
from sglang.srt.constants import HEALTH_CHECK_RID_PREFIX
from sglang.srt.disaggregation.encoder.runtime import (
    DPDispatcher,
    EncoderRuntime,
    EncoderScheduler,
    execute_encode_pipeline,
    launch_dp_runtime,
    launch_local_runtime,
)
from sglang.srt.disaggregation.encoder.server import (
    EncoderProfiler,
    MMEncoder,
    MMError,
)
from sglang.srt.managers.io_struct import (
    ProfileReq,
    ProfileReqType,
    sock_send,
    wrap_as_pickle,
)
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.runtime_context import (
    get_disagg,
    get_observability,
    get_parallel,
    get_serving,
    publish,
)
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
    publish(server_args, role="encoder")
    if get_parallel().config.dp_size > 1:
        dp_dispatcher = launch_dp_runtime(server_args)
        # runtime initializes multiprocess metrics before spawning;
        # HTTP only exposes their endpoint.
        if get_observability().enable_metrics:
            add_prometheus_middleware(app)
    else:
        local_runtime = launch_local_runtime(server_args)
        # Compatibility aliases for the existing HTTP request path. Runtime is
        # now the sole constructor and lifecycle owner of these objects.
        encoder = local_runtime.encoder
        encoder_scheduler = local_runtime.scheduler
        send_sockets = local_runtime.send_sockets
        if get_observability().enable_metrics:
            add_prometheus_middleware(app)

    # Register this encoder's URL with prefill server(s) if configured.
    if get_disagg().encoder_register_urls:
        import atexit

        _register_encoder_url_with_bootstrap(server_args)
        atexit.register(_unregister_encoder_url_from_bootstrap, server_args)

    uvicorn.run(app, host=get_serving().host, port=get_serving().port)


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
        content = result.get("content")
        return ORJSONResponse(content=content)

    try:
        if time_stats_json:
            request["time_stats_json"] = time_stats_json
        content = await execute_encode_pipeline(
            encoder,
            encoder_scheduler,
            request,
            send_sockets=send_sockets,
        )
        elapsed = time.monotonic() - start_time
        logger.info(
            f"[{req_id}] /encode completed in {elapsed:.3f}s, "
            f"modality={request.get('modality', 'image')}"
        )
        return ORJSONResponse(content=content)
    except asyncio.TimeoutError:
        return ORJSONResponse(
            status_code=HTTPStatus.GATEWAY_TIMEOUT,
            content={
                "status": "error",
                "message": "encoder batch timed out",
                "req_id": req_id,
            },
        )
    except MMError as e:
        return ORJSONResponse(
            status_code=int(e.code),
            content={"status": "error", "message": str(e), "req_id": req_id},
        )
    except Exception as e:
        error_msg = str(e)
        logger.error(f"Unexpected error in encoder logic for {req_id}: {error_msg}")
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
    """Mooncake-only: drive the RDMA push of a staged embedding. The zmq
    backends deliver embeddings inline during /encode and never call /send."""
    req_id = request["req_id"]
    receive_count = request.get("receive_count")
    if dp_dispatcher is not None:
        try:
            result = await dp_dispatcher.dispatch_send(request)
        except MMError as e:
            logger.error(f"DP dispatch_send refused req_id={req_id}: {e}")
            return Response(
                content=f"Encoder DP worker send error: {e}",
                status_code=int(e.code),
            )
        if result.get("_error"):
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
    sent = await encoder.send(
        req_id=req_id,
        prefill_host=request["prefill_host"],
        embedding_port=request["embedding_port"],
        session_id=request["session_id"],
        buffer_address=request["buffer_address"],
    )
    if not sent:
        # No transfer happened: fail fast rather than 200 + a phantom count.
        return ORJSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={
                "status": "error",
                "message": f"no staged embedding for req_id={req_id} (already released)",
                "req_id": req_id,
            },
        )
    # Sibling ranks share this embedding, so free it only once all have sent.
    # No count means a pre-refcount decoder: leave it to the sweep, as when
    # some rank never sends at all.
    if receive_count:
        await server_module.meta_registry.note_send_done(req_id, receive_count)
    return ORJSONResponse(content=None)


@app.post("/scheduler_receive_meta_data")
async def handle_scheduler_receive_meta_data(request: dict):
    """Decoder pull endpoint for the per-part encode metadata. Blocks until the
    encode publishes its sizes, so a pull that beats the encode simply waits."""
    req_id = request["req_id"]
    if dp_dispatcher is not None:
        try:
            result = await dp_dispatcher.dispatch_wait_metadata(request)
        except MMError as e:
            return ORJSONResponse(
                status_code=int(e.code),
                content={"status": "error", "message": str(e), "req_id": req_id},
            )
        if result.get("_error"):
            return ORJSONResponse(
                status_code=result.get("_error_code")
                or int(HTTPStatus.INTERNAL_SERVER_ERROR),
                content={
                    "status": "error",
                    "message": result["_error"],
                    "req_id": req_id,
                },
            )
        meta = result.get("content")
    else:
        try:
            meta = await server_module.meta_registry.wait(req_id)
        except asyncio.TimeoutError:
            logger.error(f"[{req_id}] /scheduler_receive_meta_data timed out")
            return ORJSONResponse(
                status_code=HTTPStatus.GATEWAY_TIMEOUT,
                content={
                    "status": "error",
                    "message": "encode metadata not ready",
                    "req_id": req_id,
                },
            )
    if meta is None or meta.get("error") is not None:
        message = meta["error"] if meta else "encode metadata missing"
        return ORJSONResponse(
            status_code=HTTPStatus.INTERNAL_SERVER_ERROR,
            content={"status": "error", "message": message, "req_id": req_id},
        )
    return ORJSONResponse(
        content={
            "req_id": req_id,
            "part_idx": request["part_idx"],
            "embedding_size": meta["embedding_size"],
            "embedding_len": meta["embedding_len"],
            "embedding_dim": meta["embedding_dim"],
        }
    )


@app.post("/scheduler_receive_url")
async def handle_scheduler_receive_url_request(request: dict):
    if dp_dispatcher is not None:
        try:
            result = await dp_dispatcher.dispatch_register_destinations(request)
        except MMError as e:
            return ORJSONResponse(
                status_code=int(e.code),
                content={
                    "status": "error",
                    "message": str(e),
                    "req_id": request["req_id"],
                },
            )
        if result.get("_error"):
            return ORJSONResponse(
                status_code=result.get("_error_code")
                or int(HTTPStatus.INTERNAL_SERVER_ERROR),
                content={
                    "status": "error",
                    "message": result["_error"],
                    "req_id": request["req_id"],
                },
            )
        return ORJSONResponse(content=None)
    if encoder is None:
        return ORJSONResponse(
            status_code=HTTPStatus.SERVICE_UNAVAILABLE,
            content={
                "status": "error",
                "message": "encoder not ready",
                "req_id": request["req_id"],
            },
        )
    try:
        await encoder.register_embedding_destinations(
            request["req_id"],
            request["receive_count"],
            [request["receive_url"]],
        )
    except MMError as e:
        return ORJSONResponse(
            status_code=int(e.code),
            content={
                "status": "error",
                "message": str(e),
                "req_id": request["req_id"],
            },
        )
    return ORJSONResponse(content=None)


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

        # A health encode participates in the same TP collectives as a real
        # request. Serialize its broadcast and rank-0 forward with every other
        # collective dispatch, then recheck whether traffic made the probe
        # unnecessary while it waited for the lock.
        async with encoder.encode_dispatch_lock:
            if encoder.has_pending_embeddings():
                return Response(status_code=200)
            for socket in send_sockets:
                sock_send(socket, wrap_as_pickle(dummy_request))

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
        await encoder.release_request(req_id)

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
