"""
Thin gRPC server wrapper — delegates to smg-grpc-servicer package.

A lightweight HTTP sidecar is started alongside the gRPC server to expose:
- /metrics (Prometheus, when --enable-metrics is set)
- /start_profile, /stop_profile (profiling control)

The sidecar is started on --grpc-http-sidecar-port (default: --port + 1)
once the gRPC request manager is ready, regardless of whether --enable-metrics
is set.
"""

import copy
import dataclasses
import hashlib
import inspect
import json
import logging
import os
import time

import grpc
from aiohttp import web
from smg_grpc_servicer.sglang import request_manager as grpc_request_manager
from smg_grpc_servicer.sglang import servicer as grpc_servicer

from sglang.srt.managers.io_struct import (
    ProfileReq,
    ProfileReqType,
    TokenizedGenerateReqInput,
)
from sglang.srt.managers.mm_utils import wrap_shm_features
from sglang.srt.server_args import set_global_server_args_for_tokenizer
from sglang.srt.utils.common import get_bool_env_var

logger = logging.getLogger(__name__)

# gRPC defaults to a small 4MB message cap. SMG sends already-preprocessed
# pixel tensors, so use a larger bounded default and keep an env override.
_DEFAULT_GRPC_MAX_MESSAGE_MB = 512


def _grpc_max_message_bytes() -> int:
    raw = os.environ.get("SGLANG_GRPC_MAX_MESSAGE_MB")
    mb = _DEFAULT_GRPC_MAX_MESSAGE_MB
    if raw is not None:
        try:
            mb = int(raw)
        except ValueError:
            logger.warning(
                "Invalid SGLANG_GRPC_MAX_MESSAGE_MB=%r; falling back to %dMB",
                raw,
                _DEFAULT_GRPC_MAX_MESSAGE_MB,
            )
    return max(mb, 1) * 1024 * 1024


def _with_grpc_message_size_options(options, max_message_bytes: int):
    keys = {"grpc.max_send_message_length", "grpc.max_receive_message_length"}
    merged = [(k, v) for k, v in options if k not in keys]
    merged.extend(
        [
            ("grpc.max_send_message_length", max_message_bytes),
            ("grpc.max_receive_message_length", max_message_bytes),
        ]
    )
    return merged


def _patch_grpc_message_size_options() -> None:
    if getattr(grpc, "_sglang_message_size_options_patched", False):
        return

    max_message_bytes = _grpc_max_message_bytes()
    original_aio_server = grpc.aio.server
    original_insecure_channel = grpc.insecure_channel

    def aio_server_with_message_size(*args, **kwargs):
        if "options" in kwargs:
            kwargs["options"] = [] if kwargs["options"] is None else kwargs["options"]
            kwargs["options"] = _with_grpc_message_size_options(
                kwargs["options"], max_message_bytes
            )
        elif len(args) >= 3:
            args = list(args)
            args[2] = [] if args[2] is None else args[2]
            args[2] = _with_grpc_message_size_options(args[2], max_message_bytes)
        else:
            kwargs["options"] = _with_grpc_message_size_options([], max_message_bytes)
        return original_aio_server(*args, **kwargs)

    def insecure_channel_with_message_size(target, options=None, *args, **kwargs):
        options = [] if options is None else options
        options = _with_grpc_message_size_options(options, max_message_bytes)
        return original_insecure_channel(target, options, *args, **kwargs)

    grpc.aio.server = aio_server_with_message_size
    grpc.insecure_channel = insecure_channel_with_message_size
    grpc._sglang_message_size_options_patched = True


def _patch_grpc_request_manager_shm_transport() -> None:
    """wrap shm/cuda-ipc features before sending to scheduler"""
    cls = grpc_request_manager.GrpcRequestManager
    original_send = cls._send_to_scheduler
    if getattr(original_send, "_sglang_wraps_shm_features", False):
        return

    def copy_multimodal_request(obj):
        if not isinstance(obj, TokenizedGenerateReqInput) or obj.mm_inputs is None:
            return obj

        mm_items = [copy.copy(item) for item in obj.mm_inputs.mm_items]
        for item in mm_items:
            item.set_pad_value()
        mm_inputs = dataclasses.replace(obj.mm_inputs, mm_items=mm_items)
        return dataclasses.replace(obj, mm_inputs=mm_inputs)

    async def send_to_scheduler_with_shm(self, obj):
        if obj is not None:
            obj = copy_multimodal_request(obj)
            obj = wrap_shm_features(obj)
        return await original_send(self, obj)

    send_to_scheduler_with_shm._sglang_wraps_shm_features = True
    cls._send_to_scheduler = send_to_scheduler_with_shm


def _hash_grpc_bytes(chunks) -> int:
    hasher = hashlib.sha256()
    for chunk in chunks:
        hasher.update(len(chunk).to_bytes(8, byteorder="big", signed=False))
        hasher.update(chunk)
    return int.from_bytes(hasher.digest()[:8], byteorder="big", signed=False)


def _encode_grpc_shape(shape) -> bytes:
    return b"".join(
        int(dim).to_bytes(8, byteorder="big", signed=False) for dim in shape
    )


def _hash_grpc_mm_proto(mm_proto) -> int | None:
    """hash for multimodal data"""
    if mm_proto.image_data:
        return _hash_grpc_bytes(mm_proto.image_data)

    if not mm_proto.HasField("pixel_values"):
        return None

    chunks = [
        mm_proto.pixel_values.dtype.encode(),
        _encode_grpc_shape(mm_proto.pixel_values.shape),
        mm_proto.pixel_values.data,
    ]
    for key in sorted(mm_proto.model_specific_tensors):
        tensor_data = mm_proto.model_specific_tensors[key]
        chunks.extend(
            [
                key.encode(),
                tensor_data.dtype.encode(),
                _encode_grpc_shape(tensor_data.shape),
                tensor_data.data,
            ]
        )
    return _hash_grpc_bytes(chunks)


def _patch_grpc_multimodal_hashes() -> None:
    """for processed-vlm input, hash on raw proto bytes is faster than hashing before scheduler dispatch (on python objects)"""
    cls = grpc_servicer.SGLangSchedulerServicer
    original_parse = cls._parse_mm_inputs
    if getattr(original_parse, "_sglang_sets_mm_hash", False):
        return

    def parse_mm_inputs_with_hash(self, mm_proto):
        mm_inputs = original_parse(self, mm_proto)
        mm_hash = _hash_grpc_mm_proto(mm_proto)
        if mm_hash is not None and len(mm_inputs.mm_items) == 1:
            mm_inputs.mm_items[0].hash = mm_hash
        return mm_inputs

    parse_mm_inputs_with_hash._sglang_sets_mm_hash = True
    cls._parse_mm_inputs = parse_mm_inputs_with_hash


async def _start_sidecar_server(host: str, port: int, app):
    """Start the aiohttp sidecar and return the runner for cleanup."""
    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, host, port)
        await site.start()
    except BaseException:
        await runner.cleanup()
        raise
    logger.info("HTTP sidecar server started on http://%s:%d", host, port)
    return runner


def _add_metrics_routes(app):
    """Add Prometheus /metrics endpoint to the aiohttp app."""
    from prometheus_client import (
        CollectorRegistry,
        multiprocess,
    )
    from prometheus_client.openmetrics.exposition import (
        CONTENT_TYPE_LATEST,
        generate_latest,
    )

    async def metrics_handler(request):
        try:
            registry = CollectorRegistry()
            multiprocess.MultiProcessCollector(registry)
            data = generate_latest(registry)
            return web.Response(
                body=data,
                headers={"Content-Type": CONTENT_TYPE_LATEST},
            )
        except Exception:
            logger.exception("Failed to generate Prometheus metrics")
            return web.Response(status=500, text="Failed to generate metrics")

    app.router.add_get("/metrics", metrics_handler)


def _check_communicator_results(results, action):
    """Return a web.Response error if results indicate failure, else None."""
    if not results:
        return web.Response(status=500, text="No response from scheduler\n")
    failures = [r for r in results if not r.success]
    if failures:
        msgs = " | ".join(r.message for r in failures)
        return web.Response(status=500, text=f"{action} failed: {msgs}\n")
    return None


def _add_admin_routes(app, request_manager):
    """Add admin endpoints to the aiohttp app.

    Endpoints: /start_profile, /stop_profile.
    Business logic (request construction, env var handling, response interpretation)
    lives here; request_manager only provides the transport to the scheduler.
    """

    async def start_profile_handler(request):
        try:
            if request.content_length and request.content_length > 0:
                try:
                    body = await request.json()
                except json.JSONDecodeError as e:
                    return web.Response(
                        status=400,
                        text=f"Invalid JSON in request body: {e}",
                    )
            else:
                body = {}

            # Build ProfileReq with env var overrides (same as tokenizer_communicator_mixin)
            with_stack = body.get("with_stack")
            env_with_stack = get_bool_env_var("SGLANG_PROFILE_WITH_STACK", "true")
            with_stack = (with_stack is not False) and env_with_stack
            record_shapes = body.get("record_shapes")
            env_record_shapes = get_bool_env_var("SGLANG_PROFILE_RECORD_SHAPES", "true")
            record_shapes = (record_shapes is not False) and env_record_shapes

            req = ProfileReq(
                req_type=ProfileReqType.START_PROFILE,
                output_dir=body.get("output_dir"),
                start_step=body.get("start_step"),
                num_steps=body.get("num_steps"),
                activities=body.get("activities"),
                with_stack=with_stack,
                record_shapes=record_shapes,
                profile_by_stage=body.get("profile_by_stage", False),
                profile_id=str(time.time()),
                merge_profiles=body.get("merge_profiles", False),
                profile_prefix=body.get("profile_prefix"),
                profile_stages=body.get("profile_stages"),
            )
            results = await request_manager.send_communicator_req(
                req, "profile_communicator", timeout=600.0
            )
            err = _check_communicator_results(results, "Start Profile")
            if err:
                return err
            return web.Response(text="Start profiling.\n")
        except Exception as e:
            logger.exception("Failed to start profile")
            return web.Response(
                status=500,
                text=f"Internal error: {type(e).__name__}. Check server logs.\n",
            )

    async def stop_profile_handler(request):
        try:
            req = ProfileReq(req_type=ProfileReqType.STOP_PROFILE)
            results = await request_manager.send_communicator_req(
                req, "profile_communicator", timeout=600.0
            )
            err = _check_communicator_results(results, "Stop profile")
            if err:
                return err
            return web.Response(text="Stop profiling. This will take some time.\n")
        except Exception as e:
            logger.exception("Failed to stop profile")
            return web.Response(
                status=500,
                text=f"Internal error: {type(e).__name__}. Check server logs.\n",
            )

    app.router.add_post("/start_profile", start_profile_handler)
    app.router.add_post("/stop_profile", stop_profile_handler)


async def serve_grpc(server_args, model_info=None):
    """Start the standalone gRPC server with integrated scheduler."""
    try:
        from smg_grpc_servicer.sglang.server import serve_grpc as _serve_grpc
    except ImportError as e:
        raise ImportError(
            "gRPC mode requires the smg-grpc-servicer package. "
            "If not installed, run: pip install smg-grpc-servicer[sglang]. "
            "If already installed, there may be a broken import due to a "
            "version mismatch — see the chained exception above for details."
        ) from e

    set_global_server_args_for_tokenizer(server_args)
    _patch_grpc_message_size_options()
    _patch_grpc_multimodal_hashes()
    _patch_grpc_request_manager_shm_transport()

    sidecar_app = web.Application()
    sidecar_runner = None
    sidecar_port = (
        server_args.grpc_http_sidecar_port
        if server_args.grpc_http_sidecar_port is not None
        else server_args.port + 1
    )

    # Metrics setup: must set PROMETHEUS_MULTIPROC_DIR before scheduler
    # processes import prometheus_client, since the env var is inherited
    # at fork time.
    if server_args.enable_metrics:
        try:
            from sglang.srt.observability.func_timer import enable_func_timer
            from sglang.srt.utils import set_prometheus_multiproc_dir

            set_prometheus_multiproc_dir()
            enable_func_timer()
            _add_metrics_routes(sidecar_app)
        except Exception as e:
            logger.error(
                "Failed to set up metrics: %s. Continuing without metrics.",
                e,
                exc_info=True,
            )

    async def _on_request_manager_ready(request_manager, srv_args, sched_info):
        nonlocal sidecar_runner
        try:
            _add_admin_routes(sidecar_app, request_manager)
        except Exception as e:
            logger.error(
                "Failed to set up admin routes: %s. "
                "Continuing without admin endpoints.",
                e,
                exc_info=True,
            )
        try:
            sidecar_runner = await _start_sidecar_server(
                server_args.host, sidecar_port, sidecar_app
            )
        except OSError as e:
            logger.error(
                "Failed to start HTTP sidecar server: %s. "
                "Continuing without metrics/profile endpoints.",
                e,
                exc_info=True,
            )
        except Exception as e:
            logger.error(
                "Unexpected error starting HTTP sidecar server: %s. "
                "Continuing without metrics/profile endpoints.",
                e,
                exc_info=True,
            )

    # Older smg-grpc-servicer releases (≤ 0.5.2) accept only (server_args,
    # model_info) and reject the on_request_manager_ready hook. The hook is
    # what calls _start_sidecar_server, so dropping the kwarg disables the
    # entire HTTP sidecar (Prometheus /metrics and /start_profile +
    # /stop_profile). Core gRPC serving still works without it.
    serve_kwargs: dict = {}
    sidecar_supported = (
        "on_request_manager_ready" in inspect.signature(_serve_grpc).parameters
    )
    if sidecar_supported:
        serve_kwargs["on_request_manager_ready"] = _on_request_manager_ready
    elif server_args.enable_metrics:
        # User explicitly asked for metrics but the installed servicer can't
        # start the sidecar that serves them — fail loud rather than silently
        # produce a server with no /metrics endpoint.
        raise RuntimeError(
            "--enable-metrics requires smg-grpc-servicer ≥ 0.5.3 (the version "
            "that accepts 'on_request_manager_ready'); installed version "
            "lacks the hook so the HTTP sidecar would never start. Upgrade "
            "smg-grpc-servicer or remove --enable-metrics."
        )
    else:
        logger.warning(
            "Installed smg-grpc-servicer does not accept "
            "'on_request_manager_ready'; HTTP sidecar disabled "
            "(no /metrics, /start_profile, /stop_profile). "
            "Upgrade smg-grpc-servicer to ≥ 0.5.3 to enable it."
        )

    try:
        await _serve_grpc(server_args, model_info, **serve_kwargs)
    finally:
        if sidecar_runner is not None:
            try:
                await sidecar_runner.cleanup()
            except Exception as e:
                logger.exception(
                    "Failed to cleanly shut down HTTP sidecar server: %s",
                    e,
                )
