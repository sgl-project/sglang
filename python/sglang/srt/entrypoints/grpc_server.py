"""
Thin gRPC server wrapper — delegates to smg-grpc-servicer package.

A lightweight HTTP sidecar is started alongside the gRPC server to expose:
- /metrics (Prometheus, when --enable-metrics is set)
- /start_profile, /stop_profile (profiling control)

The sidecar is started on --grpc-http-sidecar-port (default: --port + 1)
once the gRPC request manager is ready, regardless of whether --enable-metrics
is set.
"""

import inspect
import json
import logging
import time

from aiohttp import web

from sglang.srt.managers.io_struct import ProfileReq, ProfileReqType
from sglang.srt.utils.common import get_bool_env_var

logger = logging.getLogger(__name__)


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
                type=ProfileReqType.START_PROFILE,
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
            req = ProfileReq(type=ProfileReqType.STOP_PROFILE)
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
