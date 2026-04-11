"""
Thin gRPC server wrapper — delegates to smg-grpc-servicer package.

When --enable-metrics is set, a lightweight HTTP server is started on
--metrics-http-port (default: --port + 1) to expose Prometheus /metrics.
"""

import logging

logger = logging.getLogger(__name__)


async def _start_metrics_server(host: str, port: int):
    """Start an HTTP server exposing Prometheus /metrics.

    The caller is responsible for calling ``runner.cleanup()`` on the returned
    AppRunner when shutting down.  The server begins accepting requests before
    this function returns.
    """
    from aiohttp import web
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
            # Create a fresh registry and attach a MultiProcessCollector
            # on each request.  This is the recommended pattern from the
            # prometheus_client multiprocess docs to ensure up-to-date
            # data from PROMETHEUS_MULTIPROC_DIR.
            #
            # Use OpenMetrics format to match what the HTTP-mode endpoint
            # returns when Prometheus scrapes it with an OpenMetrics Accept
            # header (make_asgi_app performs content negotiation).
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

    app = web.Application()
    app.router.add_get("/metrics", metrics_handler)

    runner = web.AppRunner(app)
    await runner.setup()
    try:
        site = web.TCPSite(runner, host, port)
        await site.start()
    except BaseException:
        await runner.cleanup()
        raise
    logger.info("Prometheus metrics server started on http://%s:%d/metrics", host, port)
    return runner


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

    metrics_runner = None
    if server_args.enable_metrics:
        try:
            from sglang.srt.observability.func_timer import enable_func_timer
            from sglang.srt.utils import set_prometheus_multiproc_dir

            # Must set PROMETHEUS_MULTIPROC_DIR env var before any
            # prometheus_client import.  The env var is inherited by child
            # processes (schedulers) that import prometheus_client later.
            set_prometheus_multiproc_dir()
            enable_func_timer()

            metrics_port = (
                server_args.metrics_http_port
                if server_args.metrics_http_port is not None
                else server_args.port + 1
            )
            metrics_runner = await _start_metrics_server(server_args.host, metrics_port)
        except OSError as e:
            logger.error(
                "Failed to start metrics server: %s. " "Continuing without metrics.",
                e,
                exc_info=True,
            )
        except Exception as e:
            logger.error(
                "Unexpected error starting metrics server: %s. "
                "Continuing without metrics.",
                e,
                exc_info=True,
            )

    try:
        await _serve_grpc(server_args, model_info)
    finally:
        if metrics_runner is not None:
            try:
                await metrics_runner.cleanup()
            except Exception as e:
                logger.exception(
                    "Failed to cleanly shut down Prometheus metrics server: %s",
                    e,
                )
