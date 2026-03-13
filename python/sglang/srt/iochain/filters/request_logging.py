"""
RequestLoggingFilter — reference IOFilter implementation.

.. note::
   **Sample / placeholder only.**  This filter is a simple, benign example
   whose sole purpose is to demonstrate the value of the IOFilter framework
   (hook points, blocking vs. non-blocking dispatch, context propagation).
   It is *not* intended as production-ready observability code and should be
   treated as a starting point — replace or extend it with whatever fits your
   deployment (structured logging, OpenTelemetry spans, Prometheus histograms,
   etc.).

Logs the model name on ingress and end-to-end latency on egress for every
non-streaming request.

Non-blocking: logging is scheduled as a background task and never delays
the response to the caller.
"""

from __future__ import annotations

import logging
import time

from sglang.srt.iochain.base import IOContext, IOFilter

logger = logging.getLogger(__name__)


class RequestLoggingFilter(IOFilter):
    """
    Logs model name on request start and latency on request completion.

    blocking = False: fired as asyncio.create_task so it does not add
    latency to the inference path.

    Example output::

        DEBUG sglang.srt.iochain.filters.request_logging - request.start  model=meta-llama/Meta-Llama-3.1-8B-Instruct  request_id=a3f1...
        INFO  sglang.srt.iochain.filters.request_logging - request.complete model=meta-llama/Meta-Llama-3.1-8B-Instruct  request_id=a3f1...  latency_ms=142.3
    """

    blocking = False

    async def on_request(self, ctx: IOContext) -> None:
        model = getattr(ctx.raw_request, "model", "unknown")
        logger.debug(
            "request.start",
            extra={"request_id": ctx.request_id, "model": model},
        )

    async def on_response(self, ctx: IOContext) -> None:
        latency_ms = (time.monotonic() - ctx.start_time) * 1000
        model = getattr(ctx.raw_request, "model", "unknown")
        logger.info(
            "request.complete",
            extra={
                "request_id": ctx.request_id,
                "model": model,
                "latency_ms": round(latency_ms, 1),
            },
        )
