"""
RequestLoggingProcessor — reference IOProcessor implementation.

A non-blocking processor that emits two structured log lines per request:

* **Ingress** — model name and request ID, logged before inference.
* **Egress**  — request ID, end-to-end latency in ms, and whether the
  response was streamed, logged after the response is fully delivered.

This processor is intentionally minimal — it serves as a copy-paste starting
point for real observability integrations such as structured JSON logging,
OpenTelemetry spans, or Prometheus histograms.

Because ``blocking = False``, it is scheduled as a background task and never
adds latency to the critical path.  Errors are logged and swallowed.

TODO: move to an ``examples/`` directory once that exists upstream.

Example log output
------------------
::

    INFO IOChain ingress  request_id=a3f1... model=meta-llama/Llama-3-8B
    INFO IOChain egress   request_id=a3f1... latency_ms=312.4 streaming=False
"""

from __future__ import annotations

import logging
import time

from sglang.srt.iochain.base import IOContext, IOProcessor

logger = logging.getLogger(__name__)


class RequestLoggingProcessor(IOProcessor):
    """Log model name on ingress and end-to-end latency on egress."""

    blocking = False  # never delays inference

    async def on_request(self, ctx: IOContext) -> None:
        model = getattr(ctx.raw_request, "model", "unknown")
        logger.info("IOChain ingress  request_id=%s model=%s", ctx.request_id, model)

    async def on_response(self, ctx: IOContext) -> None:
        elapsed_ms = (time.monotonic() - ctx.start_time) * 1000
        streaming = ctx.response is None
        logger.info(
            "IOChain egress   request_id=%s latency_ms=%.1f streaming=%s",
            ctx.request_id,
            elapsed_ms,
            streaming,
        )
