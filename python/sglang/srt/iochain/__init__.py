"""
SGLang IOChain — pluggable request/response processor pipeline.

IOChain lets operators inject custom processing stages into SGLang's
OpenAI-compatible HTTP server *without* modifying core code.  It mirrors
the ``IOProcessor`` abstraction used by vLLM, so processors written for one
runtime can be adapted to the other with minimal effort.

Quick start — write a processor
--------------------------------
::

    from sglang.srt.iochain import IOProcessor, IOContext

    class MyProcessor(IOProcessor):
        # blocking = True  → awaited inline; exceptions abort the request
        # blocking = False → fire-and-forget; exceptions are logged only
        blocking = False

        async def on_request(self, ctx: IOContext) -> None:
            # Runs after tokenisation, before inference
            ...

        async def on_response(self, ctx: IOContext) -> None:
            # Runs after the response is fully delivered
            # ctx.response is None for streaming requests
            ...

Register via entry point (pyproject.toml)
------------------------------------------
Installed packages are discovered automatically at server startup — no CLI
flags required::

    [project.entry-points."sglang.io_processor_plugins"]
    my_processor = "mypackage.processors:MyProcessor"

Register via CLI flag
----------------------
Good for development or one-off deployments::

    python -m sglang.launch_server --model-path ... \\
        --io-processor mypackage.processors:MyProcessor

Multiple ``--io-processor`` flags are supported; processors are added to the
chain in the order they appear on the command line (after any entry-point
processors).

Reference implementation
-------------------------
See ``sglang/srt/iochain/processors/request_logging_processor.py`` for a
minimal non-blocking processor that logs model name on ingress and
end-to-end latency on egress.
"""

from sglang.srt.iochain.base import IOChain, IOContext, IOProcessor
from sglang.srt.iochain.processors.request_logging_processor import (
    RequestLoggingProcessor,
)

__all__ = ["IOChain", "IOContext", "IOProcessor", "RequestLoggingProcessor"]
