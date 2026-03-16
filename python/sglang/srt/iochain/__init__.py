"""
SGLang IOChain — request/response filter pipeline.

Quickstart
----------
1. Define a filter::

    from sglang.srt.iochain import IOFilter, IOContext

    class MyFilter(IOFilter):
        blocking = False  # True to block inline; False to fire-and-forget

        async def on_request(self, ctx: IOContext) -> None:
            ...  # called after tokenisation, before inference

        async def on_response(self, ctx: IOContext) -> None:
            ...  # called after the response is delivered (streaming or not)
            # ctx.response is None for streaming requests

2. Register it so the server picks it up automatically at startup::

    # Option A — Python entry point (recommended for packages / PyPI)
    # In your pyproject.toml:
    #
    #   [project.entry-points."sglang.general_plugins"]
    #   my_filter = "mypackage.filters:MyFilter"
    #
    # Install your package and SGLang will discover it on next start.

    # Option B — CLI flag (good for one-off or development use)
    #
    #   python -m sglang.launch_server \\
    #       --model-path meta-llama/... \\
    #       --iochain-filter mypackage.filters:MyFilter

Refer to ``sglang/srt/iochain/filters/request_logging.py`` for a complete
reference implementation.
"""

from sglang.srt.iochain.base import IOChain, IOContext, IOFilter
from sglang.srt.iochain.filters.request_logging import RequestLoggingFilter
from sglang.srt.iochain.mixin import IOChainMixin

# Shared default chain — empty until plugins are loaded.
# Nothing is activated implicitly; all filters are opt-in.
_default_chain: IOChain = IOChain()


def get_default_chain() -> IOChain:
    """Return the process-wide default IOChain."""
    return _default_chain


__all__ = [
    "IOChain",
    "IOContext",
    "IOFilter",
    "IOChainMixin",
    "RequestLoggingFilter",
    "get_default_chain",
]
