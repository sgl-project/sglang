"""
SGLang IOChain — request/response filter pipeline.

Quickstart
----------
1. Define a filter::

    from sglang.srt.iochain import IOFilter, IOContext

    class MyFilter(IOFilter):
        blocking = False  # True to block the request; False to fire-and-forget

        async def on_request(self, ctx: IOContext) -> None:
            ...  # called after tokenisation, before inference

        async def on_response(self, ctx: IOContext) -> None:
            ...  # called after a non-streaming response is built

2. Register it with the default chain and mix it into a serving handler::

    from sglang.srt.iochain import get_default_chain, IOChainMixin
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

    get_default_chain().add(MyFilter())

    class ChatWithFilters(IOChainMixin, OpenAIServingChat):
        _iochain = get_default_chain()

3. Use ``ChatWithFilters`` in place of ``OpenAIServingChat`` when
   constructing the server (e.g. in ``http_server.py``'s lifespan).

Refer to ``sglang/srt/iochain/filters/request_logging.py`` for a complete
reference implementation.
"""

from sglang.srt.iochain.base import IOChain, IOContext, IOFilter
from sglang.srt.iochain.filters.request_logging import RequestLoggingFilter
from sglang.srt.iochain.mixin import IOChainMixin

# Shared default chain — empty until the application adds filters.
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
