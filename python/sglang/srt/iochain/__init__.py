"""
SGLang IOChain — request/response filter pipeline.

Quick start
-----------
1. Use the pre-wired serving classes that include IOChain support:

    from sglang.srt.iochain import ChatWithIOChain, get_token_counter

    # In http_server lifespan, replace OpenAIServingChat with ChatWithIOChain:
    app.state.openai_serving_chat = ChatWithIOChain(tokenizer_manager, template_manager)

    # Read token stats at any time:
    print(get_token_counter().get_stats())

2. Add your own filter to the default chain:

    from sglang.srt.iochain import get_default_chain, IOFilter, IOContext

    class MyFilter(IOFilter):
        blocking = True   # or False for fire-and-forget

        async def on_request(self, ctx: IOContext) -> None: ...
        async def on_response(self, ctx: IOContext) -> None: ...

    get_default_chain().add(MyFilter())

3. Or build a custom chain from scratch:

    from sglang.srt.iochain import IOChain, IOChainMixin
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

    class MyChatHandler(IOChainMixin, OpenAIServingChat):
        pass

    handler = MyChatHandler(tokenizer_manager, template_manager)
    handler.set_iochain(my_chain)
"""

from sglang.srt.iochain.base import IOChain, IOContext, IOFilter
from sglang.srt.iochain.filters.token_counter import TokenCounterFilter
from sglang.srt.iochain.mixin import IOChainMixin

# ---------------------------------------------------------------------------
# Default chain — shared across all pre-built serving class variants below.
# ---------------------------------------------------------------------------
_default_chain: IOChain = IOChain()
_token_counter: TokenCounterFilter = TokenCounterFilter()
_default_chain.add(_token_counter)


def get_default_chain() -> IOChain:
    """Return the shared default IOChain."""
    return _default_chain


def get_token_counter() -> TokenCounterFilter:
    """Return the built-in token counter attached to the default chain."""
    return _token_counter


# ---------------------------------------------------------------------------
# Pre-built serving class variants — drop-in replacements that include IOChain.
# Import lazily to avoid circular imports; only the classes you use are loaded.
# ---------------------------------------------------------------------------

def _make_iochain_variant(base_cls):  # type: ignore[return]
    """Dynamically create an IOChain-enabled subclass of base_cls."""
    cls = type(
        f"{base_cls.__name__}WithIOChain",
        (IOChainMixin, base_cls),
        {"_iochain": _default_chain},
    )
    return cls


def _get_chat_cls():
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
    return _make_iochain_variant(OpenAIServingChat)


def _get_completion_cls():
    from sglang.srt.entrypoints.openai.serving_completions import OpenAIServingCompletion
    return _make_iochain_variant(OpenAIServingCompletion)


__all__ = [
    "IOChain",
    "IOContext",
    "IOFilter",
    "IOChainMixin",
    "TokenCounterFilter",
    "get_default_chain",
    "get_token_counter",
]
