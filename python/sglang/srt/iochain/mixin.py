"""
IOChainMixin — wires an IOChain into OpenAIServingBase hook methods.

Usage
-----
Mix into any OpenAIServingBase subclass (or use the pre-built variants in
`sglang.srt.iochain`):

    from sglang.srt.iochain import IOChainMixin, get_default_chain
    from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat

    class ChatWithIOChain(IOChainMixin, OpenAIServingChat):
        pass

Or wire a custom chain:

    class ChatWithCustomChain(IOChainMixin, OpenAIServingChat):
        _iochain = my_chain

The mixin uses contextvars.ContextVar to carry the IOContext between the
ingress and egress hooks across concurrent async requests safely — no locks,
no per-instance state per request.
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import Any, Optional

from sglang.srt.iochain.base import IOChain, IOContext

# Per-async-task context variable — each concurrent request gets its own slot.
_ctx_var: ContextVar[Optional[IOContext]] = ContextVar("iochain_ctx", default=None)


class IOChainMixin:
    """
    Mixin that overrides the _before_inference / _after_inference hooks
    defined in OpenAIServingBase to drive an IOChain pipeline.

    Class attribute
    ---------------
    _iochain : IOChain | None
        Set on the class or instance to use a specific chain.
        Defaults to None; call set_iochain() or override _iochain after
        instantiation, or use get_default_chain() via the module-level helpers.
    """

    _iochain: Optional[IOChain] = None

    def set_iochain(self, chain: IOChain) -> None:
        self._iochain = chain

    async def _before_inference(
        self,
        request: Any,
        adapted_request: Any,
    ) -> None:
        if self._iochain is None:
            return
        ctx = self._iochain.make_context(request, adapted_request)
        _ctx_var.set(ctx)
        await self._iochain.run_ingress(ctx)

    async def _after_inference(
        self,
        request: Any,
        adapted_request: Any,
        response: Any,
    ) -> None:
        if self._iochain is None:
            return
        ctx = _ctx_var.get()
        if ctx is None:
            return
        ctx.response = response
        await self._iochain.run_egress(ctx)
