"""SGLANG_INPUT_LOGPROB_MAX_CHUNK_TOKENS: the input-logprob rows one prefill batch admits."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional, TypeVar

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_policy import AddReqResult

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.managers.schedule_policy import PrefillAdder

_T = TypeVar("_T")


def _reads_input_logprobs(req: Req) -> bool:
    return req.return_logprob and 0 <= req.logprob_start_len < len(req.origin_input_ids)


class InputLogprobChunkCap:
    """Admits prefills so that the requests returning input logprobs take at most the
    cap in chunk tokens per batch; other requests keep the batch's chunk budget.

    Input logprobs materialize fp32 full-vocab logits for each of those rows, outside
    the prefill graph.
    """

    def __init__(self, room: Optional[int]):
        # chunk tokens left for input-logprob requests in this batch; None: no cap
        self.room = room

    @classmethod
    def for_batch(cls, chunked_prefill_size: Optional[int]) -> InputLogprobChunkCap:
        cap = envs.SGLANG_INPUT_LOGPROB_MAX_CHUNK_TOKENS.get()
        capped = (
            cap > 0 and chunked_prefill_size is not None and chunked_prefill_size > cap
        )
        return cls(cap if capped else None)

    def add_chunked_req(self, adder: PrefillAdder, req: Req) -> Optional[Req]:
        return self._admit(adder, req, lambda: adder.add_chunked_req(req))

    def add_one_req(self, adder: PrefillAdder, req: Req, **kwargs) -> AddReqResult:
        if self._capped(adder, req) and self.room <= 0:
            return AddReqResult.OTHER
        return self._admit(adder, req, lambda: adder.add_one_req(req, **kwargs))

    def _capped(self, adder: PrefillAdder, req: Req) -> bool:
        return (
            self.room is not None
            and adder.rem_chunk_tokens is not None
            and _reads_input_logprobs(req)
        )

    def _admit(self, adder: PrefillAdder, req: Req, add: Callable[[], _T]) -> _T:
        if not self._capped(adder, req):
            return add()
        rem = adder.rem_chunk_tokens
        limit = min(rem, self.room)
        adder.rem_chunk_tokens = limit
        try:
            return add()
        finally:
            taken = limit - adder.rem_chunk_tokens
            self.room -= taken
            adder.rem_chunk_tokens = rem - taken
