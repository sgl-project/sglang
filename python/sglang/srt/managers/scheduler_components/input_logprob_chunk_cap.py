"""SGLANG_INPUT_LOGPROB_MAX_CHUNK_TOKENS: the input-logprob rows one prefill batch admits."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Callable, Optional, TypeVar

from sglang.srt.environ import envs
from sglang.srt.managers.schedule_policy import AddReqResult, PrefillAdder

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

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

    @classmethod
    def check_fits_alignment(
        cls,
        *,
        chunked_prefill_size: Optional[int],
        page_size: int,
        truncation_align_size: Optional[int],
        exact_chunk_fill: bool,
    ) -> None:
        """A first chunk is cut down to the truncation alignment, and to whole pages
        unless admission fills chunks exactly (`uses_exact_chunk_fill`); a prefix
        hit's encoder replay rows also take chunk room ahead of its extend. A cap
        below the replay rows plus that step cuts an input-logprob prompt longer
        than the cap to zero tokens, and the request would wait in the queue
        forever."""
        cap = cls.for_batch(chunked_prefill_size).room
        align = truncation_align_size or 1
        if not exact_chunk_fill:
            align = math.lcm(page_size, align)
        replay = PrefillAdder.max_replay_len()
        if cap is None or cap >= replay + align:
            return
        page_note = "" if exact_chunk_fill else f"page size {page_size}, "
        step = (
            f"a {align}-token chunk step ({page_note}truncation alignment "
            f"{truncation_align_size})"
        )
        parts = f"{replay} encoder replay tokens plus {step}" if replay else step
        raise ValueError(
            f"SGLANG_INPUT_LOGPROB_MAX_CHUNK_TOKENS={cap} is below the smallest first "
            f"chunk of {replay + align} tokens: {parts}. A request returning input "
            f"logprobs longer than {cap} tokens could never be scheduled. Set it to "
            f"at least {replay + align}, or to 0 to disable the cap."
        )

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
