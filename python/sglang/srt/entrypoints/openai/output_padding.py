"""Constant-length output padding for ``/v1/completions`` streams.

A classifier-style deployment emits tiny structured completions whose token
count correlates with the answer, so an observer who can only count tokens or
SSE frames on an otherwise encrypted stream recovers it without decrypting
anything. Padding every completion out to one common length closes that channel.

Equalized: the emitted token count, the content-frame count, the reported
``completion_tokens``, and the terminal frame. Not equalized: the byte width of
an individual frame -- token ids and logprobs are decimal text, so a pad frame
is a different size from a real one -- and inter-frame timing.

The pad token is the model's own terminal token, which keeps the feature to a
single knob: no second flag has to agree with the tokenizer on a pad id.
"""

from __future__ import annotations

from typing import List, Optional

import msgspec

from sglang.srt.entrypoints.openai.protocol import LogProbs

# A pad position's logprob must be finite. These frames serialize through
# pydantic's model_dump_json, whose default ser_json_inf_nan renders a non-finite
# float as `null` -- a shape no real position ever has, so a pad frame would be
# identifiable by that alone.
_PAD_LOGPROB = 0.0

# This path never resolves output text offsets; real positions carry -1 too.
_NO_TEXT_OFFSET = -1


class OutputPaddingPlan(msgspec.Struct, frozen=True):
    """Immutable padding plan for one request.

    Built once from the server flag plus the request's own logprob settings, so
    the streaming loop reads a single object instead of threading each knob
    through separately and branching on whether padding is on.
    """

    target_tokens: int
    pad_token_id: int
    pad_token_text: str
    return_logprob: bool
    top_logprobs_width: int

    @classmethod
    def build(
        cls,
        *,
        target_tokens: Optional[int],
        pad_token_id: int,
        pad_token_text: str,
        return_logprob: bool,
        top_logprobs_width: int,
    ) -> Optional[OutputPaddingPlan]:
        """The plan, or None when ``--padded-output-tokens`` is unset."""
        if target_tokens is None:
            return None
        return cls(
            target_tokens=target_tokens,
            pad_token_id=pad_token_id,
            pad_token_text=pad_token_text,
            return_logprob=return_logprob,
            top_logprobs_width=top_logprobs_width,
        )

    def pad_count(self, wire_tokens: int) -> int:
        return max(0, self.target_tokens - wire_tokens)

    def filler_logprobs(self) -> LogProbs:
        """One synthesized position, shaped like a real one."""
        return LogProbs(
            text_offset=[_NO_TEXT_OFFSET],
            token_logprobs=[_PAD_LOGPROB],
            tokens=[self.pad_token_text],
            top_logprobs=[self._filler_top_row()],
        )

    def _filler_top_row(self) -> Optional[dict]:
        # None omits the key entirely, which is what a real position carries
        # when the request asked for no top-k.
        if self.top_logprobs_width <= 0:
            return None
        # A real row holds top_logprobs_width entries, but LogProbs.top_logprobs
        # is keyed by token *text*, so repeating the one token we know collapses
        # to a single entry. Widening it would mean inventing token strings.
        return {self.pad_token_text: _PAD_LOGPROB}

    def per_position_logprobs(
        self, source: Optional[LogProbs], count: int
    ) -> List[LogProbs]:
        """Split ``source`` into ``count`` single-position payloads.

        Positions the manager did not supply fall back to the filler, per index,
        so a backend that under-supplies logprobs cannot make the frame shape
        track the real completion length.
        """
        available = len(source.token_logprobs) if source is not None else 0
        top_available = len(source.top_logprobs) if source is not None else 0
        out = []
        for i in range(count):
            if i >= available:
                out.append(self.filler_logprobs())
                continue
            top_row = (
                source.top_logprobs[i] if i < top_available else self._filler_top_row()
            )
            out.append(
                LogProbs(
                    text_offset=[
                        source.text_offset[i]
                        if i < len(source.text_offset)
                        else _NO_TEXT_OFFSET
                    ],
                    token_logprobs=[source.token_logprobs[i]],
                    tokens=[source.tokens[i] if i < len(source.tokens) else ""],
                    top_logprobs=[top_row],
                )
            )
        return out
