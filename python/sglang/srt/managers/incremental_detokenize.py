"""The detokenizer's incremental decoding: the per-request state and step that
turn each row's decode ids into text."""

from __future__ import annotations

from collections import OrderedDict
from typing import Callable, Dict, List, Optional, Tuple

import msgspec

from sglang.srt.environ import envs
from sglang.utils import find_printable_text

DETOKENIZER_MAX_STATES = envs.SGLANG_DETOKENIZER_MAX_STATES.get()


def _clamp_decode_ids(ids: List[int], vocab_size: Optional[int]) -> List[int]:
    """Map out-of-range token ids to 0 so the tokenizer can decode them.

    Multimodal placeholder ids (e.g. Inkling's negative -101/-102, or radix-cache
    pad-value hashes) are not real vocab tokens; tiktoken-style backends raise
    OverflowError on negative / out-of-range ids. These only appear in the
    surrogate-context prefix (before read_offset) and carry no text, and the clamp
    is applied identically to surr_ids and read_ids, so the incremental
    (read-minus-surr) output text is unchanged.
    """
    hi = vocab_size if vocab_size else None
    return [t if (0 <= t and (hi is None or t < hi)) else 0 for t in ids]


class DetokenizeState(msgspec.Struct):
    """Store the status of incremental decoding."""

    # ----- | --------- read_ids -------|
    # ----- |   surr_ids  |
    # xxxxx | xxxxxxxxxxx | xxxxxxxxxxx |
    # ----- ^ ----------- ^ ----------- ^
    # ----- 1 ----------- 2 ----------- 3
    # 1: surr_offset
    # 2: read_offset
    # 3: last token
    decode_ids: List[int]
    surr_offset: int
    read_offset: int
    # Offset that's sent to tokenizer for incremental update.
    sent_offset: int = 0
    decoded_text: str = ""
    decoded_text_len: int = 0
    decoded_text_chunks: List[str] = []

    def decode_inputs(
        self,
        *,
        finished_reason: Optional[Dict],
        no_stop_trim: bool,
        trim_matched_stop: Callable,
    ) -> Tuple[List[int], List[int]]:
        """The (read_ids, surr_ids) this row decodes."""
        read_ids = trim_matched_stop(
            self.decode_ids[self.surr_offset :],
            finished_reason,
            no_stop_trim,
        )
        surr_ids = self.decode_ids[self.surr_offset : self.read_offset]
        return read_ids, surr_ids

    def step(
        self,
        *,
        read_text: str,
        surr_text: str,
        finished_reason: Optional[Dict],
        no_stop_trim: bool,
        trim_matched_stop: Callable,
    ) -> str:
        """Advance by one row's decoded texts; return the text to send."""
        new_text = read_text[len(surr_text) :]
        if finished_reason is None:
            # Streaming. Invariant: sent_offset >= decoded_text_len. The
            # gap (`pending`) is "printable but uncommitted" text emitted
            # in a prior "�" recovery step; we skip it from this step's
            # emission so we don't double-send.
            pending = self.sent_offset - self.decoded_text_len
            if new_text and not new_text.endswith("�"):
                # Clean text: commit to decoded_text and advance offsets.
                self.append_decoded_text(new_text)
                self.surr_offset = self.read_offset
                self.read_offset = len(self.decode_ids)
                self.sent_offset = self.decoded_text_len
                return new_text[pending:] if pending else new_text
            else:
                # Incomplete UTF-8: emit the printable prefix only; do not
                # commit (token offsets stay so the next iteration retries
                # with more tokens).
                printable = find_printable_text(new_text)
                self.sent_offset = self.decoded_text_len + len(printable)
                return printable[pending:] if pending else printable

        # Finished: materialize once, trim the matched stop, emit the tail.
        output_str = trim_matched_stop(
            self.get_decoded_text() + new_text,
            finished_reason,
            no_stop_trim,
        )
        incremental_output = output_str[self.sent_offset :]
        self.sent_offset = len(output_str)
        return incremental_output

    def append_decoded_text(self, text: str):
        if text:
            self.decoded_text_chunks.append(text)
            self.decoded_text_len += len(text)

    def get_decoded_text(self) -> str:
        if self.decoded_text_chunks:
            self.decoded_text += "".join(self.decoded_text_chunks)
            self.decoded_text_chunks.clear()
        return self.decoded_text


class LimitedCapacityDict(OrderedDict):
    def __init__(self, capacity: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.capacity = capacity

    def __setitem__(self, key, value):
        if len(self) >= self.capacity:
            # Remove the oldest element (first item in the dict)
            self.popitem(last=False)
        # Set the new item
        super().__setitem__(key, value)


class DetokenizeStateTable:
    """The detokenizer's DetokenizeState by rid: started by a request's first row,
    released by its finished row; past capacity, the oldest is evicted."""

    def __init__(self, capacity: int):
        self._states = LimitedCapacityDict(capacity=capacity)

    def start_or_extend(
        self,
        *,
        rid: str,
        decode_ids: List[int],
        read_offset: int,
        vocab_size: Optional[int],
    ) -> DetokenizeState:
        if rid not in self._states:
            s = DetokenizeState(
                decode_ids=_clamp_decode_ids(decode_ids, vocab_size),
                surr_offset=0,
                read_offset=read_offset,
            )
            self._states[rid] = s
        else:
            s = self._states[rid]
            s.decode_ids.extend(_clamp_decode_ids(decode_ids, vocab_size))
        return s

    def get(self, rid: str) -> Optional[DetokenizeState]:
        return self._states.get(rid)

    def release(self, rid: str) -> None:
        self._states.pop(rid, None)
