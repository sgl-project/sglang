"""The detokenizer's incremental decoding: the DetokenizeHeader the scheduler
builds and sends on a request's first output, and the per-request state, kept
by rid, that turns each row's output ids into text."""

from __future__ import annotations

import logging
from array import array
from collections import OrderedDict
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Tuple

import msgspec

from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req

logger = logging.getLogger(__name__)


PROMPT_TAIL_LEN = 5

DETOKENIZER_MAX_STATES = envs.SGLANG_DETOKENIZER_MAX_STATES.get()


def _is_chinese_char(cp: int):
    """Checks whether CP is the codepoint of a CJK character."""
    # This defines a "chinese character" as anything in the CJK Unicode block:
    #   https://en.wikipedia.org/wiki/CJK_Unified_Ideographs_(Unicode_block)
    #
    # Note that the CJK Unicode block is NOT all Japanese and Korean characters,
    # despite its name. The modern Korean Hangul alphabet is a different block,
    # as is Japanese Hiragana and Katakana. Those alphabets are used to write
    # space-separated words, so they are not treated specially and handled
    # like the all of the other languages.
    if (
        (cp >= 0x4E00 and cp <= 0x9FFF)
        or (cp >= 0x3400 and cp <= 0x4DBF)  #
        or (cp >= 0x20000 and cp <= 0x2A6DF)  #
        or (cp >= 0x2A700 and cp <= 0x2B73F)  #
        or (cp >= 0x2B740 and cp <= 0x2B81F)  #
        or (cp >= 0x2B820 and cp <= 0x2CEAF)  #
        or (cp >= 0xF900 and cp <= 0xFAFF)
        or (cp >= 0x2F800 and cp <= 0x2FA1F)  #
    ):  #
        return True

    return False


def find_printable_text(text: str):
    """Returns the longest printable substring of text that contains only entire words."""
    # Borrowed from https://github.com/huggingface/transformers/blob/061580c82c2db1de9139528243e105953793f7a2/src/transformers/generation/streamers.py#L99

    # After the symbol for a new line, we flush the cache.
    if text.endswith("\n"):
        return text
    # If the last token is a CJK character, we print the characters.
    elif len(text) > 0 and _is_chinese_char(ord(text[-1])):
        return text
    # Otherwise if the penultimate token is a CJK character, we print the characters except for the last one.
    elif len(text) > 1 and _is_chinese_char(ord(text[-2])):
        return text[:-1]
    # Otherwise, prints until the last space char (simple heuristic to avoid printing incomplete words,
    # which may change with the subsequent token -- there are probably smarter ways to do this!)
    else:
        return text[: text.rfind(" ") + 1]


class DetokenizeHeader(
    msgspec.Struct, frozen=True, kw_only=True, array_like=True, gc=False
):
    """A request's detokenization inputs. The scheduler sends it on the request's
    first output only; the detokenizer keeps it for the request's lifetime and
    decodes each row, the first included, from that row's output_ids."""

    # The prompt tail decoded ahead of the output as context; the output text
    # starts after it.
    prompt_tail_ids: array
    skip_special_tokens: bool
    spaces_between_special_tokens: bool
    no_stop_trim: bool
    http_worker_ipc: Optional[str]


def build_detokenize_header(
    req: Req, *, output_prefix_len: int = 0
) -> DetokenizeHeader:
    prompt_ids = req.origin_input_ids_unpadded
    surr_offset = max(len(prompt_ids) - PROMPT_TAIL_LEN, 0)
    prompt_tail_ids = prompt_ids[surr_offset:]
    if output_prefix_len:
        # tokens skipped before the first emission still provide decoding
        # context, but must not become part of the answer
        prompt_tail_ids = array("q", prompt_tail_ids)
        prompt_tail_ids.extend(req.output_ids[:output_prefix_len])
    sampling_params = req.sampling_params
    return DetokenizeHeader(
        prompt_tail_ids=prompt_tail_ids,
        skip_special_tokens=sampling_params.skip_special_tokens,
        spaces_between_special_tokens=sampling_params.spaces_between_special_tokens,
        no_stop_trim=sampling_params.no_stop_trim,
        http_worker_ipc=req.http_worker_ipc,
    )


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
    header: DetokenizeHeader
    # Output ids received so far; a non-beam row's completion_tokens is this
    # plus the row's output ids.
    num_received_ids: int = 0
    # Offset that's sent to tokenizer for incremental update.
    sent_offset: int = 0
    decoded_text: str = ""
    decoded_text_len: int = 0
    decoded_text_chunks: List[str] = []

    @classmethod
    def from_header(
        cls, *, header: DetokenizeHeader, vocab_size: Optional[int]
    ) -> DetokenizeState:
        return cls(
            decode_ids=_clamp_decode_ids(header.prompt_tail_ids, vocab_size),
            surr_offset=0,
            read_offset=len(header.prompt_tail_ids),
            header=header,
        )

    def append_row(
        self,
        *,
        rid: str,
        output_ids: List[int],
        completion_tokens: int,
        is_beam: bool,
        vocab_size: Optional[int],
    ) -> None:
        if not is_beam:
            self._check_continuity(
                rid=rid,
                num_received=self.num_received_ids,
                num_new=len(output_ids),
                completion_tokens=completion_tokens,
            )
            # Resync to the scheduler's count; the two differ only after an alarm.
            self.num_received_ids = completion_tokens
        else:
            # A beam row's completion_tokens counts its returned sequences.
            self.num_received_ids += len(output_ids)
        self.decode_ids.extend(_clamp_decode_ids(output_ids, vocab_size))

    @staticmethod
    def _check_continuity(
        *, rid: str, num_received: int, num_new: int, completion_tokens: int
    ) -> None:
        if num_received + num_new != completion_tokens:
            # Outside CI an alarm only: the row is still decoded and routed.
            msg = (
                f"Output ids of {rid=} do not continue the "
                f"{num_received} received: {num_new} new for "
                f"{completion_tokens} in all."
            )
            if envs.SGLANG_IS_IN_CI.get():
                raise RuntimeError(msg)
            logger.warning(msg)

    def decode_inputs(
        self, *, finished_reason: Optional[Dict], trim_matched_stop: Callable
    ) -> Tuple[List[int], List[int]]:
        """The (read_ids, surr_ids) this row decodes."""
        read_ids = trim_matched_stop(
            self.decode_ids[self.surr_offset :],
            finished_reason,
            self.header.no_stop_trim,
        )
        surr_ids = self.decode_ids[self.surr_offset : self.read_offset]
        return read_ids, surr_ids

    def step(
        self,
        *,
        read_text: str,
        surr_text: str,
        finished_reason: Optional[Dict],
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
            self.header.no_stop_trim,
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
    """The detokenizer's DetokenizeState by rid: started by a request's
    DetokenizeHeader, released by its finished row or a DetokenizeReleaseReq;
    past capacity, the oldest is evicted."""

    def __init__(self, capacity: int):
        self._states = LimitedCapacityDict(capacity=capacity)

    def start(
        self, *, rid: str, header: DetokenizeHeader, vocab_size: Optional[int]
    ) -> DetokenizeState:
        # A request's first output; it replaces any stale state of a reused rid.
        self._states.pop(rid, None)
        s = DetokenizeState.from_header(header=header, vocab_size=vocab_size)
        self._states[rid] = s
        return s

    def get(self, rid: str) -> Optional[DetokenizeState]:
        return self._states.get(rid)

    def release(self, rid: str) -> None:
        self._states.pop(rid, None)
