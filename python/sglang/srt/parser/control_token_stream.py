import logging
from bisect import bisect_left, bisect_right
from typing import Dict, Iterable, List, Literal, Optional, Tuple, TypeVar
from weakref import WeakKeyDictionary

import msgspec
from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# Replaces a marker's first character where no control token starts. Any
# character that begins no marker works; the view is only searched, never output.
_MASK = "\x00"

# Bounds the parser-text chunks a tail check walks; arbitrary.
_COMPACT_EVERY = 64

ParserMatchMode = Literal["auto", "strict", "text"]
_T = TypeVar("_T")

_control_ids_cache: WeakKeyDictionary = WeakKeyDictionary()
_special_ids_cache: WeakKeyDictionary = WeakKeyDictionary()
_logged_reasons: set = set()


class ParserMatchError(ValueError):
    """Markers cannot be matched by output token id where matching must be."""


class MarkerSpan(msgspec.Struct, frozen=True):
    """A marker at ``[start, end)`` of the queried text."""

    kind: str
    start: int
    end: int


def _log_once(message: str) -> None:
    if message not in _logged_reasons:
        _logged_reasons.add(message)
        logger.warning(message)


def _resolve_control_ids(
    backend: Tokenizer, markers: Tuple[str, ...]
) -> Tuple[Optional[Dict[int, str]], str]:
    """The control token id of each marker, or None and why there is none."""
    added = backend.get_added_tokens_decoder()
    control_ids = {}
    for marker in markers:
        token_id = backend.token_to_id(marker)
        if (
            token_id is None
            or backend.decode([token_id], skip_special_tokens=False) != marker
        ):
            return None, f"{marker!r} is not a single token of this tokenizer"
        if token_id in added and added[token_id].special:
            # Decoding skips it, so neither its position nor, under
            # skip_special_tokens, its text ever reaches the parser.
            return None, f"{marker!r} is a special token of this tokenizer"
        control_ids[token_id] = marker
    return control_ids, ""


def _control_ids(
    tokenizer: PreTrainedTokenizerFast, markers: Tuple[str, ...]
) -> Tuple[Optional[Dict[int, str]], str]:
    per_tokenizer = _control_ids_cache.setdefault(tokenizer, {})
    if markers not in per_tokenizer:
        per_tokenizer[markers] = _resolve_control_ids(
            tokenizer.backend_tokenizer, markers
        )
    return per_tokenizer[markers]


def _special_ids(tokenizer: PreTrainedTokenizerFast) -> Dict[int, str]:
    if tokenizer not in _special_ids_cache:
        added = tokenizer.backend_tokenizer.get_added_tokens_decoder()
        _special_ids_cache[tokenizer] = {
            token_id: t.content for token_id, t in added.items() if t.special
        }
    return _special_ids_cache[tokenizer]


def _find_all(text: str, char: str) -> Iterable[int]:
    i = text.find(char)
    while i != -1:
        yield i
        i = text.find(char, i + 1)


class _Alignment:
    """The parser text matched against the decoded text, for parser text that
    shows the text of special tokens or leaves it out."""

    def __init__(self, show_special: bool):
        self.show_special = show_special
        self.matched_len = 0
        # Decoded text the parser text has not reached yet, and parser text
        # not matched to it yet.
        self.decoded_ahead = ""
        self.consumed_ahead = ""
        # Specials before this index of the stream's specials are matched.
        self.next_special = 0
        # Special-token texts in the matched parser text: starts, ends, and
        # the running total of their lengths.
        self._skip_starts: List[int] = []
        self._skip_ends: List[int] = []
        self._skip_total: List[int] = []

    def frontier(self) -> int:
        """The decoded-text position the matched parser text ends at."""
        return self.matched_len - (self._skip_total[-1] if self._skip_total else 0)

    def decoded_pos(self, pos: int) -> Optional[int]:
        """Where parser-text position ``pos`` falls in the decoded text, or
        None inside a special token's text."""
        i = bisect_right(self._skip_starts, pos) - 1
        if i < 0:
            return pos
        if pos < self._skip_ends[i]:
            return None
        return pos - self._skip_total[i]

    def match(
        self, decoded: str, consumed: str, specials: List[Tuple[int, str]]
    ) -> bool:
        """Match the next decoded and parser text; False where they disagree."""
        decoded = self.decoded_ahead + decoded
        consumed = self.consumed_ahead + consumed
        while True:
            special_pos, special = self._next_special(specials)
            if special_pos == self.frontier():
                if not consumed.startswith(special):
                    if not special.startswith(consumed):
                        return False
                    break
                self._skip(special)
                consumed = consumed[len(special) :]
                continue
            n = min(len(decoded), len(consumed))
            if special_pos >= 0:
                n = min(n, special_pos - self.frontier())
            if decoded[:n] != consumed[:n]:
                return False
            self.matched_len += n
            decoded = decoded[n:]
            consumed = consumed[n:]
            if special_pos != self.frontier():
                break
        self.decoded_ahead = decoded
        self.consumed_ahead = consumed
        return True

    def _next_special(self, specials: List[Tuple[int, str]]) -> Tuple[int, str]:
        if self.show_special and self.next_special < len(specials):
            return specials[self.next_special]
        return -1, ""

    def _skip(self, special: str) -> None:
        start = self.matched_len
        self._skip_starts.append(start)
        self._skip_ends.append(start + len(special))
        self._skip_total.append(
            (self._skip_total[-1] if self._skip_total else 0) + len(special)
        )
        self.matched_len += len(special)
        self.next_special += 1


class ControlTokenStream:
    """A request's output ids, decoded next to the text its parser receives.

    A marker such as ``</think>`` decodes to the same text whether the model
    generated the control token or spelled the string with ordinary tokens.
    ``find()`` and ``match_at()`` return only markers a control token of that
    kind wrote; ``view()`` is the same-length text with every other marker
    defused, for bounded regexes whose spans then slice the text itself.

    The ids decode without special tokens. The parser text either shows the
    text of every special token the ids hold, where they hold it (serving does
    for tool-call requests), or none; each reading is matched while the text
    agrees with it, and markers resolve only where every such reading agrees.

    Matching is by id once output ids, even none, have been observed; text the
    ids have not reached yet holds no markers. Before that, and in "auto" mode
    after the parser text disagrees with the ids, markers match as text.
    "strict" mode raises ParserMatchError instead of matching as text.
    """

    def __init__(
        self,
        backend: Tokenizer,
        control_ids: Dict[int, str],
        special_ids: Dict[int, str],
        strict: bool = False,
    ):
        self._backend = backend
        self._control_ids = control_ids
        self._special_ids = special_ids
        self._strict = strict
        self._lead_chars = sorted({marker[0] for marker in control_ids.values()})
        self._decoder = DecodeStream(skip_special_tokens=True)
        self.output_ids: List[int] = []
        self._token_starts: List[int] = []
        # Decoded-text position -> the marker its control token writes.
        self._control_starts: Dict[int, str] = {}
        self._decoded_len = 0
        self._consumed: List[str] = []
        self.consumed_len = 0
        self._alignments = [_Alignment(show_special) for show_special in (True, False)]
        # Special tokens the decoder skipped, by decoded position, in id order.
        self._specials: List[Tuple[int, str]] = []
        # Ids after the last decoded piece that decode to nothing yet, such as
        # bytes of an unfinished character and the special tokens after them.
        self._undecoded: List[int] = []
        self._fed = False
        # The last view, by where its text starts in the consumed text.
        self._view_start = -1
        self._view = ""

    @classmethod
    def create(
        cls, tokenizer, markers: Iterable[str], match_mode: ParserMatchMode = "auto"
    ) -> Optional["ControlTokenStream"]:
        """A stream matching ``markers`` by id, or None to match them as text.
        Raises ParserMatchError in "strict" mode when the tokenizer cannot."""
        if match_mode == "text":
            return None
        markers = tuple(markers)
        if isinstance(tokenizer, PreTrainedTokenizerFast):
            control_ids, reason = _control_ids(tokenizer, markers)
        elif tokenizer is None:
            control_ids, reason = None, "the parser has no tokenizer"
        else:
            control_ids, reason = None, "the tokenizer is not a fast tokenizer"
        if control_ids is None:
            if match_mode == "strict":
                raise ParserMatchError(f"parser_match_mode=strict: {reason}")
            if tokenizer is not None:
                _log_once(f"{reason}; markers are matched as text.")
            return None
        return cls(
            tokenizer.backend_tokenizer,
            control_ids,
            _special_ids(tokenizer),
            strict=match_mode == "strict",
        )

    @property
    def active(self) -> bool:
        """Whether markers match by id."""
        return self._fed and bool(self._alignments)

    def feed(self, token_ids: Iterable[int]) -> None:
        self._fed = True
        pieces = []
        for token_id in token_ids:
            self.output_ids.append(token_id)
            self._token_starts.append(self._decoded_len)
            piece = self._decoder.step(self._backend, token_id)
            if not piece:
                special = self._special_ids.get(token_id)
                if special is not None and not self._undecoded:
                    self._specials.append((self._decoded_len, special))
                else:
                    self._undecoded.append(token_id)
                continue
            self._place_held_specials(self._undecoded + [token_id], piece)
            self._undecoded = []
            pieces.append(piece)
            self._decoded_len += len(piece)
            marker = self._control_ids.get(token_id)
            if marker is not None and piece.endswith(marker):
                self._control_starts[self._decoded_len - len(marker)] = marker
        if pieces:
            self._match(decoded="".join(pieces))

    def _place_held_specials(self, token_ids: List[int], piece: str) -> None:
        """Record the special tokens among ``token_ids``, which decode to
        ``piece`` once the decoder catches up, where each falls in it."""
        parts = self._render(token_ids)
        if not any(special for _, special in parts):
            return
        # Bytes completing a character across a special token render apart;
        # the special has no position in the decoded character then.
        if "".join(text for text, special in parts if not special) != piece:
            return
        pos = self._decoded_len
        for text, special in parts:
            if special:
                self._specials.append((pos, text))
            else:
                pos += len(text)

    def invalidate(self, reason: str) -> None:
        """The observed ids no longer describe the parser text."""
        self._lose_alignment(reason)

    def consume(self, text: str) -> None:
        """Record ``text`` as the next text handed to the parser."""
        if not text:
            return
        self._require_ids()
        self._consumed.append(text)
        self.consumed_len += len(text)
        if len(self._consumed) > _COMPACT_EVERY:
            self._consumed = ["".join(self._consumed)]
        self._match(consumed=text)

    def bind(self, text: str) -> None:
        """Record ``text`` as all of the parser's text, for one-shot parsing."""
        if self.consumed_len == 0:
            self.consume(text)
            self.check_sourced()
        elif self.consumed_len != len(text) or not self._is_tail(text):
            self._lose_alignment("one-shot text is not the text parsed so far")

    def find(self, kind: str, text: str, start: int = 0) -> Optional[MarkerSpan]:
        """The first ``kind`` marker at or after ``start`` in ``text``, which
        must end the consumed text."""
        idx = text.find(kind, start)
        if not self.active:
            return None if idx == -1 else MarkerSpan(kind, idx, idx + len(kind))
        base = self._tail_start(text)
        while idx != -1:
            if self._marker_at(base + idx) == kind:
                return MarkerSpan(kind, idx, idx + len(kind))
            idx = text.find(kind, idx + 1)
        return None

    def match_at(self, kind: str, text: str, offset: int) -> Optional[MarkerSpan]:
        """The ``kind`` marker starting at ``offset`` of ``text``, which must
        end the consumed text."""
        if self.active and self._marker_at(self._tail_start(text) + offset) != kind:
            return None
        if not text.startswith(kind, offset):
            return None
        return MarkerSpan(kind, offset, offset + len(kind))

    def check_sourced(self) -> None:
        """Check, once all output is parsed, that the ids cover the parser text.
        Ids decoding past the text are fine; stop text is trimmed from it."""
        if not self._fed or not self._alignments:
            return
        sourced = [
            a
            for a in self._alignments
            if self._remaining_text(a).startswith(a.consumed_ahead)
        ]
        if sourced:
            self._alignments = sourced
            return
        if self._strict:
            raise ParserMatchError(
                "Cannot match markers by output token id: "
                "output text runs past its output ids"
            )
        _log_once("Output text runs past its output ids; no markers match in it.")

    def _remaining_text(self, alignment: _Alignment) -> str:
        """What the ids past ``alignment``'s matched text render to at the
        end of output."""
        parts = []
        pos = alignment.frontier()
        ahead = alignment.decoded_ahead
        if alignment.show_special:
            for special_pos, special in self._specials[alignment.next_special :]:
                parts.append(ahead[: special_pos - pos])
                ahead = ahead[special_pos - pos :]
                pos = special_pos
                parts.append(special)
        parts.append(ahead)
        # Unfinished bytes render as U+FFFD, as the detokenizer flushes them.
        parts.extend(
            text
            for text, special in self._render(self._undecoded)
            if alignment.show_special or not special
        )
        return "".join(parts)

    def _render(self, token_ids: List[int]) -> List[Tuple[str, bool]]:
        """``token_ids`` as (text, is_special) parts: runs of ordinary ids
        decoded on their own, and special tokens."""
        parts = []
        run = []
        for token_id in token_ids:
            special = self._special_ids.get(token_id)
            if special is None:
                run.append(token_id)
                continue
            parts.append((self._backend.decode(run), False))
            run = []
            parts.append((special, True))
        parts.append((self._backend.decode(run), False))
        return parts

    def view(self, text: str) -> str:
        """``text``, which must end the consumed text, with every marker that
        does not start at a control token defused."""
        if not self.active:
            return text
        start = self._tail_start(text)
        # A parser rescans its growing buffer every chunk; reusing the part of
        # the last view whose ids were decoded keeps the Python work per chunk small.
        reuse = 0
        if start == self._view_start:
            reuse = max(0, min(len(self._view), self._matched_len() - start))
        view = self._view[:reuse] + self._defuse(text[reuse:], start + reuse)
        self._view_start, self._view = start, view
        return view

    def _tail_start(self, text: str) -> int:
        # Positions are known only for the consumed text; any other string
        # would silently match as text.
        if not self._is_tail(text):
            raise ParserMatchError(
                "Markers can only be looked up at the end of the parsed text."
            )
        return self.consumed_len - len(text)

    def _require_ids(self) -> None:
        if self._fed:
            return
        if self._strict:
            raise ParserMatchError(
                "parser_match_mode=strict: output text has no output ids"
            )
        _log_once("Parser text came without output ids; markers are matched as text.")

    def _defuse(self, text: str, start: int) -> str:
        literal = sorted(
            i
            for lead in self._lead_chars
            for i in _find_all(text, lead)
            if self._marker_at(start + i) is None
        )
        if not literal:
            return text
        parts = []
        last = 0
        for i in literal:
            parts.append(text[last:i])
            parts.append(_MASK)
            last = i + 1
        parts.append(text[last:])
        return "".join(parts)

    def ids_from(self, text: str) -> Optional[List[int]]:
        """The output ids that decode to ``text``, which must end the consumed
        text and start at a token boundary."""
        if not self.active or not self._is_tail(text):
            return None
        start = self._agreed(
            a.decoded_pos(self.consumed_len - len(text)) for a in self._alignments
        )
        if start is None:
            return None
        i = bisect_left(self._token_starts, start)
        if i == len(self._token_starts) or self._token_starts[i] != start:
            return None
        return self.output_ids[i:]

    def _matched_len(self) -> int:
        """How much of the consumed text every reading has matched."""
        return min(a.matched_len for a in self._alignments)

    def _marker_at(self, pos: int) -> Optional[str]:
        """The marker a control token writes at parser-text position ``pos``."""
        if pos >= self._matched_len():
            return None
        return self._agreed(
            self._control_starts.get(a.decoded_pos(pos)) for a in self._alignments
        )

    @staticmethod
    def _agreed(answers: Iterable[_T]) -> Optional[_T]:
        answers = set(answers)
        return answers.pop() if len(answers) == 1 else None

    def _is_tail(self, text: str) -> bool:
        if len(text) > self.consumed_len:
            return False
        end = len(text)
        for chunk in reversed(self._consumed):
            if end == 0:
                break
            take = min(len(chunk), end)
            if chunk[len(chunk) - take :] != text[end - take : end]:
                return False
            end -= take
        return True

    def _match(self, decoded: str = "", consumed: str = "") -> None:
        if not self._alignments:
            return
        agreeing = [
            a for a in self._alignments if a.match(decoded, consumed, self._specials)
        ]
        if not agreeing:
            self._lose_alignment("parser text does not match the decoded output ids")
            return
        self._alignments = agreeing

    def _lose_alignment(self, reason: str) -> None:
        if self._strict:
            raise ParserMatchError(f"Cannot match markers by output token id: {reason}")
        if self._alignments:
            logger.warning(
                "Matching markers as text for the rest of this output: %s.", reason
            )
        self._alignments = []


class OutputIdsCursor:
    """Takes a stream's output ids chunk by chunk and returns the ids each
    chunk adds, checking that none were repeated, skipped or rewritten."""

    def __init__(self):
        self.started = False
        self.seen = 0
        self._seen_ids: List[int] = []
        # The cumulative list last read; TokenizerManager extends one list per
        # request, so the same list is trusted to keep the ids already seen.
        self._source: Optional[List[int]] = None

    def advance(
        self, output_ids: List[int], *, cumulative: bool, total: Optional[int]
    ) -> List[int]:
        """``output_ids`` are every id so far when ``cumulative`` is set, else
        the chunk's; ``total`` is the generated token count after the chunk.
        Raises ParserMatchError when the chunk does not continue the ids."""
        if cumulative:
            # The shared list may have grown since ``total`` was counted, so
            # it can run ahead of the count but not behind it.
            if len(output_ids) < max(self.seen, total or 0):
                raise ParserMatchError(
                    f"cumulative output ids stop at {len(output_ids)} of "
                    f"{max(self.seen, total or 0)} tokens"
                )
            if output_ids is not self._source and (
                output_ids[: self.seen] != self._seen_ids
            ):
                raise ParserMatchError("cumulative output ids change ids seen before")
            self._source = output_ids
            new_ids = output_ids[self.seen :]
        else:
            new_ids = output_ids
            if total is not None and self.seen + len(new_ids) != total:
                raise ParserMatchError(
                    f"a chunk of {len(new_ids)} output ids after {self.seen} "
                    f"does not end at generated token {total}"
                )
        self.started = True
        self.seen += len(new_ids)
        self._seen_ids.extend(new_ids)
        return new_ids
