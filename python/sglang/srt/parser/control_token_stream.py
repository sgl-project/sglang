import logging
from bisect import bisect_left, bisect_right
from typing import Dict, Iterable, List, Literal, Optional, Tuple
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

_control_ids_cache: WeakKeyDictionary = WeakKeyDictionary()
_special_texts_cache: WeakKeyDictionary = WeakKeyDictionary()
_logged_reasons: set = set()


class ParserMatchError(ValueError):
    """Markers cannot be matched by output token id where matching must be."""


class MarkerSpan(msgspec.Struct, frozen=True):
    """A marker at ``[start, end)`` of the queried text."""

    kind: str
    start: int
    end: int


def _log_once(reason: str) -> None:
    if reason not in _logged_reasons:
        _logged_reasons.add(reason)
        logger.warning("%s; markers are matched as text.", reason)


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


def _special_texts(tokenizer: PreTrainedTokenizerFast) -> Tuple[str, ...]:
    if tokenizer not in _special_texts_cache:
        added = tokenizer.backend_tokenizer.get_added_tokens_decoder().values()
        _special_texts_cache[tokenizer] = tuple(t.content for t in added if t.special)
    return _special_texts_cache[tokenizer]


def _find_all(text: str, char: str) -> Iterable[int]:
    i = text.find(char)
    while i != -1:
        yield i
        i = text.find(char, i + 1)


class ControlTokenStream:
    """A request's output ids, decoded next to the text its parser receives.

    A marker such as ``</think>`` decodes to the same text whether the model
    generated the control token or spelled the string with ordinary tokens.
    ``find()`` and ``match_at()`` return only markers a control token of that
    kind wrote; ``view()`` is the same-length text with every other marker
    defused, for bounded regexes whose spans then slice the text itself.

    The ids decode without special tokens, and a special token the parser text
    keeps (serving keeps them for tool-call requests) is skipped over while
    matching, so either detokenizer setting lines up.

    Matching is by id once output ids, even none, have been observed; text the
    ids have not reached yet holds no markers. Before that, and in "auto" mode
    after the parser text disagrees with the ids, markers match as text.
    "strict" mode raises ParserMatchError instead of matching as text.
    """

    def __init__(
        self,
        backend: Tokenizer,
        control_ids: Dict[int, str],
        special_texts: Tuple[str, ...],
        strict: bool = False,
    ):
        self._backend = backend
        self._control_ids = control_ids
        self._special_texts = special_texts
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
        # Decoded text the parser text has not reached yet, or parser text
        # whose ids have not decoded yet; at most one is non-empty.
        self._decoded_ahead = ""
        self._consumed_ahead = ""
        # Parser text matched against the decoded text so far, and the special
        # tokens in it that the decoded text skips: starts, ends, running total.
        self._matched_len = 0
        self._skip_starts: List[int] = []
        self._skip_ends: List[int] = []
        self._skip_total: List[int] = []
        self._fed = False
        self._aligned = True
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
                _log_once(reason)
            return None
        return cls(
            tokenizer.backend_tokenizer,
            control_ids,
            _special_texts(tokenizer),
            strict=match_mode == "strict",
        )

    @property
    def active(self) -> bool:
        """Whether markers match by id."""
        return self._fed and self._aligned

    def feed(self, token_ids: Iterable[int]) -> None:
        self._fed = True
        pieces = []
        for token_id in token_ids:
            self.output_ids.append(token_id)
            self._token_starts.append(self._decoded_len)
            piece = self._decoder.step(self._backend, token_id)
            if not piece:
                continue
            pieces.append(piece)
            self._decoded_len += len(piece)
            marker = self._control_ids.get(token_id)
            if marker is not None and piece.endswith(marker):
                self._control_starts[self._decoded_len - len(marker)] = marker
        if pieces:
            self._match(decoded="".join(pieces))

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
            if self._strict and text and not self.output_ids:
                raise ParserMatchError(
                    "parser_match_mode=strict: output text has no output ids"
                )
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
        if not text.startswith(kind, offset):
            return None
        if self.active and self._marker_at(self._tail_start(text) + offset) != kind:
            return None
        return MarkerSpan(kind, offset, offset + len(kind))

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
            reuse = max(0, min(len(self._view), self._matched_len - start))
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
        _log_once("Parser text came without output ids")

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
        start = self._decoded_pos(self.consumed_len - len(text))
        if start is None:
            return None
        i = bisect_left(self._token_starts, start)
        if i == len(self._token_starts) or self._token_starts[i] != start:
            return None
        return self.output_ids[i:]

    def _decoded_pos(self, pos: int) -> Optional[int]:
        """Where parser-text position ``pos`` falls in the decoded text, or
        None inside a special token the decoded text skips."""
        i = bisect_right(self._skip_starts, pos) - 1
        if i < 0:
            return pos
        if pos < self._skip_ends[i]:
            return None
        return pos - self._skip_total[i]

    def _marker_at(self, pos: int) -> Optional[str]:
        """The marker a control token writes at parser-text position ``pos``."""
        if pos >= self._matched_len:
            return None
        return self._control_starts.get(self._decoded_pos(pos))

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
        if not self._aligned:
            return
        decoded = self._decoded_ahead + decoded
        consumed = self._consumed_ahead + consumed
        while True:
            n = min(len(decoded), len(consumed))
            if decoded[:n] == consumed[:n]:
                break
            k = next(i for i in range(n) if decoded[i] != consumed[i])
            skipped = self._special_before(consumed, k)
            if skipped is None:
                self._lose_alignment(
                    "parser text does not match the decoded output ids"
                )
                return
            j, special = skipped
            start = self._matched_len + j
            self._skip_starts.append(start)
            self._skip_ends.append(start + len(special))
            self._skip_total.append(
                (self._skip_total[-1] if self._skip_total else 0) + len(special)
            )
            self._matched_len += j + len(special)
            decoded = decoded[j:]
            consumed = consumed[j + len(special) :]
        self._matched_len += n
        self._decoded_ahead = decoded[n:]
        self._consumed_ahead = consumed[n:]

    def _special_before(self, text: str, k: int) -> Optional[Tuple[int, str]]:
        """The earliest special token in ``text`` that covers index ``k``."""
        found = None
        for special in self._special_texts:
            for j in range(max(0, k - len(special) + 1), k + 1):
                if text.startswith(special, j) and (found is None or j < found[0]):
                    found = (j, special)
                    break
        return found

    def _lose_alignment(self, reason: str) -> None:
        if self._strict:
            raise ParserMatchError(f"Cannot match markers by output token id: {reason}")
        if self._aligned:
            logger.warning(
                "Matching markers as text for the rest of this output: %s.", reason
            )
        self._aligned = False
        self._decoded_ahead = self._consumed_ahead = ""


class OutputIdsCursor:
    """Takes a stream's output ids chunk by chunk and returns the ids each
    chunk adds, checking that none were repeated, skipped or rewritten."""

    def __init__(self):
        self.started = False
        self.seen = 0
        self._last_id: Optional[int] = None

    def advance(
        self, output_ids: List[int], *, cumulative: bool, total: Optional[int]
    ) -> List[int]:
        """``output_ids`` are every id so far when ``cumulative`` is set, else
        the chunk's; ``total`` is the generated token count after the chunk.
        Raises ParserMatchError when the chunk does not continue the ids."""
        if cumulative:
            # The list may have grown since ``total`` was counted; it shares
            # storage with the request state, so only a shrink or rewrite is wrong.
            if len(output_ids) < self.seen or (
                self.seen and output_ids[self.seen - 1] != self._last_id
            ):
                raise ParserMatchError(
                    "cumulative output ids do not extend the ids seen before"
                )
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
        if new_ids:
            self._last_id = new_ids[-1]
        return new_ids
