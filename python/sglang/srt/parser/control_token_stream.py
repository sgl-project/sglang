import logging
from bisect import bisect_left, bisect_right
from typing import Dict, Iterable, List, Optional, Set, Tuple
from weakref import WeakKeyDictionary

from tokenizers import Tokenizer
from tokenizers.decoders import DecodeStream
from transformers import PreTrainedTokenizerFast

logger = logging.getLogger(__name__)

# Replaces a marker's first character where no control token starts. Any
# character that begins no marker works; the view is only searched, never output.
_MASK = "\x00"

# Bounds the parser-text chunks a tail check walks; arbitrary.
_COMPACT_EVERY = 64

_control_ids_cache: WeakKeyDictionary = WeakKeyDictionary()
_special_texts_cache: WeakKeyDictionary = WeakKeyDictionary()


def _resolve_control_ids(
    backend: Tokenizer, markers: Tuple[str, ...]
) -> Optional[Dict[int, str]]:
    control_ids = {}
    for marker in markers:
        token_id = backend.token_to_id(marker)
        if (
            token_id is None
            or backend.decode([token_id], skip_special_tokens=False) != marker
        ):
            return None
        control_ids[token_id] = marker
    return control_ids


def _control_ids(
    tokenizer: PreTrainedTokenizerFast, markers: Tuple[str, ...]
) -> Optional[Dict[int, str]]:
    per_tokenizer = _control_ids_cache.setdefault(tokenizer, {})
    if markers not in per_tokenizer:
        per_tokenizer[markers] = _resolve_control_ids(
            tokenizer.backend_tokenizer, markers
        )
        if per_tokenizer[markers] is None:
            logger.warning(
                "%s are not single tokens of this tokenizer; they are matched as text.",
                list(markers),
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
    ``view()`` returns the parser's text with every marker spelled with ordinary
    tokens defused, so searching it finds control tokens only. The view keeps
    the text's length, so an index found in it slices the text itself.

    The ids decode without special tokens, and a special token the parser text
    keeps (serving keeps them for tool-call requests) is skipped over while
    matching, so either detokenizer setting lines up.

    The stream applies once output ids have been fed. It stops applying, and
    ``view()`` returns the text unchanged, if the parser text ever disagrees
    with the decoded ids.
    """

    def __init__(
        self,
        backend: Tokenizer,
        control_ids: Dict[int, str],
        special_texts: Tuple[str, ...],
    ):
        self._backend = backend
        self._control_ids = control_ids
        self._special_texts = special_texts
        self._lead_chars = sorted({marker[0] for marker in control_ids.values()})
        self._decoder = DecodeStream(skip_special_tokens=True)
        self.output_ids: List[int] = []
        self._token_starts: List[int] = []
        self._control_starts: Set[int] = set()
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
        cls, tokenizer, markers: Iterable[str]
    ) -> Optional["ControlTokenStream"]:
        if not isinstance(tokenizer, PreTrainedTokenizerFast):
            return None
        control_ids = _control_ids(tokenizer, tuple(markers))
        if control_ids is None:
            return None
        return cls(tokenizer.backend_tokenizer, control_ids, _special_texts(tokenizer))

    @property
    def active(self) -> bool:
        return self._fed and self._aligned

    def feed(self, token_ids: Iterable[int]) -> None:
        pieces = []
        for token_id in token_ids:
            self._fed = True
            self.output_ids.append(token_id)
            self._token_starts.append(self._decoded_len)
            piece = self._decoder.step(self._backend, token_id)
            if not piece:
                continue
            pieces.append(piece)
            self._decoded_len += len(piece)
            marker = self._control_ids.get(token_id)
            if marker is not None and piece.endswith(marker):
                self._control_starts.add(self._decoded_len - len(marker))
        if pieces:
            self._match(decoded="".join(pieces))

    def consume(self, text: str) -> None:
        """Record ``text`` as the next text handed to the parser."""
        if not text:
            return
        self._consumed.append(text)
        self.consumed_len += len(text)
        if len(self._consumed) > _COMPACT_EVERY:
            self._consumed = ["".join(self._consumed)]
        self._match(consumed=text)

    def bind(self, text: str) -> None:
        """Record ``text`` as all of the parser's text, for one-shot parsing."""
        if self.consumed_len == 0:
            self.consume(text)
        elif self.consumed_len != len(text) or not self._is_tail(text):
            self._lose_alignment()

    def view(self, text: str) -> str:
        """``text``, which must end the consumed text, with every marker that
        does not start at a control token defused."""
        if not self.active or not self._is_tail(text):
            return text
        start = self.consumed_len - len(text)
        # A parser rescans its growing buffer every chunk; reusing the part of
        # the last view whose ids were decoded keeps the Python work per chunk small.
        reuse = 0
        if start == self._view_start:
            reuse = max(0, min(len(self._view), self._matched_len - start))
        view = self._view[:reuse] + self._defuse(text[reuse:], start + reuse)
        self._view_start, self._view = start, view
        return view

    def _defuse(self, text: str, start: int) -> str:
        literal = sorted(
            i
            for lead in self._lead_chars
            for i in _find_all(text, lead)
            if not self._is_control_start(start + i)
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

    def _is_control_start(self, pos: int) -> bool:
        if pos >= self._matched_len:
            return False
        return self._decoded_pos(pos) in self._control_starts

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
                self._lose_alignment()
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

    def _lose_alignment(self) -> None:
        if self._aligned:
            logger.warning(
                "Parser text does not match the decoded output ids; "
                "matching markers as text for the rest of this output."
            )
        self._aligned = False
        self._decoded_ahead = self._consumed_ahead = ""
