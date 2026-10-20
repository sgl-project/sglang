"""Encode request text in a rendered chat prompt as plain text.

String encoders such as ``encoding_dsv41`` render a conversation to one
string, and ``tokenizer.encode`` maps every added-token spelling in that string
to its token id, including spellings that came from the request. A tool result
that spells the end-of-sentence and User tokens then ends the turn and starts
a new user turn, exactly as if the encoder had written those tokens.

DeepSeek's reference serving encoder (deepseek-recipe) keeps the two apart: the
control tokens the template writes are encoded as their ids, and text taken
from the request is encoded without the tokenizer's added vocabulary. This
module gives a string encoder the same contract in two steps:

1. ``mark_request_text`` brackets each added-token spelling in the request
   strings with two Unicode noncharacters before rendering.
2. ``encode_marked_prompt`` encodes the rendered prompt. An unmarked spelling
   can only come from the template, so it is encoded as its id; the text
   between two such spellings, marked spellings included, is encoded without
   the added vocabulary.

A prompt with no marks is passed to ``tokenizer.encode`` unchanged, so a
request whose text contains no added-token spelling gets the same ids as
before.
"""

import json
import logging
import re
import weakref
from typing import Any, Collection, Dict, List, Optional

import msgspec

logger = logging.getLogger(__name__)

# Noncharacters are reserved by Unicode for process-internal use. A request
# string that contains one gets it bracketed as well, so it survives as text
# and cannot forge or break a mark.
_OPEN = "\ufdd0"
_CLOSE = "\ufdd1"


class _Plan(msgspec.Struct):
    # Matches an added-token spelling or a marker character in request text.
    mark_re: Any
    # Matches a marked spelling, an unmarked spelling, or a stray marker
    # character in a rendered prompt.
    parse_re: Any
    token_ids: Dict[str, int]
    # The backend without its added vocabulary, built on first use.
    plain_backend: Any = None


_plans: "weakref.WeakKeyDictionary[Any, Optional[_Plan]]" = weakref.WeakKeyDictionary()


def mark_request_text(tokenizer: Any, obj: Any, keep: Collection[str] = ()) -> Any:
    """Return a copy of ``obj`` with the added-token spellings in its strings marked.

    ``obj`` is a string or a JSON-like list/dict, such as a chat message or a
    tool definition; dict keys are marked too. Spellings listed in ``keep``
    stay unmarked and are encoded as control tokens. ``image_url`` values are
    left as they are, and an ``arguments`` string that holds a JSON object is
    parsed first, so a ``\\u003c``-escaped spelling is marked as well.
    """
    plan = _plan_for(tokenizer)
    if plan is None:
        return obj
    keep = frozenset(keep)

    def mark_match(match: re.Match) -> str:
        text = match.group(0)
        return text if text in keep else f"{_OPEN}{text}{_CLOSE}"

    def mark(value: Any) -> Any:
        if isinstance(value, str):
            return plan.mark_re.sub(mark_match, value)
        if isinstance(value, dict):
            marked = {}
            for key, item in value.items():
                if key == "image_url":
                    marked[key] = item
                    continue
                if key == "arguments" and isinstance(item, str):
                    item = _json_object_or_text(item)
                marked[mark(key)] = mark(item)
            return marked
        if isinstance(value, (list, tuple)):
            return [mark(item) for item in value]
        return value

    return mark(obj)


def encode_marked_prompt(tokenizer: Any, prompt: str) -> List[int]:
    """Encode a prompt rendered from request text marked by ``mark_request_text``."""
    plan = _plan_for(tokenizer) if _OPEN in prompt else None
    if plan is None:
        return tokenizer.encode(prompt)

    # Control-token ids and plain-text regions, in prompt order.
    pieces: List[Any] = []
    region: List[str] = []
    pos = 0
    for match in plan.parse_re.finditer(prompt):
        region.append(prompt[pos : match.start()])
        pos = match.end()
        text = match.group(0)
        if len(text) > 1 and text[0] == _OPEN:
            region.append(text[1:-1])
        elif text not in (_OPEN, _CLOSE):
            pieces.append("".join(region))
            region = []
            pieces.append(plan.token_ids[text])
        # A stray marker character is dropped.
    region.append(prompt[pos:])
    pieces.append("".join(region))

    if plan.plain_backend is None:
        plan.plain_backend = _build_plain_backend(tokenizer)
    texts = [piece for piece in pieces if isinstance(piece, str) and piece]
    encodings = iter(plan.plain_backend.encode_batch(texts, add_special_tokens=False))
    ids: List[int] = []
    for piece in pieces:
        if isinstance(piece, int):
            ids.append(piece)
        elif piece:
            ids.extend(next(encodings).ids)
    return ids


def _json_object_or_text(text: str) -> Any:
    try:
        value = json.loads(text)
    except ValueError:
        return text
    return value if isinstance(value, dict) else text


def _plan_for(tokenizer: Any) -> Optional[_Plan]:
    try:
        return _plans[tokenizer]
    except KeyError:
        pass
    except TypeError:
        return _build_plan(tokenizer)
    plan = _plans[tokenizer] = _build_plan(tokenizer)
    return plan


def _build_plan(tokenizer: Any) -> Optional[_Plan]:
    """The splitting below mirrors the backend's added-token matching only for
    an HF fast tokenizer whose added tokens have no strip or word-boundary
    options and whose post-processor adds no tokens."""
    from transformers import PreTrainedTokenizerFast

    if (
        not isinstance(tokenizer, PreTrainedTokenizerFast)
        or tokenizer.split_special_tokens
        or any(
            token.lstrip or token.rstrip or token.single_word
            for token in tokenizer.backend_tokenizer.get_added_tokens_decoder().values()
        )
        or tokenizer.encode("") != []
    ):
        logger.warning(
            "Cannot encode request text as plain text with %s; added-token "
            "spellings in requests are encoded as control tokens.",
            type(tokenizer).__name__,
        )
        return None
    token_ids = tokenizer.get_added_vocab()
    spellings = [spelling for spelling in token_ids if spelling.strip()]
    if not spellings:
        return None
    spelling = _longest_match_pattern(spellings)
    return _Plan(
        mark_re=re.compile(f"{spelling}|{_OPEN}|{_CLOSE}"),
        parse_re=re.compile(
            f"{_OPEN}(?:{spelling}|{_OPEN}|{_CLOSE}){_CLOSE}"
            f"|{spelling}|{_OPEN}|{_CLOSE}"
        ),
        token_ids=token_ids,
    )


def _longest_match_pattern(words: List[str]) -> str:
    """A regex that matches the longest of ``words`` starting at a position,
    which is the backend's leftmost-longest added-token match.

    Words are merged into a trie so the engine follows one path per position;
    a flat alternation of ~1300 spellings makes a 500 KB tool result take ~1 s.
    """
    trie: Dict[str, Any] = {}
    for word in words:
        node = trie
        for char in word:
            node = node.setdefault(char, {})
        node[""] = {}

    def pattern(node: Dict[str, Any]) -> str:
        branches = [re.escape(c) + pattern(child) for c, child in node.items() if c]
        if not branches:
            return ""
        body = branches[0] if len(branches) == 1 else f"(?:{'|'.join(branches)})"
        # Greedy, so a longer word wins over a word that ends here.
        return f"(?:{body})?" if "" in node else body

    return pattern(trie)


def _build_plain_backend(tokenizer: Any) -> Any:
    from tokenizers import Tokenizer

    config = json.loads(tokenizer.backend_tokenizer.to_str())
    config["added_tokens"] = []
    config["truncation"] = None
    config["padding"] = None
    return Tokenizer.from_str(json.dumps(config))
