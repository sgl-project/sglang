r"""Encode a long rendered chat prompt in parallel with ids identical to
``tokenizer.encode(text, **encode_kwargs)``.

A HF fast tokenizer first splits the text at added tokens and runs the
normalizer, pre-tokenizer and model on each piece in between independently.
Cutting the text right after an added-token match therefore cannot change the
ids, so the pieces go through the backend's ``encode_batch``, which runs on the
tokenizers (rayon) thread pool.

A single message is often most of the prompt, so long gaps between added
tokens are also cut at pre-token boundaries, but only for the exact
Split-regex + ByteLevel + BPE pipeline of GLM / Llama-3 style tokenizers
(``_INTRA_SPLIT_REGEX``). For that regex, a match containing a letter ends at
the last letter of a run (only the ``'s``-style and ``?\p{L}+`` alternatives
contain letters) and a match containing ``\n`` never continues into a letter,
so "letter|space letter" and "\n|letter" are always pre-token boundaries.
Truncating a chunk after a letter cannot change any match; after ``\n`` the
only lookahead that could flip (``\s+(?!\S)``) is shadowed by the earlier
``\s*[\r\n]+`` alternative. BPE never merges across pre-tokens.

Any configuration where that argument does not hold falls back to the original
encode: slow or wrapped tokenizers, specials added by the post-processor,
lstrip / rstrip / single_word added tokens, pre-tokenizers that treat the first
piece of a sequence differently (Metaspace), split_special_tokens, and
truncation or padding left enabled on the backend.

Gated by SGLANG_PARALLEL_PROMPT_ENCODE and SGLANG_PARALLEL_PROMPT_ENCODE_MIN_CHARS.
"""

import json
import logging
import re
import time
import weakref
from itertools import chain

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Adjacent pieces are merged up to this size; tiny inputs only add per-input
# overhead in encode_batch.
_MIN_CHUNK_CHARS = 2048
_MAX_CHUNKS = 64

# Pre-tokenizers that act on each piece independently of its position in the
# sequence. Metaspace (prepend_scheme="first") and anything unknown are not.
_SAFE_PRE_TOKENIZERS = {
    "ByteLevel",
    "Split",
    "Digits",
    "Punctuation",
    "Whitespace",
    "WhitespaceSplit",
    "BertPreTokenizer",
    "CharDelimiterSplit",
    "UnicodeScripts",
}

# Pre-token regex for which the intra-gap cut points below were verified.
_INTRA_SPLIT_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# ASCII letters only: Python and Oniguruma agree on them for \p{L}.
_INTRA_CUT = re.compile(r"(?<=[A-Za-z])(?= [A-Za-z])|(?<=\n)(?=[A-Za-z])")

_plans = weakref.WeakKeyDictionary()
# None: not yet checked; False: encode_batch ran on the calling thread (rayon
# disabled, e.g. in a process forked after tokenizers used its pool).
_parallel_ok = None


def _pre_tokenizer_safe(cfg):
    if cfg is None:
        return True
    if cfg.get("type") == "Sequence":
        return all(_pre_tokenizer_safe(c) for c in cfg.get("pretokenizers", []))
    return cfg.get("type") in _SAFE_PRE_TOKENIZERS


def _intra_cut_safe(cfg):
    pre = cfg.get("pre_tokenizer") or {}
    subs = pre.get("pretokenizers") if pre.get("type") == "Sequence" else None
    if not subs or len(subs) != 2:
        return False
    split, bl = subs
    model = cfg.get("model") or {}
    return (
        cfg.get("normalizer") is None
        and split.get("type") == "Split"
        and split.get("pattern") == {"Regex": _INTRA_SPLIT_REGEX}
        and split.get("behavior") == "Isolated"
        and not split.get("invert")
        and bl.get("type") == "ByteLevel"
        and not bl.get("add_prefix_space")
        and not bl.get("use_regex")
        and model.get("type") == "BPE"
        and model.get("dropout") is None
    )


def _alternation(tokens):
    if not tokens:
        return None
    # Longest first: Python's leftmost-first alternation then equals the
    # backend's leftmost-longest added-token match.
    alts = sorted(set(tokens), key=len, reverse=True)
    return re.compile("|".join(re.escape(t) for t in alts))


def _build_plan(tokenizer):
    """Return (backend, specials_added, pattern, normalized_pattern, intra)
    or None."""
    try:
        from transformers import PreTrainedTokenizerBase
    except ImportError:
        return None
    backend = getattr(tokenizer, "_tokenizer", None)
    if not getattr(tokenizer, "is_fast", False) or backend is None:
        return None
    if not hasattr(backend, "encode_batch") or not hasattr(backend, "to_str"):
        return None
    # Only the stock HF fast path: encode -> _encode_plus -> backend.encode_batch.
    if type(tokenizer).encode is not PreTrainedTokenizerBase.encode:
        return None
    if "encode" in vars(tokenizer) or "_encode_plus" in vars(tokenizer):
        return None
    enc_plus = type(tokenizer)._encode_plus
    if enc_plus.__module__ != "transformers.tokenization_utils_tokenizers":
        return None
    if getattr(tokenizer, "split_special_tokens", False):
        return None
    try:
        cfg = json.loads(backend.to_str())
    except Exception:
        return None
    if not _pre_tokenizer_safe(cfg.get("pre_tokenizer")):
        return None
    added = backend.get_added_tokens_decoder().values()
    if any(t.lstrip or t.rstrip or t.single_word for t in added):
        return None
    raw = [t.content for t in added if not t.normalized and t.content]
    normed = [t.content for t in added if t.normalized and t.content]
    pattern = _alternation(raw)
    # Normalized added tokens are matched on normalized text in a second pass.
    # Without a normalizer that text is the original, so their ends are cut
    # points too; with one, only the first-pass (raw) tokens are used.
    npattern = _alternation(normed) if backend.normalizer is None else None
    intra = _INTRA_CUT if _intra_cut_safe(cfg) else None
    if pattern is None and npattern is None and intra is None:
        return None
    pp = backend.post_processor
    specials_added = pp.num_special_tokens_to_add(False) if pp is not None else 0
    return backend, specials_added, pattern, npattern, intra


def _plan_for(tokenizer):
    try:
        return _plans[tokenizer]
    except KeyError:
        pass
    except TypeError:
        return _build_plan(tokenizer)
    plan = _build_plan(tokenizer)
    _plans[tokenizer] = plan
    return plan


def _added_spans(text, pattern, npattern):
    """(start, end) of the backend's added-token matches (first pass, then
    second pass inside the gaps), in ascending order."""
    if pattern is None:
        if npattern is None:
            return []
        return [m.span() for m in npattern.finditer(text)]
    spans, pos = [], 0
    for m in pattern.finditer(text):
        if npattern is not None:
            spans.extend(m2.span() for m2 in npattern.finditer(text, pos, m.start()))
        spans.append(m.span())
        pos = m.end()
    if npattern is not None:
        spans.extend(m2.span() for m2 in npattern.finditer(text, pos))
    return spans


def split_prompt(text, pattern, npattern, intra=None, max_chunks=_MAX_CHUNKS):
    """Split text into at most ~max_chunks pieces at added-token ends and, if
    intra is given, at its matches strictly inside the gaps between them."""
    n = len(text)
    target = max(_MIN_CHUNK_CHARS, -(-n // max_chunks))
    chunks, start, pos = [], 0, 0
    for s, e in _added_spans(text, pattern, npattern) + [(n, n)]:
        # Gap [pos, s): both characters around an intra cut must lie in it.
        while intra is not None:
            m = intra.search(text, max(start + target, pos + 1), s)
            if m is None:
                break
            chunks.append(text[start : m.start()])
            start = m.start()
        if e - start >= target and e < n:
            chunks.append(text[start:e])
            start = e
        pos = e
    chunks.append(text[start:])
    return chunks


def parallel_prompt_encode(tokenizer, text, encode_kwargs=None):
    """Drop-in for ``tokenizer.encode(text, **encode_kwargs)``."""
    global _parallel_ok
    encode_kwargs = encode_kwargs or {}
    if (
        _parallel_ok is False
        or not envs.SGLANG_PARALLEL_PROMPT_ENCODE.get()
        or not isinstance(text, str)
        or len(text) < envs.SGLANG_PARALLEL_PROMPT_ENCODE_MIN_CHARS.get()
        or set(encode_kwargs) - {"add_special_tokens"}
    ):
        return tokenizer.encode(text, **encode_kwargs)
    plan = _plan_for(tokenizer)
    if plan is None:
        return tokenizer.encode(text, **encode_kwargs)
    backend, specials_added, pattern, npattern, intra = plan
    if encode_kwargs.get("add_special_tokens", True) and specials_added:
        return tokenizer.encode(text, **encode_kwargs)
    if (
        backend.truncation is not None
        or backend.padding is not None
        or backend.encode_special_tokens
    ):
        return tokenizer.encode(text, **encode_kwargs)
    chunks = split_prompt(text, pattern, npattern, intra)
    if len(chunks) < 2:
        return tokenizer.encode(text, **encode_kwargs)

    check = _parallel_ok is None
    if check:
        w0, c0 = time.perf_counter(), time.thread_time()
    encodings = backend.encode_batch(chunks, add_special_tokens=False)
    if check:
        wall, cpu = time.perf_counter() - w0, time.thread_time() - c0
        # With the rayon pool the calling thread only waits; serial encode_batch
        # burns its whole wall time on this thread.
        _parallel_ok = not (wall > 0.005 and cpu > 0.5 * wall)
        if not _parallel_ok:
            logger.warning(
                "parallel_prompt_encode: tokenizers encode_batch runs serially in "
                "this process (TOKENIZERS_PARALLELISM off or process forked after "
                "tokenizers used its pool); using the original encode."
            )
    return list(chain.from_iterable(e.ids for e in encodings))
