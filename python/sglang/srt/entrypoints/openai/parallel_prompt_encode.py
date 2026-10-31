r"""Encode a long rendered chat prompt in parallel with ids identical to
``tokenizer.encode(text, **encode_kwargs)``.

A HF fast tokenizer first splits the text at added tokens and runs the
normalizer, pre-tokenizer and model on each piece in between independently.
Cutting the text right after an added-token match therefore cannot change the
ids, so the pieces go through the backend's ``encode_batch``, which runs on the
tokenizers (rayon) thread pool.

A single message is often most of the prompt, so long gaps between added
tokens are also cut at pre-token boundaries, but only for the exact
Split-regex + ByteLevel + BPE pipelines of GLM / Llama-3 and Qwen3.5 style
tokenizers (``_INTRA_SPLIT_REGEXES``). For these regexes, a match containing a
letter ends at the last letter (or combining mark) of a run (only the
``'s``-style and ``?\p{L}+`` / ``?[\p{L}\p{M}]+`` alternatives contain
letters) and a match containing ``\n`` never continues into a letter, so
"letter|space letter" and "\n|letter" are always pre-token boundaries.
Truncating a chunk after a letter cannot change any match; after ``\n`` the
only lookahead that could flip (``\s+(?!\S)``) is shadowed by the earlier
``\s*[\r\n]+`` alternative. BPE never merges across pre-tokens. An NFC
normalizer (Qwen3.5) keeps these cuts: NFC only composes a character with
ones before it, and neither the space nor an ASCII letter ever composes with a
preceding character, so normalizing the two sides of a cut separately gives
the normalized whole. Added tokens matched on normalized text could still
straddle a cut, so a normalizer with such tokens gets no intra-gap cuts.

When tokenizers may not use its thread pool in this process (SGLang sets
TOKENIZERS_PARALLELISM=false for multimodal models, whose preprocess pool
forks), the pieces are encoded one per call from a small thread pool instead:
``encode_batch`` releases the GIL while it encodes.

Any configuration where that argument does not hold falls back to the original
encode: slow or wrapped tokenizers, specials added by the post-processor,
post-processors that can rewrite sequence ids, lstrip / rstrip / single_word
added tokens, pre-tokenizers that treat the first piece of a sequence
differently (Metaspace), split_special_tokens, and truncation or padding left
enabled on the backend.

``encode_prompt_segments`` serves renderers that hand over the prompt as
segments with their special tokens on their own. Those are the cut points, so
the prompt is not scanned and no length threshold applies, and the ids of the
text between them are cached per tokenizer: text that recurs across requests,
like the system prompt, the tool list and earlier turns, is encoded once.

Gated by SGLANG_PARALLEL_PROMPT_ENCODE and SGLANG_PARALLEL_PROMPT_ENCODE_MIN_CHARS.
"""

import json
import logging
import os
import re
import threading
import time
import weakref
from collections import OrderedDict
from itertools import chain

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Adjacent pieces are merged up to this size; tiny inputs only add per-input
# overhead in encode_batch.
_MIN_CHUNK_CHARS = 2048
_MAX_CHUNKS = 64

# Per tokenizer; a text longer than the entry limit is encoded but not kept.
_SEGMENT_CACHE_MAX_TOKENS = 1 << 20
_SEGMENT_CACHE_MAX_ENTRY_TOKENS = _SEGMENT_CACHE_MAX_TOKENS // 8

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

_ID_PRESERVING_POST_PROCESSORS = {
    "BertProcessing",
    "ByteLevel",
    "RobertaProcessing",
}

# Pre-token regexes for which the intra-gap cut points below were verified.
# GLM / Llama-3:
_INTRA_SPLIT_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# Qwen3.5: letter runs take combining marks, digits are split one by one.
_QWEN35_SPLIT_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|"
    r" ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
_INTRA_SPLIT_REGEXES = (_INTRA_SPLIT_REGEX, _QWEN35_SPLIT_REGEX)
# Normalizers that keep the intra-gap cut points (see the module docstring).
_INTRA_SAFE_NORMALIZERS = (None, {"type": "NFC"})
# Threads that encode the pieces when tokenizers' own pool is off.
_MAX_ENCODE_THREADS = 16
_TOKENIZERS_PARALLELISM_OFF = ("", "off", "false", "f", "no", "n", "0")
# ASCII letters only: Python and Oniguruma agree on them for \p{L}.
_INTRA_CUT = re.compile(r"(?<=[A-Za-z])(?= [A-Za-z])|(?<=\n)(?=[A-Za-z])")

_plans = weakref.WeakKeyDictionary()
_encode_pool = None
_encode_pool_pid = None
_segment_encoders = weakref.WeakKeyDictionary()
# None: not yet checked; False: encoding the pieces consumed only one CPU (no
# spare CPU). The PID prevents inheriting the parent's result.
_parallel_pid = os.getpid()
_parallel_ok = None


def _pre_tokenizer_safe(cfg):
    if cfg is None:
        return True
    if cfg.get("type") == "Sequence":
        return all(_pre_tokenizer_safe(c) for c in cfg.get("pretokenizers", []))
    return cfg.get("type") in _SAFE_PRE_TOKENIZERS


def _template_post_processor_safe(cfg):
    sequences = 0
    for item in cfg.get("single", []):
        if not isinstance(item, dict) or len(item) != 1:
            return False
        if isinstance(item.get("SpecialToken"), dict):
            continue
        sequence = item.get("Sequence")
        if not isinstance(sequence, dict) or sequence.get("id") != "A":
            return False
        sequences += 1
    return sequences == 1


def _post_processor_safe(cfg):
    if cfg is None:
        return True
    processor_type = cfg.get("type")
    if processor_type == "Sequence":
        return all(_post_processor_safe(c) for c in cfg.get("processors", []))
    if processor_type == "TemplateProcessing":
        return _template_post_processor_safe(cfg)
    return processor_type in _ID_PRESERVING_POST_PROCESSORS


def _intra_cut_safe(cfg):
    pre = cfg.get("pre_tokenizer") or {}
    subs = pre.get("pretokenizers") if pre.get("type") == "Sequence" else None
    if not subs or len(subs) != 2:
        return False
    split, bl = subs
    model = cfg.get("model") or {}
    return (
        cfg.get("normalizer") in _INTRA_SAFE_NORMALIZERS
        and split.get("type") == "Split"
        and split.get("pattern") in [{"Regex": r} for r in _INTRA_SPLIT_REGEXES]
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
    if not _post_processor_safe(cfg.get("post_processor")):
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
    # Under a normalizer the second-pass tokens are no cut points (npattern is
    # None), so an intra-gap cut could split one of them.
    intra_ok = _intra_cut_safe(cfg) and not (normed and backend.normalizer is not None)
    intra = _INTRA_CUT if intra_ok else None
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


def _rayon_disabled():
    # tokenizers' own reading of the variable: unset means on. It also sets
    # the variable to false in a child forked after the pool was used.
    value = os.environ.get("TOKENIZERS_PARALLELISM")
    return value is not None and value.lower() in _TOKENIZERS_PARALLELISM_OFF


def _encode_chunks(backend, chunks):
    """``backend.encode_batch(chunks)``: on tokenizers' pool when it may use it,
    else one chunk per call from a thread pool."""
    global _encode_pool, _encode_pool_pid
    if len(chunks) < 2 or not _rayon_disabled():
        return backend.encode_batch(chunks, add_special_tokens=False)
    if _encode_pool is None or _encode_pool_pid != os.getpid():
        from concurrent.futures import ThreadPoolExecutor

        _encode_pool = ThreadPoolExecutor(
            max_workers=min(_MAX_ENCODE_THREADS, os.cpu_count() or 1),
            thread_name_prefix="sglang-prompt-encode",
        )
        _encode_pool_pid = os.getpid()
    return list(
        _encode_pool.map(
            lambda chunk: backend.encode_batch([chunk], add_special_tokens=False)[0],
            chunks,
        )
    )


def parallel_prompt_encode(tokenizer, text, encode_kwargs=None):
    """Drop-in for ``tokenizer.encode(text, **encode_kwargs)``."""
    global _parallel_ok, _parallel_pid
    pid = os.getpid()
    if pid != _parallel_pid:
        _parallel_pid = pid
        _parallel_ok = None
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
        w0, c0 = time.perf_counter(), time.process_time()
    encodings = _encode_chunks(backend, chunks)
    if check:
        wall, cpu = time.perf_counter() - w0, time.process_time() - c0
        # Worker CPU (rayon or the thread pool) adds up well beyond wall time.
        # Serial stays near 1.0; 1.5 leaves headroom for scheduler and
        # measurement noise.
        _parallel_ok = not (wall > 0.005 and cpu < 1.5 * wall)
        if not _parallel_ok:
            logger.warning(
                "parallel_prompt_encode: encoding the pieces used only one CPU in "
                "this process; using the original encode."
            )
    return list(chain.from_iterable(e.ids for e in encodings))


def _cuts_at(token, added_tokens):
    """Whether encoding the whole prompt matches ``token`` wherever a renderer
    put it, whatever text surrounds it: no added token contains it, and none
    can begin in the text before it and end inside it."""
    prefixes = tuple(token[:i] for i in range(1, len(token) + 1))
    return not any(
        (other != token and token in other) or other[1:].endswith(prefixes)
        for other in added_tokens
    )


class _SegmentEncoder:
    def __init__(self, plan):
        (
            self.backend,
            self.specials_added,
            self.pattern,
            self.npattern,
            self.intra,
        ) = plan
        added = self.backend.get_added_tokens_decoder()
        self.added_tokens = [t.content for t in added.values() if t.content]
        self.raw_ids = {
            t.content: i for i, t in added.items() if t.content and not t.normalized
        }
        self.max_cut_len = max(map(len, self.raw_ids), default=0)
        self.cuts = {}
        self.lock = threading.Lock()
        self.cache = OrderedDict()
        self.cached_tokens = 0

    def cut_id(self, segment):
        token_id = self.raw_ids.get(segment)
        if token_id is None:
            return None
        cuts = self.cuts.get(segment)
        if cuts is None:
            cuts = self.cuts[segment] = _cuts_at(segment, self.added_tokens)
        return token_id if cuts else None

    def encode(self, segments):
        layout, run = [], []
        for segment in segments:
            if len(segment) <= self.max_cut_len:
                token_id = self.cut_id(segment)
                if token_id is not None:
                    if run:
                        layout.append("".join(run))
                        run = []
                    layout.append(token_id)
                    continue
            if segment:
                run.append(segment)
        if run:
            layout.append("".join(run))

        texts = [item for item in layout if isinstance(item, str)]
        known = {}
        with self.lock:
            for text in texts:
                ids = self.cache.get(text)
                if ids is not None:
                    self.cache.move_to_end(text)
                    known[text] = ids
        missing = list(dict.fromkeys(text for text in texts if text not in known))
        if missing:
            known.update(self._encode(missing))

        ids = []
        for item in layout:
            if isinstance(item, str):
                ids.extend(known[item])
            else:
                ids.append(item)
        return ids

    def _encode(self, texts):
        min_chars = envs.SGLANG_PARALLEL_PROMPT_ENCODE_MIN_CHARS.get()
        chunks, counts = [], []
        for text in texts:
            pieces = (
                split_prompt(text, self.pattern, self.npattern, self.intra)
                if len(text) >= min_chars
                else [text]
            )
            chunks.extend(pieces)
            counts.append(len(pieces))
        encodings = _encode_chunks(self.backend, chunks)
        encoded, start = {}, 0
        for text, count in zip(texts, counts):
            encoded[text] = tuple(
                chain.from_iterable(e.ids for e in encodings[start : start + count])
            )
            start += count

        with self.lock:
            for text, ids in encoded.items():
                if (
                    text not in self.cache
                    and len(ids) <= _SEGMENT_CACHE_MAX_ENTRY_TOKENS
                ):
                    self.cache[text] = ids
                    self.cached_tokens += len(ids)
            while self.cached_tokens > _SEGMENT_CACHE_MAX_TOKENS:
                self.cached_tokens -= len(self.cache.popitem(last=False)[1])
        return encoded


def _segment_encoder_for(tokenizer):
    try:
        return _segment_encoders[tokenizer]
    except KeyError:
        pass
    except TypeError:
        return None
    plan = _plan_for(tokenizer)
    encoder = None if plan is None else _SegmentEncoder(plan)
    _segment_encoders[tokenizer] = encoder
    return encoder


def encode_prompt_segments(tokenizer, segments, encode_kwargs=None):
    """Drop-in for ``parallel_prompt_encode(tokenizer, "".join(segments),
    encode_kwargs)``."""
    encode_kwargs = encode_kwargs or {}
    encoder = None
    if envs.SGLANG_PARALLEL_PROMPT_ENCODE.get() and not (
        set(encode_kwargs) - {"add_special_tokens"}
    ):
        encoder = _segment_encoder_for(tokenizer)
    if (
        encoder is None
        or (encode_kwargs.get("add_special_tokens", True) and encoder.specials_added)
        or encoder.backend.truncation is not None
        or encoder.backend.padding is not None
        or encoder.backend.encode_special_tokens
    ):
        return parallel_prompt_encode(tokenizer, "".join(segments), encode_kwargs)
    return encoder.encode(segments)
