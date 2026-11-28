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
post-processors that can rewrite sequence ids, lstrip / rstrip / single_word
added tokens, pre-tokenizers that treat the first piece of a sequence
differently (Metaspace), and split_special_tokens.

The pieces are encoded on a private copy of the backend. The HF encode sets the
shared backend's padding, truncation and encode_special_tokens on every call
(a multimodal processor call with padding=True leaves padding on), and
``encode_batch`` applies whatever is set when it starts: on the shared backend,
a call from another thread between the gates and ``encode_batch`` could pad
the pieces. Nothing configures the copy, which has padding and truncation off
like the HF encode of a single text.

``encode_prompt_segments`` serves renderers that hand over the prompt as
segments with their special tokens on their own. Those are the cut points, so
the prompt is not scanned and no length threshold applies, and the ids of the
text between them are cached per tokenizer: text that recurs across requests,
like the system prompt, the tool list and earlier turns, is encoded once. Like
the radix cache, the cache is namespaced by the request's cache_salt, so a
salted request never hits text that another salt encoded. A BPE model with
dropout samples its merges on every encode, so it never uses the cache.

Gated by SGLANG_PARALLEL_PROMPT_ENCODE and SGLANG_PARALLEL_PROMPT_ENCODE_MIN_CHARS.
"""

import json
import logging
import os
import re
import sys
import threading
import time
import weakref
from array import array
from collections import OrderedDict
from itertools import chain

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

# Adjacent pieces are merged up to this size; tiny inputs only add per-input
# overhead in encode_batch.
_MIN_CHUNK_CHARS = 2048
_MAX_CHUNKS = 64

# Per tokenizer, counting the bytes of each entry's key text and its ids: a
# text can encode to no ids at all, so a token count would not bound the
# memory. An entry over the entry limit is encoded but not kept.
_SEGMENT_CACHE_MAX_BYTES = 64 << 20
_SEGMENT_CACHE_MAX_ENTRY_BYTES = _SEGMENT_CACHE_MAX_BYTES // 8
# The dict slot, the LRU links and the key tuple of one entry, roughly.
_SEGMENT_CACHE_ENTRY_OVERHEAD = 256
# A new text is kept as the list of ints encode_batch returned (a list slot
# and an int object per id); only a text that is used again is converted to a
# uint32 array, 4 bytes per id.
_LIST_ID_BYTES = 28

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

# Pre-token regex for which the intra-gap cut points below were verified.
_INTRA_SPLIT_REGEX = (
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|"
    r" ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"
)
# ASCII letters only: Python and Oniguruma agree on them for \p{L}.
_INTRA_CUT = re.compile(r"(?<=[A-Za-z])(?= [A-Za-z])|(?<=\n)(?=[A-Za-z])")

_plans = weakref.WeakKeyDictionary()
_segment_encoders = weakref.WeakKeyDictionary()
# One build per tokenizer: the first long prompts of a burst arrive together,
# and the build (parsing the backend, copying it) takes up to about a second.
_build_lock = threading.RLock()
_MISSING = object()
# None: not yet checked; False: encode_batch consumed only one CPU (rayon
# disabled, e.g. after a fork). The PID prevents inheriting the parent's result.
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
        serialized = backend.to_str()
        cfg = json.loads(serialized)
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
    intra = _INTRA_CUT if _intra_cut_safe(cfg) else None
    if pattern is None and npattern is None and intra is None:
        return None
    # The private copy the pieces are encoded on (see the module docstring).
    try:
        backend = type(backend).from_str(serialized)
    except Exception:
        return None
    backend.no_padding()
    backend.no_truncation()
    backend.encode_special_tokens = False
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
    with _build_lock:
        plan = _plans.get(tokenizer, _MISSING)
        if plan is _MISSING:
            plan = _plans[tokenizer] = _build_plan(tokenizer)
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


def _encode_batch(backend, texts):
    """``backend.encode_batch`` without the character offsets, which only the
    ids are read from: the same ids, with less work in the pool and on the
    calling thread (building and freeing the offsets)."""
    encode = getattr(backend, "encode_batch_fast", None) or backend.encode_batch
    return encode(texts, add_special_tokens=False)


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
    if (encode_kwargs.get("add_special_tokens", True) and specials_added) or getattr(
        tokenizer, "split_special_tokens", False
    ):
        return tokenizer.encode(text, **encode_kwargs)
    chunks = split_prompt(text, pattern, npattern, intra)
    if len(chunks) < 2:
        return tokenizer.encode(text, **encode_kwargs)

    check = _parallel_ok is None
    if check:
        w0, c0 = time.perf_counter(), time.process_time()
    encodings = _encode_batch(backend, chunks)
    if check:
        wall, cpu = time.perf_counter() - w0, time.process_time() - c0
        # Rayon worker CPU adds up well beyond wall time. Serial stays near 1.0;
        # 1.5 leaves headroom for scheduler and measurement noise.
        _parallel_ok = not (wall > 0.005 and cpu < 1.5 * wall)
        if not _parallel_ok:
            logger.warning(
                "parallel_prompt_encode: tokenizers encode_batch runs serially in "
                "this process (TOKENIZERS_PARALLELISM off or process forked after "
                "tokenizers used its pool); using the original encode."
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
        # Two LRU segments under one byte budget. A text enters `probation` as
        # the list encode_batch returned; one that is used again moves to
        # `cache` as a uint32 array. Eviction empties `probation` first, so
        # prompts whose text never recurs do not pay for the conversion and
        # do not push out text that does recur.
        self.cache = OrderedDict()
        self.probation = OrderedDict()
        # Key -> _InFlightEncode of the call that is encoding the text now.
        self.in_flight = {}
        self.cached_bytes = 0

    def cut_id(self, segment):
        token_id = self.raw_ids.get(segment)
        if token_id is None:
            return None
        cuts = self.cuts.get(segment)
        if cuts is None:
            cuts = self.cuts[segment] = _cuts_at(segment, self.added_tokens)
        return token_id if cuts else None

    def encode(self, segments, cache_salt=None):
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
        return self._encode_layout(layout, cache_salt)

    def encode_text(self, text, cache_salt=None):
        """``encode`` of a whole rendered prompt, cut at its added tokens in one
        pass: the text between two cut points cannot hold a whole added token,
        or the scan would have matched it there."""
        layout, start = [], 0
        for m in self.pattern.finditer(text):
            token_id = self.cut_id(m.group())
            if token_id is None:
                continue
            if m.start() > start:
                layout.append(text[start : m.start()])
            layout.append(token_id)
            start = m.end()
        if start < len(text):
            layout.append(text[start:])
        return self._encode_layout(layout, cache_salt)

    def _encode_layout(self, layout, cache_salt):
        known = {}
        texts = [item for item in layout if isinstance(item, str)]
        while texts:
            # Distinct texts encode in parallel; a text another call is already
            # encoding is waited for, not encoded again.
            own, waits = self._claim(texts, known, cache_salt)
            if own:
                encoded = {}
                try:
                    encoded = self._encode(own, cache_salt)
                finally:
                    self._publish(own, encoded, cache_salt)
                known.update(encoded)
            for text, pending in waits.items():
                pending.done.wait()
                ids = pending.ids.get(text)
                if ids is not None:
                    known[text] = ids
            # Left only if the encode waited for raised: encode them here.
            texts = [text for text in waits if text not in known]

        ids = []
        for item in layout:
            if isinstance(item, str):
                ids.extend(known[item])
            else:
                ids.append(item)
        return ids

    def _claim(self, texts, known, cache_salt):
        """Copy the cached ids of `texts` into `known`. Of the rest, return the
        texts this call must encode (now claimed, once each) and
        {text: _InFlightEncode} of those another call is encoding."""
        own, waits, claim = [], {}, None
        with self.lock:
            for text in dict.fromkeys(texts):
                key = (cache_salt, text)
                ids = self.cache.get(key)
                if ids is not None:
                    self.cache.move_to_end(key)
                    known[text] = ids
                elif (ids := self.probation.pop(key, None)) is not None:
                    packed = array("I", ids)
                    self.cached_bytes += _entry_bytes(key, packed) - _entry_bytes(
                        key, ids
                    )
                    self.cache[key] = packed
                    known[text] = ids
                elif key in self.in_flight:
                    waits[text] = self.in_flight[key]
                else:
                    if claim is None:
                        claim = _InFlightEncode()
                    self.in_flight[key] = claim
                    own.append(text)
        return own, waits

    def _encode(self, texts, cache_salt):
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
        encodings = _encode_batch(self.backend, chunks)
        encoded, start = {}, 0
        for text, count in zip(texts, counts):
            if count == 1:
                encoded[text] = encodings[start].ids
            else:
                encoded[text] = list(
                    chain.from_iterable(e.ids for e in encodings[start : start + count])
                )
            start += count
        return encoded

    def _publish(self, texts, encoded, cache_salt):
        """Cache the ids of the claimed `texts` and release their claims, in one
        step under the lock, so a text is always cached, in flight or unclaimed.
        If the encode raised, `encoded` is empty and the waiters get no ids."""
        with self.lock:
            for text in texts:
                key = (cache_salt, text)
                ids = encoded.get(text)
                if ids is not None:
                    size = _entry_bytes(key, ids)
                    lru = self.probation
                    if size > _SEGMENT_CACHE_MAX_ENTRY_BYTES:
                        # Too big as a list: keep it packed, if that fits.
                        ids, lru = array("I", ids), self.cache
                        size = _entry_bytes(key, ids)
                    if size <= _SEGMENT_CACHE_MAX_ENTRY_BYTES:
                        lru[key] = ids
                        self.cached_bytes += size
                claim = self.in_flight.pop(key)
            claim.ids = encoded
            claim.done.set()
            while self.cached_bytes > _SEGMENT_CACHE_MAX_BYTES:
                lru = self.probation if self.probation else self.cache
                self.cached_bytes -= _entry_bytes(*lru.popitem(last=False))


class _InFlightEncode:
    """One call's encode of the texts it claimed; `ids` ({text: ids}, empty if
    the encode raised) is set before `done`."""

    def __init__(self):
        self.done = threading.Event()
        self.ids = {}


def _samples_merges(backend):
    """BPE dropout samples the merges on every encode: cached ids would hand
    one sample to every later request."""
    return bool(getattr(backend.model, "dropout", None))


def _entry_bytes(key, ids):
    cache_salt, text = key
    size = sys.getsizeof(text) + sys.getsizeof(ids) + _SEGMENT_CACHE_ENTRY_OVERHEAD
    if isinstance(ids, list):
        size += _LIST_ID_BYTES * len(ids)
    return size if cache_salt is None else size + sys.getsizeof(cache_salt)


def _segment_encoder_for(tokenizer):
    try:
        return _segment_encoders[tokenizer]
    except KeyError:
        pass
    except TypeError:
        return None
    # One encoder per tokenizer, or requests would split its cache.
    with _build_lock:
        encoder = _segment_encoders.get(tokenizer, _MISSING)
        if encoder is _MISSING:
            plan = _plan_for(tokenizer)
            encoder = None if plan is None else _SegmentEncoder(plan)
            _segment_encoders[tokenizer] = encoder
    return encoder


def prebuild_in_background(tokenizer):
    """Build the tokenizer's plan and segment encoder on a daemon thread, so the
    first long prompts after startup do not pay for it. A prompt that arrives
    during the build waits for it under the build lock; if the build fails,
    the first prompt builds lazily as before. Returns the thread, or None."""
    if tokenizer is None or not envs.SGLANG_PARALLEL_PROMPT_ENCODE.get():
        return None

    def build():
        try:
            _segment_encoder_for(tokenizer)
        except Exception:
            logger.debug("parallel_prompt_encode: prebuild failed", exc_info=True)

    thread = threading.Thread(target=build, name="prompt-encode-prebuild", daemon=True)
    thread.start()
    return thread


def _usable_segment_encoder(tokenizer, encode_kwargs):
    """The tokenizer's segment encoder, or None where its ids could differ from
    ``tokenizer.encode(text, **encode_kwargs)``."""
    if not envs.SGLANG_PARALLEL_PROMPT_ENCODE.get() or (
        set(encode_kwargs) - {"add_special_tokens"}
    ):
        return None
    encoder = _segment_encoder_for(tokenizer)
    if (
        encoder is None
        or (encode_kwargs.get("add_special_tokens", True) and encoder.specials_added)
        or getattr(tokenizer, "split_special_tokens", False)
        or _samples_merges(encoder.backend)
    ):
        return None
    return encoder


def encode_prompt_segments(tokenizer, segments, encode_kwargs=None, cache_salt=None):
    """Drop-in for ``parallel_prompt_encode(tokenizer, "".join(segments),
    encode_kwargs)``. Cached ids are shared only between requests with the same
    ``cache_salt``."""
    encode_kwargs = encode_kwargs or {}
    encoder = _usable_segment_encoder(tokenizer, encode_kwargs)
    if encoder is None:
        return parallel_prompt_encode(tokenizer, "".join(segments), encode_kwargs)
    return encoder.encode(segments, cache_salt)


class _RenderedPromptChecks:
    """Per tokenizer: prompts that passed the check against a whole encode,
    checks still running, and whether a check found a mismatch (terminal)."""

    def __init__(self):
        self.passed = 0
        self.running = 0
        self.failed = False


_rendered_prompt_checks = weakref.WeakKeyDictionary()
_rendered_prompt_checks_lock = threading.Lock()
_RENDERED_PROMPT_CHECKS = 2


def _rendered_checks(tokenizer):
    """The tokenizer's check state; call with the lock held."""
    try:
        state = _rendered_prompt_checks.get(tokenizer)
        if state is None:
            state = _rendered_prompt_checks[tokenizer] = _RenderedPromptChecks()
        return state
    except TypeError:
        # Nothing can be remembered for it, so every prompt is checked.
        return _RenderedPromptChecks()


def _rendered_split_failed(tokenizer):
    try:
        state = _rendered_prompt_checks.get(tokenizer)
    except TypeError:
        return False
    return state is not None and state.failed


def encode_rendered_prompt(tokenizer, text, encode_kwargs=None, cache_salt=None):
    """Encode a rendered chat prompt, splitting it at added (special) tokens so
    the text between them goes through the segment cache; earlier turns of a
    conversation are then encoded once. Prompts are checked against a whole
    encode, and return it, until enough checks passed; a mismatch disables the
    split for good, also for prompts already split but not yet returned. After
    that no new check starts, and prompts return the whole encode until the
    checks still running finish, so no split ids are returned while a check
    that could still fail is in flight."""
    encode_kwargs = encode_kwargs or {}
    encoder = None
    if (
        isinstance(text, str)
        and len(text) >= envs.SGLANG_PARALLEL_PROMPT_ENCODE_MIN_CHARS.get()
        and not _rendered_split_failed(tokenizer)
    ):
        encoder = _usable_segment_encoder(tokenizer, encode_kwargs)
    # Only prompts the segment encoder serves are checked: a fallback compared
    # with the whole encode would count as a pass without testing the split.
    # The encoder gated here encodes them, so none can fall back after this.
    if encoder is None or encoder.pattern is None:
        return parallel_prompt_encode(tokenizer, text, encode_kwargs)
    ids = encoder.encode_text(text, cache_salt)
    with _rendered_prompt_checks_lock:
        state = _rendered_checks(tokenizer)
        check = not state.failed and state.passed < _RENDERED_PROMPT_CHECKS
        # Checks stop starting once enough passed, so `running` drains; the
        # split ids are used only when no check that could still fail runs.
        whole = state.failed or (not check and state.running > 0)
        if check:
            state.running += 1
    if whole:
        return tokenizer.encode(text, **encode_kwargs)
    if not check:
        return ids
    matched = None
    try:
        expected = tokenizer.encode(text, **encode_kwargs)
        matched = list(ids) == list(expected)
    finally:
        with _rendered_prompt_checks_lock:
            state.running -= 1
            if matched:
                state.passed += 1
            elif matched is False:
                state.failed = True
    if not matched:
        logger.error(
            "encode_rendered_prompt: segment-cached ids differ from a whole encode for %s; using the whole encode from now on.",
            type(tokenizer).__name__,
        )
    return expected
