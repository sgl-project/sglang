# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Route a HuggingFace tokenizer's encode/decode hot paths through gigatoken.

gigatoken (https://github.com/marcelroed/gigatoken) is a SIMD byte-pair encoder
that encodes a single document 50-200x faster than `tokenizers` does, which
matters here because SGLang tokenizes one prompt per request on the
TokenizerManager event loop and detokenizes every streamed token.

Rather than replace the tokenizer object with gigatoken's own HF-compatible
shim, this keeps the loaded tokenizer and overrides only the four methods on
the serving hot path, so the chat template, added-token bookkeeping, xgrammar's
TokenizerInfo, multimodal processors and `save_pretrained` all keep working
against the real implementation. Every call shape gigatoken has not been
verified byte-identical on -- sequence pairs, padding, truncation, tensor
returns, token_type_ids, `clean_up_tokenization_spaces` -- falls through to the
original method, so a request can be slower than it could be but is never
tokenized differently.
"""

from __future__ import annotations

import copy
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Encode kwargs the fast path understands. Anything else means a call shape
# gigatoken parity was not established for, so the call goes to transformers.
_KNOWN_ENCODE_KWARGS = frozenset(
    {
        "add_special_tokens",
        "return_attention_mask",
        "return_token_type_ids",
        "verbose",
    }
)

# Encode kwargs that must be absent or falsy to stay on the fast path: each one
# asks for row assembly (padding, truncation, tensors, segment ids) that
# gigatoken is not driving here.
_ENCODE_KWARGS_MUST_BE_FALSY = (
    "text_pair",
    "padding",
    "truncation",
    "max_length",
    "return_tensors",
    "return_token_type_ids",
    "is_split_into_words",
)

# Decode kwargs the fast path understands.
_KNOWN_DECODE_KWARGS = frozenset(
    {
        "skip_special_tokens",
        # PreTrainedTokenizerFast._decode ignores this outright (it is a slow
        # tokenizer concept), so gigatoken dropping it matches HF exactly.
        "spaces_between_special_tokens",
        "clean_up_tokenization_spaces",
    }
)

# The BatchEncoding keys the fast path produces. A tokenizer whose
# model_input_names asks for more (BERT-style `token_type_ids`) would silently
# lose a key, so those keep using transformers for `__call__`.
_FAST_PATH_MODEL_INPUTS = frozenset({"input_ids", "attention_mask"})

# Probes used to discover what the post-processor prepends/appends. Two
# unrelated strings that tokenize to different ids, so agreement between them
# is evidence the affixes are constant rather than content-dependent.
_AFFIX_PROBES = ("gigatoken affix probe", "下一个 42 probe")

# Probe for the decode check below: ASCII, a CJK run and a 4-byte emoji, so
# that cutting it at token boundaries produces incomplete UTF-8 sequences.
_DECODE_PROBE = "ok 日本語 🚀 end"

# Cache of generated subclasses, keyed by the class being accelerated: swapping
# __class__ per instance keeps the patch off every other tokenizer of the same
# type living in this process (e.g. a multimodal processor's own copy).
_accelerated_classes: dict[type, type] = {}


class GigatokenUnavailable(RuntimeError):
    """gigatoken cannot back this tokenizer; the caller should not use it."""


def _load_gigatoken(tokenizer):
    try:
        import gigatoken
    except ImportError as e:
        raise GigatokenUnavailable(
            "The gigatoken package is required when --tokenizer-backend=gigatoken. "
            "Install it with: pip install 'sglang[gigatoken]'"
        ) from e
    try:
        return gigatoken.Tokenizer(tokenizer)
    except Exception as e:
        raise GigatokenUnavailable(
            f"gigatoken cannot load this tokenizer ({type(tokenizer).__name__}): {e}"
        ) from e


def _find_subsequence(haystack: list[int], needle: list[int]) -> Optional[int]:
    """Index where `needle` occurs contiguously in `haystack`, else None."""
    first = needle[0]
    for i in range(len(haystack) - len(needle) + 1):
        if haystack[i] == first and haystack[i : i + len(needle)] == needle:
            return i
    return None


def _discover_special_affixes(tokenizer) -> Optional[tuple[list[int], list[int]]]:
    """The ids the post-processor puts before/after a single sequence.

    Read off the tokenizer itself rather than parsed out of tokenizer.json, by
    encoding a probe both ways and diffing. Returns None when the difference is
    not a constant prefix/suffix (the probes disagree, or the specials are
    interleaved), which disables only the add_special_tokens=True fast path.
    """
    affixes = set()
    for probe in _AFFIX_PROBES:
        bare = tokenizer.encode(probe, add_special_tokens=False)
        with_specials = tokenizer.encode(probe, add_special_tokens=True)
        if not bare or len(with_specials) < len(bare):
            return None
        start = _find_subsequence(with_specials, bare)
        if start is None:
            return None
        affixes.add(
            (
                tuple(with_specials[:start]),
                tuple(with_specials[start + len(bare) :]),
            )
        )
    if len(affixes) != 1:
        return None
    prefix, suffix = affixes.pop()
    return list(prefix), list(suffix)


def _decode_matches_on_partial_characters(tokenizer, backend) -> bool:
    """Whether gigatoken detokenizes every sub-range of a probe like HF does.

    DetokenizerManager decodes overlapping [surr:read] token windows, so a
    window can end mid-character and the two implementations have to agree on
    the resulting U+FFFD run. They do for byte-level BPE, but byte-fallback
    (SentencePiece) tokenizers emit one U+FFFD per undecodable byte in HF and
    one for the whole truncated sequence in gigatoken -- a difference that
    would show up in streamed chunk boundaries. Probing beats testing for
    byte_fallback because it checks the behavior we actually depend on, and
    stops gating decode if gigatoken's replacement policy ever changes.
    """
    ids = tokenizer.encode(_DECODE_PROBE, add_special_tokens=False)
    for read in range(1, len(ids) + 1):
        for surr in range(read):
            window = ids[surr:read]
            expected = tokenizer.decode(window, skip_special_tokens=False)
            got = backend.decode(window).decode("utf-8", errors="replace")
            if expected != got:
                return False
    return True


def _is_plain_text_input(text) -> bool:
    """A lone string or a list of strings -- no pairs, no pre-tokenized input."""
    if isinstance(text, str):
        return True
    if isinstance(text, (list, tuple)):
        return all(isinstance(item, str) for item in text)
    return False


def _is_plain_encode_call(kwargs: dict[str, Any]) -> bool:
    """Whether this is the bare "just tokenize it" shape gigatoken covers."""
    if not _KNOWN_ENCODE_KWARGS.issuperset(kwargs):
        return False
    return not any(kwargs.get(name) for name in _ENCODE_KWARGS_MUST_BE_FALSY)


class _GigatokenMethods:
    """encode/decode overrides mixed under the tokenizer's original class.

    Injected via `__class__` reassignment (see `accelerate_with_gigatoken`), so
    `super()` reaches the real transformers implementation and every fallback
    behaves exactly as an unaccelerated server would. Only `text` is accepted
    positionally: a call that passes anything else by position is handed
    straight to transformers rather than risking a different argument binding.
    """

    # Set on the instance by accelerate_with_gigatoken.
    _gigatoken: Any
    _gigatoken_prefix_ids: Optional[list[int]]
    _gigatoken_suffix_ids: Optional[list[int]]
    _gigatoken_special_ids: frozenset
    _gigatoken_plain_model_inputs: bool
    _gigatoken_decode_ok: bool

    def __call__(self, text=None, *args: Any, **kwargs: Any):
        affixes = self._gigatoken_affixes(kwargs)
        if (
            args
            or affixes is None
            or not self._gigatoken_plain_model_inputs
            or not _is_plain_text_input(text)
            or not _is_plain_encode_call(kwargs)
        ):
            return super().__call__(text, *args, **kwargs)

        from transformers.tokenization_utils_base import BatchEncoding

        prefix, suffix = affixes
        with_mask = kwargs.get("return_attention_mask") is not False
        if isinstance(text, str):
            ids = prefix + self._gigatoken.encode(text).tolist() + suffix
            data: dict[str, Any] = {"input_ids": ids}
            if with_mask:
                data["attention_mask"] = [1] * len(ids)
        else:
            rows = [
                prefix + row + suffix
                for row in self._gigatoken.encode_batch_list(list(text), parallel=False)
            ]
            data = {"input_ids": rows}
            if with_mask:
                data["attention_mask"] = [[1] * len(row) for row in rows]
        return BatchEncoding(data)

    def encode(self, text=None, *args: Any, **kwargs: Any) -> list[int]:
        affixes = self._gigatoken_affixes(kwargs)
        if (
            args
            or affixes is None
            or not isinstance(text, str)
            or not _is_plain_encode_call(kwargs)
        ):
            return super().encode(text, *args, **kwargs)
        prefix, suffix = affixes
        return prefix + self._gigatoken.encode(text).tolist() + suffix

    def decode(self, token_ids=None, *args: Any, **kwargs: Any) -> str:
        if args or not self._gigatoken_can_decode(kwargs):
            return super().decode(token_ids, *args, **kwargs)
        return self._gigatoken_decode(
            token_ids, kwargs.get("skip_special_tokens", False)
        )

    def __deepcopy__(self, memo):
        """Copy the tokenizer while sharing the gigatoken backend.

        The backend is a Rust object that cannot be pickled, and the default
        deepcopy reaches it through this instance's `__dict__`. That matters
        because `MultimodalProcessorExecutor` deepcopies the whole processor to
        get one clone per worker: without this, the copy raises and sglang
        silently falls back to synchronous multimodal processing.

        Sharing rather than rebuilding is safe — gigatoken's encode holds the
        GIL, so concurrent callers of one backend serialize — and it keeps the
        clones from each allocating another pretoken cache.
        """
        cls = type(self)
        clone = cls.__new__(cls)
        memo[id(self)] = clone
        for name, value in self.__dict__.items():
            object.__setattr__(
                clone,
                name,
                value if name == "_gigatoken" else copy.deepcopy(value, memo),
            )
        return clone

    def batch_decode(self, sequences=None, *args: Any, **kwargs: Any) -> list[str]:
        if args or not self._gigatoken_can_decode(kwargs):
            return super().batch_decode(sequences, *args, **kwargs)
        skip = kwargs.get("skip_special_tokens", False)
        return [self._gigatoken_decode(ids, skip) for ids in sequences]

    # -- helpers ------------------------------------------------------------

    def _gigatoken_affixes(
        self, kwargs: dict[str, Any]
    ) -> Optional[tuple[list[int], list[int]]]:
        """(prefix, suffix) to wrap the ids in, or None to fall back."""
        if not kwargs.get("add_special_tokens", True):
            return [], []
        if self._gigatoken_prefix_ids is None:
            return None
        return self._gigatoken_prefix_ids, self._gigatoken_suffix_ids

    def _gigatoken_can_decode(self, kwargs: dict[str, Any]) -> bool:
        if not self._gigatoken_decode_ok:
            return False
        if not _KNOWN_DECODE_KWARGS.issuperset(kwargs):
            return False
        # transformers v5 skips cleanup for BPE models, but honor the request by
        # falling back rather than reasoning about which model kind this is.
        cleanup = kwargs.get("clean_up_tokenization_spaces")
        if cleanup is None:
            cleanup = self.clean_up_tokenization_spaces
        return not cleanup

    def _gigatoken_decode(self, token_ids, skip_special_tokens: bool) -> str:
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        if skip_special_tokens and self._gigatoken_special_ids:
            special = self._gigatoken_special_ids
            token_ids = [i for i in token_ids if i not in special]
        # Without filtering the sequence goes through untouched: numpy arrays
        # and array('q') buffers are borrowed by the Rust decoder, so no
        # per-token Python int is built.
        return self._gigatoken.decode(token_ids).decode("utf-8", errors="replace")


def _accelerated_class(base: type) -> type:
    cls = _accelerated_classes.get(base)
    if cls is None:
        cls = type(f"Gigatoken{base.__name__}", (_GigatokenMethods, base), {})
        _accelerated_classes[base] = cls
    return cls


def accelerate_with_gigatoken(tokenizer):
    """Back `tokenizer`'s encode/decode hot paths with gigatoken, in place.

    Returns the same object with its class swapped for a generated subclass, so
    `isinstance` checks, attribute assignment (e.g. `tokenizer.chat_template =
    ...`) and every method not overridden behave exactly as before. Raises
    GigatokenUnavailable if gigatoken is missing or cannot load this tokenizer.
    """
    if isinstance(tokenizer, _GigatokenMethods):
        return tokenizer

    base_name = type(tokenizer).__name__
    backend = _load_gigatoken(tokenizer)
    affixes = _discover_special_affixes(tokenizer)
    if affixes is None:
        logger.info(
            "gigatoken: %s adds special tokens in a shape that is not a constant "
            "prefix/suffix; add_special_tokens=True will use the HuggingFace path.",
            base_name,
        )
        prefix_ids = suffix_ids = None
    else:
        prefix_ids, suffix_ids = affixes

    plain_model_inputs = _FAST_PATH_MODEL_INPUTS.issuperset(tokenizer.model_input_names)
    if not plain_model_inputs:
        logger.info(
            "gigatoken: %s requires model inputs %s beyond input_ids/attention_mask; "
            "encoding will use the HuggingFace path, decoding still uses gigatoken.",
            base_name,
            sorted(set(tokenizer.model_input_names) - _FAST_PATH_MODEL_INPUTS),
        )

    decode_ok = _decode_matches_on_partial_characters(tokenizer, backend.backend)
    if not decode_ok:
        logger.info(
            "gigatoken: %s detokenizes truncated multi-byte characters differently "
            "from HuggingFace (byte-fallback vocabularies do); decoding will use the "
            "HuggingFace path, encoding still uses gigatoken.",
            base_name,
        )

    tokenizer._gigatoken = backend
    tokenizer._gigatoken_prefix_ids = prefix_ids
    tokenizer._gigatoken_suffix_ids = suffix_ids
    tokenizer._gigatoken_plain_model_inputs = plain_model_inputs
    tokenizer._gigatoken_decode_ok = decode_ok
    tokenizer._gigatoken_special_ids = frozenset(
        token_id
        for token_id, token in tokenizer.added_tokens_decoder.items()
        if token.special
    )
    tokenizer.__class__ = _accelerated_class(type(tokenizer))

    logger.info(
        "gigatoken backend enabled for %s (encode=%s, decode=%s, "
        "special affixes: prefix=%s suffix=%s)",
        base_name,
        "gigatoken" if plain_model_inputs else "huggingface",
        "gigatoken" if decode_ok else "huggingface",
        prefix_ids,
        suffix_ids,
    )
    return tokenizer
