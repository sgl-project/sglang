import logging
import re
from typing import List

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def patch_tokenizer(tokenizer):
    if not envs.SGLANG_PATCH_TOKENIZER.get():
        return tokenizer

    if _is_kimi_tiktoken_tokenizer(tokenizer):
        logger.info(
            f"Applying special tokens cache patch for Kimi tokenizer: {type(tokenizer)}"
        )
        _SpecialTokensCachePatcher.patch(tokenizer)
        logger.info(
            f"Applying encode-piece fast path patch for Kimi tokenizer: {type(tokenizer)}"
        )
        return _EncodePieceFastPathPatcher.patch(tokenizer)

    return tokenizer


def unpatch_tokenizer(tokenizer):
    _EncodePieceFastPathPatcher.unpatch(tokenizer)
    return _SpecialTokensCachePatcher.unpatch(tokenizer)


def _is_kimi_tiktoken_tokenizer(tokenizer):
    cls = type(tokenizer)
    class_name = cls.__name__
    module_name = cls.__module__ or ""
    return class_name == "TikTokenTokenizer" and "tokenization_kimi" in module_name


def decode_without_hf_kwargs(tokenizer, token_ids, skip_special_tokens):
    if skip_special_tokens:
        special_ids = getattr(tokenizer, "all_special_ids_set", None)
        if special_ids is None:
            special_ids = getattr(tokenizer, "all_special_ids", None)
        if special_ids is not None:
            special_ids_set = set(special_ids)
            token_ids = [tid for tid in token_ids if tid not in special_ids_set]
    return tokenizer.decode(token_ids)


class _SpecialTokensCachePatcher:
    _PATCHED_FLAG = "_sglang_special_tokens_patched"
    _CACHED_TOKENS_ATTR = "_sglang_cached_special_tokens"
    _CACHED_IDS_ATTR = "_sglang_cached_special_ids"

    @classmethod
    def patch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls._original_all_special_tokens = (
            tokenizer_cls.all_special_tokens.fget
        )
        tokenizer_cls._original_all_special_ids = tokenizer_cls.all_special_ids.fget
        tokenizer_cls._original_add_special_tokens = tokenizer_cls.add_special_tokens
        tokenizer_cls._original_add_tokens = tokenizer_cls.add_tokens

        patched_all_special_tokens = _make_cached_property(
            cls._CACHED_TOKENS_ATTR, tokenizer_cls._original_all_special_tokens
        )
        patched_all_special_ids = _make_cached_property(
            cls._CACHED_IDS_ATTR, tokenizer_cls._original_all_special_ids
        )

        def patched_add_special_tokens(self, *args, **kwargs):
            assert False, (
                "Cannot modify special tokens after patch. Call unpatch_tokenizer first."
            )

        def patched_add_tokens(self, new_tokens, special_tokens=False):
            assert not special_tokens, (
                "Cannot add special tokens after patch. Call unpatch_tokenizer first."
            )
            return tokenizer_cls._original_add_tokens(
                self, new_tokens, special_tokens=False
            )

        tokenizer_cls.all_special_tokens = patched_all_special_tokens
        tokenizer_cls.all_special_ids = patched_all_special_ids
        tokenizer_cls.add_special_tokens = patched_add_special_tokens
        tokenizer_cls.add_tokens = patched_add_tokens
        setattr(tokenizer_cls, cls._PATCHED_FLAG, True)

        return tokenizer

    @classmethod
    def unpatch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if not getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls.all_special_tokens = property(
            tokenizer_cls._original_all_special_tokens
        )
        tokenizer_cls.all_special_ids = property(
            tokenizer_cls._original_all_special_ids
        )
        tokenizer_cls.add_special_tokens = tokenizer_cls._original_add_special_tokens
        tokenizer_cls.add_tokens = tokenizer_cls._original_add_tokens

        del tokenizer_cls._original_all_special_tokens
        del tokenizer_cls._original_all_special_ids
        del tokenizer_cls._original_add_special_tokens
        del tokenizer_cls._original_add_tokens
        delattr(tokenizer_cls, cls._PATCHED_FLAG)

        for attr in [cls._CACHED_TOKENS_ATTR, cls._CACHED_IDS_ATTR]:
            if hasattr(tokenizer, attr):
                delattr(tokenizer, attr)

        logger.info(f"Unpatched special tokens cache for {tokenizer_cls.__name__}")
        return tokenizer


class _EncodePieceFastPathPatcher:
    """Short-circuit ``TikTokenTokenizer._encode_text_piece`` for the two segment
    shapes that dominate Kimi-K3 chat encoding.

    ``encoding_k3.build_chat_segments`` renders a conversation into tens of
    thousands of tiny segments (one per control token, tag name, or tool-call
    attribute) and ``_encode_text_piece`` is called once per segment.  Two costs
    make that path CPU-bound on the API server for agentic conversations:

    * control segments call ``tiktoken.Encoding.encode(allowed_special="all")``,
      which rebuilds the allowed-special set on every call -- ~30us per call
      regardless of text length -- for what is a dictionary lookup;
    * text segments run a pure-Python per-character splitter that is a no-op
      for anything shorter than ``MAX_NO_WHITESPACES_CHARS``.

    The patched method keeps the original as the fallback, so token ids are
    unchanged: a special-token literal inside a text segment, a control segment
    that is not exactly one special token, and long text all take the original
    path.
    """

    _PATCHED_FLAG = "_sglang_encode_piece_patched"
    # Mirrors MAX_NO_WHITESPACES_CHARS in tokenization_kimi.py: below this length
    # the original splitter yields the input unchanged, so skipping it is exact.
    _MAX_UNSPLIT_TEXT_CHARS = 25_000

    @classmethod
    def patch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        original_encode_text_piece = tokenizer_cls._encode_text_piece
        max_unsplit_text_chars = cls._MAX_UNSPLIT_TEXT_CHARS

        def patched_encode_text_piece(
            self, text: str, allow_special_tokens: bool = True
        ) -> List[int]:
            if allow_special_tokens:
                special_id = self.special_tokens.get(text)
                if special_id is not None:
                    return [special_id]
                return original_encode_text_piece(self, text, allow_special_tokens)
            if len(text) <= max_unsplit_text_chars and not _special_literal_regex(
                self
            ).search(text):
                # disallowed_special=() encodes special literals as plain text,
                # so with none present encode() == encode_ordinary().
                return self.model._core_bpe.encode_ordinary(text)
            return original_encode_text_piece(self, text, allow_special_tokens)

        tokenizer_cls._original_encode_text_piece = original_encode_text_piece
        tokenizer_cls._encode_text_piece = patched_encode_text_piece
        setattr(tokenizer_cls, cls._PATCHED_FLAG, True)
        return tokenizer

    @classmethod
    def unpatch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if not getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        tokenizer_cls._encode_text_piece = tokenizer_cls._original_encode_text_piece
        del tokenizer_cls._original_encode_text_piece
        delattr(tokenizer_cls, cls._PATCHED_FLAG)
        if hasattr(tokenizer, _SPECIAL_LITERAL_REGEX_ATTR):
            delattr(tokenizer, _SPECIAL_LITERAL_REGEX_ATTR)

        logger.info(f"Unpatched encode-piece fast path for {tokenizer_cls.__name__}")
        return tokenizer


_SPECIAL_LITERAL_REGEX_ATTR = "_sglang_special_literal_regex"


def _special_literal_regex(tokenizer):
    regex = getattr(tokenizer, _SPECIAL_LITERAL_REGEX_ATTR, None)
    if regex is None:
        regex = re.compile(
            "|".join(
                re.escape(token)
                for token in sorted(tokenizer.special_tokens, key=len, reverse=True)
            )
        )
        setattr(tokenizer, _SPECIAL_LITERAL_REGEX_ATTR, regex)
    return regex


def _make_cached_property(cache_attr, original_fn):
    @property
    def cached_prop(self):
        if getattr(self, cache_attr, None) is None:
            setattr(self, cache_attr, original_fn(self))
        return getattr(self, cache_attr)

    return cached_prop
