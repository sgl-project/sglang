import logging

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def patch_tokenizer(tokenizer):
    if not envs.SGLANG_PATCH_TOKENIZER.get():
        return tokenizer

    if _is_kimi_tokenizer(tokenizer):
        logger.info(
            f"Applying special tokens cache patch for Kimi tokenizer: {type(tokenizer).__name__}"
        )
        return _patch_special_tokens_cache(tokenizer)

    return tokenizer


def unpatch_tokenizer(tokenizer):
    cls = type(tokenizer)

    if not getattr(cls, "_sglang_special_tokens_patched", False):
        return tokenizer

    cls.all_special_tokens = property(cls._original_all_special_tokens)
    cls.all_special_ids = property(cls._original_all_special_ids)
    cls.add_special_tokens = cls._original_add_special_tokens
    cls.add_tokens = cls._original_add_tokens

    del cls._original_all_special_tokens
    del cls._original_all_special_ids
    del cls._original_add_special_tokens
    del cls._original_add_tokens
    del cls._sglang_special_tokens_patched

    if hasattr(tokenizer, "_sglang_cached_special_tokens"):
        del tokenizer._sglang_cached_special_tokens
    if hasattr(tokenizer, "_sglang_cached_special_ids"):
        del tokenizer._sglang_cached_special_ids

    logger.info(f"Unpatched special tokens cache for {cls.__name__}")
    return tokenizer


def _is_kimi_tokenizer(tokenizer):
    name = getattr(tokenizer, "name_or_path", "") or ""
    return "kimi" in name.lower()


def _patch_special_tokens_cache(tokenizer):
    cls = type(tokenizer)

    if getattr(cls, "_sglang_special_tokens_patched", False):
        return tokenizer

    cls._original_all_special_tokens = cls.all_special_tokens.fget
    cls._original_all_special_ids = cls.all_special_ids.fget
    cls._original_add_special_tokens = cls.add_special_tokens
    cls._original_add_tokens = cls.add_tokens

    @property
    def patched_all_special_tokens(self):
        if getattr(self, "_sglang_cached_special_tokens", None) is None:
            self._sglang_cached_special_tokens = cls._original_all_special_tokens(self)
        return self._sglang_cached_special_tokens

    @property
    def patched_all_special_ids(self):
        if getattr(self, "_sglang_cached_special_ids", None) is None:
            self._sglang_cached_special_ids = cls._original_all_special_ids(self)
        return self._sglang_cached_special_ids

    def patched_add_special_tokens(self, *args, **kwargs):
        assert (
            False
        ), "Cannot modify special tokens after patch. Call unpatch_tokenizer first."

    def patched_add_tokens(self, new_tokens, special_tokens=False):
        assert (
            not special_tokens
        ), "Cannot add special tokens after patch. Call unpatch_tokenizer first."
        return cls._original_add_tokens(self, new_tokens, special_tokens=False)

    cls.all_special_tokens = patched_all_special_tokens
    cls.all_special_ids = patched_all_special_ids
    cls.add_special_tokens = patched_add_special_tokens
    cls.add_tokens = patched_add_tokens
    cls._sglang_special_tokens_patched = True

    return tokenizer

