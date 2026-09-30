import inspect
import logging

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)


def patch_tokenizer(tokenizer):
    if not envs.SGLANG_PATCH_TOKENIZER.get():
        return tokenizer

    if _is_kimi_tiktoken_tokenizer(tokenizer):
        logger.info(
            f"Applying special tokens cache patch for Kimi tokenizer: {type(tokenizer)}"
        )
        return _SpecialTokensCachePatcher.patch(tokenizer)

    if _needs_pad_padding_side_shim(tokenizer):
        logger.info(
            f"Applying _pad(padding_side=...) compat shim for {type(tokenizer)} "
            "(custom remote-code tokenizer's _pad() override predates "
            "transformers passing padding_side to it)"
        )
        return _PadPaddingSideShim.patch(tokenizer)

    return tokenizer


def _needs_pad_padding_side_shim(tokenizer) -> bool:
    """
    Some custom remote-code tokenizers (e.g. zai-org/chatglm2-6b's
    ChatGLMTokenizer, unchanged for years -- see tokenization_chatglm.py's
    own _pad() override) declare a fixed _pad() signature with no
    padding_side parameter and no **kwargs catch-all, predating a newer
    transformers version's PreTrainedTokenizerBase.pad() always forwarding
    padding_side down to _pad(). Every tokenizer.encode()/pad() call then
    raises "TypeError: <Tokenizer>._pad() got an unexpected keyword argument
    'padding_side'" -- not an sglang bug, a real installed-transformers-
    version vs custom-tokenizer mismatch that breaks EVERY generation
    request for that model regardless of memory/backend settings (confirmed:
    SGLANGT-1689, chatglm2-6b). Detected structurally via inspect rather
    than a name/class check, so any other custom tokenizer with the same
    stale-signature bug is covered too, not just this one model.
    """
    pad_fn = getattr(type(tokenizer), "_pad", None)
    if pad_fn is None:
        return False
    try:
        sig = inspect.signature(pad_fn)
    except (TypeError, ValueError):
        return False
    params = sig.parameters
    if "padding_side" in params:
        return False
    return not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values())


def unpatch_tokenizer(tokenizer):
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


class _PadPaddingSideShim:
    """
    Wraps a stale custom _pad() override (see _needs_pad_padding_side_shim's
    docstring) so it tolerates the padding_side kwarg newer transformers
    versions always pass -- dropped before delegating to the original
    _pad(), which already reads self.padding_side directly (the standard
    HF tokenizer instance attribute), so behavior is unchanged from before
    transformers started passing it explicitly.
    """

    _PATCHED_FLAG = "_sglang_pad_padding_side_patched"

    @classmethod
    def patch(cls, tokenizer):
        tokenizer_cls = type(tokenizer)

        if getattr(tokenizer_cls, cls._PATCHED_FLAG, False):
            return tokenizer

        original_pad = tokenizer_cls._pad

        def patched_pad(self, *args, **kwargs):
            kwargs.pop("padding_side", None)
            return original_pad(self, *args, **kwargs)

        tokenizer_cls._original_pad = original_pad
        tokenizer_cls._pad = patched_pad
        setattr(tokenizer_cls, cls._PATCHED_FLAG, True)

        return tokenizer


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


def _make_cached_property(cache_attr, original_fn):
    @property
    def cached_prop(self):
        if getattr(self, cache_attr, None) is None:
            setattr(self, cache_attr, original_fn(self))
        return getattr(self, cache_attr)

    return cached_prop
