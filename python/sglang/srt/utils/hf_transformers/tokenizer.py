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
"""Tokenizer loading utilities."""

import json
import logging
import warnings
from pathlib import Path
from typing import Optional, Union

from transformers import (
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)

from sglang.srt.connector import create_remote_connector
from sglang.srt.utils import is_remote_url, logger
from sglang.srt.utils.patch_tokenizer import patch_tokenizer
from sglang.srt.utils.runai_utils import ObjectStorageModel, is_runai_obj_uri

from ..hf_transformers_patches import _ensure_gguf_version
from .common import (
    _resolve_local_or_cached_file,
    attach_additional_stop_token_ids,
    check_gguf_file,
)
from .mistral_utils import (
    _MISTRAL_TOKENIZER_REDIRECTS,
    patch_mistral_common_tokenizer,
    retry_without_mistral_common_kwargs,
)

# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"

# Class name used by transformers v5 when no tokenizer mapping exists for a model_type.
_TOKENIZERS_BACKEND = "TokenizersBackend"


def _load_tokenizer_by_declared_class(tokenizer_name, *args, **kwargs):
    """Load tokenizer by the class declared in tokenizer_config.json.

    AutoTokenizer resolves to TokenizersBackend when the model's config
    model_type has no tokenizer class mapping (e.g. deepseek_vl_v2), even
    though tokenizer_config.json declares a standard class like
    LlamaTokenizerFast.  Returns None if it cannot improve on AutoTokenizer.
    """
    import transformers

    try:
        revision = kwargs.get("revision") or kwargs.get("tokenizer_revision")
        config_file = _resolve_local_or_cached_file(
            tokenizer_name, "tokenizer_config.json", revision
        )
        with open(config_file) as f:
            tok_config = json.load(f)
        tok_class_name = tok_config.get("tokenizer_class")
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError) as e:
        logger.debug(
            "Failed to read tokenizer_config.json for %s: %s", tokenizer_name, e
        )
        return None

    if not tok_class_name:
        return None

    # Skip base classes that don't implement required methods (e.g. get_vocab)
    if tok_class_name in ("PreTrainedTokenizer", "PreTrainedTokenizerBase"):
        return None

    tok_cls = getattr(transformers, tok_class_name, None)
    if tok_cls is None and kwargs.get("trust_remote_code"):
        # Class not in transformers — try loading via auto_map.
        try:
            auto_map = tok_config.get("auto_map", {})
            auto_tok_ref = auto_map.get("AutoTokenizer")
            if isinstance(auto_tok_ref, (list, tuple)):
                auto_tok_ref = auto_tok_ref[0]
            if auto_tok_ref:
                from transformers.dynamic_module_utils import (
                    get_class_from_dynamic_module,
                )

                tok_cls = get_class_from_dynamic_module(
                    auto_tok_ref,
                    tokenizer_name,
                    code_revision=revision,
                )
        except (OSError, ImportError, ValueError, RuntimeError) as e:
            logger.debug("Dynamic module lookup for %s failed: %s", tok_class_name, e)
    if tok_cls is None:
        return None

    logger.info(
        "Loading tokenizer for %s directly as %s (bypassing AutoTokenizer)",
        tokenizer_name,
        tok_class_name,
    )
    try:
        return tok_cls.from_pretrained(tokenizer_name, *args, **kwargs)
    except (OSError, ValueError, TypeError, ImportError) as e:
        logger.warning(
            "Direct load as %s failed for %s: %s. "
            "Falling back to AutoTokenizer result.",
            tok_class_name,
            tokenizer_name,
            e,
        )
        return None


# Filter warnings like: https://github.com/sgl-project/sglang/issues/8082
class TokenizerWarningsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Calling super().encode with" not in record.getMessage()


# ---------------------------------------------------------------------------
# Helpers for get_tokenizer
# ---------------------------------------------------------------------------


def _resolve_tokenizer_name(tokenizer_name, kwargs):
    """Resolve special name formats (GGUF, remote URLs, etc.) to a local path.

    May mutate *kwargs* (e.g. to add ``gguf_file``).
    """
    tokenizer_name = _MISTRAL_TOKENIZER_REDIRECTS.get(tokenizer_name, tokenizer_name)

    if check_gguf_file(tokenizer_name):
        _ensure_gguf_version()
        kwargs["gguf_file"] = tokenizer_name
        tokenizer_name = Path(tokenizer_name).parent

    if is_runai_obj_uri(tokenizer_name):
        tokenizer_name = ObjectStorageModel.get_path(tokenizer_name)

    if is_remote_url(tokenizer_name):
        # BaseConnector implements __del__() to clean up the local dir.
        # Since config files need to exist all the time, so we DO NOT use
        # with statement to avoid closing the client.
        client = create_remote_connector(tokenizer_name)
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
        tokenizer_name = client.get_local_dir()

    return tokenizer_name


def _auto_tokenizer_from_pretrained(tokenizer_name, *args, **common_kwargs):
    """Call ``AutoTokenizer.from_pretrained`` with error handling."""
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, *args, **common_kwargs
        )
        logging.getLogger(tokenizer.__class__.__module__).addFilter(
            TokenizerWarningsFilter()
        )
        return tokenizer
    except TypeError as e:
        err_msg = (
            "Failed to load the tokenizer. If you are using a LLaMA V1 model "
            f"consider using '{_FAST_LLAMA_TOKENIZER}' instead of the "
            "original tokenizer."
        )
        raise RuntimeError(err_msg) from e
    except ValueError as e:
        # MistralCommon tokenizers reject standard HF kwargs like
        # trust_remote_code, use_fast etc. Retry without them.
        if "are not supported by" in str(e) and "MistralCommon" in str(e):
            return retry_without_mistral_common_kwargs(
                tokenizer_name, *args, **common_kwargs
            )
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        if not common_kwargs.get("trust_remote_code") and (
            "does not exist or is not currently imported." in str(e)
            or "requires you to execute the tokenizer file" in str(e)
        ):
            err_msg = (
                "Failed to load the tokenizer. If the tokenizer is a custom "
                "tokenizer not yet available in the HuggingFace transformers "
                "library, consider setting `trust_remote_code=True` in LLM "
                "or using the `--trust-remote-code` flag in the CLI."
            )
            raise RuntimeError(err_msg) from e
        raise


def _resolve_tokenizers_backend(tokenizer_name, *args, **common_kwargs):
    """Resolve generic ``TokenizersBackend`` to a proper tokenizer class.

    In transformers v5, ``AutoTokenizer`` falls back to ``TokenizersBackend``
    when the model_type has no tokenizer mapping.  This retries with
    ``use_fast=False``, then attempts loading by the class declared in
    ``tokenizer_config.json``.  May still return a ``TokenizersBackend``
    if all retries fail (with a warning).
    """
    logger.warning(
        "Tokenizer loaded as generic TokenizersBackend for %s, "
        "retrying with use_fast=False",
        tokenizer_name,
    )
    common_kwargs = {**common_kwargs, "use_fast": False}
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, *args, **common_kwargs
        )
    except (ValueError, TypeError, OSError, ImportError, RuntimeError) as e:
        raise RuntimeError(
            f"Retry with use_fast=False for {tokenizer_name} also failed "
            f"(initial load returned TokenizersBackend): {e}"
        ) from e

    if type(tokenizer).__name__ == _TOKENIZERS_BACKEND:
        tokenizer = (
            _load_tokenizer_by_declared_class(tokenizer_name, *args, **common_kwargs)
            or tokenizer
        )

    if type(tokenizer).__name__ == _TOKENIZERS_BACKEND:
        if common_kwargs.get("trust_remote_code"):
            logger.warning(
                "Tokenizer for %s is still TokenizersBackend after retries "
                "with --trust-remote-code. Model-specific tokenizer attributes "
                "may be missing.",
                tokenizer_name,
            )
        else:
            logger.warning(
                "Tokenizer for %s loaded as generic TokenizersBackend. "
                "Set --trust-remote-code to load the model-specific tokenizer.",
                tokenizer_name,
            )

    return tokenizer


# ---------------------------------------------------------------------------
# Post-load fixups
# ---------------------------------------------------------------------------


def _fix_v5_tokenizer_components(tokenizer, model_name_or_path, revision=None):
    """Fix pre_tokenizer/decoder when a v5 tokenizer class overwrites them.

    In transformers v5, some tokenizer classes (e.g. LlamaTokenizer) have a
    custom __init__ that rebuilds the pre_tokenizer and decoder from scratch
    with class-specific components, discarding the originals from tokenizer.json.
    This breaks models that specify LlamaTokenizerFast but actually use a
    different tokenizer architecture (e.g. DeepSeek-V3.2 uses ByteLevel).

    Detects the mismatch by comparing against the raw tokenizer.json and
    restores the original components when they differ.
    """
    backend = getattr(tokenizer, "_tokenizer", None)
    if backend is None:
        return

    try:
        from tokenizers import Tokenizer as RawTokenizer

        tok_file = _resolve_local_or_cached_file(
            model_name_or_path, "tokenizer.json", revision
        )
        raw = RawTokenizer.from_file(tok_file)
    except FileNotFoundError:
        return
    except (OSError, ValueError, RuntimeError) as e:
        logger.warning(
            "_fix_v5_tokenizer_components: unexpected error loading tokenizer.json "
            "for %s, v5 component fix will not be applied: %s",
            model_name_or_path,
            e,
        )
        return

    raw_pre = type(raw.pre_tokenizer).__name__ if raw.pre_tokenizer else None
    loaded_pre = type(backend.pre_tokenizer).__name__ if backend.pre_tokenizer else None

    if raw_pre and loaded_pre and raw_pre != loaded_pre:
        logger.info(
            "Fixing v5 tokenizer component mismatch for %s: "
            "pre_tokenizer %s -> %s, decoder %s -> %s",
            model_name_or_path,
            loaded_pre,
            raw_pre,
            type(backend.decoder).__name__ if backend.decoder else None,
            type(raw.decoder).__name__ if raw.decoder else None,
        )
        backend.pre_tokenizer = raw.pre_tokenizer
        backend.decoder = raw.decoder


def _fix_v5_add_bos_eos_token(tokenizer, model_name_or_path, revision=None):
    """Restore add_bos_token/add_eos_token stripped by transformers v5.

    In transformers v5, _from_pretrained() strips add_bos_token and
    add_eos_token from init kwargs when a tokenizer.json file is present,
    assuming the tokenizer.json post-processor handles BOS/EOS addition.
    However, many models (e.g. DeepSeek-V3) have a tokenizer.json whose
    post-processor does NOT add BOS/EOS, and rely on the add_bos_token flag
    from tokenizer_config.json instead. This causes silent accuracy regressions.

    This function reads the tokenizer_config.json and restores the values,
    but only for tokenizer classes that actually supported these flags in v4.
    Classes like Qwen2Tokenizer did not support add_bos_token/add_eos_token
    in v4, so restoring them would change behavior.
    """
    # In transformers v4, only certain tokenizer classes supported
    # add_bos_token / add_eos_token as init parameters.  Restoring these
    # flags for classes that never supported them (e.g. Qwen2Tokenizer)
    # would incorrectly change tokenization behavior.
    _V4_CLASSES_WITH_BOS_EOS_FLAGS = frozenset(
        {
            "LlamaTokenizer",
            "LlamaTokenizerFast",
            "CodeLlamaTokenizer",
            "CodeLlamaTokenizerFast",
            "GemmaTokenizer",
            "GemmaTokenizerFast",
            "CohereTokenizerFast",
        }
    )

    try:
        config_file = _resolve_local_or_cached_file(
            model_name_or_path, "tokenizer_config.json", revision
        )
        with open(config_file) as f:
            config = json.load(f)
    except FileNotFoundError:
        return
    except (OSError, json.JSONDecodeError, ValueError) as e:
        logger.warning(
            "_fix_v5_add_bos_eos_token: failed to read tokenizer_config.json "
            "for %s, BOS/EOS token restoration will not be applied: %s",
            model_name_or_path,
            e,
        )
        return

    tokenizer_class = config.get("tokenizer_class", "")
    if tokenizer_class not in _V4_CLASSES_WITH_BOS_EOS_FLAGS:
        logger.debug(
            "_fix_v5_add_bos_eos_token: skipping %s (tokenizer_class=%s "
            "did not support add_bos/eos_token in v4)",
            model_name_or_path,
            tokenizer_class,
        )
        return

    # In v4, Llama/Gemma tokenizers defaulted add_bos_token=True.
    # When the config omits the key or has null, use the v4 default so that
    # update_post_processor() doesn't drop BOS/EOS that was there before.
    _V4_DEFAULTS = {"add_bos_token": True, "add_eos_token": False}

    changed = False
    for attr in ("add_bos_token", "add_eos_token"):
        config_val = config.get(attr)
        if config_val is None:
            # Key missing or null -> use v4 default for this tokenizer class
            config_val = _V4_DEFAULTS.get(attr, False)
        # Fast tokenizers in v4 used tokenizer.json post-processor for EOS —
        # the add_eos_token Python attribute was set but the post-processor
        # came from tokenizer.json, not from the attribute.  In v5, the flag is
        # stripped and both sglang and HF reference end up with add_eos_token=False.
        # Restoring add_eos_token for fast tokenizers makes sglang diverge from
        # the HF reference, breaking embedding models like e5-mistral-7b-instruct.
        if attr == "add_eos_token" and isinstance(tokenizer, PreTrainedTokenizerFast):
            config_val = _V4_DEFAULTS["add_eos_token"]  # False
        current_val = getattr(tokenizer, attr, None)
        if current_val != config_val:
            logger.info(
                "Restoring %s=%s for %s (was %s after v5 loading)",
                attr,
                config_val,
                model_name_or_path,
                current_val,
            )
            # Set the private backing attribute (not the property) because
            # transformers tokenizers expose add_bos/eos_token as properties
            # that read from the underscore-prefixed attribute.
            setattr(tokenizer, f"_{attr}", config_val)
            changed = True

    # Rebuild the post-processor so it respects the restored flags
    if changed and hasattr(tokenizer, "update_post_processor"):
        tokenizer.update_post_processor()


def _fix_special_tokens_pattern(tokenizer):
    """Fix https://github.com/huggingface/transformers/pull/42563 which defaults
    special_tokens_pattern to "cls_sep", inserting None into token IDs when
    cls_token/sep_token are undefined (e.g. Kimi-VL's TikTokenTokenizer).
    """
    pattern = getattr(tokenizer, "special_tokens_pattern", None)
    if pattern == "cls_sep" and (
        tokenizer.cls_token_id is None or tokenizer.sep_token_id is None
    ):
        tokenizer.special_tokens_pattern = "none"


def _apply_post_load_fixes(tokenizer, tokenizer_name, revision):
    """Apply all post-load patches and return the final tokenizer."""
    _fix_v5_tokenizer_components(tokenizer, tokenizer_name, revision)
    _fix_v5_add_bos_eos_token(tokenizer, tokenizer_name, revision)

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        warnings.warn(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    patch_mistral_common_tokenizer(tokenizer)
    _fix_special_tokens_pattern(tokenizer)
    attach_additional_stop_token_ids(tokenizer)
    return patch_tokenizer(tokenizer)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------


_fastokens_patched = False


def _ensure_fastokens_patched():
    """Monkey-patch transformers to use the fastokens backend (once)."""
    global _fastokens_patched
    if _fastokens_patched:
        return
    try:
        import fastokens
    except ImportError:
        raise ImportError(
            "The fastokens package is required when --tokenizer-backend=fastokens. "
            "Install it with: pip install 'sglang[fastokens]'"
        ) from None

    fastokens.patch_transformers()
    _fastokens_patched = True
    logger.info("fastokens backend enabled - transformers patched successfully")


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    tokenizer_backend: str = "huggingface",
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    # Tiktoken format has its own backend — no fastokens patching needed.
    if tokenizer_name.endswith(".json"):
        from sglang.srt.tokenizer.tiktoken_tokenizer import TiktokenTokenizer

        return TiktokenTokenizer(tokenizer_name)

    if tokenizer_backend == "fastokens":
        _ensure_fastokens_patched()

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False
    elif tokenizer_mode == "auto":
        # Transformers v5 AutoTokenizer ignores use_fast (always fast), but
        # some code paths pass kwargs to non-AutoTokenizer loaders where
        # use_fast still matters. Set explicitly for those fallback paths.
        if "use_fast" not in kwargs:
            kwargs["use_fast"] = True

    tokenizer_name = _resolve_tokenizer_name(tokenizer_name, kwargs)

    common_kwargs = dict(
        trust_remote_code=trust_remote_code,
        tokenizer_revision=tokenizer_revision,
        clean_up_tokenization_spaces=False,
        **kwargs,
    )

    try:
        tokenizer = _auto_tokenizer_from_pretrained(
            tokenizer_name, *args, **common_kwargs
        )

        # With fastokens, the patched TokenizersBackend.from_pretrained already
        # returned a tokenizer whose backend is a fastokens shim. Re-resolving via
        # the declared class (e.g. Qwen2Tokenizer) would discard that work.
        if (
            type(tokenizer).__name__ == _TOKENIZERS_BACKEND
            and tokenizer_backend != "fastokens"
        ):
            tokenizer = _resolve_tokenizers_backend(
                tokenizer_name, *args, **common_kwargs
            )

        return _apply_post_load_fixes(tokenizer, tokenizer_name, tokenizer_revision)
    except Exception as e:
        if tokenizer_backend == "fastokens":
            raise RuntimeError(
                f"fastokens failed to load tokenizer for {tokenizer_name!r}. "
                f"This model's tokenizer may not be supported by fastokens — "
                f"see https://github.com/crusoecloud/fastokens. "
                f"Re-run without --tokenizer-backend=fastokens to use the default backend."
            ) from e
        raise


# ---------------------------------------------------------------------------
# Exported helpers (used by processor.py, etc.)
# ---------------------------------------------------------------------------


def _fix_added_tokens_encoding(tokenizer):
    """Ensure special tokens encode as single tokens in transformers v5.

    Some model tokenizers (e.g. MiniCPM-V-4) define special tokens like <image>,
    <slice> as attributes on the tokenizer class with corresponding IDs in the
    vocabulary (via tokenizer.json's added_tokens). In transformers v5, these
    tokens may not appear in get_added_vocab() and encode() splits them into
    subwords, breaking multimodal pipelines that rely on finding them in input_ids.

    This function discovers such tokens by scanning tokenizer attributes, checks
    if they encode correctly, and re-registers any that don't.
    """

    # Discover special token strings from tokenizer attributes.
    # Model tokenizers (e.g. MiniCPMVTokenizerFast) store them as attributes
    # like im_start="<image>", slice_start="<slice>", etc.
    def _is_special_token_attr(val):
        return (
            isinstance(val, str)
            and val.startswith("<")
            and val.endswith(">")
            and len(val) <= 20
        )

    candidates = {}
    for attr in dir(tokenizer):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(tokenizer, attr)
        except (AttributeError, TypeError, ValueError):
            continue
        if not _is_special_token_attr(val):
            continue
        token_id = tokenizer.convert_tokens_to_ids(val)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            candidates[val] = token_id

    if not candidates:
        return

    def _encodes_correctly(token_str, expected_id):
        try:
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            return len(ids) == 1 and ids[0] == expected_id
        except (ValueError, OverflowError, RuntimeError) as e:
            logger.debug("Token %s encode check failed: %s", token_str, e)
            return False

    broken = [
        tok for tok, eid in candidates.items() if not _encodes_correctly(tok, eid)
    ]

    if not broken:
        return

    from transformers import AddedToken

    tokens_to_add = [AddedToken(tok, special=True, normalized=False) for tok in broken]
    tokenizer.add_tokens(tokens_to_add, special_tokens=True)
    logger.info(
        "Re-registered %d special tokens for correct v5 encoding: %s",
        len(broken),
        broken[:10],
    )
