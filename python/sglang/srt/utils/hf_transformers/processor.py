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
"""Processor loading utilities."""

import json
from pathlib import Path
from typing import Optional

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

from sglang.srt.multimodal.customized_mm_processor_utils import _CUSTOMIZED_MM_PROCESSOR
from sglang.srt.utils import logger

from .common import (
    AutoConfig,
    _is_deepseek_ocr2_model,
    _is_deepseek_ocr_model,
    _override_v_head_dim_if_zero,
    _resolve_local_or_cached_file,
    attach_additional_stop_token_ids,
    download_from_hf,
    get_tokenizer_from_processor,
    resolve_runai_obj_uri,
)
from .mistral_utils import (
    is_mistral_model,
    load_mistral_config,
    patch_mistral_common_tokenizer,
    wrap_as_pixtral,
)
from .tokenizer import (
    _TOKENIZERS_BACKEND,
    _fix_added_tokens_encoding,
    _fix_special_tokens_pattern,
)


def _build_processor_manually(
    model_path, config, trust_remote_code, revision, **kwargs
):
    """Build processor when AutoProcessor fails to resolve feature_extractor_type.

    In transformers v5, AutoProcessor.from_pretrained calls
    AutoFeatureExtractor.from_pretrained which fails if
    preprocessor_config.json lacks 'feature_extractor_type'. This resolves
    the processor class via dynamic module resolution and constructs it with
    individually-loaded components.
    """
    import transformers
    from transformers import AutoImageProcessor, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    # Resolve processor class from auto_map -- check both the model config
    # and the preprocessor_config.json (some models like MiniCPM-o only
    # declare AutoProcessor in the latter).
    auto_map = getattr(config, "auto_map", None) or {}
    proc_ref = auto_map.get("AutoProcessor")
    if not proc_ref:
        try:
            pp_file = _resolve_local_or_cached_file(
                model_path, "preprocessor_config.json", revision
            )
            with open(pp_file) as f:
                pp_auto_map = json.load(f).get("auto_map", {})
            proc_ref = pp_auto_map.get("AutoProcessor")
        except (OSError, json.JSONDecodeError, ValueError) as e:
            logger.warning(
                "_build_processor_manually: could not read preprocessor_config.json "
                "for %s: %s",
                model_path,
                e,
            )
    if not proc_ref:
        raise ValueError(f"Cannot determine processor class for {model_path}")

    proc_cls = get_class_from_dynamic_module(
        proc_ref, model_path, code_revision=revision
    )

    # Load sub-components individually (these succeed)
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=trust_remote_code, revision=revision
    )
    init_kwargs = {"tokenizer": tokenizer}

    if "image_processor" in getattr(proc_cls, "attributes", []):
        try:
            init_kwargs["image_processor"] = AutoImageProcessor.from_pretrained(
                model_path, trust_remote_code=trust_remote_code, revision=revision
            )
        except (ImportError, OSError, ValueError) as e:
            raise RuntimeError(
                f"Failed to load image_processor for {model_path}: {e}. "
                f"This model requires an image processor for multimodal features. "
                f"Check that the model files are complete and accessible."
            ) from e

    # Instantiate feature extractor from its declared class
    fe_class_name = getattr(proc_cls, "feature_extractor_class", None)
    if fe_class_name:
        fe_class = getattr(transformers, fe_class_name, None)
        if fe_class is not None:
            try:
                init_kwargs["feature_extractor"] = fe_class()
            except TypeError as e:
                logger.warning(
                    "Cannot instantiate feature extractor %s with no arguments "
                    "for %s: %s",
                    fe_class_name,
                    model_path,
                    e,
                )
        else:
            logger.warning(
                "Feature extractor class %s not found in transformers for %s",
                fe_class_name,
                model_path,
            )

    return proc_cls(**init_kwargs)


def get_processor(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    use_fast: Optional[bool] = True,
    tokenizer_backend: str = "huggingface",
    **kwargs,
):
    if tokenizer_backend == "fastokens":
        from .tokenizer import _ensure_fastokens_patched

        _ensure_fastokens_patched()

    revision = kwargs.pop("revision", tokenizer_revision)
    tokenizer_name = resolve_runai_obj_uri(tokenizer_name)

    if is_mistral_model(tokenizer_name):
        config = load_mistral_config(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    else:
        config = AutoConfig.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )
    is_ocr2 = _is_deepseek_ocr2_model(config)
    if _is_deepseek_ocr_model(config) or is_ocr2:
        config.model_type = "deepseek-ocr"
        config.update({"architectures": ["DeepseekOCRForCausalLM"]})
        if is_ocr2:
            _override_v_head_dim_if_zero(config)

    if config.model_type in {"qwen2_vl", "sarashina2_vision"}:
        if "size" not in kwargs:
            kwargs["size"] = {"shortest_edge": 3136, "longest_edge": 1003520}

    if config.model_type not in {"llava", "clip"}:
        kwargs["use_fast"] = use_fast
    try:
        if "InternVL3_5" in tokenizer_name:
            processor = AutoTokenizer.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )
        else:
            if config.model_type in _CUSTOMIZED_MM_PROCESSOR:
                processor = _CUSTOMIZED_MM_PROCESSOR[config.model_type].from_pretrained(
                    tokenizer_name,
                    *args,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    **kwargs,
                )
            else:
                processor = AutoProcessor.from_pretrained(
                    tokenizer_name,
                    *args,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    **kwargs,
                )

    except ValueError as e:
        error_message = str(e)
        if "does not have a slow version" in error_message:
            logger.info(
                "Processor %s does not have a slow version. Automatically use fast version",
                tokenizer_name,
            )
            kwargs["use_fast"] = True
            processor = AutoProcessor.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )
        elif "Unrecognized feature extractor" in error_message:
            logger.info(
                "AutoProcessor failed on feature extractor for %s, "
                "constructing processor manually",
                tokenizer_name,
            )
            processor = _build_processor_manually(
                tokenizer_name,
                config,
                trust_remote_code,
                revision,
                **kwargs,
            )
        elif (
            "are not supported by" in error_message and "MistralCommon" in error_message
        ):
            logger.info(
                "AutoProcessor for %s rejected standard kwargs, "
                "retrying without trust_remote_code/use_fast",
                tokenizer_name,
            )
            kwargs.pop("use_fast", None)
            kwargs.pop("_from_auto", None)
            processor = AutoProcessor.from_pretrained(
                tokenizer_name,
                *args,
                revision=revision,
                **kwargs,
            )
        else:
            raise
    if (
        isinstance(processor, PreTrainedTokenizerBase)
        and getattr(config, "model_type", None) == "pixtral"
    ):
        processor = wrap_as_pixtral(processor, config)

    tokenizer = get_tokenizer_from_processor(processor)

    # AutoProcessor may internally create a TokenizersBackend tokenizer
    # (same issue as get_tokenizer). Replace it with a properly loaded one.
    if type(tokenizer).__name__ == _TOKENIZERS_BACKEND:
        from .tokenizer import get_tokenizer

        logger.warning(
            "Processor tokenizer for %s is TokenizersBackend, "
            "reloading via get_tokenizer",
            tokenizer_name,
        )
        tokenizer = get_tokenizer(
            tokenizer_name,
            tokenizer_mode=tokenizer_mode,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=revision,
            tokenizer_backend=tokenizer_backend,
        )
        if isinstance(processor, PreTrainedTokenizerBase):
            processor = tokenizer
        else:
            processor.tokenizer = tokenizer

    if tokenizer.chat_template is None:
        local_path = download_from_hf(
            tokenizer_name, allow_patterns=["*.json", "*.jinja", "*.model"]
        )
        jinja_path = Path(local_path) / "chat_template.jinja"
        if jinja_path.is_file():
            tokenizer.chat_template = jinja_path.read_text()
            logger.info("Loaded chat_template from %s", jinja_path)

    patch_mistral_common_tokenizer(tokenizer)
    _fix_special_tokens_pattern(tokenizer)
    _fix_added_tokens_encoding(tokenizer)
    attach_additional_stop_token_ids(tokenizer)
    return processor
