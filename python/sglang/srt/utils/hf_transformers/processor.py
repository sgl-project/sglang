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

from sglang.srt.multimodal.customized_mm_processor_utils import _CUSTOMIZED_MM_PROCESSOR
from sglang.srt.utils import logger
from transformers import (
    AutoImageProcessor,
    AutoProcessor,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

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
    _install_tokenizer_warnings_filter,
)

_IMAGE_PROCESSOR_BACKENDS = {"auto", "torchvision", "pil"}


def resolve_image_processor_backend(mm_config) -> str:
    """Resolve the new backend option while honoring the legacy disable flag.

    Takes the `mm` config bag (`get_mm()`): both leaves are resolved config, and
    every caller is past publish. `getattr` with a default keeps it working for a
    stand-in that carries only one of the two.
    """
    if getattr(mm_config, "disable_fast_image_processor", False):
        return "pil"
    return getattr(mm_config, "image_processor_backend", "auto")


def _normalize_image_processor_backend(
    image_processor_backend: Optional[str], use_fast: Optional[bool]
) -> str:
    backend = image_processor_backend or "auto"
    if backend not in _IMAGE_PROCESSOR_BACKENDS:
        raise ValueError(
            f"Unsupported image processor backend: {backend}. "
            f"Expected one of {sorted(_IMAGE_PROCESSOR_BACKENDS)}."
        )

    if use_fast is not None:
        legacy_backend = "torchvision" if use_fast else "pil"
        if backend not in {"auto", legacy_backend}:
            raise ValueError(
                f"use_fast={use_fast} conflicts with "
                f"image_processor_backend={backend!r}."
            )
        backend = legacy_backend
    return backend


def _apply_image_processor_backend(
    processor,
    tokenizer_name,
    args,
    trust_remote_code,
    revision,
    backend,
    kwargs,
):
    """Apply an explicit backend only to the image sub-processor.

    ProcessorMixin forwards generic kwargs to every sub-processor. Passing
    ``backend`` through AutoProcessor therefore also reaches tokenizers and
    video processors, where it has different semantics or may be read-only.
    """
    if backend == "auto" or not hasattr(processor, "image_processor"):
        return processor

    image_processor = processor.image_processor
    if getattr(image_processor, "backend", None) == backend:
        return processor

    image_processor_kwargs = dict(kwargs)
    image_processor_kwargs.pop("backend", None)
    image_processor_kwargs.pop("use_fast", None)
    processor.image_processor = AutoImageProcessor.from_pretrained(
        tokenizer_name,
        *args,
        trust_remote_code=trust_remote_code,
        revision=revision,
        backend=backend,
        **image_processor_kwargs,
    )
    return processor


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


def _build_glm5_next_processor(model_path, tokenizer, revision):
    """Build the GLM-5 Next processor until Transformers ships one.

    Current GLM-5 Next checkpoints use a nested ``processor_config.json``.
    Transformers versions without the corresponding processor class silently
    return the tokenizer from ``AutoProcessor`` instead, dropping all visual
    inputs. Reuse the compatible GLM-4V components, but fail loudly if the
    checkpoint does not provide the component configuration needed to do so.
    """
    from transformers import (
        Glm4vImageProcessor,
        Glm4vProcessor,
        Glm4vVideoProcessor,
    )

    try:
        config_file = _resolve_local_or_cached_file(
            model_path, "processor_config.json", revision
        )
        with open(config_file) as file:
            processor_config = json.load(file)
    except (OSError, json.JSONDecodeError, ValueError) as e:
        raise RuntimeError(
            "Cannot construct the GLM-5 Next multimodal processor: "
            f"failed to load processor_config.json for {model_path}: {e}"
        ) from e
    if not isinstance(processor_config, dict):
        raise TypeError(
            "Cannot construct the GLM-5 Next multimodal processor: "
            "processor_config.json must contain an object"
        )

    def build_component(component_class, name):
        raw_config = processor_config.get(name)
        if not isinstance(raw_config, dict):
            raise TypeError(
                "Cannot construct the GLM-5 Next multimodal processor: "
                f"processor_config.json must contain a {name!r} object"
            )
        component_config = dict(raw_config)
        component_config.pop(f"{name}_type", None)
        min_tokens = int(component_config.pop("min_image_tokens", 16))
        max_tokens = int(component_config.pop("max_image_tokens", 8000))
        patch_expand_factor = int(component_config.pop("patch_expand_factor", 1))
        patch_size = int(component_config.get("patch_size", 14))
        merge_size = int(component_config.get("merge_size", 2))
        temporal_patch_size = int(component_config.get("temporal_patch_size", 2))
        if not (0 < min_tokens <= max_tokens):
            raise RuntimeError(
                f"Invalid {name} token budget in processor_config.json: "
                f"min_image_tokens={min_tokens}, max_image_tokens={max_tokens}"
            )
        if min(patch_size, merge_size, temporal_patch_size, patch_expand_factor) <= 0:
            raise RuntimeError(
                f"Invalid {name} patch geometry in processor_config.json"
            )
        pixels_per_token = temporal_patch_size * (patch_size * merge_size) ** 2
        component_config["size"] = {
            "shortest_edge": min_tokens * pixels_per_token,
            "longest_edge": max_tokens * pixels_per_token,
        }
        component = component_class(**component_config)
        component.min_image_tokens = min_tokens
        component.max_image_tokens = max_tokens
        component.patch_expand_factor = patch_expand_factor
        return component

    image_processor = build_component(Glm4vImageProcessor, "image_processor")
    video_processor = build_component(Glm4vVideoProcessor, "video_processor")
    return Glm4vProcessor(
        image_processor=image_processor,
        tokenizer=tokenizer,
        video_processor=video_processor,
        chat_template=getattr(tokenizer, "chat_template", None),
    )


def get_processor(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    use_fast: Optional[bool] = None,
    image_processor_backend: Optional[str] = None,
    tokenizer_backend: str = "huggingface",
    model_name: Optional[str] = None,
    **kwargs,
):
    if tokenizer_backend == "fastokens":
        from .tokenizer import _ensure_fastokens_patched

        _ensure_fastokens_patched()

    revision = kwargs.pop("revision", tokenizer_revision)
    image_processor_backend = _normalize_image_processor_backend(
        image_processor_backend, use_fast
    )
    tokenizer_name = resolve_runai_obj_uri(tokenizer_name)
    if model_name is not None:
        model_name = resolve_runai_obj_uri(model_name)

    if is_mistral_model(tokenizer_name):
        config = load_mistral_config(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    elif model_name is not None:
        config = AutoConfig.from_pretrained(
            model_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
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

    # Checkpoints with language_model_only=True are text-only despite their
    # multimodal-family config; route to tokenizer instead of the mm processor.
    if getattr(config, "language_model_only", False):
        return AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )

    if config.model_type in {"qwen2_vl", "sarashina2_vision"}:
        if "size" not in kwargs:
            kwargs["size"] = {"shortest_edge": 3136, "longest_edge": 1003520}

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
                if config.model_type == "glm5_next" and isinstance(
                    processor, PreTrainedTokenizerBase
                ):
                    processor = _build_glm5_next_processor(
                        tokenizer_name, processor, revision
                    )

    except ValueError as e:
        error_message = str(e)
        if "Unrecognized feature extractor" in error_message:
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
                "retrying without trust_remote_code",
                tokenizer_name,
            )
            kwargs.pop("_from_auto", None)
            processor = AutoProcessor.from_pretrained(
                tokenizer_name,
                *args,
                revision=revision,
                **kwargs,
            )
        else:
            raise

    processor = _apply_image_processor_backend(
        processor,
        tokenizer_name,
        args,
        trust_remote_code,
        revision,
        image_processor_backend,
        kwargs,
    )
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

    _install_tokenizer_warnings_filter(tokenizer)

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
