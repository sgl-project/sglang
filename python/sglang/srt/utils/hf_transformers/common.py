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
"""Shared helpers used by config, tokenizer, and processor modules."""

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Type, Union

import torch
from huggingface_hub import snapshot_download

from sglang.srt.configs import (
    AfmoeConfig,
    BailingHybridConfig,
    ChatGLMConfig,
    DbrxConfig,
    DeepseekVL2Config,
    DotsOCRConfig,
    DotsVLMConfig,
    ExaoneConfig,
    FalconH1Config,
    GraniteMoeHybridConfig,
    JetNemotronConfig,
    JetVLMConfig,
    KimiK25Config,
    KimiLinearConfig,
    KimiVLConfig,
    LongcatFlashConfig,
    MultiModalityConfig,
    NemotronH_Nano_VL_V2_Config,
    NemotronHConfig,
    Olmo3Config,
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3NextConfig,
    Step3p5Config,
    Step3VLConfig,
)
from sglang.srt.configs.deepseek_ocr import DeepseekVLV2Config
from sglang.srt.configs.internvl import InternVLChatConfig
from sglang.srt.utils import get_bool_env_var, logger, lru_cache_frozenset

from ..hf_transformers_patches import normalize_rope_scaling_compat

if get_bool_env_var("SGLANG_USE_MODELSCOPE"):
    from modelscope import AutoConfig, GenerationConfig
else:
    from transformers import AutoConfig, GenerationConfig

from transformers import PretrainedConfig

# ---------------------------------------------------------------------------
# Config registry
# ---------------------------------------------------------------------------

_CONFIG_REGISTRY: Dict[str, Type[PretrainedConfig]] = {
    cls.model_type: cls
    for cls in [
        AfmoeConfig,
        BailingHybridConfig,
        ChatGLMConfig,
        DbrxConfig,
        ExaoneConfig,
        DeepseekVL2Config,
        MultiModalityConfig,
        KimiVLConfig,
        InternVLChatConfig,
        Step3VLConfig,
        LongcatFlashConfig,
        Olmo3Config,
        KimiLinearConfig,
        Qwen3NextConfig,
        FalconH1Config,
        GraniteMoeHybridConfig,
        DotsVLMConfig,
        DotsOCRConfig,
        NemotronH_Nano_VL_V2_Config,
        NemotronHConfig,
        DeepseekVLV2Config,
        Qwen3_5Config,
        Qwen3_5MoeConfig,
        JetNemotronConfig,
        JetVLMConfig,
        KimiK25Config,
        Step3p5Config,
    ]
}

for name, cls in _CONFIG_REGISTRY.items():
    try:
        AutoConfig.register(name, cls)
    except ValueError as e:
        err = str(e).lower()
        if "already registered" not in err and "already used" not in err:
            logger.warning("Failed to register config %s: %s", name, e)


# ---------------------------------------------------------------------------
# Download / path helpers
# ---------------------------------------------------------------------------


def download_from_hf(
    model_path: str,
    allow_patterns: Optional[Union[str, list]] = None,
):
    if os.path.exists(model_path):
        return model_path

    if not allow_patterns:
        allow_patterns = ["*.json", "*.bin", "*.model"]

    return snapshot_download(model_path, allow_patterns=allow_patterns)


def _resolve_local_or_cached_file(model_name_or_path, filename, revision=None):
    """Resolve a file from a local directory or HF hub cache (no network)."""
    local_path = Path(model_name_or_path) / filename
    if local_path.is_file():
        return str(local_path)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        model_name_or_path, filename, revision=revision, local_files_only=True
    )


def check_gguf_file(model: Union[str, os.PathLike]) -> bool:
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True

    with open(model, "rb") as f:
        header = f.read(4)
    return header == b"GGUF"


# ---------------------------------------------------------------------------
# Rope / text config helpers
# ---------------------------------------------------------------------------


def get_rope_config(config):
    """Get (rope_theta, rope_params) from config, supporting both v4 and v5.

    Trust-remote-code configs or parent configs passed to sub-models may not
    have the v5 ``rope_parameters`` property, so we fall back to the v4-style
    ``config.rope_theta`` / ``config.rope_scaling`` attributes.

    Returns:
        (rope_theta, rope_params): In v5, rope_params is the full
        rope_parameters dict (which subsumes rope_scaling and includes
        rope_theta). In v4, rope_params is the rope_scaling dict or None.
    """
    rope_params = getattr(config, "rope_parameters", None)
    if rope_params is not None:
        return rope_params["rope_theta"], rope_params
    return config.rope_theta, getattr(config, "rope_scaling", None)


def _patch_text_config(parent_config: PretrainedConfig, text_config):
    """Synchronize standard attributes between parent config and text sub-config.

    In transformers v5, the "untangle config" refactor removed automatic
    inheritance of top-level PretrainedConfig attributes (pad_token_id,
    tie_word_embeddings, etc.) from sub-configs. Downstream code expects
    these attributes to be present on both configs (some models pass the
    parent directly to the language model, others pass the text sub-config),
    so we propagate in both directions when an attribute is missing.
    (See https://github.com/huggingface/transformers/pull/41541)
    """
    _ATTRS_TO_PROPAGATE = [
        "pad_token_id",
        "bos_token_id",
        "eos_token_id",
        "tie_word_embeddings",
    ]
    for attr in _ATTRS_TO_PROPAGATE:
        parent_has = hasattr(parent_config, attr)
        text_has = hasattr(text_config, attr)
        if parent_has and not text_has:
            setattr(text_config, attr, getattr(parent_config, attr))
        elif text_has and not parent_has:
            setattr(parent_config, attr, getattr(text_config, attr))
    return text_config


def get_hf_text_config(config: PretrainedConfig):
    """Get the "sub" config relevant to llm for multi modal models.
    No op for pure text models.
    """
    if config.architectures is not None:
        class_name = config.architectures[0]
        if class_name.startswith("Llava") and class_name.endswith("ForCausalLM"):
            # We support non-hf version of llava models, so we do not want to
            # read the wrong values from the unused default text_config.
            # NOTE(HandH1998): We set `torch_dtype` of config to `torch.float16` for the weights, as
            # `torch.float16` is default used for image features in `python/sglang/srt/models/llava.py`.
            setattr(config, "dtype", torch.float16)
            return config

    text_config = None

    # Some models (e.g. DeepSeek-OCR) store sub-configs as plain dicts.
    # Convert to PretrainedConfig early so hasattr() checks and asserts work.
    parent_dtype = getattr(config, "dtype", None)
    for _attr in ("text_config", "llm_config", "language_config", "thinker_config"):
        _sub = getattr(config, _attr, None)
        if isinstance(_sub, dict):
            _converted = PretrainedConfig(**_sub)
            if getattr(_converted, "dtype", None) is None and parent_dtype is not None:
                _converted.dtype = parent_dtype
            setattr(config, _attr, _converted)

    # Priority: thinker_config > llm_config > language_config > text_config
    if hasattr(config, "thinker_config"):
        # qwen2.5 omni
        thinker_config = config.thinker_config
        if hasattr(thinker_config, "text_config"):
            setattr(
                thinker_config.text_config,
                "dtype",
                getattr(thinker_config, "dtype", None),
            )
            text_config = thinker_config.text_config
        else:
            text_config = thinker_config
    elif hasattr(config, "llm_config"):
        # PointsV1.5 Chat Model
        assert hasattr(config.llm_config, "num_attention_heads")
        text_config = config.llm_config
    elif hasattr(config, "language_config"):
        text_config = config.language_config
    elif hasattr(config, "text_config"):
        # The code operates under the assumption that text_config should have
        # `num_attention_heads` (among others). Assert here to fail early
        # if transformers config doesn't align with this assumption.
        assert hasattr(config.text_config, "num_attention_heads")
        text_config = config.text_config

    # Ensure rope_scaling dicts have "type" for remote-code compat (v5).
    normalize_rope_scaling_compat(config)

    if text_config is not None:
        return _patch_text_config(config, text_config)
    return config


# ---------------------------------------------------------------------------
# Model-specific helpers
# ---------------------------------------------------------------------------


def _ensure_sub_configs(config: PretrainedConfig, *attr_names: str) -> None:
    """Convert dict-valued sub-configs to proper AutoConfig objects in-place."""
    for attr in attr_names:
        sub = getattr(config, attr, None)
        if sub is not None and isinstance(sub, dict):
            setattr(config, attr, AutoConfig.for_model(**sub))


def _is_deepseek_ocr_model(config: PretrainedConfig) -> bool:
    # TODO: Remove this workaround once AutoConfig correctly identifies deepseek-ocr.
    # Hugging Face's AutoConfig currently misidentifies it as deepseekvl2.
    auto_map = getattr(config, "auto_map", None) or {}
    return auto_map.get("AutoModel") == "modeling_deepseekocr.DeepseekOCRForCausalLM"


def _is_deepseek_ocr2_model(config: PretrainedConfig) -> bool:
    auto_map = getattr(config, "auto_map", None) or {}
    return auto_map.get("AutoModel") == "modeling_deepseekocr2.DeepseekOCR2ForCausalLM"


def _override_v_head_dim_if_zero(config: PretrainedConfig, patch: int = 128) -> None:
    patched = False
    for attr in ("text_config", "language_config"):
        sub = getattr(config, attr, None)
        if sub is None:
            continue
        if isinstance(sub, dict):
            if sub.get("v_head_dim") == 0:
                sub["v_head_dim"] = patch
                patched = True
        elif getattr(sub, "v_head_dim", None) == 0:
            sub.v_head_dim = patch
            patched = True
    if patched:
        logger.warning(
            f"Overriding v_head_dim from 0 to {patch} to avoid potential issues."
        )


def _load_deepseek_v32_model(
    model_path: str,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    **kwargs,
):
    import tempfile

    local_path = download_from_hf(model_path)
    config_file = os.path.join(local_path, "config.json")
    if not os.path.exists(config_file):
        raise RuntimeError(f"Can't find config file in {local_path}.")

    with open(config_file, "r") as f:
        config_json = json.load(f)

    config_json["architectures"] = ["DeepseekV3ForCausalLM"]
    config_json["model_type"] = "deepseek_v3"

    tmp_path = os.path.join(tempfile.gettempdir(), "_tmp_config_folder")
    os.makedirs(tmp_path, exist_ok=True)

    unique_path = os.path.join(tmp_path, f"deepseek_v32_{os.getpid()}")
    with open(unique_path, "w") as f:
        json.dump(config_json, f)

    return AutoConfig.from_pretrained(
        unique_path, trust_remote_code=trust_remote_code, revision=revision, **kwargs
    )


# ---------------------------------------------------------------------------
# Context length / generation config / sparse attention
# ---------------------------------------------------------------------------

# Models don't use the same configuration key for determining the maximum
# context length.  Store them here so we can sanely check them.
# NOTE: The ordering here is important. Some models have two of these and we
# have a preference for which value gets used.
CONTEXT_LENGTH_KEYS = [
    "max_sequence_length",
    "seq_length",
    "max_seq_len",
    "model_max_length",
    "max_position_embeddings",
]


def get_context_length(config):
    """Get the context length of a model from a huggingface model configs."""
    text_config = config
    rope_scaling = getattr(text_config, "rope_scaling", None)
    if rope_scaling:
        rope_scaling_factor = rope_scaling.get("factor", 1)
        if "original_max_position_embeddings" in rope_scaling:
            rope_scaling_factor = 1
        if rope_scaling.get("rope_type", None) == "llama3":
            rope_scaling_factor = 1
    else:
        rope_scaling_factor = 1

    for key in CONTEXT_LENGTH_KEYS:
        val = getattr(text_config, key, None)
        if val is not None:
            return int(rope_scaling_factor * val)
    return 2048


@lru_cache_frozenset(maxsize=32)
def get_generation_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    **kwargs,
):
    try:
        return GenerationConfig.from_pretrained(
            model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
        )
    except FileNotFoundError:
        return None
    except OSError as e:
        logger.warning(
            "Failed to load generation config for %s: %s. "
            "Proceeding without generation config.",
            model,
            e,
        )
        return None


# Qwen-1M related
def get_sparse_attention_config(
    model: str,
    sparse_attention_config_filename: str = "sparse_attention_config.json",
) -> Dict[str, Any]:
    is_local = os.path.isdir(model)
    if not is_local:
        model = download_from_hf(model, allow_patterns=["*.json"])

    config_file = os.path.join(model, sparse_attention_config_filename)
    if not os.path.exists(config_file):
        return {}

    with open(config_file) as f:
        config = json.load(f)
    return config


# ---------------------------------------------------------------------------
# Tokenizer / processor helpers
# ---------------------------------------------------------------------------


# Some models don't have an available processor, e.g.: InternVL
def get_tokenizer_from_processor(processor):
    from transformers import PreTrainedTokenizerBase

    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


def attach_additional_stop_token_ids(tokenizer):
    added = tokenizer.get_added_vocab()
    if "<|eom_id|>" in added:
        tokenizer.additional_stop_token_ids = {added["<|eom_id|>"]}
    else:
        tokenizer.additional_stop_token_ids = None
