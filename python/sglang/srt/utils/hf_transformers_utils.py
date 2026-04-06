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
"""Utilities for Huggingface Transformers."""

import contextlib
import json
import logging
import os
import tempfile
import warnings
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, Union

import torch
from huggingface_hub import snapshot_download

from sglang.srt.utils import get_bool_env_var
from sglang.srt.utils.runai_utils import ObjectStorageModel, is_runai_obj_uri

# Compatibility shim: flash-attn-4 registers a bare ``flash_attn`` namespace
# that makes ``is_flash_attn_2_available()`` return True, but lacks the v2 API
# (``flash_attn_func``, etc.).  HuggingFace remote model code (e.g. Kimi-VL)
# guarded by that check will crash with ImportError at module load time.
# Force it to False when the real v2 API is absent.
try:
    import flash_attn as _flash_attn_mod

    if not hasattr(_flash_attn_mod, "flash_attn_func"):
        import transformers.utils as _hf_utils
        import transformers.utils.import_utils as _hf_import_utils

        _hf_import_utils.is_flash_attn_2_available = lambda: False
        _hf_utils.is_flash_attn_2_available = lambda: False
    del _flash_attn_mod
except ImportError:
    pass

# Conditional import based on SGLANG_USE_MODELSCOPE environment variable
if get_bool_env_var("SGLANG_USE_MODELSCOPE"):
    from modelscope import AutoConfig, GenerationConfig
else:
    from transformers import AutoConfig, GenerationConfig

from transformers import (
    AutoProcessor,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
)
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

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
from sglang.srt.connector import create_remote_connector
from sglang.srt.multimodal.customized_mm_processor_utils import _CUSTOMIZED_MM_PROCESSOR
from sglang.srt.utils import is_remote_url, logger, lru_cache_frozenset, mistral_utils
from sglang.srt.utils.patch_tokenizer import patch_tokenizer

_CONFIG_REGISTRY: List[Type[PretrainedConfig]] = [
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

_CONFIG_REGISTRY = {
    config_cls.model_type: config_cls for config_cls in _CONFIG_REGISTRY
}

for name, cls in _CONFIG_REGISTRY.items():
    with contextlib.suppress(ValueError):
        AutoConfig.register(name, cls)


def download_from_hf(
    model_path: str,
    allow_patterns: Optional[Union[str, list]] = None,
):
    if os.path.exists(model_path):
        return model_path

    if not allow_patterns:
        allow_patterns = ["*.json", "*.bin", "*.model"]

    return snapshot_download(model_path, allow_patterns=allow_patterns)


def get_rope_config(config):
    """Get (rope_theta, rope_scaling) from config, supporting both v4 and v5.

    In transformers v5, rope_theta/rope_scaling are accessed via the computed
    property config.rope_parameters. Trust-remote-code configs or parent configs
    passed to sub-models may not have this property or may return None.
    Falls back to the v4-style config.rope_theta / config.rope_scaling attributes.
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
    for _attr in ("text_config", "llm_config", "language_config", "thinker_config"):
        _sub = getattr(config, _attr, None)
        if isinstance(_sub, dict):
            _converted = PretrainedConfig(**_sub)
            # Propagate torch_dtype from parent so weight loading uses correct precision.
            if (
                getattr(_converted, "torch_dtype", None) is None
                and getattr(config, "torch_dtype", None) is not None
            ):
                _converted.torch_dtype = config.torch_dtype
            setattr(config, _attr, _converted)

    # Priority: thinker_config > llm_config > language_config > text_config
    if hasattr(config, "thinker_config"):
        # qwen2.5 omni
        thinker_config = config.thinker_config
        if hasattr(thinker_config, "text_config"):
            setattr(
                thinker_config.text_config,
                "torch_dtype",
                getattr(thinker_config, "torch_dtype", None),
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


# Temporary hack for DeepSeek-V3.2 model
def _load_deepseek_v32_model(
    model_path: str,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
    **kwargs,
):
    # first get the local path
    local_path = download_from_hf(model_path)
    # then load the config file in json
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


# Temporary hack for Mistral Large
@lru_cache(maxsize=2)
def _load_mistral_large_3_for_causal_LM(
    model_path: str,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
):
    # first get the local path
    local_path = download_from_hf(model_path)
    # then load the config file in json
    parser = mistral_utils.MistralConfigParser()
    config_dict, _ = parser.parse(local_path)

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as f:
        json.dump(config_dict, f)
        f.flush()
        loaded_config = AutoConfig.from_pretrained(
            f.name, trust_remote_code=trust_remote_code, revision=revision
        )
    text_config = getattr(loaded_config, "text_config", None)
    if text_config is not None and isinstance(text_config, dict):
        text_config = AutoConfig.for_model(**text_config)
        setattr(loaded_config, "text_config", text_config)
    vision_config = getattr(loaded_config, "vision_config", None)
    if vision_config is not None and isinstance(vision_config, dict):
        vision_config = AutoConfig.for_model(**vision_config)
        setattr(loaded_config, "vision_config", vision_config)

    return loaded_config


def _is_deepseek_ocr_model(config: PretrainedConfig) -> bool:
    # TODO: Remove this workaround related when AutoConfig correctly identifies deepseek-ocr.
    # Hugging Face's AutoConfig currently misidentifies it as deepseekvl2.
    auto_map = getattr(config, "auto_map", None) or {}
    return auto_map.get("AutoModel") == "modeling_deepseekocr.DeepseekOCRForCausalLM"


def _is_deepseek_ocr2_model(config: PretrainedConfig) -> bool:
    auto_map = getattr(config, "auto_map", None) or {}
    return auto_map.get("AutoModel") == "modeling_deepseekocr2.DeepseekOCR2ForCausalLM"


def _override_deepseek_ocr_v_head_dim(config: DeepseekVLV2Config) -> None:
    # FIXME: deepseek-ocr's v_head_dim is set to 0 in its config file.
    # https://huggingface.co/deepseek-ai/DeepSeek-OCR/blob/main/config.json#L116
    if config.text_config.v_head_dim == 0:
        V_HEAD_DIM_PATCH = 128
        config.text_config.v_head_dim = V_HEAD_DIM_PATCH
        # Also fix language_config so get_hf_text_config (which may prefer it
        # over text_config) stays consistent.
        lc = getattr(config, "language_config", None)
        if isinstance(lc, dict):
            lc["v_head_dim"] = V_HEAD_DIM_PATCH
        elif hasattr(lc, "v_head_dim"):
            lc.v_head_dim = V_HEAD_DIM_PATCH
        logger.warning(
            f"Overriding deepseek-ocr's v_head_dim from 0 to {V_HEAD_DIM_PATCH} to avoid potential issues."
        )


def _override_v_head_dim_if_zero(config: PretrainedConfig, patch: int = 128) -> None:
    text_config = getattr(config, "text_config", None)
    language_config = getattr(config, "language_config", None)
    target = text_config or language_config
    if target is None:
        return
    if getattr(target, "v_head_dim", None) == 0:
        setattr(target, "v_head_dim", patch)
        logger.warning(
            f"Overriding v_head_dim from 0 to {patch} to avoid potential issues."
        )


def _ensure_clean_up_tokenization_compat() -> None:
    """Re-add ``clean_up_tokenization`` removed in transformers v5.

    Remote-code tokenizers (e.g. InternLM2Tokenizer) call
    ``self.clean_up_tokenization()`` which was a static method on
    ``PreTrainedTokenizerBase`` in v4 but removed in v5. Patch it back
    so existing HuggingFace Hub tokenizer code keeps working.
    """
    if hasattr(PreTrainedTokenizerBase, "clean_up_tokenization"):
        return

    @staticmethod
    def clean_up_tokenization(out_string: str) -> str:
        out_string = (
            out_string.replace(" .", ".")
            .replace(" ?", "?")
            .replace(" !", "!")
            .replace(" ,", ",")
            .replace(" ' ", "'")
            .replace(" n't", "n't")
            .replace(" 'm", "'m")
            .replace(" 's", "'s")
            .replace(" 've", "'ve")
            .replace(" 're", "'re")
        )
        return out_string

    PreTrainedTokenizerBase.clean_up_tokenization = clean_up_tokenization


# Apply immediately so all code paths (get_tokenizer, get_processor,
# and any external callers) benefit without needing an explicit call.
_ensure_clean_up_tokenization_compat()


def _ensure_is_torch_fx_available_compat() -> None:
    """Re-add ``is_torch_fx_available`` removed in transformers v5.

    Remote-code models (e.g. MiniCPM-V) import ``is_torch_fx_available``
    from ``transformers.utils.import_utils``.  The function was removed
    in v5.  Patch it back so existing HuggingFace Hub model code keeps
    working.  torch.fx is always available in PyTorch >= 2.0.
    """
    import transformers.utils.import_utils as _import_utils

    if hasattr(_import_utils, "is_torch_fx_available"):
        return

    _import_utils.is_torch_fx_available = lambda: True


_ensure_is_torch_fx_available_compat()


def normalize_rope_scaling_compat(config: "PretrainedConfig") -> None:
    """Ensure rope_scaling dicts have ``"type"`` alongside ``"rope_type"``.

    Transformers v5 standardises rope_scaling to use ``"rope_type"`` and may
    omit the legacy ``"type"`` key.  Remote-code models (e.g. Kimi-VL) still
    read ``rope_scaling["type"]``, causing a ``KeyError``.  This helper adds
    ``"type"`` from ``"rope_type"`` whenever it is missing, recursively across
    the config and all its sub-configs.
    """

    def _patch(cfg):
        try:
            rs = getattr(cfg, "rope_scaling", None)
        except AttributeError:
            rs = None
        if isinstance(rs, dict) and "rope_type" in rs and "type" not in rs:
            rs["type"] = rs["rope_type"]
        # Recurse into sub-configs
        for attr in (
            "text_config",
            "llm_config",
            "language_config",
            "vision_config",
            "thinker_config",
        ):
            sub = getattr(cfg, attr, None)
            if sub is not None:
                _patch(sub)

    _patch(config)


def _ensure_llama_flash_attention2_compat() -> None:
    """Ensure LlamaFlashAttention2 symbol exists for remote code compatibility."""
    try:
        from transformers.models.llama import modeling_llama
    except (ImportError, ModuleNotFoundError):
        return
    if not hasattr(modeling_llama, "LlamaFlashAttention2"):
        if hasattr(modeling_llama, "LlamaAttention"):
            modeling_llama.LlamaFlashAttention2 = modeling_llama.LlamaAttention


def _ensure_gguf_version():
    """Workaround for transformers v5 bug where is_gguf_available() fails
    when the gguf package lacks __version__ and metadata lookup also fails,
    resulting in packaging.version.InvalidVersion: Invalid version: 'N/A'."""
    try:
        import gguf

        if not hasattr(gguf, "__version__"):
            import importlib.metadata

            try:
                gguf.__version__ = importlib.metadata.version("gguf")
            except Exception:
                gguf.__version__ = "0.0.0"
    except ImportError:
        pass


@lru_cache_frozenset(maxsize=32)
def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_override_args: Optional[dict] = None,
    **kwargs,
):
    is_gguf = check_gguf_file(model)
    if is_gguf:
        _ensure_gguf_version()
        kwargs["gguf_file"] = model
        model = Path(model).parent

    if is_runai_obj_uri(model):
        model = ObjectStorageModel.get_path(model)

    if is_remote_url(model):
        # BaseConnector implements __del__() to clean up the local dir.
        # Since config files need to exist all the time, so we DO NOT use
        # with statement to avoid closing the client.
        client = create_remote_connector(model)
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
        model = client.get_local_dir()

    if (
        "mistral-large-3" in str(model).lower()
        or "mistral-small-4" in str(model).lower()
        or "leanstral" in str(model).lower()
    ):
        config = _load_mistral_large_3_for_causal_LM(
            model, trust_remote_code=trust_remote_code, revision=revision
        )
    else:
        _ensure_llama_flash_attention2_compat()
        try:
            config = AutoConfig.from_pretrained(
                model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
            )
        except ValueError as e:
            if not "deepseek_v32" in str(e):
                raise e
            config = _load_deepseek_v32_model(
                model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
            )
        except KeyError as e:
            # Transformers v5 may register a built-in config class that
            # conflicts with sglang's custom one (e.g. NemotronHConfig
            # doesn't handle '-' in hybrid_override_pattern). Fall back
            # to loading the raw config dict and using sglang's class.
            # Also handle deepseek_v32 which v5 doesn't recognize.
            if "deepseek_v32" in str(e):
                config = _load_deepseek_v32_model(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    **kwargs,
                )
            else:
                config_dict, _ = PretrainedConfig.get_config_dict(
                    model,
                    trust_remote_code=trust_remote_code,
                    revision=revision,
                    **kwargs,
                )
                model_type = config_dict.get("model_type")
                if model_type in _CONFIG_REGISTRY:
                    config = _CONFIG_REGISTRY[model_type].from_dict(config_dict)
                    config._name_or_path = model
                else:
                    raise

    if (
        config.architectures is not None
        and config.architectures[0] == "Phi4MMForCausalLM"
    ):
        # Phi4MMForCausalLM uses a hard-coded vision_config. See:
        # https://github.com/vllm-project/vllm/blob/6071e989df1531b59ef35568f83f7351afb0b51e/vllm/model_executor/models/phi4mm.py#L71
        # We set it here to support cases where num_attention_heads is not divisible by the TP size.
        from transformers import SiglipVisionConfig

        vision_config = {
            "hidden_size": 1152,
            "image_size": 448,
            "intermediate_size": 4304,
            "model_type": "siglip_vision_model",
            "num_attention_heads": 16,
            "num_hidden_layers": 26,
            # Model is originally 27-layer, we only need the first 26 layers for feature extraction.
            "patch_size": 14,
        }
        config.vision_config = SiglipVisionConfig(**vision_config)

    if config.architectures in [
        ["LongcatCausalLM"],
        ["LongcatFlashForCausalLM"],
        ["LongcatFlashNgramForCausalLM"],
    ]:
        config.model_type = "longcat_flash"

    text_config = get_hf_text_config(config=config)

    if isinstance(model, str) and text_config is not None:
        items = (
            text_config.items()
            if hasattr(text_config, "items")
            else vars(text_config).items()
        )
        for key, val in items:
            if not hasattr(config, key) and val is not None:
                setattr(config, key, val)

    if _is_deepseek_ocr2_model(config):
        _override_v_head_dim_if_zero(config)
        # Temporary hack for load deepseek-ocr2
        config.model_type = "deepseek-ocr"
        config.update({"architectures": ["DeepseekOCRForCausalLM"]})
        config = DeepseekVLV2Config.from_pretrained(model, revision=revision)
        _override_v_head_dim_if_zero(config)
        config.update({"architectures": ["DeepseekOCRForCausalLM"]})
        setattr(config, "_name_or_path", model)
    elif config.model_type in _CONFIG_REGISTRY:
        model_type = config.model_type
        if model_type == "deepseek_vl_v2":
            if _is_deepseek_ocr_model(config) or _is_deepseek_ocr2_model(config):
                model_type = "deepseek-ocr"
        config_class = _CONFIG_REGISTRY[model_type]
        config = config_class.from_pretrained(model, revision=revision)

        if _is_deepseek_ocr_model(config):
            _override_deepseek_ocr_v_head_dim(config)
            config.update({"architectures": ["DeepseekOCRForCausalLM"]})
        elif _is_deepseek_ocr2_model(config):
            _override_v_head_dim_if_zero(config)
            config.update({"architectures": ["DeepseekOCRForCausalLM"]})

        # NOTE(HandH1998): Qwen2VL requires `_name_or_path` attribute in `config`.
        setattr(config, "_name_or_path", model)

    if isinstance(model, str) and config.model_type == "internvl_chat":
        for key, val in config.llm_config.__dict__.items():
            if not hasattr(config, key):
                setattr(config, key, val)

    if config.model_type == "multi_modality":
        config.update({"architectures": ["MultiModalityCausalLM"]})

    if config.model_type == "longcat_flash":
        config.update({"architectures": ["LongcatFlashForCausalLM"]})

    if model_override_args:
        config.update(model_override_args)

    # Special architecture mapping check for GGUF models
    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        model_type = MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type]
        config.update({"architectures": [model_type]})

    return config


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
    except OSError as e:
        return None


# Qwen-1M related
def get_sparse_attention_config(
    model: str,
    sparse_attention_config_filename: str = "sparse_attention_config.json",
) -> Dict[str, Any]:
    is_local = os.path.isdir(model)
    if not is_local:
        # Download the config files.
        model = download_from_hf(model, allow_patterns=["*.json"])

    config_file = os.path.join(model, sparse_attention_config_filename)
    if not os.path.exists(config_file):
        return {}

    # Load the sparse attention config.
    with open(config_file) as f:
        config = json.load(f)
    return config


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


# A fast LLaMA tokenizer with the pre-processed `tokenizer.json` file.
_FAST_LLAMA_TOKENIZER = "hf-internal-testing/llama-tokenizer"


# Filter warnings like: https://github.com/sgl-project/sglang/issues/8082
class TokenizerWarningsFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return "Calling super().encode with" not in record.getMessage()


_is_base_mistral_patched = False

# transformers version where _patch_mistral_regex calls model_info() on every tokenizer load
_TRANSFORMERS_PATCHED_VERSION = "5.3.0"


def _patch_is_base_mistral_in_ci():
    """Patch transformers' _patch_mistral_regex to avoid HF API calls in CI.

    transformers defines is_base_mistral as a local function inside
    _patch_mistral_regex, so it cannot be patched via module attribute.
    Instead we replace the entire _patch_mistral_regex classmethod with a
    version that simply returns the tokenizer unchanged.

    In CI this prevents exhausting the 3000 req/5min HF API rate limit.
    """
    global _is_base_mistral_patched
    if _is_base_mistral_patched:
        return

    from sglang.srt.environ import envs

    if not envs.SGLANG_IS_IN_CI.get():
        return

    import transformers

    if transformers.__version__ != _TRANSFORMERS_PATCHED_VERSION:
        logger.warning(
            "transformers version changed to %s (expected %s), "
            "_patch_mistral_regex patch skipped — may need update if 429 errors recur",
            transformers.__version__,
            _TRANSFORMERS_PATCHED_VERSION,
        )
        _is_base_mistral_patched = True  # don't warn repeatedly
        return

    from transformers import PreTrainedTokenizerFast

    if hasattr(PreTrainedTokenizerFast, "_patch_mistral_regex"):

        @classmethod
        def _noop_patch_mistral_regex(cls, tokenizer, *args, **kwargs):
            return tokenizer

        PreTrainedTokenizerFast._patch_mistral_regex = _noop_patch_mistral_regex
        logger.info("CI: patched _patch_mistral_regex to skip HF API calls")

    _is_base_mistral_patched = True


def get_tokenizer(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    **kwargs,
) -> Union[PreTrainedTokenizer, PreTrainedTokenizerFast]:
    """Gets a tokenizer for the given model name via Huggingface."""
    if tokenizer_name.endswith(".json"):
        from sglang.srt.tokenizer.tiktoken_tokenizer import TiktokenTokenizer

        return TiktokenTokenizer(tokenizer_name)

    if tokenizer_mode == "slow":
        if kwargs.get("use_fast", False):
            raise ValueError("Cannot use the fast tokenizer in slow tokenizer mode.")
        kwargs["use_fast"] = False
    elif tokenizer_mode == "auto":
        # In Transformers v5, the default for use_fast changed from True to False.
        # Explicitly set use_fast=True for "auto" mode to maintain previous behavior
        # and avoid issues with models that have incorrect tokenizer_class values.
        if "use_fast" not in kwargs:
            kwargs["use_fast"] = True

    # TODO(Xinyuan): Remove this once we have a proper tokenizer for Devstral
    if tokenizer_name == "mistralai/Devstral-Small-2505":
        tokenizer_name = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"

    is_gguf = check_gguf_file(tokenizer_name)
    if is_gguf:
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

    _patch_is_base_mistral_in_ci()

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=trust_remote_code,
            tokenizer_revision=tokenizer_revision,
            clean_up_tokenization_spaces=False,
            **kwargs,
        )
        # Filter tokenizer warnings
        logging.getLogger(tokenizer.__class__.__module__).addFilter(
            TokenizerWarningsFilter()
        )
    except TypeError as e:
        # The LLaMA tokenizer causes a protobuf error in some environments.
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
            for k in (
                "trust_remote_code",
                "tokenizer_revision",
                "use_fast",
                "_from_auto",
                "clean_up_tokenization_spaces",
            ):
                kwargs.pop(k, None)
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_name,
                *args,
                **kwargs,
            )
        # If the error pertains to the tokenizer class not existing or not
        # currently being imported, suggest using the --trust-remote-code flag.
        elif not trust_remote_code and (
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
        else:
            raise e

    # Transformers v5 may silently fall back to a generic TokenizersBackend
    # when trust_remote_code=False and the model requires a custom tokenizer.
    # Detect this and auto-retry with trust_remote_code=True.
    if not trust_remote_code and type(tokenizer).__name__ == "TokenizersBackend":
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name,
            *args,
            trust_remote_code=True,
            tokenizer_revision=tokenizer_revision,
            clean_up_tokenization_spaces=False,
            **kwargs,
        )

    _fix_v5_tokenizer_components(tokenizer, tokenizer_name, tokenizer_revision)
    _fix_v5_add_bos_eos_token(tokenizer, tokenizer_name, tokenizer_revision)

    if not isinstance(tokenizer, PreTrainedTokenizerFast):
        warnings.warn(
            "Using a slow tokenizer. This might cause a significant "
            "slowdown. Consider using a fast tokenizer instead."
        )

    _patch_mistral_common_tokenizer(tokenizer)
    _fix_special_tokens_pattern(tokenizer)
    attach_additional_stop_token_ids(tokenizer)
    tokenizer = patch_tokenizer(tokenizer)
    return tokenizer


def _resolve_local_or_cached_file(model_name_or_path, filename, revision=None):
    """Resolve a file from a local directory or HF hub cache (no network)."""
    local_path = Path(model_name_or_path) / filename
    if local_path.is_file():
        return str(local_path)
    from huggingface_hub import hf_hub_download

    return hf_hub_download(
        model_name_or_path, filename, revision=revision, local_files_only=True
    )


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
    except Exception as e:
        logger.debug(
            "_fix_v5_tokenizer_components: could not load tokenizer.json for %s: %s",
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
    except Exception as e:
        logger.debug(
            "_fix_v5_add_bos_eos_token: could not read tokenizer_config.json "
            "for %s: %s",
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
            # Key missing or null → use v4 default for this tokenizer class
            config_val = _V4_DEFAULTS.get(attr, False)
        # Fast tokenizers in v4 used tokenizer.json post-processor for EOS —
        # the add_eos_token Python attribute was set but the post-processor
        # came from tokenizer.json, not from the attribute. In v5, the flag is
        # stripped and both sglang and HF reference end up with add_eos_token=False.
        # Restoring add_eos_token for fast tokenizers makes sglang diverge from
        # the HF reference (which doesn't restore it), breaking embedding models
        # like intfloat/e5-mistral-7b-instruct (cosine similarity drops to ~0.33).
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
    candidates = {}
    for attr in dir(tokenizer):
        if attr.startswith("_"):
            continue
        try:
            val = getattr(tokenizer, attr)
        except Exception:
            continue
        if (
            not isinstance(val, str)
            or not val.startswith("<")
            or not val.endswith(">")
            or len(val) > 20
        ):
            continue
        token_id = tokenizer.convert_tokens_to_ids(val)
        if token_id is not None and token_id != tokenizer.unk_token_id:
            candidates[val] = token_id

    if not candidates:
        return

    # Check which tokens fail to encode as single tokens.
    broken = []
    for token_str, expected_id in candidates.items():
        try:
            ids = tokenizer.encode(token_str, add_special_tokens=False)
            if len(ids) != 1 or ids[0] != expected_id:
                broken.append(token_str)
        except Exception:
            broken.append(token_str)

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


# Some models doesn't have an available processor, e.g.: InternVL
def get_tokenizer_from_processor(processor):
    if isinstance(processor, PreTrainedTokenizerBase):
        return processor
    return processor.tokenizer


def _build_processor_manually(
    model_path, config, trust_remote_code, revision, **kwargs
):
    """Build processor when AutoProcessor fails to resolve feature_extractor_type.

    In transformers v5, AutoProcessor.from_pretrained calls
    AutoFeatureExtractor.from_pretrained which fails if
    preprocessor_config.json lacks 'feature_extractor_type'. This loads the
    processor class from the hub and constructs it with individually-loaded
    components.
    """
    import transformers
    from transformers import AutoImageProcessor, AutoTokenizer
    from transformers.dynamic_module_utils import get_class_from_dynamic_module

    # Resolve processor class from auto_map — check both the model config
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
        except Exception as e:
            logger.debug(
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
        except Exception as e:
            logger.warning("Failed to load image_processor for %s: %s", model_path, e)

    # Instantiate feature extractor from its declared class
    fe_class_name = getattr(proc_cls, "feature_extractor_class", None)
    if fe_class_name:
        fe_class = getattr(transformers, fe_class_name, None)
        if fe_class is not None:
            init_kwargs["feature_extractor"] = fe_class()

    return proc_cls(**init_kwargs)


def get_processor(
    tokenizer_name: str,
    *args,
    tokenizer_mode: str = "auto",
    trust_remote_code: bool = False,
    tokenizer_revision: Optional[str] = None,
    use_fast: Optional[bool] = True,
    **kwargs,
):
    # pop 'revision' from kwargs if present.
    revision = kwargs.pop("revision", tokenizer_revision)
    if (
        "mistral-large-3" in str(tokenizer_name).lower()
        or "mistral-small-4" in str(tokenizer_name).lower()
        or "leanstral" in str(tokenizer_name).lower()
    ):
        config = _load_mistral_large_3_for_causal_LM(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
        )
    else:
        _ensure_llama_flash_attention2_compat()
        config = AutoConfig.from_pretrained(
            tokenizer_name,
            trust_remote_code=trust_remote_code,
            revision=revision,
            **kwargs,
        )
    if _is_deepseek_ocr_model(config):
        # Temporary hack for load deepseek-ocr
        config.model_type = "deepseek-ocr"
        config.update({"architectures": ["DeepseekOCRForCausalLM"]})
    elif _is_deepseek_ocr2_model(config):
        # Temporary hack for load deepseek-ocr2
        config.model_type = "deepseek-ocr"
        config.update({"architectures": ["DeepseekOCRForCausalLM"]})
        _override_v_head_dim_if_zero(config)

    # fix: for Qwen2-VL and Sarashina2Vision models, inject default 'size' if not provided.
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
                f"Processor {tokenizer_name} does not have a slow version. Automatically use fast version"
            )
            kwargs["use_fast"] = True
            processor = AutoProcessor.from_pretrained(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
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
        else:
            raise e
    # If processor is a bare tokenizer (e.g. Mistral-Small-4 has no processor_config.json)
    # and the model is a vision model (pixtral), wrap it in a proper PixtralProcessor
    # so that image data is actually processed through the image processor.
    if (
        isinstance(processor, PreTrainedTokenizerBase)
        and getattr(config, "model_type", None) == "pixtral"
    ):
        from transformers.models.pixtral.image_processing_pixtral import (
            PixtralImageProcessor,
        )
        from transformers.models.pixtral.processing_pixtral import (
            PixtralProcessor as HFPixtralProcessor,
        )

        vision_config = config.vision_config
        patch_size = vision_config.patch_size
        image_size = vision_config.image_size
        spatial_merge_size = getattr(vision_config, "spatial_merge_size", 1)

        effective_patch = patch_size * spatial_merge_size
        image_processor = PixtralImageProcessor(
            do_resize=True,
            size={"longest_edge": image_size},
            patch_size={"height": effective_patch, "width": effective_patch},
        )
        processor = HFPixtralProcessor(
            image_processor=image_processor,
            tokenizer=processor,
            patch_size=patch_size,
            spatial_merge_size=spatial_merge_size,
        )

    tokenizer = get_tokenizer_from_processor(processor)
    _patch_mistral_common_tokenizer(tokenizer)

    if tokenizer.chat_template is None:
        local_path = download_from_hf(
            tokenizer_name, allow_patterns=["*.json", "*.jinja", "*.model"]
        )
        jinja_path = Path(local_path) / "chat_template.jinja"
        if jinja_path.is_file():
            tokenizer.chat_template = jinja_path.read_text()
            logger.info("Loaded chat_template from %s", jinja_path)

    _fix_special_tokens_pattern(tokenizer)
    _fix_added_tokens_encoding(tokenizer)
    attach_additional_stop_token_ids(tokenizer)
    return processor


def attach_additional_stop_token_ids(tokenizer):
    # Special handling for stop token <|eom_id|> generated by llama 3 tool use.
    if "<|eom_id|>" in tokenizer.get_added_vocab():
        tokenizer.additional_stop_token_ids = set(
            [tokenizer.get_added_vocab()["<|eom_id|>"]]
        )
    else:
        tokenizer.additional_stop_token_ids = None


def _patch_mistral_common_tokenizer(tokenizer):
    """Patch MistralCommonTokenizer/Backend to be compatible with HF tokenizer API.

    MistralCommon tokenizers (used by Voxtral, Pixtral, etc.) reject several
    standard kwargs and lack some attributes that sglang expects.  We wrap the
    offending methods once at load time so that the rest of the codebase does
    not need any special-casing.
    """
    cls_name = type(tokenizer).__name__
    if "MistralCommon" not in cls_name:
        return tokenizer
    if getattr(tokenizer, "_mistral_common_patched", False):
        return tokenizer
    tokenizer._mistral_common_patched = True

    # Missing attributes
    if not hasattr(tokenizer, "get_added_vocab"):
        tokenizer.get_added_vocab = lambda: {}

    # Set a chat_template containing "audio" so that sglang's content format
    # detector returns "openai" (which preserves audio_url extraction).
    # The actual template rendering is done by MistralCommon's apply_chat_template.
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        tokenizer.chat_template = "<!-- audio/image multimodal -->"

    # convert_tokens_to_ids asserts on multi-token strings
    _orig_convert = tokenizer.convert_tokens_to_ids

    def _safe_convert(val):
        try:
            return _orig_convert(val)
        except AssertionError:
            return getattr(tokenizer, "unk_token_id", None)

    tokenizer.convert_tokens_to_ids = _safe_convert

    # Wrap methods that reject certain kwargs
    def _drop_kwargs(fn, keys):
        def wrapper(*args, **kwargs):
            for k in keys:
                kwargs.pop(k, None)
            return fn(*args, **kwargs)

        return wrapper

    tokenizer.decode = _drop_kwargs(tokenizer.decode, ["spaces_between_special_tokens"])
    tokenizer.batch_decode = _drop_kwargs(
        tokenizer.batch_decode, ["spaces_between_special_tokens"]
    )

    # Save original apply_chat_template for processors that need it (e.g. Voxtral)
    tokenizer._orig_apply_chat_template = tokenizer.apply_chat_template

    def _safe_apply_chat_template(messages, **kwargs):
        """Wrapper that strips unsupported kwargs and non-text content parts.

        When sglang extracts audio/image URLs, it replaces content blocks with
        {"type": "audio"} or {"type": "image"} (no URL).  MistralCommon fails
        on these stripped blocks.  We convert them to text-only messages.
        """
        kwargs.pop("add_generation_prompt", None)
        cleaned = []
        for msg in messages:
            if isinstance(msg, dict):
                content = msg.get("content", "")
                if isinstance(content, list):
                    text_parts = [
                        p.get("text", "")
                        for p in content
                        if isinstance(p, dict) and p.get("type") == "text"
                    ]
                    msg = {**msg, "content": " ".join(text_parts) if text_parts else ""}
                cleaned.append(msg)
            else:
                cleaned.append(msg)
        return tokenizer._orig_apply_chat_template(cleaned, **kwargs)

    tokenizer.apply_chat_template = _safe_apply_chat_template


def check_gguf_file(model: Union[str, os.PathLike]) -> bool:
    """Check if the file is a GGUF model."""
    model = Path(model)
    if not model.is_file():
        return False
    elif model.suffix == ".gguf":
        return True

    with open(model, "rb") as f:
        header = f.read(4)
    return header == b"GGUF"
