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
"""Config loading utilities."""

from pathlib import Path
from typing import Optional

from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from sglang.srt.configs.model_config_parser_registry import (
    ModelConfigParserBase,
    get_model_config_parser,
    register_model_config_parser,
)
from sglang.srt.connector import create_remote_connector
from sglang.srt.utils import is_remote_url, lru_cache_frozenset

from ..hf_transformers_patches import _ensure_gguf_version
from .common import (
    _CONFIG_REGISTRY,
    AutoConfig,
    DeepseekVLV2Config,
    _is_deepseek_ocr2_model,
    _is_deepseek_ocr_model,
    _override_v_head_dim_if_zero,
    check_gguf_file,
    get_hf_text_config,
    resolve_runai_obj_uri,
)
from .mistral_utils import is_mistral_model, load_mistral_config


def _set_architectures(config, arch_name):
    config.update({"architectures": [arch_name]})


def _apply_deepseek_ocr_overrides(config, model):
    _override_v_head_dim_if_zero(config)
    _set_architectures(config, "DeepseekOCRForCausalLM")
    config._name_or_path = model


def _is_legacy_glm_moe_dsa_layer_types_error(error: Exception) -> bool:
    error_msg = str(error)
    return (
        "validate_layer_type" in error_msg and "deepseek_sparse_attention" in error_msg
    )


def _load_glm_moe_dsa_config_without_legacy_layer_types(
    model,
    revision: Optional[str] = None,
    **kwargs,
):
    from transformers import PretrainedConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING

    raw_config, unused_kwargs = PretrainedConfig.get_config_dict(
        model, revision=revision, **kwargs
    )
    if raw_config.get("model_type") != "glm_moe_dsa" or raw_config.get(
        "architectures"
    ) != ["GlmMoeDsaForCausalLM"]:
        return None

    layer_types = raw_config.get("layer_types")
    if not isinstance(layer_types, list) or any(
        layer_type != "deepseek_sparse_attention" for layer_type in layer_types
    ):
        return None

    raw_config = dict(raw_config)
    raw_config.pop("layer_types", None)
    config = CONFIG_MAPPING[raw_config["model_type"]].from_dict(
        raw_config, **unused_kwargs
    )
    config._name_or_path = model
    return config


@register_model_config_parser("hf")
class HfModelConfigParser(ModelConfigParserBase):
    def parse(
        self,
        model,
        trust_remote_code: bool,
        revision: Optional[str] = None,
        **kwargs,
    ):
        try:
            config = AutoConfig.from_pretrained(
                model,
                trust_remote_code=trust_remote_code,
                revision=revision,
                **kwargs,
            )
        except Exception as e:
            config = (
                _load_glm_moe_dsa_config_without_legacy_layer_types(
                    model, revision, **kwargs
                )
                if _is_legacy_glm_moe_dsa_layer_types_error(e)
                else None
            )
            if config is None:
                raise

        if (
            config.architectures is not None
            and config.architectures[0] == "GlmMoeDsaForCausalLM"
        ):
            # GlmMoeDsaConfig drops/clobbers raw checkpoint fields the DSA path
            # needs, so re-read them from config.json and restore. Fixed upstream
            # by https://github.com/huggingface/transformers/pull/46338; remove
            # this block once SGLang requires transformers >= 5.10.
            from transformers import PretrainedConfig

            raw_config, _ = PretrainedConfig.get_config_dict(model, revision=revision)
            for key in (
                "qk_rope_head_dim",
                "index_topk_freq",
            ):
                if key in raw_config:
                    setattr(config, key, raw_config[key])
            if hasattr(config, "qk_head_dim") and hasattr(config, "qk_nope_head_dim"):
                config.qk_head_dim = config.qk_nope_head_dim + config.qk_rope_head_dim

        if (
            config.architectures is not None
            and config.architectures[0] == "Phi4MMForCausalLM"
        ):
            from transformers import SiglipVisionConfig

            config.vision_config = SiglipVisionConfig(
                hidden_size=1152,
                image_size=448,
                intermediate_size=4304,
                model_type="siglip_vision_model",
                num_attention_heads=16,
                num_hidden_layers=26,
                patch_size=14,
            )

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

        is_ocr = _is_deepseek_ocr_model(config)
        is_ocr2 = _is_deepseek_ocr2_model(config)

        if is_ocr2:
            _override_v_head_dim_if_zero(config)
            config.model_type = "deepseek-ocr"
            _set_architectures(config, "DeepseekOCRForCausalLM")
            config = DeepseekVLV2Config.from_pretrained(model, revision=revision)
            _apply_deepseek_ocr_overrides(config, model)
        elif config.model_type in _CONFIG_REGISTRY:
            model_type = config.model_type
            if model_type == "deepseek_vl_v2" and is_ocr:
                model_type = "deepseek-ocr"
            config = _CONFIG_REGISTRY[model_type].from_pretrained(
                model, revision=revision
            )

            # Re-check after reloading config from registry
            if _is_deepseek_ocr_model(config) or _is_deepseek_ocr2_model(config):
                _apply_deepseek_ocr_overrides(config, model)
            else:
                config._name_or_path = model

        if isinstance(model, str) and config.model_type == "internvl_chat":
            for key, val in config.llm_config.__dict__.items():
                if not hasattr(config, key):
                    setattr(config, key, val)

        if config.model_type == "multi_modality":
            _set_architectures(config, "MultiModalityCausalLM")

        if config.model_type in (
            "gemma4",
            "gemma4_assistant",
            "gemma4_unified",
            "gemma4_unified_assistant",
            "diffusion_gemma",
        ):
            # Gemma4 configs use base attributes for SWA layers and `global_*`
            # variants for full-attention layers.  SGLang expects the opposite:
            # base = full-attention, `swa_*` = sliding-window overrides.
            text_config = config.text_config
            global_head_dim = getattr(text_config, "global_head_dim", None)
            global_kv_heads = getattr(text_config, "num_global_key_value_heads", None)

            swa_head_dim = text_config.head_dim
            swa_kv_heads = text_config.num_key_value_heads

            text_config.swa_head_dim = swa_head_dim
            text_config.swa_v_head_dim = swa_head_dim
            text_config.swa_num_key_value_heads = swa_kv_heads

            if global_head_dim is not None:
                text_config.head_dim = global_head_dim
            if global_kv_heads is not None:
                text_config.num_key_value_heads = global_kv_heads

            if not hasattr(text_config, "v_head_dim"):
                text_config.v_head_dim = text_config.head_dim
            if not hasattr(text_config, "swa_v_head_dim"):
                text_config.swa_v_head_dim = text_config.swa_head_dim

            # Unified Gemma4 names the end-of-audio token `eoa_token_index`,
            # but the multimodal processor expects `eoa_token_id`.
            if not hasattr(config, "eoa_token_id") and hasattr(
                config, "eoa_token_index"
            ):
                config.eoa_token_id = config.eoa_token_index

        if config.model_type == "longcat_flash":
            _set_architectures(config, "LongcatFlashForCausalLM")

        return config


@register_model_config_parser("mistral")
class MistralModelConfigParser(ModelConfigParserBase):
    def parse(
        self,
        model,
        trust_remote_code: bool,
        revision: Optional[str] = None,
        **kwargs,
    ):
        del kwargs
        return load_mistral_config(
            model, trust_remote_code=trust_remote_code, revision=revision
        )


@lru_cache_frozenset(maxsize=32)
def get_config(
    model: str,
    trust_remote_code: bool,
    revision: Optional[str] = None,
    model_override_args: Optional[dict] = None,
    model_config_parser: str = "auto",
    **kwargs,
):
    is_gguf = check_gguf_file(model)
    if is_gguf:
        if model_config_parser not in ("auto", "hf"):
            raise ValueError(
                f"model_config_parser={model_config_parser!r} is incompatible "
                "with GGUF inputs; only 'hf' (or 'auto') is supported."
            )
        _ensure_gguf_version()
        kwargs["gguf_file"] = model
        model = Path(model).parent
        # Skip auto-resolution for GGUF: the name-based Mistral heuristic
        # would misfire on the rewritten parent dir.
        model_config_parser = "hf"

    model = resolve_runai_obj_uri(model)

    if is_remote_url(model):
        client = create_remote_connector(model)
        client.pull_files(ignore_pattern=["*.pt", "*.safetensors", "*.bin"])
        model = client.get_local_dir()

    if model_config_parser == "auto":
        # `model` is post-rewrite (gguf parent / runai uri / remote pull).
        model_config_parser = "mistral" if is_mistral_model(model) else "hf"

    parser = get_model_config_parser(model_config_parser)
    config = parser.parse(
        model, trust_remote_code=trust_remote_code, revision=revision, **kwargs
    )

    if model_override_args:
        config.update(model_override_args)

    if is_gguf:
        if config.model_type not in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES:
            raise RuntimeError(f"Can't get gguf config for {config.model_type}.")
        _set_architectures(config, MODEL_FOR_CAUSAL_LM_MAPPING_NAMES[config.model_type])

    return config
