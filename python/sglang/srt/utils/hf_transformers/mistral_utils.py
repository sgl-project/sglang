# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/mistral.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import json
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from transformers import AutoConfig, PretrainedConfig, WhisperConfig

from sglang.srt.utils import logger

from .common import (
    _ensure_sub_configs,
    _resolve_local_or_cached_file,
    download_from_hf,
)


def adapt_config_dict(
    config_dict: dict[str, Any], model: str, **kwargs
) -> tuple[dict, PretrainedConfig]:
    config_dict.update(kwargs)
    config_dict = _remap_general_mistral_args(config_dict)

    if bool(config_dict.get("quantization")):
        config_dict = _remap_mistral_quantization_args(config_dict)

    is_moe = bool(config_dict.get("moe"))
    is_mistral_large_3 = (
        is_moe and (config_dict["moe"].get("num_shared_experts") or 0) > 0
    )
    is_eagle = "eagle" in model.lower()
    is_mla_eagle = is_eagle and any(
        config_dict.get(k) is not None
        for k in ("kv_lora_rank", "q_lora_rank", "v_head_dim")
    )
    if is_eagle and not is_moe and is_mla_eagle:
        # Dense MLA EAGLE draft model (e.g. Mistral Small 4 EAGLE).
        # Uses MLA attention like MistralLarge3 but has no MoE layers.
        # Set model_type to deepseek_v3 for MLA support, and override
        # MoE fields so all layers are dense.
        config_dict["model_type"] = "deepseek_v3"
        config_dict["architectures"] = ["MistralLarge3ForCausalLMEagle"]
        num_layers = config_dict.get("num_hidden_layers", 0)
        config_dict["n_routed_experts"] = 1
        config_dict["first_k_dense_replace"] = num_layers
        config_dict["moe_layer_freq"] = 1
        config_dict["n_shared_experts"] = 0
        config_dict["n_group"] = 1
        config_dict["topk_group"] = 1
        config_dict["num_experts_per_tok"] = 1
        config_dict["moe_intermediate_size"] = 1
        config_dict["routed_scaling_factor"] = 1.0
        config_dict["topk_method"] = None
        config_dict["scoring_func"] = "softmax"
        config_dict["routing_method_type"] = 1
    elif is_eagle and not is_moe:
        # Dense GQA EAGLE draft model (e.g. Mistral Medium 3.5 EAGLE).
        # Routes to a Llama-backbone draft body — no MoE shimming required.
        config_dict["architectures"] = ["MistralForCausalLMEagle"]
        config_dict["model_type"] = "mistral"
        config_dict["rope_is_neox_style"] = False
        for mla_key in (
            "q_lora_rank",
            "qk_rope_head_dim",
            "qk_nope_head_dim",
            "kv_lora_rank",
            "v_head_dim",
        ):
            if config_dict.get(mla_key) is None:
                config_dict.pop(mla_key, None)
    elif is_moe:
        if is_mistral_large_3:
            config_dict = _remap_moe_args(config_dict)
            config_dict["model_type"] = "deepseek_v3"
            if is_eagle:
                config_dict["architectures"] = ["MistralLarge3ForCausalLMEagle"]
            else:
                config_dict["architectures"] = ["MistralLarge3ForCausalLM"]

            assert (
                "llama_4_scaling" in config_dict
            ), "MistralLarge3 expect llama4 scaling config."
            llama_4_scaling_config_keys = ["original_max_position_embeddings", "beta"]
            assert all(
                [
                    key in config_dict["llama_4_scaling"]
                    for key in llama_4_scaling_config_keys
                ]
            ), (
                "llama_4_scaling config should define the keys: "
                f"{','.join(llama_4_scaling_config_keys)}"
            )
        else:
            config_dict["architectures"] = ["MixtralForCausalLM"]
    else:
        config_dict["architectures"] = ["MistralForCausalLM"]
        config_dict["model_type"] = "mistral"
        # Mistral models use non-interleaved RoPE (is_neox_style=False),
        # unlike Llama which defaults to True.
        config_dict["rope_is_neox_style"] = False
        # Remove None-valued MLA fields that would shadow defaults in
        # model_config._derive_model_shapes (getattr returns None instead
        # of the fallback when the attribute exists but is None).
        for mla_key in (
            "q_lora_rank",
            "qk_rope_head_dim",
            "qk_nope_head_dim",
            "kv_lora_rank",
            "v_head_dim",
        ):
            if config_dict.get(mla_key) is None:
                config_dict.pop(mla_key, None)

    if bool(config_dict.get("yarn")):
        config_dict = _remap_mistral_yarn_args(config_dict)

    is_vision = bool(
        (config_dict.get("multimodal") or {}).get("vision_encoder_args")
        or config_dict.get("vision_encoder")
    )
    is_audio = bool(
        ((config_dict.get("multimodal") or {}).get("whisper_model_args") or {}).get(
            "encoder_args"
        )
    )

    assert not (is_vision and is_audio), "Vision and audio are mutually exclusive"

    if is_vision:
        config_dict = _remap_mistral_vision_args(config_dict)
    if is_audio:
        config_dict = _remap_mistral_audio_args(config_dict)

    config = PretrainedConfig.from_dict(config_dict)

    logger.debug("Initialized config %s", config)

    return config_dict, config


def _remap_mistral_vision_args(config: dict) -> dict:
    if config.get("multimodal"):
        vision_config = config.pop("multimodal")
    else:
        vision_config = config.pop("vision_encoder")

    quant_config = config.get("quantization_config")

    config = {
        "model_type": "pixtral",
        "architectures": ["PixtralForConditionalGeneration"],
        "text_config": config,
        "vision_config": {"model_type": "pixtral", **vision_config},
    }
    if quant_config:
        config["quantization_config"] = quant_config
    return config


def _remap_mistral_yarn_args(config: dict) -> dict:
    yarn_config_map = {
        "factor": "factor",
        "original_max_position_embeddings": "original_max_position_embeddings",
        "beta": "beta_fast",
        "alpha": "beta_slow",
        "apply_scale": "apply_yarn_scaling",
    }
    yarn_config = config.get("yarn") or {}
    config["rope_scaling"] = {
        "rope_type": "deepseek_yarn",
        "mscale_all_dim": 1,
    }
    # Include rope_theta in rope_scaling if present at the top level,
    # as transformers yarn validation requires it.
    if "rope_theta" in config:
        config["rope_scaling"]["rope_theta"] = config["rope_theta"]
    for old_name, new_name in yarn_config_map.items():
        if old_name in yarn_config:
            value = yarn_config.pop(old_name)
            if new_name is not None:
                config["rope_scaling"][new_name] = value

    assert len(yarn_config) == 0, f"Unparsed yarn config: {yarn_config}"

    return config


def _remap_general_mistral_args(config: dict) -> dict:
    # Mistral key -> HF key
    config_mapping = {
        "dim": "hidden_size",
        "norm_eps": "rms_norm_eps",
        "n_kv_heads": "num_key_value_heads",
        "n_layers": "num_hidden_layers",
        "n_heads": "num_attention_heads",
        "hidden_dim": "intermediate_size",
    }
    # HF key -> (Mistral key, default value)
    top_level_mapping_with_default = {
        "model_type": ("model_type", "transformer"),
        "hidden_act": ("activation", "silu"),
        "tie_word_embeddings": ("tied_embeddings", False),
        "max_seq_len": ("max_seq_len", 128_000),
        "max_position_embeddings": ("max_position_embeddings", 128_000),
    }

    for key, new_key in config_mapping.items():
        if key in config:
            config[new_key] = config.pop(key)

    for new_key, (key, default_value) in top_level_mapping_with_default.items():
        config[new_key] = config.pop(key, default_value)

    return config


def _remap_mistral_quantization_args(config: dict) -> dict:
    if config.get("quantization"):
        quantization = config.pop("quantization", {})
        if quantization.get("qformat_weight") == "fp8_e4m3":
            qscheme_act = quantization.get("qscheme_act")
            assert qscheme_act in (
                "NO_SCALES",
                "TENSOR",
                None,
            ), "Only NO_SCALES and TENSOR (default) are supported for qscheme_act"
            is_dynamic = qscheme_act == "NO_SCALES"
            config["quantization_config"] = {
                "quant_method": "fp8",
                "activation_scheme": "dynamic" if is_dynamic else "static",
            }
        else:
            raise ValueError(f"Found unknown quantization='{quantization}' in config")

    return config


def _remap_mistral_audio_args(config: dict) -> dict:
    whisper_args = config["multimodal"].pop("whisper_model_args")
    encoder_args = whisper_args["encoder_args"]
    downsample_args = whisper_args["downsample_args"]

    quant_config = config.get("quantization_config")
    config = {
        "model_type": "whixtral",
        "architectures": ["VoxtralForConditionalGeneration"],
        "text_config": PretrainedConfig.from_dict(config),
        "audio_config": WhisperConfig(
            num_mel_bins=encoder_args["audio_encoding_args"]["num_mel_bins"],
            window_size=encoder_args["audio_encoding_args"]["window_size"],
            sampling_rate=encoder_args["audio_encoding_args"]["sampling_rate"],
            hop_length=encoder_args["audio_encoding_args"]["hop_length"],
            downsample_factor=downsample_args["downsample_factor"],
            d_model=encoder_args["dim"],
            encoder_layers=encoder_args["n_layers"],
            encoder_ffn_dim=encoder_args["hidden_dim"],
            encoder_attention_heads=encoder_args["n_heads"],
            vocab_size=encoder_args["vocab_size"],
            max_source_positions=encoder_args["max_source_positions"],
            is_encoder_decoder=False,  # Override WhisperConfig default
        ),
    }
    if quant_config:
        config["quantization_config"] = quant_config
    return config


def _remap_moe_args(config: dict) -> dict:
    moe_config_map = {
        "route_every_n": "moe_layer_freq",
        "first_k_dense_replace": "first_k_dense_replace",
        "num_experts_per_tok": "num_experts_per_tok",
        "num_experts": "n_routed_experts",
        "expert_hidden_dim": "moe_intermediate_size",
        "routed_scale": "routed_scaling_factor",
        "num_shared_experts": "n_shared_experts",
        "num_expert_groups": "n_group",
        "num_expert_groups_per_tok": "topk_group",
    }
    moe_config = config.get("moe", {})
    for old_name, new_name in moe_config_map.items():
        if old_name in moe_config:
            value = moe_config.pop(old_name)
            config[new_name] = value

    config["topk_method"] = None
    config["scoring_func"] = "softmax"
    config["routing_method_type"] = 1  # RoutingMethodType.Renormalize

    return config


class MistralConfigParser:
    def get_hf_file_to_dict(
        self, file_name: str, model: str | Path, revision: str | None = "main"
    ):
        file_path = Path(model) / file_name
        if not file_path.is_file():
            raise FileNotFoundError(f"File not found {model}, {file_name}")

        with open(file_path) as file:
            return json.load(file)

    def _download_mistral_config_file(self, model, revision) -> dict:
        config_file_name = "params.json"
        config_dict = self.get_hf_file_to_dict(config_file_name, model, revision)
        if config_dict is None:
            raise ValueError(
                f"Failed to load mistral '{config_file_name}' config for model "
                f"{model}. Please check if the model is a mistral-format model "
                f"and if the config file exists."
            )
        assert isinstance(config_dict, dict)
        return config_dict

    def parse(
        self,
        model: str | Path,
        revision: str | None = None,
        **kwargs,
    ) -> tuple[dict, PretrainedConfig]:
        config_dict = self._download_mistral_config_file(model, revision)
        if config_dict.get("max_position_embeddings") is None:
            logger.warning(
                "The params.json file is missing 'max_position_embeddings'"
                " and could not get a value from the HF config."
                " Defaulting to 128000"
            )
            config_dict["max_position_embeddings"] = 128_000

        config_dict, config = adapt_config_dict(config_dict, model)

        # Mistral configs may define sliding_window as list[int]. Convert it
        # to int and add the layer_types list[str] to make it HF compatible
        if (sliding_window := getattr(config, "sliding_window", None)) and isinstance(
            sliding_window, list
        ):
            pattern_repeats = config.num_hidden_layers // len(sliding_window)
            layer_types = sliding_window * pattern_repeats
            config.layer_types = [
                "full_attention" if layer_type is None else "sliding_attention"
                for layer_type in layer_types
            ]
            config.sliding_window = next(filter(None, sliding_window), None)

        return config_dict, config


def is_mistral_model(name) -> bool:
    """Return True if *name* refers to a Mistral model needing the custom parser."""
    lower = str(name).lower()
    if "mistral-large-3" in lower or "mistral-small-4" in lower or "leanstral" in lower:
        return True
    # EAGLE drafts for Mistral targets ship native-format only (params.json +
    # consolidated.safetensors, no config.json), so route them through the
    # custom parser regardless of the base model name.
    if "eagle" in lower and "mistral" in lower:
        return True
    return False


@lru_cache(maxsize=2)
def load_mistral_config(
    model_path: str,
    trust_remote_code: bool = False,
    revision: Optional[str] = None,
):
    """Load and parse a Mistral model config via the custom params.json format.

    Returns a ``PretrainedConfig`` with dict sub-configs (text_config,
    vision_config) converted to proper AutoConfig objects.
    """
    local_path = download_from_hf(model_path)
    parser = MistralConfigParser()
    config_dict, _ = parser.parse(local_path)

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as f:
        json.dump(config_dict, f)
        f.flush()
        loaded_config = AutoConfig.from_pretrained(
            f.name, trust_remote_code=trust_remote_code, revision=revision
        )
    _ensure_sub_configs(loaded_config, "text_config", "vision_config")

    return loaded_config


def wrap_as_pixtral(processor, config):
    """Wrap a tokenizer as a PixtralProcessor for Mistral vision models."""
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
    return HFPixtralProcessor(
        image_processor=image_processor,
        tokenizer=processor,
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
    )


# kwargs that MistralCommon tokenizers reject.
_MISTRAL_COMMON_REJECTED_KWARGS = frozenset(
    {
        "trust_remote_code",
        "tokenizer_revision",
        "use_fast",
        "_from_auto",
        "clean_up_tokenization_spaces",
    }
)

# Models whose tokenizer should be loaded from a different checkpoint.
_MISTRAL_TOKENIZER_REDIRECTS = {
    # TODO(Xinyuan): Remove this once we have a proper tokenizer for Devstral
    "mistralai/Devstral-Small-2505": "mistralai/Mistral-Small-3.1-24B-Instruct-2503",
}


def is_bare_tekken_checkpoint(tokenizer_name, revision=None) -> bool:
    """True iff the checkpoint ships tekken.json but no tokenizer.json.

    AutoTokenizer converts tekken.json on the fly, but the converter assigns
    BPE ids from rank 0, dropping the 1000 special-token slots that precede
    the BPE vocab in tekken's id space — every encoded id is shifted and
    generation produces garbage. Such checkpoints must load through the
    mistral-common backed tokenizer instead.
    """

    def _has(filename):
        try:
            _resolve_local_or_cached_file(tokenizer_name, filename, revision)
            return True
        except Exception:
            return False

    return _has("tekken.json") and not _has("tokenizer.json")


def retry_without_mistral_common_kwargs(tokenizer_name, *args, **common_kwargs):
    """Retry ``AutoTokenizer.from_pretrained`` without kwargs that MistralCommon rejects.

    Returns the loaded tokenizer, or *None* if the error is not a
    MistralCommon kwargs rejection.
    """
    from transformers import AutoTokenizer

    stripped = {
        k: v
        for k, v in common_kwargs.items()
        if k not in _MISTRAL_COMMON_REJECTED_KWARGS
    }
    return AutoTokenizer.from_pretrained(tokenizer_name, *args, **stripped)


def patch_mistral_common_tokenizer(tokenizer):
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

    if not hasattr(tokenizer, "get_added_vocab"):
        tokenizer.get_added_vocab = lambda: {}

    # Keep the old no-op pad add working on transformers 5.12 MistralCommon.
    _orig_add_special_tokens = tokenizer.add_special_tokens

    def _safe_add_special_tokens(special_tokens_dict, *args, **kwargs):
        if set(special_tokens_dict) == {"pad_token"}:
            tokenizer.pad_token = special_tokens_dict["pad_token"]
            return 0
        return _orig_add_special_tokens(special_tokens_dict, *args, **kwargs)

    tokenizer.add_special_tokens = _safe_add_special_tokens

    # Set a chat_template containing "audio" so that sglang's content format
    # detector returns "openai" (which preserves audio_url extraction).
    if not hasattr(tokenizer, "chat_template") or tokenizer.chat_template is None:
        tokenizer.chat_template = "<!-- audio/image multimodal -->"

    _orig_convert = tokenizer.convert_tokens_to_ids

    def _safe_convert(val):
        try:
            return _orig_convert(val)
        except AssertionError:
            logger.debug(
                "convert_tokens_to_ids failed for %r, returning unk_token_id", val
            )
            return getattr(tokenizer, "unk_token_id", None)

    tokenizer.convert_tokens_to_ids = _safe_convert

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

    if hasattr(tokenizer, "_text_to_ids"):
        _orig_text_to_ids = tokenizer._text_to_ids
        marker_to_id = {
            "[IMG]": tokenizer.convert_tokens_to_ids("[IMG]"),
            "[IMG_BREAK]": tokenizer.convert_tokens_to_ids("[IMG_BREAK]"),
            "[IMG_END]": tokenizer.convert_tokens_to_ids("[IMG_END]"),
        }

        def _text_to_ids_with_pixtral_markers(text, add_special_tokens):
            if not isinstance(text, str) or not any(
                marker in text for marker in marker_to_id
            ):
                return _orig_text_to_ids(text, add_special_tokens)

            ids = []
            pos = 0
            while pos < len(text):
                next_marker = None
                next_idx = len(text)
                for marker in marker_to_id:
                    marker_idx = text.find(marker, pos)
                    if marker_idx != -1 and marker_idx < next_idx:
                        next_marker = marker
                        next_idx = marker_idx

                if next_marker is None:
                    ids.extend(_orig_text_to_ids(text[pos:], False))
                    break
                if next_idx > pos:
                    ids.extend(_orig_text_to_ids(text[pos:next_idx], False))
                ids.append(marker_to_id[next_marker])
                pos = next_idx + len(next_marker)

            if add_special_tokens:
                return tokenizer.build_inputs_with_special_tokens(ids)
            return ids

        tokenizer._text_to_ids = _text_to_ids_with_pixtral_markers

    tokenizer._orig_apply_chat_template = tokenizer.apply_chat_template

    def _adapt_placeholder_content_for_mistral_common(content):
        if not isinstance(content, list):
            return content

        rendered_parts = []
        has_placeholder = False
        for part in content:
            if not isinstance(part, dict):
                return content
            part_type = part.get("type")
            if part_type in ("text", "input_text"):
                rendered_parts.append(part.get("text", ""))
            elif part_type == "image" and not any(
                key in part for key in ("url", "path", "base64")
            ):
                has_placeholder = True
                rendered_parts.append("[IMG]")
            elif part_type in ("audio", "video") and not any(
                key in part for key in ("url", "path", "base64")
            ):
                has_placeholder = True
                continue
            else:
                return content

        return "".join(rendered_parts) if has_placeholder else content

    def _adapt_placeholder_messages_for_mistral_common(messages):
        if not isinstance(messages, (list, tuple)):
            return messages

        adapted = []
        for msg in messages:
            if isinstance(msg, (list, tuple)):
                adapted.append(_adapt_placeholder_messages_for_mistral_common(msg))
            elif isinstance(msg, dict):
                adapted.append(
                    {
                        **msg,
                        "content": _adapt_placeholder_content_for_mistral_common(
                            msg.get("content", "")
                        ),
                    }
                )
            else:
                adapted.append(msg)
        return adapted

    def _safe_apply_chat_template(messages, **kwargs):
        kwargs.pop("add_generation_prompt", None)
        messages = _adapt_placeholder_messages_for_mistral_common(messages)
        return tokenizer._orig_apply_chat_template(messages, **kwargs)

    tokenizer.apply_chat_template = _safe_apply_chat_template
    return tokenizer
