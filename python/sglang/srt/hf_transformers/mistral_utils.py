# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/transformers_utils/configs/mistral.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import copy
import glob
import importlib.metadata
import json
import math
import os
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

import numpy as np
from packaging.version import Version
from transformers import AutoConfig, PretrainedConfig, WhisperConfig

from sglang.srt.utils import logger
from sglang.srt.utils.common import find_local_repo_dir

from .common import (
    _cached_file_exists,
    _ensure_sub_configs,
    _remote_file_exists,
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
    has_shared_experts = (
        is_moe and (config_dict["moe"].get("num_shared_experts") or 0) > 0
    )
    dsa_args = config_dict.get("dsa")
    if isinstance(dsa_args, dict):
        dsa_args = config_dict["dsa"] = _canonical_dsa_block(dsa_args)
    is_dsa = isinstance(dsa_args, dict) and dsa_args.get("index_topk") is not None
    config_dict = _remap_mistral_nope_args(config_dict)
    config_dict = _remap_mistral_kv_sharing_args(config_dict)
    config_dict = _remap_mistral_dsa_args(config_dict)

    is_eagle = "eagle" in model.lower()
    is_mla_eagle = is_eagle and any(
        config_dict.get(k) is not None
        for k in ("kv_lora_rank", "q_lora_rank", "v_head_dim")
    )
    if is_dsa:
        if not config_dict.get("qk_nope_head_dim"):
            raise ValueError("Mistral DSA requires an MLA configuration.")
        if not has_shared_experts:
            raise ValueError(
                "Mistral DSA is only supported on MoE configs with shared experts."
            )
        config_dict = _remap_moe_args(config_dict)
        config_dict = _remap_mistral_large_4_args(config_dict)
    elif is_eagle and not is_moe and is_mla_eagle:
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
        if has_shared_experts:
            config_dict = _remap_moe_args(config_dict)
            config_dict["model_type"] = "deepseek_v3"
            if is_eagle:
                config_dict["architectures"] = ["MistralLarge3ForCausalLMEagle"]
            else:
                config_dict["architectures"] = ["MistralLarge3ForCausalLM"]

            assert "llama_4_scaling" in config_dict, (
                "MistralLarge3 expect llama4 scaling config."
            )
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
        # The speculative hooks find a bundled DSpark draft on the outer config.
        **{key: value for key, value in config.items() if key.startswith("dspark_")},
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
        "max_seq_len": ("max_seq_len", config.get("max_position_embeddings", 128_000)),
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
            qscheme_weight = quantization.get("qscheme_weight")
            if qscheme_act not in ("NO_SCALES", "TENSOR", "BLOCK", None):
                raise ValueError(
                    f"Unsupported qscheme_act {qscheme_act!r}; "
                    "expected NO_SCALES, TENSOR, or BLOCK"
                )
            if qscheme_act == "BLOCK" and qscheme_weight != "BLOCK":
                raise ValueError("BLOCK qscheme_act requires BLOCK qscheme_weight")
            is_dynamic = qscheme_act in ("NO_SCALES", "BLOCK") or (
                qscheme_weight == "BLOCK"
            )
            config["quantization_config"] = {
                "quant_method": "fp8",
                "activation_scheme": "dynamic" if is_dynamic else "static",
            }
            if qscheme_weight == "BLOCK":
                config["quantization_config"]["weight_block_size"] = [
                    quantization["block_size_out"],
                    quantization["block_size_in"],
                ]
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


def _remap_mistral_large_4_args(config: dict) -> dict:
    quant_config = config.get("quantization_config")
    if quant_config and quant_config["quant_method"] == "fp8":
        # The checkpoint keeps the MLA KV projections in BF16 without scales.
        quant_config["ignored_layers"] = quant_config.get("ignored_layers", []) + [
            f"{prefix}.layers.{layer_idx}.self_attn.{projection}"
            for prefix in ("model", "language_model.model")
            for layer_idx in range(config["num_hidden_layers"])
            for projection in ("kv_a_proj_with_mqa", "kv_b_proj")
        ]
    if quant_config and quant_config["quant_method"] == "modelopt":
        # Without this key the loader looks for an hf_quant_config.json that
        # Mistral-format checkpoints do not ship.
        quant_config.setdefault("kv_cache_quant_algo", None)
    # Routing renormalizes the top-k weights; don't inherit it from a transformers default.
    config["norm_topk_prob"] = True
    config["model_type"] = "mistral_large_4"
    config["architectures"] = ["MistralLarge4ForCausalLM"]
    return _remap_mistral_dspark_args(config)


_MTP_KEYS = frozenset(
    {
        "n_layers",
        "depth",
        "ragged_attention",
        "target_activation_layers",
        "norm_strategy",
        "dspark",
    }
)
_DSPARK_HEAD_TYPES = ("vanilla", "rnn")
_DSPARK_DRAFT_INPUTS = ("shared_mask", "position_specific")


def _remap_mistral_dspark_args(config: dict) -> dict:
    """Record the DSpark draft bundled with the checkpoint (``mtp_dspark.*``)."""
    mtp = config.pop("mtp", None)
    if mtp is None or mtp.get("dspark") is None:
        return config
    if unknown := set(mtp) - _MTP_KEYS:
        raise ValueError(
            f"Unknown keys in the params.json mtp block: {sorted(unknown)}"
        )
    dspark = dict(mtp["dspark"])
    markov_rank = int(dspark.pop("markov_rank"))
    head_type = dspark.pop("head_type", "vanilla")
    draft_input = dspark.pop("draft_input", "shared_mask")
    if dspark:
        raise ValueError(f"Unknown keys in the mtp.dspark block: {sorted(dspark)}")
    if head_type not in _DSPARK_HEAD_TYPES:
        raise ValueError(f"Unsupported DSpark head_type {head_type!r}")
    if draft_input not in _DSPARK_DRAFT_INPUTS:
        raise ValueError(f"Unsupported DSpark draft_input {draft_input!r}")
    if mtp.get("norm_strategy", "eagle") != "eagle":
        raise ValueError(
            f"Unsupported DSpark norm_strategy {mtp['norm_strategy']!r}; "
            "only eagle (project, then normalize the target taps) is implemented."
        )

    depth = int(mtp["depth"])
    if draft_input == "position_specific" and depth < 2:
        raise ValueError(f"Position-specific DSpark masks need depth > 1, got {depth}")
    target_layers = [int(v) for v in mtp["target_activation_layers"].split(",")]
    if target_layers != sorted(set(target_layers)) or not (
        0 <= target_layers[0] and target_layers[-1] < config["num_hidden_layers"]
    ):
        raise ValueError(
            "DSpark target_activation_layers must be unique ascending target layers, "
            f"got {target_layers}"
        )
    windows = {int(v) for v in (mtp.get("ragged_attention") or "").split(",") if v}
    if len(windows) > 1:
        raise NotImplementedError(
            f"DSpark draft layers must share one window, got {mtp['ragged_attention']}"
        )

    config.update(
        {
            "dspark_block_size": depth,
            "dspark_markov_rank": markov_rank,
            "dspark_markov_head_type": head_type,
            "dspark_target_layer_ids": target_layers,
            # Mask slots read the learned mask_embedding, so this id is never embedded.
            "dspark_noise_token_id": 0,
            "dspark_draft_input": draft_input,
            "dspark_num_layers": int(mtp["n_layers"]),
            "dspark_sliding_window": windows.pop() if windows else None,
        }
    )
    return config


def mistral_large_4_dspark_draft_config(
    text_config: PretrainedConfig,
) -> PretrainedConfig:
    """The bundled draft: dense BF16 MLA layers with the target's attention
    geometry, FFN width and YaRN RoPE."""
    config = copy.deepcopy(text_config)
    for key in ("nope_layer_flags", "indexer_nope_layer_flags", "kv_sharing_config"):
        if hasattr(config, key):
            delattr(config, key)
    num_layers = config.dspark_num_layers
    window = config.dspark_sliding_window
    config.update(
        {
            "architectures": ["MistralLarge4DSparkModel"],
            "num_hidden_layers": num_layers,
            "layer_types": ["full_attention" if window is None else "sliding_attention"]
            * num_layers,
            "sliding_window": window,
            "is_causal": False,
            "quantization_config": None,
            "markov_rank": config.dspark_markov_rank,
            "markov_head_type": config.dspark_markov_head_type,
            "sample_from_anchor": True,
            "dflash_config": {
                "block_size": config.dspark_block_size,
                "target_layer_ids": config.dspark_target_layer_ids,
                "mask_token_id": config.dspark_noise_token_id,
            },
        }
    )
    return config


def _expand_mistral_pattern(pattern: list[Any], num_layers: int) -> list[Any]:
    assert pattern, "Per-layer patterns must not be empty."
    return (pattern * math.ceil(num_layers / len(pattern)))[:num_layers]


def _remap_mistral_nope_args(config: dict) -> dict:
    nope = config.pop("nope", None)
    if nope:
        if not isinstance(nope, str):
            raise ValueError(f"nope must be a comma-separated string, got {nope!r}")
        config["nope_layer_flags"] = _expand_mistral_pattern(
            [value == "nope" for value in nope.split(",")],
            config["num_hidden_layers"],
        )
    return config


def _remap_mistral_kv_sharing_args(config: dict) -> dict:
    kv_sharing = config.pop("kv_sharing", None)
    if kv_sharing is None:
        return config
    if not isinstance(kv_sharing, dict):
        raise ValueError(f"kv_sharing must be a dict, got {kv_sharing!r}")
    if set(kv_sharing) != {"source_layers_str"}:
        raise ValueError(
            f"Only kv_sharing.source_layers_str is supported, got {sorted(kv_sharing)}"
        )
    source_layers_str = kv_sharing["source_layers_str"]
    if not isinstance(source_layers_str, str):
        raise ValueError("kv_sharing.source_layers_str must be a string")
    if isinstance(config.get("sliding_window"), list):
        raise ValueError("KV sharing with per-layer sliding windows is not supported")

    num_layers = config["num_hidden_layers"]
    _validate_kv_sharing_sources(
        num_layers=num_layers,
        layer_types=config.get("layer_types") or ["full_attention"] * num_layers,
        sources=[int(source) for source in source_layers_str.split(",")],
        nope_layer_flags=config.get("nope_layer_flags"),
    )
    config["kv_sharing_config"] = {"source_layers_str": source_layers_str}
    return config


def _validate_kv_sharing_sources(
    num_layers: int,
    layer_types: list[str],
    sources: list[int],
    nope_layer_flags: Optional[list[bool]],
) -> None:
    if not len(layer_types) == len(sources) == num_layers:
        raise ValueError(
            f"Expected {num_layers} KV sharing entries, "
            f"got {len(layer_types)} layer types and {len(sources)} sources"
        )
    assert nope_layer_flags is None or len(nope_layer_flags) == num_layers

    providers = {source for layer, source in enumerate(sources) if source != layer}
    full_attn_provider: Optional[int] = None
    for layer, source in enumerate(sources):
        if source == layer and layer not in providers:
            continue
        if layer_types[layer] != "full_attention":
            raise ValueError(
                f"KV sharing layer {layer} is {layer_types[layer]}; "
                "only full-attention layers can share KV"
            )
        if source == layer:
            full_attn_provider = layer
            continue
        if source != full_attn_provider:
            raise ValueError(
                f"KV consumer {layer} must reference the latest KV provider "
                f"{full_attn_provider}, got {source}"
            )
        if (
            nope_layer_flags is not None
            and nope_layer_flags[layer] != nope_layer_flags[source]
        ):
            raise ValueError(
                f"KV source {source} and consumer {layer} must have the same "
                "NoPE/RoPE setting"
            )


_DSA_ARG_KEYS = (
    "index_topk",
    "index_n_heads",
    "index_head_dim",
    "indexer_norm_eps",
    "index_nope",
    "index_rope_head_dim",
    "index_sharing",
    "training_stage",
    "dsa_detach_indexer_input",
    "dsa_training_stage",
    "dsa_fp8_indexer",
    "dsa_fp8_kv_cache",
    "indexer_match_nope",
    "indexer_ssmax_scaling",
)

# Converted checkpoints name the dsa keys either with the index prefix or with the
# trainer's unprefixed names.
_DSA_UNPREFIXED_KEYS = {
    "topk": "index_topk",
    "n_heads": "index_n_heads",
    "head_dim": "index_head_dim",
    "rope_head_dim": "index_rope_head_dim",
    "match_nope": "indexer_match_nope",
    "ssmax_scaling": "indexer_ssmax_scaling",
    "norm_eps": "indexer_norm_eps",
    "sharing": "index_sharing",
}
_DSA_PREFIXED_KEYS = frozenset(_DSA_UNPREFIXED_KEYS.values()) | {"index_nope"}
# The dsa_-prefixed precision names are this converter's outputs, not
# params.json spellings; inside the block they would otherwise be dropped.
_DSA_PRECISION_OUTPUT_KEYS = {
    "dsa_fp8_indexer": "fp8_indexer",
    "dsa_fp8_kv_cache": "fp8_kv_cache",
}
_DSA_SHARED_KEYS = (
    frozenset(_DSA_ARG_KEYS) - _DSA_PREFIXED_KEYS - set(_DSA_PRECISION_OUTPUT_KEYS)
) | set(_DSA_PRECISION_OUTPUT_KEYS.values())


def _canonical_dsa_block(dsa: dict) -> dict:
    for key, spelling in _DSA_PRECISION_OUTPUT_KEYS.items():
        if key in dsa:
            raise ValueError(
                f"The params.json dsa block sets {key!r}; the checkpoint "
                f"spelling is {spelling!r}."
            )
    unknown = (
        set(dsa) - _DSA_PREFIXED_KEYS - set(_DSA_UNPREFIXED_KEYS) - _DSA_SHARED_KEYS
    )
    if unknown:
        raise ValueError(
            f"Unknown keys in the params.json dsa block: {sorted(unknown)}."
        )
    prefixed = sorted(set(dsa) & _DSA_PREFIXED_KEYS)
    unprefixed = sorted(set(dsa) & set(_DSA_UNPREFIXED_KEYS))
    if prefixed and unprefixed:
        raise ValueError(
            "The params.json dsa block mixes prefixed keys "
            f"{prefixed} with unprefixed keys {unprefixed}; use one spelling."
        )
    return {_DSA_UNPREFIXED_KEYS.get(key, key): value for key, value in dsa.items()}


def _remap_mistral_dsa_args(config: dict) -> dict:
    dsa = config.pop("dsa", None)
    for key in _DSA_ARG_KEYS:
        config.pop(key, None)
    if dsa is None:
        return config
    if not isinstance(dsa, dict):
        raise ValueError(f"Expected the `dsa` block to be a dict, got {dsa!r}")
    dsa = _canonical_dsa_block(dsa)
    if dsa.get("index_topk") is None:
        return config

    config.update({key: dsa[key] for key in _DSA_ARG_KEYS if key in dsa})
    if config.pop("index_nope", False):
        raise ValueError("DSA indexer with index_nope=True is not supported.")
    index_rope_head_dim = config.pop("index_rope_head_dim", -1)
    if index_rope_head_dim not in (-1, config["qk_rope_head_dim"]):
        raise ValueError(
            f"DSA index_rope_head_dim={index_rope_head_dim} must be -1 or equal "
            f"qk_rope_head_dim={config['qk_rope_head_dim']}."
        )

    if config.pop("indexer_match_nope", False):
        config["indexer_nope_layer_flags"] = list(
            config.get("nope_layer_flags", [False] * config["num_hidden_layers"])
        )
    config["indexer_ssmax_scaling"] = bool(config.get("indexer_ssmax_scaling", False))
    config["indexer_rope_interleave"] = True
    config["index_kv_dtype"] = "fp8" if dsa.get("fp8_indexer", True) else "bf16"
    if "fp8_kv_cache" in dsa:
        config["dsa_fp8_kv_cache"] = dsa["fp8_kv_cache"]

    index_sharing = config.pop("index_sharing", None)
    if index_sharing is not None:
        if not isinstance(index_sharing, str):
            raise ValueError("index_sharing must be a comma-separated string")
        sharing = index_sharing.split(",")
        if not set(sharing) <= {"full", "shared"}:
            raise ValueError(f"Unexpected index_sharing values: {set(sharing)}")
        config["use_index_cache"] = True
        config["index_topk_pattern"] = _expand_mistral_pattern(
            ["S" if value == "shared" else "F" for value in sharing],
            config["num_hidden_layers"],
        )

    config.pop("training_stage", None)
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


def is_mistral_native_checkpoint(model_path: str, revision: Optional[str]) -> bool:
    """True iff the checkpoint requires load_format=mistral.

    Looks for consolidated*.safetensors with no competing
    model*.safetensors; when both weight formats ship in the
    same checkpoint (e.g. Mistral-7B-Instruct-v0.3) the HF path is
    preferred to avoid loading Mistral-named weights into an
    HF-named architecture.

    Name override: mistral-large-3 / mistral-small-4 /
    leanstral always treat as Mistral-native when params.json
    is present -- those families need Mistral weight loading
    regardless of which weight files happen to be present.
    """
    _MISTRAL_NATIVE_PATTERNS = (
        "mistral-large-3",
        "mistral-small-4",
        "leanstral",
    )
    name_matches = any(p in str(model_path).lower() for p in _MISTRAL_NATIVE_PATTERNS)

    def _check_format(has_params, has_consolidated, has_hf_weights) -> bool:
        if has_params and name_matches:
            return True
        return has_consolidated and not has_hf_weights

    if os.path.isdir(model_path):
        return _check_format(
            has_params=os.path.exists(os.path.join(model_path, "params.json")),
            has_consolidated=bool(
                glob.glob(os.path.join(model_path, "consolidated*.safetensors"))
            ),
            has_hf_weights=bool(
                glob.glob(os.path.join(model_path, "model*.safetensors"))
            ),
        )

    try:
        from huggingface_hub import HfApi

        files = {
            s.rfilename
            for s in HfApi().model_info(model_path, revision=revision).siblings
        }
    except Exception:
        # Hub unreachable (offline, rate-limited): judge from the local snapshot.
        snapshot_dir = find_local_repo_dir(model_path, revision)
        if snapshot_dir is None:
            return False
        files = set(os.listdir(snapshot_dir))
    return _check_format(
        has_params="params.json" in files,
        has_consolidated=any(
            f.startswith("consolidated") and f.endswith(".safetensors") for f in files
        ),
        has_hf_weights=any(
            f.startswith("model") and f.endswith(".safetensors") and "/" not in f
            for f in files
        ),
    )


def is_mistral_model(name, revision: Optional[str] = None) -> bool:
    """Return True if *name* refers to a Mistral model needing the custom parser."""
    lower = str(name).lower()
    if "mistral-large-3" in lower or "mistral-small-4" in lower or "leanstral" in lower:
        return True
    # EAGLE drafts for Mistral targets ship native-format only (params.json +
    # consolidated.safetensors, no config.json), so route them through the
    # custom parser regardless of the base model name.
    if "eagle" in lower and "mistral" in lower:
        return True
    return _is_mistral_format_checkpoint(name, revision)


def _is_mistral_format_checkpoint(name, revision: Optional[str] = None) -> bool:
    # A config.json next to Mistral-native consolidated weights is only a
    # transformers view of the checkpoint; the weights decide the format.
    local_dir = Path(name)
    if local_dir.is_dir():
        if not (local_dir / "params.json").is_file():
            return False
        return not (local_dir / "config.json").is_file() or (
            is_mistral_native_checkpoint(name, revision)
        )

    config_cached = _cached_file_exists(name, "config.json", revision)
    has_params = _cached_file_exists(
        name, "params.json", revision
    ) or _remote_file_exists(name, "params.json", revision)
    if not has_params:
        return False
    if not config_cached and not _remote_file_exists(name, "config.json", revision):
        return True
    return is_mistral_native_checkpoint(name, revision)


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
    local_path = download_from_hf(model_path, revision=revision)
    parser = MistralConfigParser()
    config_dict, _ = parser.parse(local_path)

    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json") as f:
        json.dump(config_dict, f)
        f.flush()
        loaded_config = AutoConfig.from_pretrained(
            f.name, trust_remote_code=trust_remote_code
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

    local_dir = Path(tokenizer_name)
    if local_dir.is_dir():
        return (local_dir / "tekken.json").is_file() and not (
            local_dir / "tokenizer.json"
        ).is_file()

    if _cached_file_exists(tokenizer_name, "tokenizer.json", revision):
        return False
    if _cached_file_exists(tokenizer_name, "tekken.json", revision):
        return True

    # Cold cache: the tokenizer loads before weights, so tekken.json isn't
    # cached yet on a first launch — HEAD-probe the hub to still detect it.
    if not _remote_file_exists(tokenizer_name, "tekken.json", revision):
        return False
    return not _remote_file_exists(tokenizer_name, "tokenizer.json", revision)


_MISTRAL_COMMON_MODEL_SETTINGS_MIN_VERSION = Version("1.13.0")


def load_mistral_common_tokenizer(tokenizer_name, revision=None):
    from transformers.tokenization_mistral_common import MistralCommonTokenizer

    try:
        return MistralCommonTokenizer.from_pretrained(tokenizer_name, revision=revision)
    except Exception as e:
        installed = Version(importlib.metadata.version("mistral-common"))
        if installed >= _MISTRAL_COMMON_MODEL_SETTINGS_MIN_VERSION:
            raise
        raise RuntimeError(
            f"Failed to load the mistral-common tokenizer for {tokenizer_name} "
            f"with mistral-common {installed}. Tokenizers whose tekken.json "
            "declares model settings (e.g. json_schema) require mistral-common>="
            f"{_MISTRAL_COMMON_MODEL_SETTINGS_MIN_VERSION}."
        ) from e


def to_mistral_common_response_format(response_format) -> Optional[dict]:
    if response_format is None or response_format.type in ("text", "structural_tag"):
        return None
    if response_format.type == "json_object":
        return {"type": "json_object"}
    assert response_format.type == "json_schema", response_format.type
    json_schema = response_format.json_schema
    if json_schema is None or json_schema.schema_ is None:
        raise ValueError("response_format json_schema must define a schema")
    return {
        "type": "json_schema",
        "json_schema": {
            "name": json_schema.name,
            "schema": json_schema.schema_,
            "strict": bool(json_schema.strict),
        },
    }


def check_mistral_common_reasoning_effort(reasoning_effort) -> None:
    from mistral_common.protocol.instruct.request import ReasoningEffort

    supported = [e.value for e in ReasoningEffort]
    if reasoning_effort is not None and reasoning_effort not in supported:
        raise ValueError(
            f"reasoning_effort={reasoning_effort} is not supported by Mistral "
            f"models. Supported values are: {supported}."
        )


def mistral_common_default_reasoning_effort(tokenizer) -> Optional[str]:
    """The effort mistral-common renders when a request leaves it unset."""
    instruct = getattr(
        getattr(tokenizer, "tokenizer", None), "instruct_tokenizer", None
    )
    builder = getattr(
        getattr(instruct, "tokenizer", None), "model_settings_builder", None
    )
    default = getattr(getattr(builder, "reasoning_effort", None), "default", None)
    default = getattr(default, "value", default)
    return default if isinstance(default, str) else None


def attach_mistral_common_image_urls(messages: list[dict], image_data: list) -> None:
    images = iter(image_data)
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            continue
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "image":
                image = next(images, None)
                if image is None:
                    raise ValueError("More image placeholders than images")
                part = {"type": "image", "url": image.url}
            parts.append(part)
        message["content"] = parts
    if next(images, None) is not None:
        raise ValueError("More images than image placeholders")


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

    # MistralCommonBackend.add_special_tokens raises NotImplementedError by design;
    # a pad-token-only call sets `pad_token` directly and adds nothing.
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

    def _assistant_content_is_empty(content):
        if content is None:
            return True
        if isinstance(content, str):
            return not content.strip()
        if isinstance(content, list):
            return all(
                isinstance(part, dict)
                and part.get("type") in ("text", "input_text")
                and not str(part.get("text") or "").strip()
                for part in content
            )
        return False

    def _drop_empty_assistant_messages(messages):
        """Drop assistant turns with neither content nor tool calls, which
        mistral_common rejects while other chat templates ignore them. A trailing
        assistant turn is consumed upstream as the continue_final_message prefix,
        so this cannot drop a prefill.
        """
        if not isinstance(messages, (list, tuple)):
            return messages

        kept = []
        for msg in messages:
            if isinstance(msg, (list, tuple)):
                kept.append(_drop_empty_assistant_messages(msg))
                continue
            if (
                isinstance(msg, dict)
                and msg.get("role") == "assistant"
                and not msg.get("tool_calls")
                and _assistant_content_is_empty(msg.get("content"))
            ):
                continue
            kept.append(msg)
        return kept

    def _safe_apply_chat_template(messages, **kwargs):
        kwargs.pop("add_generation_prompt", None)
        messages = _adapt_placeholder_messages_for_mistral_common(messages)
        messages = _drop_empty_assistant_messages(messages)
        return tokenizer._orig_apply_chat_template(messages, **kwargs)

    tokenizer.apply_chat_template = _safe_apply_chat_template

    def init_xgrammar():
        from xgrammar import TokenizerInfo

        tekken = getattr(
            getattr(tokenizer.tokenizer, "instruct_tokenizer", None), "tokenizer", None
        )
        if tekken is None or not hasattr(tekken, "id_to_byte_piece"):
            logger.warning(
                "Cannot build XGrammar TokenizerInfo: no Tekkenizer found under %s",
                type(tokenizer).__name__,
            )
            return None, None

        try:
            placeholder = "<|xg_special_token_{}|>"
            encoded_vocab = []
            for token_id in range(tekken.n_words):
                piece = (
                    tekken.id_to_piece(token_id)
                    if token_id < tekken.num_special_tokens
                    else tekken.id_to_byte_piece(token_id)
                )
                # XGrammar reserves b"\x00"-prefixed tokens as special markers.
                if isinstance(piece, bytes) and piece.startswith(b"\x00"):
                    piece = placeholder.format(f"nul{token_id}")
                encoded_vocab.append(piece)

            eos_token_id = getattr(tokenizer, "eos_token_id", None)
            override_stop_tokens = [eos_token_id] if eos_token_id is not None else None
            tokenizer_info = TokenizerInfo(
                encoded_vocab, stop_token_ids=override_stop_tokens
            )
        except Exception as e:
            logger.warning(
                "Failed to build XGrammar TokenizerInfo for %s: %s",
                type(tokenizer).__name__,
                e,
            )
            return None, None
        return tokenizer_info, override_stop_tokens

    tokenizer.init_xgrammar = init_xgrammar
    _use_placeholder_image_encoder(tokenizer)
    return tokenizer


def mistral_common_image_encoder(tokenizer):
    instruct = getattr(
        getattr(tokenizer, "tokenizer", None), "instruct_tokenizer", None
    )
    return getattr(instruct, "image_encoder", None)


def _use_placeholder_image_encoder(tokenizer) -> None:
    encoder = mistral_common_image_encoder(tokenizer)
    if encoder is None:
        return

    from mistral_common.tokens.tokenizers.image import ImageEncoder, ImageEncoding

    class PlaceholderImageEncoder(ImageEncoder):
        # The multimodal processor loads each image under the media URL policy and
        # expands this placeholder into its grid; rendering must not touch the source.
        def __call__(self, content) -> ImageEncoding:
            return ImageEncoding(
                tokens=[self.special_ids.img],
                image=np.empty((3, 0, 0), dtype=np.float32),
            )

    tokenizer.tokenizer.instruct_tokenizer.image_encoder = PlaceholderImageEncoder(
        encoder.image_config, encoder.special_ids
    )
