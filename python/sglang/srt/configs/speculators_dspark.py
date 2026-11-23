"""Loader for speculators-format DSpark draft checkpoints."""

from __future__ import annotations

from typing import Optional

from transformers import AutoConfig, PretrainedConfig

_COPIED_DSPARK_KEYS = (
    "block_size",
    "markov_rank",
    "markov_head_type",
    "sample_from_anchor",
    "enable_confidence_head",
    "confidence_head_with_markov",
    "target_hidden_size",
    "draft_vocab_size",
    "tie_word_embeddings",
    "dtype",
    "mask_token_id",
)


def try_load_speculators_dspark_config(
    model, revision: Optional[str], **kwargs
) -> Optional[PretrainedConfig]:
    """Flatten a ``speculators`` DSpark checkpoint config into the HF config of its
    layer backbone (``transformer_layer_config.model_type``).

    Speculators checkpoints omit top-level ``model_type``, nest layer fields
    under ``transformer_layer_config``, and index ``aux_hidden_state_layer_ids``
    starting at 1 (0 is embeddings), one above SGLang's layer indices.
    """
    config_dict, _ = PretrainedConfig.get_config_dict(
        model, revision=revision, **kwargs
    )
    if config_dict.get("speculators_model_type") != "dspark":
        return None
    layer_config = config_dict.get("transformer_layer_config")
    layer_model_type = (
        layer_config.get("model_type") if isinstance(layer_config, dict) else None
    )
    if not isinstance(layer_model_type, str):
        raise ValueError(
            "speculators DSpark config needs transformer_layer_config.model_type"
        )

    layer_types = set(layer_config.get("layer_types") or ())
    if layer_types == {"sliding_attention"}:
        is_causal = not bool(config_dict.get("sliding_window_non_causal", False))
    elif layer_types == {"full_attention"}:
        is_causal = bool(config_dict.get("full_attention_causal", False))
    else:
        is_causal = True

    flat = dict(layer_config)
    flat.pop("model_type", None)
    flat["architectures"] = list(
        config_dict.get("architectures") or ["DSparkDraftModel"]
    )
    flat["is_causal"] = is_causal
    if flat.get("sliding_window") is not None:
        flat["use_sliding_window"] = True
    flat["dflash_config"] = {
        "mask_token_id": int(config_dict["mask_token_id"]),
        "target_layer_ids": [
            int(idx) - 1 for idx in config_dict.get("aux_hidden_state_layer_ids", [])
        ],
        "aux_hidden_stream": "prefix",
    }
    for key in _COPIED_DSPARK_KEYS:
        if key in config_dict and config_dict[key] is not None:
            flat[key] = config_dict[key]

    return AutoConfig.for_model(layer_model_type, **flat)
