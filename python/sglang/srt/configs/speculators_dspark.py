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
    if not isinstance(layer_model_type, str) or not layer_model_type:
        raise ValueError(
            "speculators DSpark config needs transformer_layer_config.model_type"
        )

    is_causal = _speculators_is_causal(config_dict, layer_config)
    layer_ids = [int(idx) for idx in config_dict.get("aux_hidden_state_layer_ids", [])]
    if any(idx < 1 for idx in layer_ids):
        raise ValueError(
            "aux_hidden_state_layer_ids count layers from 1 (0 is the embeddings), "
            f"got {layer_ids}"
        )

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
        "target_layer_ids": [idx - 1 for idx in layer_ids],
        "aux_hidden_stream": "prefix",
    }
    for key in _COPIED_DSPARK_KEYS:
        if key in config_dict and config_dict[key] is not None:
            flat[key] = config_dict[key]

    return AutoConfig.for_model(layer_model_type, **flat)


def _speculators_is_causal(config_dict: dict, layer_config: dict) -> bool:
    """One is_causal for the draft: sliding layers are causal unless
    sliding_window_non_causal, full layers only with full_attention_causal, other
    layer types always. Raises when the layer types disagree."""
    causal_by_type = {
        "sliding_attention": not bool(
            config_dict.get("sliding_window_non_causal", False)
        ),
        "full_attention": bool(config_dict.get("full_attention_causal", False)),
    }
    layer_types = set(layer_config.get("layer_types") or ())
    # Other or missing layer types are causal, as before.
    causality = {causal_by_type.get(t, True) for t in layer_types} or {True}
    if len(causality) > 1:
        raise ValueError(
            "speculators DSpark config mixes causal and non-causal layer types "
            f"{sorted(layer_types)}, which one is_causal cannot express"
        )
    return causality.pop()
