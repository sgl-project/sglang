"""Translate the Speculators dense DSpark export into SGLang's config layout."""

from copy import deepcopy
from typing import Any, Optional


def _require_int(config: dict, name: str, minimum: int = 1) -> int:
    value = config.get(name)
    if type(value) is not int or value < minimum:
        raise ValueError(f"Speculators DSpark {name} must be an integer >= {minimum}.")
    return value


def _check_alias(config: dict, name: str, expected: Any, source: str) -> None:
    value = config.get(name)
    if value is not None and value != expected:
        raise ValueError(f"Speculators DSpark {name} conflicts with {source}.")


def normalize_speculators_dspark_config(config: dict) -> Optional[dict]:
    """Return a non-mutating translation, or None for an existing HF layout.

    Speculators stores the decoder under ``transformer_layer_config`` and
    numbers auxiliary hidden states one above SGLang's DFlash capture IDs.
    The result uses the native HF layout, so reloading it never subtracts
    twice. This only handles the Qwen3-style dense draft, not the verifier.
    """
    if "transformer_layer_config" not in config:
        return None
    if config.get("architectures") != ["DSparkDraftModel"]:
        return None
    if config.get("speculators_model_type", "dspark") != "dspark":
        raise ValueError("Speculators DSpark speculators_model_type must be 'dspark'.")

    speculators = config.get("speculators_config", {})
    if not isinstance(speculators, dict):
        raise ValueError("Speculators DSpark speculators_config must be an object.")
    if speculators.get("algorithm", "dspark") != "dspark":
        raise ValueError(
            "Speculators DSpark speculators_config.algorithm must be 'dspark'."
        )

    decoder = config["transformer_layer_config"]
    if not isinstance(decoder, dict) or decoder.get("model_type") != "qwen3":
        raise ValueError(
            "Speculators DSpark requires a qwen3 transformer_layer_config."
        )
    if config.get("model_type") not in (None, "dspark", "qwen3"):
        raise ValueError("Speculators DSpark model_type conflicts with the decoder.")
    if config.get("text_config") is not None:
        raise ValueError(
            "Speculators DSpark cannot use both text_config "
            "and transformer_layer_config."
        )

    normalized = deepcopy(config)
    for key, value in decoder.items():
        # The outer architecture selects DSpark; the inner type selects its
        # decoder configuration. Neither should select the target network.
        if key in ("architectures", "model_type", "auto_map", "_name_or_path"):
            continue
        _check_alias(normalized, key, value, f"transformer_layer_config.{key}")
        normalized[key] = deepcopy(value)
    normalized["model_type"] = "qwen3"

    for name in (
        "hidden_size",
        "intermediate_size",
        "num_hidden_layers",
        "num_attention_heads",
        "num_key_value_heads",
        "vocab_size",
        "block_size",
        "markov_rank",
    ):
        _require_int(normalized, name)
    if "head_dim" in normalized:
        # head_dim need not equal hidden_size // num_attention_heads.
        _require_int(normalized, "head_dim")
    # The input-only mask can use a padded row of the shared target embedding;
    # the runtime checks its upper bound against the actual target vocabulary.
    _require_int(normalized, "mask_token_id", minimum=0)
    _check_alias(normalized, "draft_vocab_size", normalized["vocab_size"], "vocab_size")
    _check_alias(
        normalized, "target_hidden_size", normalized["hidden_size"], "hidden_size"
    )
    if normalized.get("markov_head_type") not in ("vanilla", "gated", "rnn"):
        raise ValueError("Speculators DSpark has an unsupported markov_head_type.")
    for name in (
        "enable_confidence_head",
        "confidence_head_with_markov",
        "sample_from_anchor",
    ):
        if name in normalized and type(normalized[name]) is not bool:
            raise ValueError(f"Speculators DSpark {name} must be a boolean.")

    aux_ids = normalized.get("aux_hidden_state_layer_ids")
    if (
        not isinstance(aux_ids, list)
        or not aux_ids
        or any(type(layer) is not int or layer < 1 for layer in aux_ids)
    ):
        raise ValueError(
            "Speculators DSpark aux_hidden_state_layer_ids must be a non-empty "
            "list of positive integers."
        )
    # Target hooks append features in model execution order. Sorting a trained
    # list here would silently change the FC input, and duplicates cannot be
    # captured by their membership-based hooks.
    if any(left >= right for left, right in zip(aux_ids, aux_ids[1:])):
        raise ValueError(
            "Speculators DSpark aux_hidden_state_layer_ids must be strictly increasing."
        )
    # Capture count need not equal draft depth.
    target_ids = [layer - 1 for layer in aux_ids]
    _check_alias(
        normalized, "target_layer_ids", target_ids, "aux_hidden_state_layer_ids"
    )
    normalized["target_layer_ids"] = target_ids

    # Existing readers give these legacy aliases priority over the fields above.
    # Accept matching aliases, but never silently override the export contract.
    for alias, source in (
        ("dspark_target_layer_ids", "target_layer_ids"),
        ("dspark_block_size", "block_size"),
        ("dspark_noise_token_id", "mask_token_id"),
        ("dspark_markov_rank", "markov_rank"),
        ("dspark_markov_head_type", "markov_head_type"),
    ):
        _check_alias(normalized, alias, normalized[source], source)
    for section in ("dflash_config", "dspark_config"):
        aliases = normalized.get(section, {})
        if not isinstance(aliases, dict):
            raise ValueError(f"Speculators DSpark {section} must be an object.")
        for name in (
            "target_layer_ids",
            "block_size",
            "mask_token_id",
            "markov_rank",
            "markov_head_type",
        ):
            # Native readers use dict.get(key, canonical), so an explicit null
            # would hide the canonical value rather than use that fallback.
            # In particular, null target IDs would silently enable auto-picking.
            if name in aliases and aliases[name] is None:
                del aliases[name]
            _check_alias(aliases, name, normalized[name], f"top-level {name}")

    # Keep the canonical IDs as provenance, but emit just one decoder layout.
    # HF expands defaults (e.g. RoPE parameters) during construction, so keeping
    # a second raw decoder would make a saved config conflict with itself.
    normalized.pop("transformer_layer_config")
    normalized.pop("auto_map", None)
    return normalized
