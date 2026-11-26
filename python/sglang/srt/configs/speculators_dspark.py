"""Loader for speculators-format DSpark/DFlash draft checkpoints."""

from __future__ import annotations

from numbers import Integral
from typing import Any, Mapping, Optional

from transformers import AutoConfig, PretrainedConfig

_SUPPORTED_SPECULATORS_MODEL_TYPES = frozenset({"dspark", "dflash"})

_COPIED_DSPARK_KEYS = (
    "speculators_model_type",
    "block_size",
    "markov_rank",
    "markov_head_type",
    "sample_from_anchor",
    "enable_confidence_head",
    "confidence_head_with_markov",
    "sliding_window_non_causal",
    "full_attention_causal",
    "target_hidden_size",
    "draft_vocab_size",
    "tie_word_embeddings",
    "dtype",
    "mask_token_id",
)

_FALLBACK_QUANT_KEYS = (
    "quantization_config",
    "compression_config",
)

_CAUSALITY_OVERRIDE_KEYS = (
    "is_causal",
    "sliding_window_non_causal",
    "full_attention_causal",
    "layer_types",
)


def _is_supported_speculators_type(spec_type: object) -> bool:
    return (
        isinstance(spec_type, str)
        and spec_type.lower() in _SUPPORTED_SPECULATORS_MODEL_TYPES
    )


def _config_text_subconfig(config: Any) -> Any:
    if isinstance(config, PretrainedConfig):
        resolved = config.get_text_config()
        if resolved is not config and not isinstance(resolved, Mapping):
            return resolved
        return None
    sub = getattr(config, "text_config", None)
    if sub is not None and sub is not config and not isinstance(sub, Mapping):
        return sub
    return None


def _first_non_none(*mappings: Optional[Mapping], key: str) -> Any:
    for mapping in mappings:
        if mapping is not None:
            value = mapping.get(key)
            if value is not None:
                return value
    return None


def _first_attr_non_none(*objs: Any, name: str) -> Any:
    for obj in objs:
        if obj is not None:
            value = getattr(obj, name, None)
            if value is not None:
                return value
    return None


def _set_config_attr(config: Any, name: str, value: Any) -> None:
    setattr(config, name, value)
    text_config = _config_text_subconfig(config)
    if text_config is not None:
        setattr(text_config, name, value)


def is_speculators_wrapper_config(config: Any) -> bool:
    if getattr(config, "_speculators_wrapper", False):
        return True
    text_config = _config_text_subconfig(config)
    return text_config is not None and getattr(
        text_config, "_speculators_wrapper", False
    )


def prepare_speculators_override_args(
    config: PretrainedConfig, model_override_args: Mapping
) -> Mapping:
    if not is_speculators_wrapper_config(config):
        return model_override_args
    if _config_text_subconfig(config) is not None:
        return model_override_args
    text_overrides = model_override_args.get("text_config")
    if not isinstance(text_overrides, Mapping):
        return model_override_args
    merged = dict(model_override_args)
    merged.pop("text_config", None)
    for key, value in text_overrides.items():
        if key not in merged:
            merged[key] = value
    return merged


def parse_int_layer_ids(
    raw_layer_ids: Any, *, field_name: str, min_value: Optional[int] = None
) -> list[int]:
    if not isinstance(raw_layer_ids, (list, tuple)):
        raise ValueError(
            f"{field_name} must be a list of ints, "
            f"got type={type(raw_layer_ids).__name__}."
        )
    if not raw_layer_ids:
        raise ValueError(f"{field_name} must be non-empty.")
    layer_ids: list[int] = []
    for idx in raw_layer_ids:
        if not isinstance(idx, Integral) or isinstance(idx, bool):
            raise ValueError(
                f"{field_name} entries must be integers, "
                f"got {idx!r} (type={type(idx).__name__})."
            )
        layer_ids.append(int(idx))
    if min_value is not None and any(idx < min_value for idx in layer_ids):
        if min_value == 1:
            raise ValueError(
                f"{field_name} count layers from 1 (0 is the embeddings), "
                f"got {layer_ids}"
            )
        raise ValueError(
            f"{field_name} entries must be integers >= {min_value}, got {layer_ids}"
        )
    return layer_ids


def _resolve_unique_layer_ids(
    sources: tuple[Any, ...], *, field_name: str, min_value: int
) -> Optional[list[int]]:
    resolved: Optional[list[int]] = None
    for raw in sources:
        if raw is None:
            continue
        parsed = parse_int_layer_ids(raw, field_name=field_name, min_value=min_value)
        if resolved is not None and parsed != resolved:
            raise ValueError(
                f"speculators config has conflicting {field_name}: "
                f"{resolved} vs {parsed}."
            )
        resolved = parsed
    return resolved


def _build_dflash_config(config_dict: Mapping, layer_config: Mapping) -> dict:
    dflash_config: dict = {}
    inner_dflash = layer_config.get("dflash_config")
    if isinstance(inner_dflash, Mapping):
        dflash_config.update(inner_dflash)
    outer_dflash = config_dict.get("dflash_config")
    if isinstance(outer_dflash, Mapping):
        dflash_config.update(outer_dflash)
    if dflash_config.get("aux_hidden_stream") is None:
        dflash_config["aux_hidden_stream"] = "prefix"

    explicit_target_ids = _resolve_unique_layer_ids(
        (
            dflash_config.get("target_layer_ids"),
            config_dict.get("target_layer_ids"),
            layer_config.get("target_layer_ids"),
        ),
        field_name="target_layer_ids",
        min_value=0,
    )

    dflash_aux_ids = dflash_config.pop("aux_hidden_state_layer_ids", None)
    aux_layer_ids = _resolve_unique_layer_ids(
        (
            config_dict.get("aux_hidden_state_layer_ids"),
            dflash_aux_ids,
            layer_config.get("aux_hidden_state_layer_ids"),
        ),
        field_name="aux_hidden_state_layer_ids",
        min_value=1,
    )
    if aux_layer_ids is not None:
        derived_target_ids = [idx - 1 for idx in aux_layer_ids]
        if (
            explicit_target_ids is not None
            and explicit_target_ids != derived_target_ids
        ):
            raise ValueError(
                "speculators config has conflicting target_layer_ids="
                f"{explicit_target_ids} and aux_hidden_state_layer_ids={aux_layer_ids} "
                f"(expected target_layer_ids={derived_target_ids})."
            )
        dflash_config["target_layer_ids"] = derived_target_ids
    elif explicit_target_ids is not None:
        dflash_config["target_layer_ids"] = explicit_target_ids

    mask_token_id = config_dict.get("mask_token_id")
    if mask_token_id is not None:
        if not isinstance(mask_token_id, Integral) or isinstance(mask_token_id, bool):
            raise ValueError(
                f"speculators mask_token_id must be an integer, got {mask_token_id!r}."
            )
        dflash_config["mask_token_id"] = int(mask_token_id)

    return dflash_config


def _apply_outer_quant_config(flat: dict, config_dict: Mapping) -> None:
    if not any(config_dict.get(key) is not None for key in _FALLBACK_QUANT_KEYS):
        return
    nested_text_config = flat.get("text_config")
    if isinstance(nested_text_config, Mapping):
        nested_text_config = dict(nested_text_config)
        for key in _FALLBACK_QUANT_KEYS:
            nested_text_config.pop(key, None)
        flat["text_config"] = nested_text_config
    for key in _FALLBACK_QUANT_KEYS:
        flat.pop(key, None)
        if config_dict.get(key) is not None:
            flat[key] = config_dict[key]


def try_load_speculators_dspark_config(
    model,
    revision: Optional[str],
    *,
    config_registry: Optional[Mapping[str, type[PretrainedConfig]]] = None,
    config_dict: Optional[Mapping] = None,
    **kwargs,
) -> Optional[PretrainedConfig]:
    """Flatten a ``speculators`` DSpark/DFlash checkpoint config into the HF config of
    its layer backbone (``transformer_layer_config.model_type``).

    Speculators checkpoints omit top-level ``model_type``, nest layer fields
    under ``transformer_layer_config``, and index ``aux_hidden_state_layer_ids``
    starting at 1 (0 is embeddings), one above SGLang's layer indices.
    """
    if config_dict is None:
        config_dict, _ = PretrainedConfig.get_config_dict(
            model, revision=revision, **kwargs
        )
    spec_model_type = config_dict.get("speculators_model_type")
    if not _is_supported_speculators_type(spec_model_type):
        return None
    layer_config = config_dict.get("transformer_layer_config")
    layer_model_type = (
        layer_config.get("model_type") if isinstance(layer_config, Mapping) else None
    )
    if not isinstance(layer_model_type, str) or not layer_model_type:
        raise ValueError(
            "speculators DSpark config needs transformer_layer_config.model_type"
        )

    raw_text_config = layer_config.get("text_config")
    text_config_dict = raw_text_config if isinstance(raw_text_config, Mapping) else None
    causality_sources = (config_dict, layer_config, text_config_dict)
    explicit_is_causal = _get_explicit_is_causal(
        config_dict, layer_config, text_config_dict
    )
    effective_layer_types = _first_non_none(*causality_sources, key="layer_types")
    effective_sw_non_causal = _first_non_none(
        *causality_sources, key="sliding_window_non_causal"
    )
    effective_full_causal = _first_non_none(
        *causality_sources, key="full_attention_causal"
    )
    is_causal = _speculators_is_causal(
        layer_types=effective_layer_types,
        explicit_is_causal=explicit_is_causal,
        raw_sw_non_causal=effective_sw_non_causal,
        raw_full_causal=effective_full_causal,
    )
    dflash_config = _build_dflash_config(config_dict, layer_config)

    spec_type = spec_model_type.lower()
    default_arch = (
        ["DFlashDraftModel"] if spec_type == "dflash" else ["DSparkDraftModel"]
    )
    flat = dict(layer_config)
    flat.pop("model_type", None)
    flat.pop("auto_map", None)
    flat["architectures"] = list(
        config_dict.get("architectures")
        or layer_config.get("architectures")
        or default_arch
    )
    flat["is_causal"] = is_causal
    if flat.get("sliding_window") is not None:
        flat["use_sliding_window"] = True
    flat["dflash_config"] = dflash_config
    for key in _COPIED_DSPARK_KEYS:
        if key in config_dict and config_dict[key] is not None:
            flat[key] = config_dict[key]
    _apply_outer_quant_config(flat, config_dict)

    if config_registry is not None and layer_model_type in config_registry:
        config = config_registry[layer_model_type](**flat)
    else:
        config = AutoConfig.for_model(layer_model_type, **flat)
    config._name_or_path = model
    _set_config_attr(config, "_speculators_wrapper", True)
    _set_config_attr(config, "_speculators_explicit_is_causal", explicit_is_causal)
    _set_config_attr(config, "is_causal", is_causal)
    if effective_layer_types is not None:
        _set_config_attr(config, "layer_types", list(effective_layer_types))
    if effective_sw_non_causal is not None:
        _set_config_attr(config, "sliding_window_non_causal", effective_sw_non_causal)
    if effective_full_causal is not None:
        _set_config_attr(config, "full_attention_causal", effective_full_causal)
    return config


def _get_explicit_is_causal(
    config_dict: Mapping,
    layer_config: Mapping,
    text_config_dict: Optional[Mapping] = None,
) -> Optional[bool]:
    sources = (
        ("speculators is_causal", config_dict.get("is_causal")),
        (
            "speculators transformer_layer_config.is_causal",
            layer_config.get("is_causal"),
        ),
        (
            "speculators transformer_layer_config.text_config.is_causal",
            text_config_dict.get("is_causal") if text_config_dict is not None else None,
        ),
    )
    for label, value in sources:
        if value is not None and not isinstance(value, bool):
            raise ValueError(f"{label} must be a bool, got {value!r}.")
    for _, value in sources:
        if value is not None:
            return value
    return None


def _validate_explicit_causality(
    explicit_is_causal: bool,
    layer_types: set[str],
    raw_sw_non_causal: Any,
    raw_full_causal: Any,
) -> bool:
    if (
        raw_sw_non_causal is True
        and ("sliding_attention" in layer_types or not layer_types)
        and explicit_is_causal is True
    ):
        raise ValueError(
            f"speculators config has conflicting is_causal={explicit_is_causal} "
            f"and sliding_window_non_causal={raw_sw_non_causal!r}."
        )
    if (
        raw_full_causal is True
        and "full_attention" in layer_types
        and explicit_is_causal is False
    ):
        raise ValueError(
            f"speculators config has conflicting is_causal={explicit_is_causal} "
            f"and full_attention_causal={raw_full_causal!r}."
        )
    return explicit_is_causal


def _derive_layer_types_is_causal(
    layer_types: set[str],
    *,
    sliding_causal: bool,
    full_causal: bool,
    default_causal: bool = True,
) -> bool:
    causal_by_type = {
        "sliding_attention": sliding_causal,
        "full_attention": full_causal,
    }
    causality = {
        causal_by_type.get(layer_type, default_causal) for layer_type in layer_types
    } or {default_causal}
    if len(causality) > 1:
        raise ValueError(
            "speculators DSpark config mixes causal and non-causal layer types "
            f"{sorted(layer_types)}, which one is_causal cannot express"
        )
    return causality.pop()


def _speculators_is_causal(
    *,
    layer_types: Any,
    explicit_is_causal: Optional[bool],
    raw_sw_non_causal: Any,
    raw_full_causal: Any,
) -> bool:
    layer_type_set = set(layer_types or ())
    if explicit_is_causal is not None:
        return _validate_explicit_causality(
            explicit_is_causal, layer_type_set, raw_sw_non_causal, raw_full_causal
        )
    return _derive_layer_types_is_causal(
        layer_type_set,
        sliding_causal=not bool(raw_sw_non_causal),
        full_causal=bool(raw_full_causal),
    )


def _extract_causality_overrides(model_override_args: Mapping) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    text_overrides = model_override_args.get("text_config")
    if isinstance(text_overrides, Mapping):
        for key in _CAUSALITY_OVERRIDE_KEYS:
            if key in text_overrides:
                overrides[key] = text_overrides[key]
    for key in _CAUSALITY_OVERRIDE_KEYS:
        if key in model_override_args:
            overrides[key] = model_override_args[key]
    return overrides


def _sync_alias_overrides(config: Any, overrides: Mapping) -> None:
    for key in ("sliding_window_non_causal", "full_attention_causal"):
        if key in overrides:
            _set_config_attr(config, key, overrides[key])


def _clear_contradictory_inherited_aliases(
    config: Any, overrides: Mapping, *, resolved_is_causal: bool
) -> None:
    targets = (config, _config_text_subconfig(config))
    if "sliding_window_non_causal" not in overrides and resolved_is_causal is True:
        for target in targets:
            if (
                target is not None
                and getattr(target, "sliding_window_non_causal", None) is True
            ):
                target.sliding_window_non_causal = None
    if "full_attention_causal" not in overrides and resolved_is_causal is False:
        for target in targets:
            if (
                target is not None
                and getattr(target, "full_attention_causal", None) is True
            ):
                target.full_attention_causal = None


def reconcile_speculators_causality_overrides(
    config: PretrainedConfig, model_override_args: Mapping
) -> None:
    if not is_speculators_wrapper_config(config):
        return
    overrides = _extract_causality_overrides(model_override_args)
    if not overrides:
        return

    text_config = _config_text_subconfig(config)
    if "layer_types" in overrides:
        raw_layer_types = overrides["layer_types"]
        _set_config_attr(config, "layer_types", raw_layer_types)
    else:
        raw_layer_types = _first_attr_non_none(config, text_config, name="layer_types")
    layer_types = set(raw_layer_types or ())

    override_is_causal = overrides.get("is_causal")
    if override_is_causal is not None:
        if not isinstance(override_is_causal, bool):
            raise ValueError(
                f"speculators is_causal must be a bool, got {override_is_causal!r}."
            )
        resolved = _validate_explicit_causality(
            override_is_causal,
            layer_types,
            overrides.get("sliding_window_non_causal"),
            overrides.get("full_attention_causal"),
        )
        _set_config_attr(config, "_speculators_explicit_is_causal", resolved)
        _set_config_attr(config, "is_causal", resolved)
        _sync_alias_overrides(config, overrides)
        _clear_contradictory_inherited_aliases(
            config, overrides, resolved_is_causal=resolved
        )
        return

    if "is_causal" in overrides:
        explicit_base = None
        _set_config_attr(config, "_speculators_explicit_is_causal", None)
    else:
        explicit_base = _first_attr_non_none(
            config, text_config, name="_speculators_explicit_is_causal"
        )
    if "sliding_window_non_causal" in overrides:
        sliding_causal = not bool(overrides["sliding_window_non_causal"])
    elif explicit_base is not None:
        sliding_causal = explicit_base
    else:
        sliding_causal = not bool(
            _first_attr_non_none(config, text_config, name="sliding_window_non_causal")
        )

    if "full_attention_causal" in overrides:
        full_causal = bool(overrides["full_attention_causal"])
    elif explicit_base is not None:
        full_causal = explicit_base
    else:
        full_causal = bool(
            _first_attr_non_none(config, text_config, name="full_attention_causal")
        )

    default_causal = explicit_base if explicit_base is not None else True
    resolved = _derive_layer_types_is_causal(
        layer_types,
        sliding_causal=sliding_causal,
        full_causal=full_causal,
        default_causal=default_causal,
    )
    _sync_alias_overrides(config, overrides)
    _set_config_attr(config, "is_causal", resolved)
