"""Attention helpers based on duck typing for the MLX backend."""

from __future__ import annotations

from typing import Any, Iterable

# ``rope`` and a softmax scale are required by MLXAttentionWrapper. Keeping
# them in the contract also prevents recurrent mixers such as DeltaNet from
# being mistaken for softmax attention just because they expose projections.
ATTENTION_API_ATTRS = ("q_proj", "k_proj", "v_proj", "o_proj", "rope")
# Any one of these satisfies the scale requirement (gpt_oss uses ``sm_scale``).
SCALE_ATTRS = ("scale", "sm_scale")
NUM_HEAD_ATTRS = ("n_heads", "num_heads", "num_attention_heads")
NUM_KV_HEAD_ATTRS = ("n_kv_heads", "num_k_heads", "num_kv_heads", "num_key_value_heads")
SLIDING_ATTENTION_ATTRS = (
    "is_sliding",
    "use_sliding",
    "is_sliding_window",
    "use_sliding_window",
    "is_swa",
)
# mlx-lm containers name their scalar sliding window either ``window_size``
# (gpt_oss, gemma4) or ``sliding_window`` (olmo3, llama SWA variants, ...).
WINDOW_SIZE_ATTRS = ("window_size", "sliding_window")


def first_present_attr(module: Any, names: Iterable[str]) -> Any | None:
    """Return the first present attribute value without treating 0 as absent."""
    for name in names:
        if hasattr(module, name):
            return getattr(module, name)
    return None


def get_num_heads(module: Any) -> int | None:
    return first_present_attr(module, NUM_HEAD_ATTRS)


def get_num_kv_heads(module: Any) -> int | None:
    return first_present_attr(module, NUM_KV_HEAD_ATTRS)


def get_head_dim(module: Any) -> int | None:
    head_dim = first_present_attr(module, ("head_dim",))
    if head_dim is not None:
        return head_dim

    n_kv_heads = get_num_kv_heads(module)
    if n_kv_heads is None:
        return None
    if hasattr(module, "hidden_size") and hasattr(module, "num_k_heads"):
        return module.hidden_size // module.num_k_heads
    if hasattr(module, "k_proj") and hasattr(module.k_proj, "weight"):
        return module.k_proj.weight.shape[0] // n_kv_heads
    return None


def get_attention_scale(module: Any) -> float | None:
    return first_present_attr(module, SCALE_ATTRS)


def is_attention_module(module: Any) -> bool:
    return (
        all(hasattr(module, attr) for attr in ATTENTION_API_ATTRS)
        and any(hasattr(module, attr) for attr in SCALE_ATTRS)
        and get_num_heads(module) is not None
        and get_num_kv_heads(module) is not None
    )


def language_model_container(model: Any) -> Any:
    """The mlx-lm container carrying the layer list and model-level config."""
    root = getattr(model, "language_model", model)
    return getattr(root, "model", root)


def get_container_window_size(model: Any) -> int | None:
    """The container-level scalar sliding window, if the model declares one."""
    return first_present_attr(language_model_container(model), WINDOW_SIZE_ATTRS)


def get_layer_window_sizes(model: Any) -> dict[int, int | None]:
    """Per-layer sliding-window sizes from the mlx-lm container convention.

    Containers such as gpt_oss or olmo3 expose ``layer_types`` (one entry
    per layer, ``"sliding_attention"`` marking windowed layers) plus a
    scalar window (see ``WINDOW_SIZE_ATTRS``).  gemma3 exposes
    ``sliding_window_pattern`` instead: every pattern-th layer is global.
    Returns ``{layer_idx: window or None}``, or ``{}`` when the model
    follows neither convention.
    """
    container = language_model_container(model)
    window_size = first_present_attr(container, WINDOW_SIZE_ATTRS)
    if window_size is None:
        return {}
    layer_types = getattr(container, "layer_types", None)
    if layer_types:
        return {
            idx: window_size if layer_type == "sliding_attention" else None
            for idx, layer_type in enumerate(layer_types)
        }
    pattern = getattr(container, "sliding_window_pattern", None)
    num_layers = len(getattr(container, "layers", None) or ())
    # gemma4_text also carries a pattern, but only as a default; its real
    # per-layer types are in config.layer_types, so leave such models alone.
    config = getattr(container, "config", None) or getattr(container, "args", None)
    if (
        isinstance(pattern, int)
        and pattern > 0
        and num_layers
        and not getattr(config, "layer_types", None)
    ):
        # As in mlx_lm gemma3_text: layer i is global when (i + 1) % pattern == 0.
        return {
            idx: window_size if (idx + 1) % pattern else None
            for idx in range(num_layers)
        }
    return {}


def uses_sliding_window_attention(*modules: Any) -> bool:
    return any(
        bool(getattr(module, attr, False))
        for module in modules
        for attr in SLIDING_ATTENTION_ATTRS
    )
