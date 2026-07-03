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


# mlx-lm containers name their scalar sliding window either ``window_size``
# (gpt_oss, gemma4) or ``sliding_window`` (olmo3, llama SWA variants, ...).
WINDOW_SIZE_ATTRS = ("window_size", "sliding_window")


def get_container_window_size(model: Any) -> int | None:
    """The container-level scalar sliding window, if the model declares one."""
    root = getattr(model, "language_model", model)
    container = getattr(root, "model", root)
    return first_present_attr(container, WINDOW_SIZE_ATTRS)


def get_layer_window_sizes(model: Any) -> dict[int, int | None]:
    """Per-layer sliding-window sizes from the mlx-lm container convention.

    Containers such as gpt_oss or olmo3 expose ``layer_types`` (one entry
    per layer, ``"sliding_attention"`` marking windowed layers) plus a
    scalar window (see ``WINDOW_SIZE_ATTRS``).  Returns
    ``{layer_idx: window or None}``, or ``{}`` when the model does not
    follow the convention.
    """
    root = getattr(model, "language_model", model)
    container = getattr(root, "model", root)
    layer_types = getattr(container, "layer_types", None)
    window_size = get_container_window_size(model)
    if not layer_types or window_size is None:
        return {}
    return {
        idx: window_size if layer_type == "sliding_attention" else None
        for idx, layer_type in enumerate(layer_types)
    }


def uses_sliding_window_attention(*modules: Any) -> bool:
    return any(
        bool(getattr(module, attr, False))
        for module in modules
        for attr in SLIDING_ATTENTION_ATTRS
    )
