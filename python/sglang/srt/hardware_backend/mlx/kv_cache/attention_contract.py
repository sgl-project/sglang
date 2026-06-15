"""Attention helpers based on duck typing for the MLX backend."""

from __future__ import annotations

from typing import Any, Iterable

# ``rope`` and ``scale`` are required by MLXAttentionWrapper. Keeping them in
# the contract also prevents recurrent mixers such as DeltaNet from being
# mistaken for softmax attention just because they expose projection layers.
ATTENTION_API_ATTRS = ("q_proj", "k_proj", "v_proj", "o_proj", "rope", "scale")
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


def is_attention_module(module: Any) -> bool:
    return (
        all(hasattr(module, attr) for attr in ATTENTION_API_ATTRS)
        and get_num_heads(module) is not None
        and get_num_kv_heads(module) is not None
    )


def uses_sliding_window_attention(*modules: Any) -> bool:
    return any(
        bool(getattr(module, attr, False))
        for module in modules
        for attr in SLIDING_ATTENTION_ATTRS
    )
