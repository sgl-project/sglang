"""Model introspection and attention patching."""

from typing import Any

from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import (
    MLXAttentionWrapper,
)


def find_attention_layers(model: Any) -> tuple[list[Any], str]:
    """Find transformer layers and the attention attribute name."""
    root = getattr(model, "language_model", model)
    container = getattr(root, "model", root)
    layer_list = getattr(container, "layers", None) or getattr(root, "layers", [])

    if layer_list:
        sample = layer_list[0]
        if hasattr(sample, "self_attn"):
            return layer_list, "self_attn"
        if hasattr(sample, "attention"):
            return layer_list, "attention"
        raise ValueError(f"No attention attribute in layer type {type(sample)}")
    return layer_list, "self_attn"


def patch_model_attention(model: Any) -> int:
    """Install MLXAttentionWrapper on all attention layers (idempotent).

    The wrapper delegates to the inner module when no BatchedDecodeContext
    is set, so it is always installed and never removed.
    """
    layer_list, attn_attr = find_attention_layers(model)
    patched = 0
    for idx, layer in enumerate(layer_list):
        attn = getattr(layer, attn_attr)
        if isinstance(attn, MLXAttentionWrapper):
            continue
        setattr(layer, attn_attr, MLXAttentionWrapper(attn, idx))
        patched += 1
    return patched


def get_num_layers(model: Any) -> int:
    """Return the number of transformer layers."""
    layer_list, _ = find_attention_layers(model)
    return len(layer_list)
