"""Model introspection and attention patching."""

from typing import Any

from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import (
    MLXAttentionWrapper,
)

_ATTENTION_API_ATTRS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "rope",
    "scale",
)


def _is_attention_module(module: Any) -> bool:
    """Return whether a module matches MLXAttentionWrapper's inner API."""
    if isinstance(module, MLXAttentionWrapper):
        module = module._inner
    return (
        all(hasattr(module, attr) for attr in _ATTENTION_API_ATTRS)
        and (
            hasattr(module, "n_heads")
            or hasattr(module, "num_heads")
            or hasattr(module, "num_attention_heads")
        )
        and (
            hasattr(module, "n_kv_heads")
            or hasattr(module, "num_k_heads")
            or hasattr(module, "num_key_value_heads")
        )
    )


def _find_attention_attr(layer: Any) -> str | None:
    if hasattr(layer, "items"):
        child_modules = layer.items()
    else:
        child_modules = vars(layer).items()
    for name, module in child_modules:
        if _is_attention_module(module):
            return name
    return None


def find_attention_layers(model: Any) -> tuple[list[Any], list[str | None]]:
    """Find transformer layers and per layer attention attribute names."""
    root = getattr(model, "language_model", model)
    container = getattr(root, "model", root)
    layer_list = getattr(container, "layers", None) or getattr(root, "layers", [])

    if layer_list:
        attn_attrs = [_find_attention_attr(layer) for layer in layer_list]
        if any(attr is not None for attr in attn_attrs):
            return layer_list, attn_attrs
        raise ValueError(f"No attention attribute in layer type {type(layer_list[0])}")
    return layer_list, []


def patch_model_attention(model: Any) -> int:
    """Install MLXAttentionWrapper on all attention layers (idempotent).

    The wrapper delegates to the inner module when no BatchedDecodeContext
    is set, so it is always installed and never removed.
    """
    layer_list, attn_attrs = find_attention_layers(model)
    patched = 0
    for idx, (layer, attn_attr) in enumerate(zip(layer_list, attn_attrs)):
        if attn_attr is None:
            continue
        attn = getattr(layer, attn_attr, None)
        if attn is None or isinstance(attn, MLXAttentionWrapper):
            continue
        if not _is_attention_module(attn):
            raise ValueError(
                f"Attribute {attn_attr!r} in layer type {type(layer)} does not "
                "match the MLX attention API"
            )
        setattr(layer, attn_attr, MLXAttentionWrapper(attn, idx))
        patched += 1
    return patched


def get_num_layers(model: Any) -> int:
    """Return the number of transformer layers."""
    layer_list, _ = find_attention_layers(model)
    return len(layer_list)
