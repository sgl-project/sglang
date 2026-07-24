"""Model introspection and attention patching."""

import logging
from typing import Any

import mlx.nn as nn

from sglang.srt.hardware_backend.mlx.kv_cache.attention_contract import (
    get_container_window_size,
    get_layer_window_sizes,
    is_attention_module,
)
from sglang.srt.hardware_backend.mlx.kv_cache.attention_wrapper import (
    MLXAttentionWrapper,
)

logger = logging.getLogger(__name__)


def _find_attention_attr(layer: Any) -> str | None:
    """Return the direct child name that satisfies the attention contract."""
    if not isinstance(layer, nn.Module):
        raise TypeError(f"Expected mlx.nn.Module layer, got {type(layer)}")
    for name, module in layer.children().items():
        if isinstance(module, MLXAttentionWrapper) or is_attention_module(module):
            return name
    return None


def find_attention_layers(model: Any) -> tuple[list[Any], list[str | None]]:
    """Find transformer layers and per-layer attention attribute names."""
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
    window_sizes = get_layer_window_sizes(model)
    if not window_sizes and get_container_window_size(model) is not None:
        # e.g. gemma3-style containers derive per-layer windows from a
        # pattern instead of ``layer_types``. Prefill masks (delegated to
        # the container) honor the window, but batched decode cannot
        # without a per-layer map, so outputs would diverge past the
        # window. Surface it instead of silently splitting semantics.
        logger.warning(
            "Model %s declares a sliding window but no per-layer "
            "layer_types map; MLX batched decode will not apply the "
            "window and long-context output may be incorrect.",
            type(model).__name__,
        )
    patched = 0
    for idx, (layer, attn_attr) in enumerate(zip(layer_list, attn_attrs)):
        if attn_attr is None:
            continue
        attn = getattr(layer, attn_attr)
        if isinstance(attn, MLXAttentionWrapper):
            continue
        setattr(
            layer,
            attn_attr,
            MLXAttentionWrapper(attn, idx, window_size=window_sizes.get(idx)),
        )
        patched += 1
    return patched


def get_num_layers(model: Any) -> int:
    """Return the number of transformer layers."""
    layer_list, _ = find_attention_layers(model)
    return len(layer_list)
