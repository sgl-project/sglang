"""Utility functions for vision attention layers."""

import torch

from sglang.srt.layers.dp_attention import get_attention_tp_size


def update_vit_attn_dummy_heads_config(config):
    """Update HF config to ensure vision attention num_attention_heads is divisible by tp_size"""
    tp_size = get_attention_tp_size()
    num_heads = getattr(
        config.vision_config,
        "num_heads",
        getattr(config.vision_config, "num_attention_heads", None),
    )
    head_dim = config.vision_config.hidden_size // num_heads
    num_dummy_heads = 0

    if num_heads % tp_size != 0:
        num_dummy_heads = ((num_heads + tp_size - 1) // tp_size) * tp_size - num_heads

    setattr(config.vision_config, "head_dim", head_dim)
    setattr(config.vision_config, "num_dummy_heads", num_dummy_heads)


def pad_vit_attn_dummy_heads(config, name: str, loaded_weight: torch.Tensor):
    """Pad attention qkv weights for dummy heads"""
    num_dummy_heads = config.vision_config.num_dummy_heads
    if num_dummy_heads == 0:
        return loaded_weight
    head_dim = config.vision_config.head_dim

    if "attn.qkv_proj" in name:
        wq, wk, wv = loaded_weight.chunk(3, dim=0)
        if name.endswith(".weight"):
            dummy_shape = [num_dummy_heads, head_dim, wq.shape[-1]]
        elif name.endswith(".bias"):
            dummy_shape = [num_dummy_heads, head_dim]
        else:
            raise RuntimeError(f"Unsupported weight with name={name}")
        pad_func = lambda x: torch.cat(
            [x.unflatten(0, (-1, head_dim)), x.new_zeros(dummy_shape)], dim=0
        ).flatten(0, 1)
        wq, wk, wv = pad_func(wq), pad_func(wk), pad_func(wv)
        loaded_weight = torch.cat([wq, wk, wv], dim=0)
    elif any([_ in name for _ in ["attn.q_proj", "attn.k_proj", "attn.v_proj"]]):
        if name.endswith(".weight"):
            dummy_shape = [num_dummy_heads, head_dim, loaded_weight.shape[-1]]
        elif name.endswith(".bias"):
            dummy_shape = [num_dummy_heads, head_dim]
        else:
            raise RuntimeError(f"Unsupported weight with name={name}")
        padded_weight = loaded_weight.new_zeros(dummy_shape)
        loaded_weight = torch.cat(
            [loaded_weight.unflatten(0, (-1, head_dim)), padded_weight], dim=0
        ).flatten(0, 1)
    elif "attn.proj.weight" in name:
        padded_weight = loaded_weight.new_zeros(
            loaded_weight.shape[0], head_dim * num_dummy_heads
        )
        loaded_weight = torch.cat([loaded_weight, padded_weight], dim=-1)
    elif "attn.q_norm.weight" in name or "attn.k_norm.weight" in name:
        padded_weight = loaded_weight.new_zeros(head_dim * num_dummy_heads)
        loaded_weight = torch.cat([loaded_weight, padded_weight], dim=0)
    return loaded_weight
