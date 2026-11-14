"""Utility functions for vision attention layers."""

import torch

from sglang.srt.layers.vocab_parallel_embedding import pad_vocab_size
from sglang.srt.runtime_context import get_parallel
from sglang.srt.utils import is_cpu

_is_cpu = is_cpu()


def update_vit_attn_dummy_heads_config(config):
    """Update HF config to ensure vision attention num_attention_heads is divisible by tp_size"""
    tp_size = get_parallel().attn_tp_size
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


def update_vit_mlp_dummy_neurons_config(config):
    """CPU-only: pad the vision MLP hidden width to a multiple of tp_size.

    Some models (e.g. GLM4V) reuse `vision_config.out_hidden_size` as both the
    vision block MLP's hidden width and the vision->language projection width;
    only the former may be padded, so the padded value is stored under a
    separate `mlp_hidden_size` attribute instead of mutating `out_hidden_size`.
    The checkpoint's narrower gate_up_proj/down_proj weights are zero-padded
    automatically at load time by the generic CPU path in
    `sglang.srt.layers.linear` (gate_proj/up_proj are loaded as separate
    shards), so no custom weight-loading code is needed here.

    The per-rank shard width (padded size / tp_size) must also be a multiple
    of 32: it is used as the AMX GEMM's IC (down_proj input) and, doubled, as
    its OC (gate_up_proj's two concatenated per-rank outputs, so 32 keeps
    each individual chunk divisible by 16 too). Padding only to a multiple of
    tp_size (ignoring this) leaves the AMX weight-prepacking check failing
    silently, falling back to a slow non-AMX GEMM for every vision block.
    """
    AMX_TILE_ALIGN = 32
    mlp_hidden_size = config.vision_config.out_hidden_size
    if _is_cpu:
        tp_size = get_parallel().tp_size
        mlp_hidden_size = pad_vocab_size(mlp_hidden_size, tp_size * AMX_TILE_ALIGN)
    setattr(config.vision_config, "mlp_hidden_size", mlp_hidden_size)


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
        if loaded_weight.numel() == head_dim:
            return loaded_weight
        padded_weight = loaded_weight.new_zeros(head_dim * num_dummy_heads)
        loaded_weight = torch.cat([loaded_weight, padded_weight], dim=0)
    return loaded_weight
