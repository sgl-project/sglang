"""Factory functions: get_rope, get_rope_cpu, get_rope_wrapper."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch

from sglang.srt.layers.rotary_embedding.base import (
    LinearScalingRotaryEmbedding,
    RotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.mrope import (
    MRotaryEmbedding,
    YaRNScalingMRotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.rope_variant import (
    DeepseekScalingRotaryEmbedding,
    DualChunkRotaryEmbedding,
    DynamicNTKAlphaRotaryEmbedding,
    DynamicNTKScalingRotaryEmbedding,
    FourierRotaryEmbedding,
    Llama3RotaryEmbedding,
    Phi3LongRoPEScaledRotaryEmbedding,
)
from sglang.srt.layers.rotary_embedding.yarn import YaRNScalingRotaryEmbedding
from sglang.srt.utils import get_bool_env_var, is_hip

_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip

if _use_aiter:
    from aiter.rotary_embedding import get_rope as aiter_get_rope

_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
    dual_chunk_attention_config: Optional[Dict[str, Any]] = None,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None

    if dual_chunk_attention_config is not None:
        dual_chunk_attention_tuple = {
            k: tuple(v) if isinstance(v, list) else v
            for k, v in dual_chunk_attention_config.items()
            if k != "sparse_attention_config"
        }
        dual_chunk_attention_args = tuple(dual_chunk_attention_tuple.items())
    else:
        dual_chunk_attention_args = None

    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dual_chunk_attention_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    if dual_chunk_attention_config is not None:
        extra_kwargs = {
            k: v
            for k, v in dual_chunk_attention_config.items()
            if k in ("chunk_size", "local_size")
        }
        rotary_emb = DualChunkRotaryEmbedding(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            dtype,
            **extra_kwargs,
        )
    elif rope_scaling is None:
        rotary_emb = RotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype
        )
    else:
        if "rope_type" in rope_scaling:
            scaling_type = rope_scaling["rope_type"]
        elif "type" in rope_scaling:
            scaling_type = rope_scaling["type"]
        else:
            raise ValueError(
                f"Unknown RoPE scaling type, rope_scaling is {rope_scaling}"
            )

        if scaling_type == "llama3":
            scaling_factor = rope_scaling["factor"]
            low_freq_factor = rope_scaling["low_freq_factor"]
            high_freq_factor = rope_scaling["high_freq_factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            rotary_emb = Llama3RotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                dtype,
                scaling_factor,
                low_freq_factor,
                high_freq_factor,
                original_max_position,
            )
        elif scaling_type == "default":
            if "mrope_section" in rope_scaling:
                rotary_emb = MRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                    mrope_section=rope_scaling["mrope_section"],
                    mrope_interleaved=rope_scaling.get("mrope_interleaved", False),
                )
            elif rope_scaling.get("use_fope", False):
                rotary_emb = FourierRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                    num_kv_heads=rope_scaling["num_kv_heads"],
                    fope_init_factor=rope_scaling.get("fope_init_factor", 0.1),
                    fope_sep_head=rope_scaling.get("fope_sep_head", True),
                    num_inv_freq=rope_scaling.get("num_inv_freq", None),
                )
            else:
                rotary_emb = RotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    dtype,
                )
        elif scaling_type == "linear":
            scaling_factor = rope_scaling["factor"]
            rotary_emb = LinearScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
            )
        elif scaling_type == "dynamic":
            scaling_factor = rope_scaling["factor"]
            if "alpha" in rope_scaling:
                rotary_emb = DynamicNTKAlphaRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    rope_scaling["alpha"],
                    dtype,
                )
            else:
                rotary_emb = DynamicNTKScalingRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    max_position,
                    base,
                    is_neox_style,
                    scaling_factor,
                    dtype,
                )
        elif scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k
                in ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow")
            }
            extra_kwargs["truncate"] = rope_scaling.get("truncate", True)
            if "mrope_section" in rope_scaling:
                rotary_emb = YaRNScalingMRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    original_max_position,
                    base,
                    is_neox_style,
                    scaling_factor,
                    dtype,
                    mrope_section=rope_scaling["mrope_section"],
                    mrope_interleaved=rope_scaling.get("mrope_interleaved", False),
                    **extra_kwargs,
                )
            else:
                rotary_emb = YaRNScalingRotaryEmbedding(
                    head_size,
                    rotary_dim,
                    original_max_position,
                    base,
                    is_neox_style,
                    scaling_factor,
                    dtype,
                    **extra_kwargs,
                )
        elif scaling_type == "deepseek_yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k
                in (
                    "extrapolation_factor",
                    "attn_factor",
                    "beta_fast",
                    "beta_slow",
                    "mscale",
                    "mscale_all_dim",
                )
            }
            rotary_emb = DeepseekScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                original_max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
                **extra_kwargs,
            )
        elif scaling_type == "longrope":
            short_factor = rope_scaling["short_factor"]
            long_factor = rope_scaling["long_factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k in ("short_mscale", "long_mscale")
            }
            rotary_emb = Phi3LongRoPEScaledRotaryEmbedding(
                head_size,
                rotary_dim,
                max_position,
                original_max_position,
                base,
                is_neox_style,
                dtype,
                short_factor,
                long_factor,
                **extra_kwargs,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb


def get_rope_cpu(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
    device: Optional[str] = None,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dtype,
    )
    if key in _ROPE_DICT:
        return _ROPE_DICT[key]

    assert rope_scaling is not None
    scaling_type = rope_scaling["rope_type"]
    assert (
        scaling_type == "deepseek_yarn"
    ), "Only deepseek_yarn is supported for CPU for now"

    scaling_factor = rope_scaling["factor"]
    original_max_position = rope_scaling["original_max_position_embeddings"]
    extra_kwargs = {
        k: v
        for k, v in rope_scaling.items()
        if k
        in (
            "extrapolation_factor",
            "attn_factor",
            "beta_fast",
            "beta_slow",
            "mscale",
            "mscale_all_dim",
        )
    }
    extra_kwargs["device"] = device
    rotary_emb = DeepseekScalingRotaryEmbedding(
        head_size,
        rotary_dim,
        original_max_position,
        base,
        is_neox_style,
        scaling_factor,
        dtype,
        **extra_kwargs,
    )
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb


def get_rope_wrapper(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
    device: Optional[str] = None,
):
    if device != "cpu":
        wrapper = aiter_get_rope if _use_aiter else get_rope
        return wrapper(
            head_size,
            rotary_dim,
            max_position,
            base,
            is_neox_style,
            rope_scaling,
            dtype,
            partial_rotary_factor,
        )

    return get_rope_cpu(
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling,
        dtype,
        partial_rotary_factor,
        device,
    )
