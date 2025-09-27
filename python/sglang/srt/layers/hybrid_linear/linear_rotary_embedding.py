# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.6.6.post1/vllm/model_executor/layers/rotary_embedding.py

"""Rotary Positional Embeddings for support fp32."""
import itertools
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from vllm._custom_ops import rotary_embedding

from sglang.srt.layers.rotary_embedding import (
    RotaryEmbedding,
    _yarn_find_correction_range,
    _yarn_get_mscale,
    _yarn_linear_ramp_mask,
)


class Fp32RotaryEmbedding(RotaryEmbedding):
    """rotary positional embedding support fp32."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )
        self.vllm_rotary_embedding = rotary_embedding

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg=None,  # Optional[FusedSetKVBufferArg]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            fused_set_kv_buffer_arg is None
        ), "save kv cache is not supported for vllm_rotary_embedding."
        self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)
        self.vllm_rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )
        return query, key

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s


class Fp32YaRNScalingRotaryEmbedding(Fp32RotaryEmbedding):
    """RotaryEmbedding extended with YaRN method.

    Credits to Peng et al. github.com/jquesnelle/yarn
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype: torch.dtype,
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation
        self.mscale = float(_yarn_get_mscale(self.scaling_factor) * attn_factor)
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)

        low, high = _yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        # Get n-d rotational scaling corrected for extrapolation
        inv_freq_mask = (
            1
            - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=torch.float)
        ) * self.extrapolation_factor
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(
            self.max_position_embeddings * self.scaling_factor, dtype=torch.float32
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cache = torch.cat((cos, sin), dim=-1)
        return cache


_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}


def get_linear_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype: Optional[torch.dtype] = None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    if dtype is None:
        dtype = torch.get_default_dtype()
    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
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

    if rope_scaling is None:
        rotary_emb = Fp32RotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype
        )
    else:
        if "rope_type" in rope_scaling:
            scaling_type = rope_scaling["rope_type"]
        elif "type" in rope_scaling:
            scaling_type = rope_scaling["type"]
        else:
            raise ValueError("Unknown RoPE scaling type")

        if scaling_type == "yarn":
            scaling_factor = rope_scaling["factor"]
            original_max_position = rope_scaling["original_max_position_embeddings"]
            extra_kwargs = {
                k: v
                for k, v in rope_scaling.items()
                if k
                in ("extrapolation_factor", "attn_factor", "beta_fast", "beta_slow")
            }
            rotary_emb = Fp32YaRNScalingRotaryEmbedding(
                head_size,
                rotary_dim,
                original_max_position,
                base,
                is_neox_style,
                scaling_factor,
                dtype,
                **extra_kwargs,
            )
        else:
            raise ValueError(f"Unknown RoPE scaling type {scaling_type}")
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb
