# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the SGLang project
import math
from typing import Optional, Tuple, Type, Union

import numpy as np
from mindspore import Tensor, from_numpy, mint, nn, ops

from sglang.srt.layers.rotary_embedding import (
    _yarn_find_correction_dim,
    _yarn_find_correction_range,
    yarn_get_mscale,
)


def _yarn_get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


def _yarn_find_correction_dim(
    num_rotations: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> float:
    return (dim * math.log(max_position_embeddings / (num_rotations * 2 * math.pi))) / (
        2 * math.log(base)
    )


# Find dim range bounds based on rotations
def _yarn_find_correction_range(
    low_rot: int,
    high_rot: int,
    dim: int,
    base: float = 10000,
    max_position_embeddings: int = 2048,
) -> Tuple[int, int]:
    low = math.floor(
        _yarn_find_correction_dim(low_rot, dim, base, max_position_embeddings)
    )
    high = math.ceil(
        _yarn_find_correction_dim(high_rot, dim, base, max_position_embeddings)
    )
    return max(low, 0), min(high, dim - 1)  # Clamp values just in case


def _yarn_linear_ramp_mask(
    low: float, high: float, dim: int, dtype: np.dtype
) -> np.ndarray:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (np.arange(dim, dtype=dtype) - low) / (high - low)
    ramp_func = np.clip(linear_func, 0, 1)
    return ramp_func


class YaRNScalingRotaryEmbedding(nn.Cell):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factor: float,
        dtype,
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

        super().__init__()

        self.rotary_embedding_op = ops.ApplyRotaryPosEmb(2)
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.freqs_cos, self.freqs_sin = self._compute_cos_sin_cache()

    def _compute_inv_freq(self, scaling_factor: float) -> Tensor:
        pos_freqs = self.base ** (
            np.arange(0, self.rotary_dim, 2, dtype=np.float32) / self.rotary_dim
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
            - _yarn_linear_ramp_mask(
                low,
                high,
                self.rotary_dim // 2,
                dtype=np.float32,  # type: ignore[arg-type]
            )
        ) * self.extrapolation_factor
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> Tuple[Tensor, Tensor]:
        freqs = self._compute_inv_freq(self.scaling_factor)
        t = np.arange(self.max_position_embeddings * self.scaling_factor).astype(
            np.float32
        )
        self.freqs = Tensor(freqs.reshape(1, 1, 1, -1), dtype=self.dtype)
        freqs = np.outer(t, freqs)  # (max_position_embedding, head_dim // 2)
        emb = np.concatenate((freqs, freqs), axis=-1)
        freqs_cos = np.cos(emb) * self.mscale  # (seq_len, head_dim)
        freqs_sin = np.sin(emb) * self.mscale  # (seq_len, head_dim)
        freqs_cos = Tensor(freqs_cos, dtype=self.dtype)
        freqs_sin = Tensor(freqs_sin, dtype=self.dtype)
        return freqs_cos, freqs_sin

    def construct(
        self,
        positions: Tensor,
        query: Tensor,
        key: Tensor,
        batch_valid_length: Tensor,
        is_prefill: bool,
        offsets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if is_prefill:
            freqs_cos = self.freqs_cos
            freqs_sin = self.freqs_sin
        else:
            freqs_cos = mint.index_select(self.freqs_cos, 0, positions.view(-1))
            freqs_sin = mint.index_select(self.freqs_sin, 0, positions.view(-1))

        return self.rotary_embedding_op(
            query, key, freqs_cos, freqs_sin, batch_valid_length
        )


# Adapt from: https://gitee.com/mindspore/vllm-mindspore/blob/master/vllm_mindspore/model_executor/layers/rotary_embedding.py
class BaseRotaryEmbedding(nn.Cell):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        dtype: Optional[Type],
    ) -> None:
        super().__init__()

        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.dtype = dtype

        self.rotary_embedding_op = ops.ApplyRotaryPosEmb(2)

        self.freqs_cos, self.freqs_sin = self._compute_cos_sin_cache()

    def _compute_inv_freq(self, base: Union[int, float]) -> Tensor:
        freqs_base = mint.arange(0, self.rotary_dim, 2).astype(
            np.float32
        )  # (head_dim // 2, )
        freqs = 1.0 / (base ** (freqs_base / self.rotary_dim))  # (head_dim // 2, )
        return freqs

    def _compute_cos_sin_cache(self) -> Tuple[Tensor, Tensor]:
        freqs = self._compute_inv_freq(self.base)
        t = np.arange(0, self.max_position_embeddings, 1).astype(np.float32)
        freqs = np.outer(t, freqs)  # (max_position_embedding, head_dim // 2)
        emb = np.concatenate((freqs, freqs), axis=-1)
        freqs_cos = np.cos(emb)  # (seq_len, head_dim)
        freqs_sin = np.sin(emb)  # (seq_len, head_dim)
        freqs_cos = Tensor(freqs_cos, dtype=self.dtype)
        freqs_sin = Tensor(freqs_sin, dtype=self.dtype)
        return freqs_cos, freqs_sin

    def construct(
        self,
        positions: Tensor,
        query: Tensor,
        key: Tensor,
        batch_valid_length: Tensor,
        is_prefill: bool,
        offsets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if is_prefill:
            freqs_cos = self.freqs_cos
            freqs_sin = self.freqs_sin
        else:
            freqs_cos = mint.index_select(self.freqs_cos, 0, positions.view(-1))
            freqs_sin = mint.index_select(self.freqs_sin, 0, positions.view(-1))

        return self.rotary_embedding_op(
            query, key, freqs_cos, freqs_sin, batch_valid_length
        )


class DeepseekScalingRotaryEmbedding(BaseRotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        scaling_factor: float,
        dtype: Optional[Type],
        *,
        extrapolation_factor: float = 1,
        attn_factor: float = 1,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale))
            / yarn_get_mscale(self.scaling_factor, float(mscale_all_dim))
            * attn_factor
        )
        super().__init__(head_size, rotary_dim, max_position_embeddings, base, dtype)

    def _compute_inv_freq(self, scaling_factor: float):
        pos_freqs = self.base ** (
            np.arange(0, self.rotary_dim, 2, dtype=np.float32) / self.rotary_dim
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
            - _yarn_linear_ramp_mask(low, high, self.rotary_dim // 2, dtype=np.float32)
        ) * self.extrapolation_factor
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq

    def _compute_cos_sin_cache(self):
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = np.arange(self.max_position_embeddings * self.scaling_factor).astype(
            np.float32
        )
        freqs = np.outer(t, inv_freq)
        emb = np.concatenate((freqs, freqs), axis=-1)
        emb = from_numpy(emb)
        freqs_cos = mint.cos(emb) * self.mscale
        freqs_sin = mint.sin(emb) * self.mscale
        return freqs_cos, freqs_sin
