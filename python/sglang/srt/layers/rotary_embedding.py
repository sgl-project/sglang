# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.6.6.post1/vllm/model_executor/layers/rotary_embedding.py

"""Rotary Positional Embeddings."""
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from transformers.models.qwen2_5_omni import Qwen2_5OmniThinkerConfig

from sglang.srt.custom_op import CustomOp
from sglang.srt.utils import is_cuda_available
from sglang.utils import logger

_is_cuda_available = is_cuda_available()

if _is_cuda_available:
    from sgl_kernel import apply_rope_with_cos_sin_cache_inplace
else:
    from vllm._custom_ops import rotary_embedding as vllm_rotary_embedding


def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)


class RotaryEmbedding(CustomOp):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        # NOTE(ByronHsu): cache needs to be in FP32 for numerical stability
        if not _is_cuda_available:
            cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if _is_cuda_available and (self.head_size in [64, 128, 256, 512]):
            apply_rope_with_cos_sin_cache_inplace(
                positions=positions,
                query=query,
                key=key,
                head_size=self.head_size,
                cos_sin_cache=self.cos_sin_cache,
                is_neox=self.is_neox_style,
            )
        else:
            self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)
            vllm_rotary_embedding(
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


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with linear scaling.

    It supports multiple scaling factors. Since multiple LoRA adapters may have
    different scaling factors, we need multiple cos/sin caches. In this way,
    instead of running rotary embedding kernel per lora, we can run multiple
    lora in a batched way.

    In addition to that, we also keep the cos/sin cache for the scaling factor
    of 1 (default) at all times.

    Exemplary for two scaling factors x=1, y and z with embeddings
    [[x11, x12, ... x1m], ..., [xn1, xn2, ..., xnm]] and
    [[y11, y12, ... y1o], ..., [yn1, yn2, ..., yno]], and
    [[z11, z12, ... z1p], ..., [zn1, zn2, ..., znp]],

    we construct the cos/sin cache as follows:
    [[x11, x12, ... x1m, y11, y12, ... y1o, z11, z12, ... z1p],
        ...
     [xn1, xn2, ... xnm, yn1, yn2, ... yno, zn1, zn2, ... znp]]

    We then use offsets to index into the cos/sin cache for
    the respective scaling factors.

    The offset to cache can be accessed via `scaling_factor_to_offset` API.

    Credits to the Reddit user /u/kaiokendev
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        scaling_factors: Union[List[float], float],
        dtype: torch.dtype,
    ) -> None:
        if isinstance(scaling_factors, float):
            scaling_factors = [scaling_factors]
        self.scaling_factors: List[float] = scaling_factors  # noqa
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )
        # Lazy initialized.
        self._scaling_factor_to_offset: Dict[float, int]

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        cache_list: List[torch.Tensor] = []
        # offsets to the next cache in a tensor.
        # Each offset corresponds to the same index in scaling_factors.
        offsets: List[int] = []
        for scaling_factor in self.scaling_factors:
            # NOTE(woosuk): self.max_position_embeddings is the original
            # maximum length before applying the rope scaling.
            # Thus, the maximum length after applying the rope scaling is
            # self.max_position_embeddings * self.scaling_factor.
            max_len = self.max_position_embeddings * scaling_factor
            t = torch.arange(max_len, dtype=torch.float)
            t = t / scaling_factor

            freqs = torch.einsum("i,j -> ij", t, inv_freq)
            cos = freqs.cos()
            sin = freqs.sin()
            cache = torch.cat((cos, sin), dim=-1)
            if not cache_list:
                offset = 0
            else:
                last_offset = offsets[-1]
                next_max_len = cache_list[-1].shape[0]
                offset = last_offset + next_max_len
            offsets.append(offset)
            cache_list.append(cache)
        self._scaling_factor_to_offset = {
            float(scaling_factor): offsets[i]
            for i, scaling_factor in enumerate(self.scaling_factors)
        }
        assert len(self.scaling_factors) == len(offsets)
        return torch.cat(cache_list, dim=0)

    @property
    def scaling_factor_to_offset(self) -> Dict[float, int]:
        return self._scaling_factor_to_offset


class DynamicNTKScalingRotaryEmbedding(RotaryEmbedding):
    """RotaryEmbedding extended with Dynamic NTK scaling.

    Credits to the Reddit users /u/bloc97 and /u/emozilla
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
    ) -> None:
        self.scaling_factor = scaling_factor
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        # NOTE(woosuk): self.max_position_embeddings is the original
        # maximum length before applying the rope scaling.
        # Thus, the maximum length after applying the rope scaling is
        # self.max_position_embeddings * self.scaling_factor.
        max_len = self.max_position_embeddings * self.scaling_factor
        base = self.base * (
            (self.scaling_factor * max_len / self.max_position_embeddings)
            - (self.scaling_factor - 1)
        ) ** (self.rotary_dim / (self.rotary_dim - 2))
        inv_freq = self._compute_inv_freq(base)
        t = torch.arange(max_len, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache


# Inverse dim formula to find dim based on number of rotations
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
    low: float, high: float, dim: int, dtype: torch.dtype, device: torch.device = None
) -> torch.Tensor:
    if low == high:
        high += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=dtype, device=device) - low) / (high - low)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func


def _yarn_get_mscale(scale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0


class YaRNScalingRotaryEmbedding(RotaryEmbedding):
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


class Phi3LongRoPEScaledRotaryEmbedding(nn.Module):
    """Phi3 family of models scaled rotary embedding.

    Based on the original RotaryEmbedding implementation.
    """

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        original_max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        short_factor: List[float],
        long_factor: List[float],
        short_mscale: Optional[float] = None,
        long_mscale: Optional[float] = None,
    ):
        super().__init__()

        if is_neox_style is False:
            raise ValueError(
                "`Phi3LongRoPEScaledRotaryEmbedding` only supports neox_style."
            )

        self.rotary_dim = rotary_dim
        self.head_size = head_size
        self.max_position_embeddings = max_position_embeddings
        self.original_max_position_embeddings = original_max_position_embeddings
        self.base = base
        self.short_factor = short_factor
        self.long_factor = long_factor

        scale = self.max_position_embeddings / self.original_max_position_embeddings
        if scale <= 1.0:
            scaling_factor = 1.0
        else:
            scaling_factor = math.sqrt(
                1 + math.log(scale) / math.log(self.original_max_position_embeddings)
            )
        if short_mscale is None:
            short_mscale = scaling_factor
        if long_mscale is None:
            long_mscale = scaling_factor

        self.short_mscale = short_mscale
        self.long_mscale = long_mscale

        short_cache = self._compute_cos_sin_cache(
            original_max_position_embeddings, short_factor, short_mscale
        )
        short_cache = short_cache.to(dtype)
        self.register_buffer("short_cos_sin_cache", short_cache, persistent=False)

        long_cache = self._compute_cos_sin_cache(
            max_position_embeddings, long_factor, long_mscale
        )
        long_cache = long_cache.to(dtype)
        self.register_buffer("long_cos_sin_cache", long_cache, persistent=False)

        long_short_cache = torch.cat(
            [self.short_cos_sin_cache, self.long_cos_sin_cache], dim=0
        )
        self.register_buffer(
            "long_short_cos_sin_cache", long_short_cache, persistent=False
        )

    def _compute_inv_freq(self, rescale_factors: List[float]) -> torch.Tensor:
        rescale_factors = torch.tensor(rescale_factors, dtype=torch.float32)
        inv_freq = 1.0 / (
            rescale_factors
            * (
                self.base
                ** (
                    torch.arange(0, self.rotary_dim, 2, dtype=torch.float)
                    / self.rotary_dim
                )
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(
        self,
        max_position_embeddings: int,
        rescale_factors: List[float],
        mscale: float,
    ) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(rescale_factors)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * mscale
        sin = freqs.sin() * mscale
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = query.view(*query.shape[:-1], -1, self.head_size)
        key = key.view(*key.shape[:-1], -1, self.head_size)

        k = self.original_max_position_embeddings
        long_prompt_offset = (
            torch.any(positions > k).float() * torch.full_like(positions, k)
        ).long()
        idx = (
            torch.add(positions, long_prompt_offset)
            if long_prompt_offset is not None
            else positions
        )
        self.long_short_cos_sin_cache: torch.Tensor = self.long_short_cos_sin_cache.to(
            idx.device
        )
        idx = torch.add(idx, offsets) if offsets is not None else idx
        cos_sin = torch.index_select(self.long_short_cos_sin_cache, 0, idx)

        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.repeat(1, 2).unsqueeze(-2)
        sin = sin.repeat(1, 2).unsqueeze(-2)

        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = query_rot * cos + _rotate_neox(query_rot) * sin
        query = torch.cat((query_rot, query_pass), dim=-1)

        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = key_rot * cos + _rotate_neox(key_rot) * sin
        key = torch.cat((key_rot, key_pass), dim=-1)

        return query.flatten(-2), key.flatten(-2)


def yarn_get_mscale(scale: float = 1, mscale: float = 1) -> float:
    if scale <= 1:
        return 1.0
    return 0.1 * mscale * math.log(scale) + 1.0


class DeepseekScalingRotaryEmbedding(RotaryEmbedding):
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
        mscale: float = 1,
        mscale_all_dim: float = 0,
        device: Optional[str] = "cuda",
    ) -> None:
        self.scaling_factor = scaling_factor
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        # Get n-d magnitude scaling corrected for interpolation.
        self.mscale = float(
            yarn_get_mscale(self.scaling_factor, float(mscale))
            / yarn_get_mscale(self.scaling_factor, float(mscale_all_dim))
            * attn_factor
        )
        self.device = device
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=self.device)
            / self.rotary_dim
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
                low, high, self.rotary_dim // 2, dtype=torch.float, device=self.device
            )
        ) * self.extrapolation_factor
        inv_freq = (
            inv_freq_interpolation * (1 - inv_freq_mask)
            + inv_freq_extrapolation * inv_freq_mask
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.scaling_factor)
        t = torch.arange(
            self.max_position_embeddings * self.scaling_factor,
            device=self.device,
            dtype=torch.float32,
        )
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos() * self.mscale
        sin = freqs.sin() * self.mscale
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward_hip(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward(self, *args, **kwargs):
        if torch.compiler.is_compiling():
            return self.forward_native(*args, **kwargs)
        if _is_cuda_available:
            return self.forward_cuda(*args, **kwargs)
        else:
            return self.forward_native(*args, **kwargs)

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(positions.device)
        cos_sin = self.cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)

        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        query_rot = query_rot * cos + rotate_fn(query_rot) * sin
        key_rot = key_rot * cos + rotate_fn(key_rot) * sin

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key


class Llama3RotaryEmbedding(RotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        scaling_factor: float,
        low_freq_factor: float,
        high_freq_factor: float,
        orig_max_position: int,
    ) -> None:
        self.scaling_factor = scaling_factor
        self.low_freq_factor = low_freq_factor
        self.high_freq_factor = high_freq_factor
        self.orig_max_position = orig_max_position
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        inv_freqs = super()._compute_inv_freq(base)
        low_freq_wavelen = self.orig_max_position / self.low_freq_factor
        high_freq_wavelen = self.orig_max_position / self.high_freq_factor

        wave_len = 2 * math.pi / inv_freqs
        if self.low_freq_factor != self.high_freq_factor:
            smooth = (self.orig_max_position / wave_len - self.low_freq_factor) / (
                self.high_freq_factor - self.low_freq_factor
            )
        else:
            smooth = 0
        new_freqs = torch.where(
            wave_len < high_freq_wavelen,
            inv_freqs,
            torch.where(
                wave_len > low_freq_wavelen,
                inv_freqs / self.scaling_factor,
                (1 - smooth) * inv_freqs / self.scaling_factor + smooth * inv_freqs,
            ),
        )
        return new_freqs


class Llama4VisionRotaryEmbedding(RotaryEmbedding):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
    ):
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        inv_freqs = super()._compute_inv_freq(base)
        inv_freqs = inv_freqs[: (self.rotary_dim // 2)]
        return inv_freqs

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)

        # self.max_position_embeddings here is number of image patches
        # i.e. (image_size // patch_size) ** 2
        num_patches = self.max_position_embeddings
        img_idx = torch.arange(num_patches, dtype=torch.int32).reshape(num_patches, 1)
        img_idx = torch.cat([img_idx, img_idx[:1]], dim=0)
        img_idx[-1, -1] = -2  # set to ID_CLS_TOKEN
        num_patches_single_dim = int(math.sqrt(num_patches))
        frequencies_x = img_idx % num_patches_single_dim
        frequencies_y = img_idx // num_patches_single_dim
        freqs_x = (
            (frequencies_x + 1)[..., None] * inv_freq[None, None, :]
        ).repeat_interleave(2, dim=-1)
        freqs_y = (
            (frequencies_y + 1)[..., None] * inv_freq[None, None, :]
        ).repeat_interleave(2, dim=-1)
        freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
        freqs = freqs.masked_fill(img_idx.reshape(-1, 1, 1) < 0, 0)
        cache = torch.view_as_complex(
            torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1)
        )
        return cache

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(query.device)
        query_ = torch.view_as_complex(query.float().reshape(*query.shape[:-1], -1, 2))
        key_ = torch.view_as_complex(key.float().reshape(*key.shape[:-1], -1, 2))
        broadcast_shape = [
            d if i == 1 or i == (query_.ndim - 1) else 1
            for i, d in enumerate(query_.shape)
        ]
        freqs_ci = self.cos_sin_cache.view(*broadcast_shape)
        query_out = torch.view_as_real(query_ * freqs_ci).flatten(3)
        key_out = torch.view_as_real(key_ * freqs_ci).flatten(3)
        return query_out.type_as(query), key_out.type_as(key)


class MRotaryEmbedding(RotaryEmbedding):
    """Rotary Embedding with Multimodal Sections."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        mrope_section: Optional[List[int]] = None,
    ) -> None:
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

        self.mrope_section = mrope_section
        if self.mrope_section:
            assert sum(self.mrope_section) == rotary_dim // 2

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward().

        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        assert positions.ndim == 1 or positions.ndim == 2

        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if positions.ndim == 2:
            assert self.mrope_section

            cos = torch.cat(
                [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
                dim=-1,
            )
            sin = torch.cat(
                [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],
                dim=-1,
            )

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    @staticmethod
    def get_input_positions(
        input_tokens: List[int],
        image_grid_thw: Union[List[List[int]], torch.Tensor],
        video_grid_thw: Union[List[List[int]], torch.Tensor],
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        vision_end_token_id: int,
        spatial_merge_size: int,
        context_len: int = 0,
        seq_len: Optional[int] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        use_audio_in_video: bool = False,
        audio_seqlens: Optional[torch.LongTensor] = None,
        tokens_per_second: Optional[int] = None,
    ) -> Tuple[torch.Tensor, int]:
        """
        Get mrope input positions and delta value.

        :arg
            second_per_grid_ts (`torch.Tensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        """

        if isinstance(image_grid_thw, torch.Tensor):
            image_grid_thw = image_grid_thw.tolist()
        if isinstance(video_grid_thw, torch.Tensor):
            video_grid_thw = video_grid_thw.tolist()

        input_tokens_tensor = torch.tensor(input_tokens)
        vision_start_indices = torch.argwhere(
            input_tokens_tensor == vision_start_token_id
        ).squeeze(1)
        vision_tokens = input_tokens_tensor[vision_start_indices + 1]
        image_nums = (vision_tokens == image_token_id).sum()
        video_nums = (vision_tokens == video_token_id).sum()
        llm_pos_ids_list: list = []

        st = 0
        remain_images, remain_videos = image_nums, video_nums

        image_index, video_index = 0, 0
        for _ in range(image_nums + video_nums):
            if image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            if video_token_id in input_tokens and remain_videos > 0:
                ed_video = input_tokens.index(video_token_id, st)
            else:
                ed_video = len(input_tokens) + 1
            if ed_image < ed_video:
                t, h, w = (
                    image_grid_thw[image_index][0],
                    image_grid_thw[image_index][1],
                    image_grid_thw[image_index][2],
                )
                image_index += 1
                remain_images -= 1
                second_per_grid_t = 0
                ed = ed_image
            else:
                t, h, w = (
                    video_grid_thw[video_index][0],
                    video_grid_thw[video_index][1],
                    video_grid_thw[video_index][2],
                )
                if second_per_grid_ts is not None:
                    second_per_grid_t = second_per_grid_ts[video_index]
                else:
                    second_per_grid_t = 1.0
                video_index += 1
                remain_videos -= 1
                ed = ed_video
            llm_grid_t, llm_grid_h, llm_grid_w = (
                t,
                h // spatial_merge_size,
                w // spatial_merge_size,
            )
            text_len = ed - st

            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

            t_index = (
                torch.arange(llm_grid_t).view(-1, 1).expand(-1, llm_grid_h * llm_grid_w)
                * second_per_grid_t
                * tokens_per_second
            ).flatten()

            h_index = (
                torch.arange(llm_grid_h)
                .view(1, -1, 1)
                .expand(llm_grid_t, -1, llm_grid_w)
                .flatten()
            )
            w_index = (
                torch.arange(llm_grid_w)
                .view(1, 1, -1)
                .expand(llm_grid_t, llm_grid_h, -1)
                .flatten()
            )
            llm_pos_ids_list.append(
                torch.stack([t_index, h_index, w_index]) + text_len + st_idx
            )
            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = llm_pos_ids_list[-1].max() + 1 if len(llm_pos_ids_list) > 0 else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
            )

        llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
        mrope_position_delta = (llm_positions.max() + 1 - len(input_tokens)).item()
        llm_positions = llm_positions[:, context_len:seq_len]

        return llm_positions, mrope_position_delta

    @staticmethod
    def get_next_input_positions(
        mrope_position_delta: int,
        context_len: int,
        seq_len: int,
    ) -> torch.Tensor:
        return torch.tensor(
            [
                list(
                    range(
                        context_len + mrope_position_delta,
                        seq_len + mrope_position_delta,
                    )
                )
                for _ in range(3)
            ]
        )

    @staticmethod
    def get_llm_pos_ids_for_vision(
        start_idx: int,
        vision_idx: int,
        spatial_merge_size: int,
        t_index: List[int],
        grid_hs: List[int],
        grid_ws: List[int],
    ):
        llm_pos_ids_list = []
        llm_grid_h = grid_hs[vision_idx] // spatial_merge_size
        llm_grid_w = grid_ws[vision_idx] // spatial_merge_size
        h_index = (
            torch.arange(llm_grid_h, device=llm_grid_h.device)
            .view(1, -1, 1)
            .expand(len(t_index), -1, llm_grid_w)
            .flatten()
        )
        w_index = (
            torch.arange(llm_grid_w, device=llm_grid_w.device)
            .view(1, 1, -1)
            .expand(len(t_index), llm_grid_h, -1)
            .flatten()
        )
        t_index = (
            torch.Tensor(t_index)
            .to(llm_grid_h.device)
            .view(-1, 1)
            .expand(-1, llm_grid_h * llm_grid_w)
            .flatten()
            .long()
        )
        _llm_pos_ids = torch.stack([t_index, h_index, w_index]).to("cuda")
        llm_pos_ids_list.append(_llm_pos_ids + start_idx)  # + 1 ) # 12.09 by malinhan
        llm_pos_ids = torch.cat(llm_pos_ids_list, dim=1)
        return llm_pos_ids

    @staticmethod
    def get_rope_index(
        input_ids: Optional[torch.Tensor],
        config: Qwen2_5OmniThinkerConfig,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        use_audio_in_video: bool = False,
        audio_seqlens: Optional[torch.LongTensor] = None,
        second_per_grids: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Calculate the 3D rope index based on image and video's temporal, height and width in LLM.

        Explanation:
            Each embedding sequence contains vision embedding and text embedding or just contains text embedding.

            For pure text embedding sequence, the rotary position embedding has no difference with modern LLMs.
            Examples:
                input_ids: [T T T T T], here T is for text.
                temporal position_ids: [0, 1, 2, 3, 4]
                height position_ids: [0, 1, 2, 3, 4]
                width position_ids: [0, 1, 2, 3, 4]

            For vision and text embedding sequence, we calculate 3D rotary position embedding for vision part
            and 1D rotary position embedding for text part.
            Examples:
                Temporal (Time): 3 patches, representing different segments of the video in time.
                Height: 2 patches, dividing each frame vertically.
                Width: 2 patches, dividing each frame horizontally.
                We also have some important parameters:
                fps (Frames Per Second): The video's frame rate, set to 1. This means one frame is processed each second.
                tokens_per_second: This is a crucial parameter. It dictates how many "time-steps" or "temporal tokens" are conceptually packed into a one-second interval of the video. In this case, we have 25 tokens per second. So each second of the video will be represented with 25 separate time points. It essentially defines the temporal granularity.
                temporal_patch_size: The number of frames that compose one temporal patch. Here, it's 2 frames.
                interval: The step size for the temporal position IDs, calculated as tokens_per_second * temporal_patch_size / fps. In this case, 25 * 2 / 1 = 50. This means that each temporal patch will be have a difference of 50 in the temporal position IDs.
                input_ids: [V V V V V V V V V V V V T T T T T], here V is for vision.
                vision temporal position_ids: [0, 0, 0, 0, 50, 50, 50, 50, 100, 100, 100, 100]
                vision height position_ids: [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1]
                vision width position_ids: [0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1]
                text temporal position_ids: [101, 102, 103, 104, 105]
                text height position_ids: [101, 102, 103, 104, 105]
                text width position_ids: [101, 102, 103, 104, 105]
                Here we calculate the text start position_ids as the max vision position_ids plus 1.

        Args:
            input_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you provide
                it.
            image_grid_thw (`torch.LongTensor` of shape `(num_images, 3)`, *optional*):
                The temporal, height and width of feature shape of each image in LLM.
            video_grid_thw (`torch.LongTensor` of shape `(num_videos, 3)`, *optional*):
                The temporal, height and width of feature shape of each video in LLM.

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.
            use_audio_in_video (`bool`, *optional*):
                 If set to `True`, use the audio in video.
            audio_seqlens (`torch.LongTensor` of shape `(num_audios)`, *optional*):
                The length of feature shape of each audio in LLM.
            second_per_grids (`torch.LongTensor` of shape `(num_videos)`, *optional*):
                The time interval (in seconds) for each grid along the temporal dimension in the 3D position IDs.

        Returns:
            position_ids (`torch.LongTensor` of shape `(3, batch_size, sequence_length)`)
            mrope_position_deltas (`torch.Tensor` of shape `(batch_size)`)
        """
        image_token_id = config.image_token_index
        video_token_id = config.video_token_index
        audio_token_id = config.audio_token_index
        vision_start_token_id = config.vision_start_token_id
        vision_end_token_id = config.vision_end_token_id
        audio_start_token_id = config.audio_start_token_id
        audio_end_token_id = config.audio_end_token_id
        position_id_per_seconds = config.position_id_per_seconds
        seconds_per_chunk = config.seconds_per_chunk
        spatial_merge_size = config.vision_config.spatial_merge_size
        try:
            mrope_position_deltas = []
            if input_ids is not None and (
                image_grid_thw is not None or video_grid_thw is not None
            ):
                total_input_ids = input_ids
                attention_mask = torch.ones_like(total_input_ids).to("cuda")
                position_ids = torch.ones(
                    3,
                    input_ids.shape[0],
                    input_ids.shape[1],
                    dtype=input_ids.dtype,
                    device=input_ids.device,
                ).to("cuda")
                image_idx, video_idx, audio_idx = 0, 0, 0
                attention_mask = attention_mask.to(total_input_ids.device)
                for i, input_ids in enumerate(total_input_ids):
                    input_ids = input_ids[attention_mask[i] == 1]
                    image_nums, video_nums, audio_nums = 0, 0, 0
                    vision_start_indices = torch.argwhere(
                        input_ids == vision_start_token_id
                    ).squeeze(1)
                    vision_tokens = input_ids[vision_start_indices + 1]
                    audio_nums = torch.sum(input_ids == audio_start_token_id)
                    image_nums = (vision_tokens == image_token_id).sum()
                    video_nums = (
                        (vision_tokens == audio_start_token_id).sum()
                        if use_audio_in_video
                        else (vision_tokens == video_token_id).sum()
                    )
                    input_tokens = input_ids.tolist()
                    llm_pos_ids_list: list = []
                    st = 0
                    remain_images, remain_videos, remain_audios = (
                        image_nums,
                        video_nums,
                        audio_nums,
                    )
                    multimodal_nums = (
                        image_nums + audio_nums
                        if use_audio_in_video
                        else image_nums + video_nums + audio_nums
                    )
                    for _ in range(multimodal_nums):
                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        if image_token_id in input_tokens and remain_images > 0:
                            ed_image = input_tokens.index(image_token_id, st)
                        else:
                            ed_image = len(input_tokens) + 1
                        if video_token_id in input_tokens and remain_videos > 0:
                            ed_video = input_tokens.index(video_token_id, st)
                        else:
                            ed_video = len(input_tokens) + 1
                        if audio_token_id in input_tokens and remain_audios > 0:
                            ed_audio = input_tokens.index(audio_token_id, st)
                        else:
                            ed_audio = len(input_tokens) + 1
                        min_ed = min(ed_image, ed_video, ed_audio)
                        if min_ed == ed_audio:
                            text_len = min_ed - st - 1
                            if text_len != 0:
                                st_idx = (
                                    llm_pos_ids_list[-1].max() + 1
                                    if len(llm_pos_ids_list) > 0
                                    else 0
                                )
                                llm_pos_ids_list.append(
                                    torch.arange(text_len)
                                    .view(1, -1)
                                    .expand(3, -1)
                                    .to("cuda")
                                    + st_idx
                                )

                            st_idx = (
                                llm_pos_ids_list[-1].max() + 1
                                if len(llm_pos_ids_list) > 0
                                else 0
                            )
                            bos_len = 1
                            llm_pos_ids_list.append(
                                torch.arange(bos_len)
                                .view(1, -1)
                                .expand(3, -1)
                                .to("cuda")
                                + st_idx
                            )

                            st_idx = (
                                llm_pos_ids_list[-1].max() + 1
                                if len(llm_pos_ids_list) > 0
                                else 0
                            )
                            audio_len = (
                                (audio_seqlens[audio_idx] - 1) // 2 + 1 - 2
                            ) // 2 + 1
                            llm_pos_ids = (
                                torch.arange(audio_len)
                                .view(1, -1)
                                .expand(3, -1)
                                .to("cuda")
                                + st_idx
                            )
                            llm_pos_ids_list.append(llm_pos_ids)

                            st_idx = (
                                llm_pos_ids_list[-1].max() + 1
                                if len(llm_pos_ids_list) > 0
                                else 0
                            )
                            eos_len = 1
                            llm_pos_ids_list.append(
                                torch.arange(eos_len)
                                .view(1, -1)
                                .expand(3, -1)
                                .to("cuda")
                                + st_idx
                            )

                            st += text_len + bos_len + audio_len + eos_len
                            audio_idx += 1
                            remain_audios -= 1

                        elif min_ed == ed_image:
                            text_len = min_ed - st - 1
                            if text_len != 0:
                                st_idx = (
                                    llm_pos_ids_list[-1].max() + 1
                                    if len(llm_pos_ids_list) > 0
                                    else 0
                                )
                                llm_pos_ids_list.append(
                                    torch.arange(text_len)
                                    .view(1, -1)
                                    .expand(3, -1)
                                    .to("cuda")
                                    + st_idx
                                )

                            st_idx = (
                                llm_pos_ids_list[-1].max() + 1
                                if len(llm_pos_ids_list) > 0
                                else 0
                            )
                            bos_len = 1
                            llm_pos_ids_list.append(
                                torch.arange(bos_len)
                                .view(1, -1)
                                .expand(3, -1)
                                .to("cuda")
                                + st_idx
                            )

                            st_idx = (
                                llm_pos_ids_list[-1].max() + 1
                                if len(llm_pos_ids_list) > 0
                                else 0
                            )
                            grid_t = image_grid_thw[image_idx][0]
                            grid_hs = image_grid_thw[:, 1]
                            grid_ws = image_grid_thw[:, 2]
                            t_index = (
                                torch.arange(grid_t).to("cuda")
                                * 1
                                * position_id_per_seconds
                            ).long()
                            llm_pos_ids = MRotaryEmbedding.get_llm_pos_ids_for_vision(
                                st_idx,
                                image_idx,
                                spatial_merge_size,
                                t_index,
                                grid_hs,
                                grid_ws,
                            )
                            image_len = image_grid_thw[image_idx].prod() // (
                                spatial_merge_size**2
                            )
                            llm_pos_ids_list.append(llm_pos_ids)

                            st_idx = (
                                llm_pos_ids_list[-1].max() + 1
                                if len(llm_pos_ids_list) > 0
                                else 0
                            )
                            eos_len = 1
                            llm_pos_ids_list.append(
                                torch.arange(eos_len)
                                .view(1, -1)
                                .expand(3, -1)
                                .to("cuda")
                                + st_idx
                            )

                            st += text_len + bos_len + image_len + eos_len
                            image_idx += 1
                            remain_images -= 1

                        elif min_ed == ed_video and not use_audio_in_video:
                            text_len = min_ed - st - 1
                            if text_len != 0:
                                st_idx = (
                                    llm_pos_ids_list[-1].max() + 1
                                    if len(llm_pos_ids_list) > 0
                                    else 0
                                )
                                llm_pos_ids_list.append(
                                    torch.arange(text_len)
                                    .view(1, -1)
                                    .expand(3, -1)
                                    .to("cuda")
                                    + st_idx
                                )

                            st_idx = (
                                llm_pos_ids_list[-1].max() + 1
                                if len(llm_pos_ids_list) > 0
                                else 0
                            ).to("cuda")
                            bos_len = 1
                            llm_pos_ids_list.append(
                                torch.arange(bos_len)
                                .view(1, -1)
                                .expand(3, -1)
                                .to("cuda")
                                + st_idx
                            )

                            st_idx = (
                                llm_pos_ids_list[-1].max() + 1
                                if len(llm_pos_ids_list) > 0
                                else 0
                            )
                            grid_t = video_grid_thw[video_idx][0]
                            grid_hs = video_grid_thw[:, 1]
                            grid_ws = video_grid_thw[:, 2]
                            t_index = (
                                torch.arange(grid_t).to("cuda")
                                * second_per_grids[video_idx].cpu().float()
                                * position_id_per_seconds
                            ).long()
                            llm_pos_ids = MRotaryEmbedding.get_llm_pos_ids_for_vision(
                                st_idx,
                                video_idx,
                                spatial_merge_size,
                                t_index,
                                grid_hs,
                                grid_ws,
                            )
                            video_len = video_grid_thw[video_idx].prod() // (
                                spatial_merge_size**2
                            )
                            llm_pos_ids_list.append(llm_pos_ids)

                            st_idx = (
                                llm_pos_ids_list[-1].max() + 1
                                if len(llm_pos_ids_list) > 0
                                else 0
                            )
                            eos_len = 1
                            llm_pos_ids_list.append(
                                torch.arange(eos_len)
                                .view(1, -1)
                                .expand(3, -1)
                                .to("cuda")
                                + st_idx
                            )

                            st += text_len + bos_len + video_len + eos_len
                            video_idx += 1
                            remain_videos -= 1

                        elif min_ed == ed_video and use_audio_in_video:
                            text_len = min_ed - st - 2
                            if text_len != 0:
                                st_idx = (
                                    llm_pos_ids_list[-1].max() + 1
                                    if len(llm_pos_ids_list) > 0
                                    else 0
                                )
                                llm_pos_ids_list.append(
                                    torch.arange(text_len)
                                    .view(1, -1)
                                    .expand(3, -1)
                                    .to("cuda")
                                    + st_idx
                                )

                            st_idx = (
                                llm_pos_ids_list[-1].max() + 1
                                if len(llm_pos_ids_list) > 0
                                else 0
                            )
                            bos_len = 1
                            llm_pos_ids_list.append(
                                torch.arange(bos_len)
                                .view(1, -1)
                                .expand(3, -1)
                                .to("cuda")
                                + st_idx
                            )
                            llm_pos_ids_list.append(
                                torch.arange(bos_len)
                                .view(1, -1)
                                .expand(3, -1)
                                .to("cuda")
                                + st_idx
                            )

                            st_idx = (
                                llm_pos_ids_list[-1].max() + 1
                                if len(llm_pos_ids_list) > 0
                                else 0
                            )
                            audio_len = (
                                (audio_seqlens[audio_idx] - 1) // 2 + 1 - 2
                            ) // 2 + 1
                            audio_llm_pos_ids = (
                                torch.arange(audio_len)
                                .view(1, -1)
                                .expand(3, -1)
                                .to("cuda")
                                + st_idx
                            )
                            grid_t = video_grid_thw[video_idx][0]
                            grid_hs = video_grid_thw[:, 1]
                            grid_ws = video_grid_thw[:, 2]

                            t_index = (
                                (
                                    torch.arange(grid_t).to("cuda")
                                    * second_per_grids[video_idx].cpu().float()
                                    * position_id_per_seconds
                                )
                                .long()
                                .to("cuda")
                            )
                            video_llm_pos_ids = (
                                MRotaryEmbedding.get_llm_pos_ids_for_vision(
                                    st_idx,
                                    video_idx,
                                    spatial_merge_size,
                                    t_index,
                                    grid_hs,
                                    grid_ws,
                                )
                            )

                            t_ntoken_per_chunk = int(
                                position_id_per_seconds * seconds_per_chunk
                            )
                            video_chunk_indexes = MRotaryEmbedding.get_chunked_index(
                                video_llm_pos_ids, t_ntoken_per_chunk, st_idx
                            )
                            audio_chunk_indexes = MRotaryEmbedding.get_chunked_index(
                                audio_llm_pos_ids, t_ntoken_per_chunk, st_idx
                            )
                            sub_len = 0
                            for j in range(
                                max(len(video_chunk_indexes), len(audio_chunk_indexes))
                            ):
                                video_chunk_index = (
                                    video_chunk_indexes[j]
                                    if j < len(video_chunk_indexes)
                                    else None
                                )
                                audio_chunk_index = (
                                    audio_chunk_indexes[j]
                                    if j < len(audio_chunk_indexes)
                                    else None
                                )
                                if video_chunk_index is not None:
                                    sub_len += (
                                        video_chunk_index[1] - video_chunk_index[0]
                                    )

                                    llm_pos_ids_list.append(
                                        video_llm_pos_ids[
                                            :,
                                            video_chunk_index[0] : video_chunk_index[1],
                                        ]
                                    )
                                if audio_chunk_index is not None:
                                    sub_len += (
                                        audio_chunk_index[1] - audio_chunk_index[0]
                                    )

                                    llm_pos_ids_list.append(
                                        audio_llm_pos_ids[
                                            :,
                                            audio_chunk_index[0] : audio_chunk_index[1],
                                        ]
                                    )
                            video_len = video_grid_thw[video_idx].prod() // (
                                spatial_merge_size**2
                            )

                            st_idx = (
                                llm_pos_ids_list[-1].max() + 1
                                if len(llm_pos_ids_list) > 0
                                else 0
                            )
                            eos_len = 1
                            llm_pos_ids_list.append(
                                torch.arange(eos_len)
                                .view(1, -1)
                                .expand(3, -1)
                                .to("cuda")
                                + st_idx
                            )
                            llm_pos_ids_list.append(
                                torch.arange(eos_len)
                                .view(1, -1)
                                .expand(3, -1)
                                .to("cuda")
                                + st_idx
                            )

                            st += (
                                text_len
                                + bos_len * 2
                                + audio_len
                                + video_len
                                + eos_len * 2
                            )

                            audio_idx += 1
                            video_idx += 1
                            remain_videos -= 1
                            remain_audios -= 1

                    if st < len(input_tokens):
                        st_idx = (
                            llm_pos_ids_list[-1].max() + 1
                            if len(llm_pos_ids_list) > 0
                            else 0
                        )
                        text_len = len(input_tokens) - st
                        llm_pos_ids_list.append(
                            torch.arange(text_len).view(1, -1).expand(3, -1).to("cuda")
                            + st_idx
                        )

                    llm_positions = (
                        torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1).to("cuda")
                    )

                    position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                        position_ids.device
                    )
                    mrope_position_deltas.append(
                        llm_positions.max() + 1 - len(input_ids)
                    )
                mrope_position_deltas = torch.tensor(
                    mrope_position_deltas, device=input_ids.device
                ).unsqueeze(1)

                return position_ids, mrope_position_deltas
            else:
                assert input_ids is not None, input_ids
                # position_ids = attention_mask.long().cumsum(-1) - 1
                # position_ids.masked_fill_(attention_mask == 0, 1)
                s = input_ids.shape[1]
                position_ids = torch.arange(s)
                position_ids = (
                    position_ids.unsqueeze(0).expand(3, -1, -1).to(input_ids.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - s

                return position_ids, mrope_position_deltas
        except Exception as e:
            logger.info(f"Please consider disabling chunked_prefill: {e}")
            raise

    @staticmethod
    def _pad_to_list_of_tensors_1d(tensor_list, padding_value=0, padding_side="left"):
        lengths = [len(tensor) for tensor in tensor_list]
        max_length = max(lengths)
        pad_len = [max_length - leng for leng in lengths]
        for idx in range(len(tensor_list)):
            if pad_len[idx] != 0:
                if padding_side == "left":
                    tensor_list[idx] = torch.cat(
                        [
                            torch.full(
                                size=[pad_len[idx]],
                                fill_value=padding_value,
                                dtype=tensor_list[idx].dtype,
                                device=tensor_list[idx].device,
                            ),
                            tensor_list[idx],
                        ],
                        dim=0,
                    )
                else:
                    tensor_list[idx] = torch.cat(
                        [
                            tensor_list[idx],
                            torch.full(
                                size=[pad_len[idx]],
                                fill_value=padding_value,
                                dtype=tensor_list[idx].dtype,
                                device=tensor_list[idx].device,
                            ),
                        ],
                        dim=0,
                    )
        return tensor_list


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
        rotary_emb = RotaryEmbedding(
            head_size, rotary_dim, max_position, base, is_neox_style, dtype
        )
    else:
        if "rope_type" in rope_scaling:
            scaling_type = rope_scaling["rope_type"]
        elif "type" in rope_scaling:
            scaling_type = rope_scaling["type"]
        else:
            raise ValueError("Unknown RoPE scaling type")

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
            # assert max_position == original_max_position * scaling_factor
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


# Copied from transformers
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim=1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    orig_q_dtype = q.dtype
    orig_k_dtype = k.dtype
    q, k = q.float(), k.float()

    # embedding is performed in float
    cos = cos.unsqueeze(unsqueeze_dim).float()
    sin = sin.unsqueeze(unsqueeze_dim).float()
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)

    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)

    return q_embed, k_embed


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
        return get_rope(
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
