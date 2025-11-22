# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.6.6.post1/vllm/model_executor/layers/rotary_embedding.py

"""Rotary Positional Embeddings."""
import itertools
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import triton
import triton.language as tl

from sglang.srt.custom_op import CustomOp
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    cpu_has_amx_support,
    get_bool_env_var,
    get_compiler_backend,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    is_xpu,
)

_is_cuda = is_cuda()
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_cpu = is_cpu()
_is_xpu = is_xpu()

if _is_cuda:
    from sgl_kernel import FusedSetKVBufferArg, apply_rope_with_cos_sin_cache_inplace
else:
    FusedSetKVBufferArg = None

if _use_aiter:
    from aiter.rotary_embedding import get_rope as aiter_get_rope

if is_npu():
    import torch_npu

    NPU_ROTARY_MUL_MAX_NUM_HEADS = 1000
    NPU_ROTARY_MUL_MAX_HEAD_SIZE = 896


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
        if not _is_cuda:
            cache = cache.to(dtype)

        if (
            (not (_is_cuda or _is_npu) or self.head_size not in [64, 128, 256, 512])
            and not (_is_cpu and _is_cpu_amx_available)
            and not (_is_xpu)
        ):
            from vllm._custom_ops import rotary_embedding

            self.use_fallback_kernel = True
            self.fallback_rotary_embedding = rotary_embedding
        else:
            self.use_fallback_kernel = False

        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

        self._apply_rotary_emb_wrapped = _apply_rotary_emb

        if get_global_server_args().rl_on_policy_target is not None:
            self._forward_method = self.forward_native
            self._apply_rotary_emb_wrapped = torch.compile(dynamic=True)(
                self._apply_rotary_emb_wrapped
            )

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        init_device = (
            "cpu" if get_global_server_args().rl_on_policy_target is not None else None
        )
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(
                    0, self.rotary_dim, 2, dtype=torch.float, device=init_device
                )
                / self.rotary_dim
            )
        )
        if get_global_server_args().rl_on_policy_target is not None:
            inv_freq = inv_freq.cuda()
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
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        assert (
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for native implementation"

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
        query_rot = self._apply_rotary_emb_wrapped(
            query_rot, cos, sin, self.is_neox_style
        )
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = self._apply_rotary_emb_wrapped(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward_npu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-npu implementation of forward()."""
        assert (
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for npu implementation"

        if get_bool_env_var("SGLANG_ENABLE_TORCH_COMPILE"):
            return self.forward_native(
                positions, query, key, offsets, fused_set_kv_buffer_arg
            )
        else:
            rotary_mode = "half"
            if self.is_neox_style:
                rotary_mode = "half"
            else:
                rotary_mode = "interleave"
            mrope_section = [0, 0, 0]
            query_out, key_out = torch_npu.npu_mrope(
                positions,
                query,
                key,
                self.cos_sin_cache,
                self.head_size,
                mrope_section=mrope_section,
                rotary_mode=rotary_mode,
            )
            return query_out, key_out

    def forward_cpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for cpu implementation"

        positions = torch.add(positions, offsets) if offsets is not None else positions
        if _is_cpu_amx_available:
            return torch.ops.sgl_kernel.rotary_embedding_cpu(
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
            )
        else:
            return self.forward_native(
                positions, query, key, offsets, fused_set_kv_buffer_arg
            )

    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.use_fallback_kernel:
            apply_rope_with_cos_sin_cache_inplace(
                positions=positions,
                query=query,
                key=key,
                head_size=self.head_size,
                cos_sin_cache=self.cos_sin_cache,
                is_neox=self.is_neox_style,
                # Compatible with old sgl-kernel
                **(
                    dict(fused_set_kv_buffer_arg=fused_set_kv_buffer_arg)
                    if fused_set_kv_buffer_arg is not None
                    else {}
                ),
            )
        else:
            assert (
                fused_set_kv_buffer_arg is None
            ), "save kv cache is not supported for vllm_rotary_embedding."
            self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)
            self.fallback_rotary_embedding(
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

    def forward_xpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for xpu implementation"
        positions = torch.add(positions, offsets) if offsets is not None else positions
        return torch.ops.sgl_kernel.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )


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
        device: Optional[str] = "cuda" if not _is_npu else "npu",
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

        # Re-dispatch
        if _is_hip:
            self._forward_method = self.forward_native

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

    def forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward()."""
        dtype = query.dtype
        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

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
        return query.to(dtype), key.to(dtype)

    def forward_npu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        num_tokens, num_q_heads, _ = query.shape
        num_k_heads = key.shape[1]

        self.cos_sin_cache: torch.Tensor = self.cos_sin_cache.to(positions.device)
        cos_sin = self.cos_sin_cache[
            torch.add(positions, offsets) if offsets is not None else positions
        ]
        cos, sin = cos_sin.chunk(2, dim=-1)
        # Reshape to [batchsize, head_dim, seq, rotary_dim]
        cos = cos.repeat(1, 2).unsqueeze(-2).unsqueeze(-2)
        sin = sin.repeat(1, 2).unsqueeze(-2).unsqueeze(-2)

        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]

        query_rot = torch_npu.npu_interleave_rope(
            query_rot.reshape(num_tokens, num_q_heads, 1, self.rotary_dim),
            cos,
            sin,
        )
        key_rot = torch_npu.npu_interleave_rope(
            key_rot.reshape(num_tokens, num_k_heads, 1, self.rotary_dim),
            cos,
            sin,
        )
        query_rot = query_rot.reshape(num_tokens, -1, self.rotary_dim)
        key_rot = key_rot.reshape(num_tokens, -1, self.rotary_dim)

        if self.rotary_dim < self.head_size:
            query = torch.cat((query_rot, query_pass), dim=-1)
            key = torch.cat((key_rot, key_pass), dim=-1)
        else:
            query = query_rot
            key = key_rot
        return query, key

    def forward_cpu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        positions = torch.add(positions, offsets) if offsets is not None else positions
        if _is_cpu_amx_available:
            return torch.ops.sgl_kernel.rotary_embedding_cpu(
                positions, query, key, self.head_size, self.cos_sin_cache, False
            )
        else:
            return self.forward_native(positions, query, key, offsets)


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


class DynamicNTKAlphaRotaryEmbedding(RotaryEmbedding):
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
        scaling_alpha: float,
        dtype: torch.dtype,
    ) -> None:
        self.scaling_alpha = scaling_alpha
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        max_len = self.max_position_embeddings
        base = self.base * self.scaling_alpha ** (
            self.rotary_dim / (self.rotary_dim - 2)
        )

        inv_freq = self._compute_inv_freq(base)
        t = torch.arange(max_len, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache


def apply_interleaved_rope(x: torch.Tensor, mrope_section: list[int]) -> torch.Tensor:
    """Apply interleaved MRoPE to 3D rotary embeddings.
    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THTHWHTHW...TT], preserving frequency continuity.
    """
    x_t = x[0].clone()
    x_t[..., 1 : mrope_section[1] * 3 : 3] = x[1, ..., 1 : mrope_section[1] * 3 : 3]
    x_t[..., 2 : mrope_section[2] * 3 : 3] = x[2, ..., 2 : mrope_section[2] * 3 : 3]
    return x_t


@triton.jit
def _triton_mrope_forward(
    q_ptr,
    k_ptr,
    cos,
    sin,
    num_tokens,
    n_qh: tl.constexpr,
    n_kh: tl.constexpr,
    hd: tl.constexpr,
    rd: tl.constexpr,
    pad_n_qh: tl.constexpr,
    pad_n_kh: tl.constexpr,
    pad_hd: tl.constexpr,
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    is_interleaved: tl.constexpr,
    is_neox_style: tl.constexpr,
):
    # Adapted from
    # https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/ops/qwen2vl_mrope.py
    # This version supports flatten input tensors from vllm
    # and supports cos and sin cache with shape (3, num_tokens, head_dim // 2)
    # instead of (3, bsz, seq_len, head_dim), also supports interleaved rotary
    pid = tl.program_id(0)
    # locate start address
    q_ptr = q_ptr + pid * (n_qh * hd)
    k_ptr = k_ptr + pid * (n_kh * hd)

    # ####################################################################
    # get the cos(mθ_{i...d/2}) and sin(mθ_{i...d/2}) for token position
    # m of this program instance
    # ####################################################################
    # Note: cos and sin now have shape (3, num_tokens, head_dim // 2)

    # Updated stride calculation for half head_dim
    half_rd = rd // 2
    t_cos = cos + pid * half_rd
    h_cos = t_cos + num_tokens * half_rd
    w_cos = h_cos + num_tokens * half_rd
    t_sin = sin + pid * half_rd
    h_sin = t_sin + num_tokens * half_rd
    w_sin = h_sin + num_tokens * half_rd

    # Updated offsets for half head_dim
    cos_offsets = tl.arange(0, pad_hd // 2)
    if is_interleaved:
        h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
        w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        t_end = mrope_section_t
        h_end = t_end + mrope_section_h
        t_mask = cos_offsets < mrope_section_t
        h_mask = (t_end <= cos_offsets) & (cos_offsets < h_end)
        w_mask = (h_end <= cos_offsets) & (cos_offsets < half_rd)

    t_cos_row = tl.load(t_cos + cos_offsets, mask=t_mask, other=0)
    h_cos_row = tl.load(h_cos + cos_offsets, mask=h_mask, other=0)
    w_cos_row = tl.load(w_cos + cos_offsets, mask=w_mask, other=0)
    t_sin_row = tl.load(t_sin + cos_offsets, mask=t_mask, other=0)
    h_sin_row = tl.load(h_sin + cos_offsets, mask=h_mask, other=0)
    w_sin_row = tl.load(w_sin + cos_offsets, mask=w_mask, other=0)

    cos_row = t_cos_row + h_cos_row + w_cos_row
    sin_row = t_sin_row + h_sin_row + w_sin_row

    # ####################################################################
    # Load the left and right half of q and k for the current
    # program instance (i.e. for the current token) separately
    # ####################################################################
    # left half of the head
    if is_neox_style:
        first_half_q_offsets = (
            tl.arange(0, pad_n_qh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
        )
        first_half_k_offsets = (
            tl.arange(0, pad_n_kh)[:, None] * hd + tl.arange(0, pad_hd // 2)[None, :]
        )
        first_q_mask = (tl.arange(0, pad_n_qh)[:, None] < n_qh) & (
            tl.arange(0, pad_hd // 2)[None, :] < rd // 2
        )
        first_k_mask = (tl.arange(0, pad_n_kh)[:, None] < n_kh) & (
            tl.arange(0, pad_hd // 2)[None, :] < rd // 2
        )

        q_tile_1 = tl.load(q_ptr + first_half_q_offsets, mask=first_q_mask, other=0).to(
            sin_row.dtype
        )
        k_tile_1 = tl.load(k_ptr + first_half_k_offsets, mask=first_k_mask, other=0).to(
            sin_row.dtype
        )

        # right half of the head
        second_half_q_offsets = first_half_q_offsets + (rd // 2)
        second_half_k_offsets = first_half_k_offsets + (rd // 2)
        second_q_mask = first_q_mask
        second_k_mask = first_k_mask

        q_tile_2 = tl.load(
            q_ptr + second_half_q_offsets, mask=second_q_mask, other=0
        ).to(sin_row.dtype)
        k_tile_2 = tl.load(
            k_ptr + second_half_k_offsets, mask=second_k_mask, other=0
        ).to(sin_row.dtype)

        # y = [x1, x2] * [cos, cos] + [-x2, x1] * [sin, sin]
        # Since cos and sin are now half-size,
        # we use the same cos_row and sin_row for both halves
        new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
        tl.store(q_ptr + first_half_q_offsets, new_q_tile_1, mask=first_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
        tl.store(q_ptr + second_half_q_offsets, new_q_tile_2, mask=second_q_mask)

        new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
        tl.store(k_ptr + first_half_k_offsets, new_k_tile_1, mask=first_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
        tl.store(k_ptr + second_half_k_offsets, new_k_tile_2, mask=second_k_mask)
    else:
        base_q = tl.arange(0, pad_n_qh)[:, None] * hd
        base_k = tl.arange(0, pad_n_kh)[:, None] * hd
        even_idx = 2 * tl.arange(0, pad_hd // 2)[None, :]
        odd_idx = even_idx + 1

        even_q_offsets = base_q + even_idx
        odd_q_offsets = base_q + odd_idx
        even_k_offsets = base_k + even_idx
        odd_k_offsets = base_k + odd_idx

        idx_mask = tl.arange(0, pad_hd // 2)[None, :] < (rd // 2)
        qn_mask = tl.arange(0, pad_n_qh)[:, None] < n_qh
        kn_mask = tl.arange(0, pad_n_kh)[:, None] < n_kh

        even_q_mask = qn_mask & idx_mask
        odd_q_mask = qn_mask & idx_mask
        even_k_mask = kn_mask & idx_mask
        odd_k_mask = kn_mask & idx_mask

        q_tile_1 = tl.load(q_ptr + even_q_offsets, mask=even_q_mask, other=0).to(
            sin_row.dtype
        )
        k_tile_1 = tl.load(k_ptr + even_k_offsets, mask=even_k_mask, other=0).to(
            sin_row.dtype
        )

        q_tile_2 = tl.load(q_ptr + odd_q_offsets, mask=odd_q_mask, other=0).to(
            sin_row.dtype
        )
        k_tile_2 = tl.load(k_ptr + odd_k_offsets, mask=odd_k_mask, other=0).to(
            sin_row.dtype
        )

        # y = [x_even, x_odd] * [cos, cos] + [-x_odd, x_even] * [sin, sin]
        # NeoX-style rotary embedding:
        # Each (even, odd) channel pair forms one rotation arm.
        # cos_row and sin_row each have length rd//2, shared across all (even, odd) pairs.
        new_q_tile_1 = q_tile_1 * cos_row - q_tile_2 * sin_row
        tl.store(q_ptr + even_q_offsets, new_q_tile_1, mask=even_q_mask)
        new_q_tile_2 = q_tile_2 * cos_row + q_tile_1 * sin_row
        tl.store(q_ptr + odd_q_offsets, new_q_tile_2, mask=odd_q_mask)

        new_k_tile_1 = k_tile_1 * cos_row - k_tile_2 * sin_row
        tl.store(k_ptr + even_k_offsets, new_k_tile_1, mask=even_k_mask)
        new_k_tile_2 = k_tile_2 * cos_row + k_tile_1 * sin_row
        tl.store(k_ptr + odd_k_offsets, new_k_tile_2, mask=odd_k_mask)


def triton_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: list[int],
    head_size: int,
    rotary_dim: int,
    mrope_interleaved: bool,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The mrope triton kernel.

    Args:
        q: [num_tokens, num_heads * head_size]
        k: [num_tokens, num_kv_heads * head_size]
        cos: [3, num_tokens, head_size //2 ]
            (T/H/W positions with multimodal inputs)
        sin: [3, num_tokens, head_size //2 ]
            (T/H/W positions with multimodal inputs)
        mrope_section: [t, h, w]
        head_size: int
    """
    n_row, n_q_head_head_dim = q.shape
    assert (
        n_q_head_head_dim % head_size == 0
    ), f"q shape {n_q_head_head_dim} must be divisible by head_size {head_size}"
    n_q_head = n_q_head_head_dim // head_size
    assert (
        k.shape[1] % head_size == 0
    ), f"k shape {k.shape[1]} must be divisible by head_size {head_size}"
    n_kv_head = k.shape[1] // head_size
    pad_hd = triton.next_power_of_2(head_size)
    pad_n_q_head = triton.next_power_of_2(n_q_head)
    pad_n_kv_head = triton.next_power_of_2(n_kv_head)

    # ensure tensors passed into the kernel are contiguous.
    # It will be no-op if they are already contiguous
    q = q.contiguous()
    k = k.contiguous()
    cos = cos.contiguous()
    sin = sin.contiguous()

    _triton_mrope_forward[(n_row,)](
        q,
        k,
        cos,
        sin,
        n_row,
        n_q_head,
        n_kv_head,
        head_size,
        rotary_dim,
        pad_n_q_head,
        pad_n_kv_head,
        pad_hd,
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        mrope_interleaved,
        is_neox_style,
    )
    return q, k


def triton_mrope_wrapper(
    query,
    key,
    cos,
    sin,
    mrope_section,
    head_size,
    rotary_dim,
    mrope_interleaved,
    is_neox_style,
):
    return triton_mrope(
        query,
        key,
        cos,
        sin,
        mrope_section,
        head_size,
        rotary_dim,
        mrope_interleaved,
        is_neox_style,
    )


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
        mrope_interleaved: bool = False,
    ) -> None:
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )

        self.mrope_section = mrope_section
        self.mrope_interleaved = mrope_interleaved
        if self.mrope_section:
            expected_sum = rotary_dim // 2
            actual_sum = sum(self.mrope_section)
            if actual_sum != expected_sum:
                print(
                    f"MRoPE section sum mismatch: expected {expected_sum}, got {actual_sum}. "
                    f"Adjusting mrope_section to match rotary_dim // 2 = {expected_sum}"
                )
                # Auto-correct by scaling the mrope_section proportionally
                if actual_sum > 0:
                    scale_factor = expected_sum / actual_sum
                    self.mrope_section = [
                        max(1, int(section * scale_factor))
                        for section in self.mrope_section
                    ]
                    # Ensure the sum exactly matches by adjusting the last element
                    current_sum = sum(self.mrope_section)
                    if current_sum != expected_sum:
                        self.mrope_section[-1] += expected_sum - current_sum
                else:
                    # If all sections are 0, create a default distribution
                    self.mrope_section = [
                        expected_sum // len(self.mrope_section)
                    ] * len(self.mrope_section)
                    # Handle remainder
                    remainder = expected_sum % len(self.mrope_section)
                    for i in range(remainder):
                        self.mrope_section[i] += 1

                print(
                    f"Corrected mrope_section: {self.mrope_section} (sum={sum(self.mrope_section)})"
                )

    def _match_cos_sin_cache_dtype(self, query: torch.Tensor) -> None:
        # __setattr__ in nn.Module (called by `self.cos_sin_cache = ...`)
        # is expensive, so avoid calling it if possible
        if (
            self.cos_sin_cache.device != query.device
            or self.cos_sin_cache.dtype != query.dtype
        ):
            self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)

    @torch.compile(dynamic=True, backend=get_compiler_backend())
    def _forward_native(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """PyTorch-native implementation equivalent to forward().

        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        assert (
            fused_set_kv_buffer_arg is None
        ), "save kv cache is not supported for MRotaryEmbedding."
        assert positions.ndim == 1 or positions.ndim == 2

        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if positions.ndim == 2:
            assert self.mrope_section
            if self.mrope_interleaved:
                cos = apply_interleaved_rope(cos, self.mrope_section)
                sin = apply_interleaved_rope(sin, self.mrope_section)
            else:
                cos = torch.cat(
                    [m[i] for i, m in enumerate(cos.split(self.mrope_section, dim=-1))],
                    dim=-1,
                )
                sin = torch.cat(
                    [m[i] for i, m in enumerate(sin.split(self.mrope_section, dim=-1))],
                    dim=-1,
                )

        seq_len_q = query.shape[0]
        query_shape = query.shape
        query = query.view(seq_len_q, -1, self.head_size)

        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        seq_len_k = key.shape[0]
        key_shape = key.shape
        key = key.view(seq_len_k, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with optional Triton kernel acceleration.
        Args:
            positions:
                [num_tokens,] (text only) or
                [3, num_tokens] (T/H/W positions with multimodal inputs)
            query: [num_tokens, num_heads * head_size]
            key: [num_tokens, num_kv_heads * head_size]
        """
        assert positions.ndim == 1 or positions.ndim == 2

        if positions.ndim == 2 and self.mrope_section and _is_cuda:
            return self._forward_triton(positions, query, key)
        elif _is_npu:
            return self._forward_npu(positions, query, key)
        else:
            return self._forward_native(positions, query, key)

    def _forward_triton(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert positions.ndim == 1 or positions.ndim == 2
        assert key is not None

        self._match_cos_sin_cache_dtype(query)
        num_tokens = positions.shape[-1]
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        cos = cos.contiguous()
        sin = sin.contiguous()
        query_shape = query.shape
        key_shape = key.shape
        if positions.ndim == 2:
            assert self.mrope_section

            q, k = triton_mrope_wrapper(
                query,
                key,
                cos,
                sin,
                self.mrope_section,
                self.head_size,
                self.rotary_dim,
                self.mrope_interleaved,
                self.is_neox_style,
            )

            return q.reshape(query_shape), k.reshape(key_shape)

        seq_len_q = query.shape[0]
        query = query.view(seq_len_q, -1, self.head_size)

        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

    def _forward_npu(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # TODO: remove this when npu_mrope supports QNumHeads * QHeadSize > 4096
        if query.shape[1] > 4096:
            return self._forward_native(positions, query, key)
        rotary_mode = "half"
        if self.is_neox_style:
            rotary_mode = "half"
        else:
            rotary_mode = "interleave"
        mrope_section = [0, 0, 0]
        query_out, key_out = torch_npu.npu_mrope(
            positions,
            query,
            key,
            self.cos_sin_cache,
            self.head_size,
            mrope_section=mrope_section,
            rotary_mode=rotary_mode,
        )
        return query_out, key_out

    # Copied from https://github.com/huggingface/transformers/blob/c8e0e603de9b3d49161a15fe6e8ea84badfb5d02/src/transformers/models/qwen2_vl/modeling_qwen2_vl.py#L1439
    @staticmethod
    def get_rope_index(
        spatial_merge_size: int,
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        model_type: str,
        tokens_per_second: Optional[int] = None,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if model_type == "qwen3_omni_moe":
            # For qwen3-omni
            return MRotaryEmbedding.get_rope_index_qwen3_omni(
                spatial_merge_size,
                image_token_id,
                video_token_id,
                vision_start_token_id,
                tokens_per_second,
                input_ids,
                image_grid_thw,
                video_grid_thw,
                second_per_grid_ts,
                **kwargs,
            )
        if (
            model_type.startswith("qwen3_vl") or model_type.startswith("qwen3_vl_moe")
        ) and video_grid_thw is not None:
            video_grid_thw = torch.repeat_interleave(
                video_grid_thw, video_grid_thw[:, 0], dim=0
            )
            video_grid_thw[:, 0] = 1

        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            for i, input_ids in enumerate(total_input_ids):
                image_nums, video_nums = 0, 0
                vision_start_indices = torch.argwhere(
                    input_ids == vision_start_token_id
                ).squeeze(1)
                vision_tokens = input_ids[vision_start_indices + 1]
                image_nums = (vision_tokens == image_token_id).sum()
                video_nums = (vision_tokens == video_token_id).sum()
                input_tokens = input_ids.tolist()
                llm_pos_ids_list: list = []
                st = 0
                remain_images, remain_videos = image_nums, video_nums
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
                        second_per_grid_t = 0
                        image_index += 1
                        remain_images -= 1
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
                        t.item(),
                        h.item() // spatial_merge_size,
                        w.item() // spatial_merge_size,
                    )
                    text_len = ed - st

                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                    if model_type == "qwen2_5_vl":
                        range_tensor = torch.arange(llm_grid_t).view(-1, 1)
                        expanded_range = range_tensor.expand(
                            -1, llm_grid_h * llm_grid_w
                        )

                        time_tensor = (
                            expanded_range * second_per_grid_t * tokens_per_second
                        )

                        time_tensor_long = time_tensor.long()
                        t_index = time_tensor_long.flatten()
                    elif model_type in (
                        "qwen2_vl",
                        "qwen3_vl",
                        "qwen3_vl_moe",
                    ):
                        t_index = (
                            torch.arange(llm_grid_t)
                            .view(-1, 1)
                            .expand(-1, llm_grid_h * llm_grid_w)
                            .flatten()
                        )
                    else:
                        raise RuntimeError(f"Unimplemented model type: {model_type}")
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
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, :] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
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

    @staticmethod
    def get_rope_index_qwen3_omni(
        spatial_merge_size: int,
        image_token_id: int,
        video_token_id: int,
        vision_start_token_id: int,
        tokens_per_second: Optional[int] = None,
        input_ids: Optional[torch.LongTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # For qwen3-omni
        audio_token_id = kwargs["audio_token_id"]
        audio_start_token_id = kwargs["audio_start_token_id"]
        position_id_per_seconds = kwargs["position_id_per_seconds"]
        use_audio_in_video = kwargs.get("use_audio_in_video", False)
        audio_seqlens = kwargs.get("audio_seqlens", None)
        second_per_grids = second_per_grid_ts

        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            position_ids = torch.zeros(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=torch.float,
                device=input_ids.device,
            )
            image_idx, video_idx, audio_idx = 0, 0, 0
            for i, current_input_ids in enumerate(total_input_ids):
                image_nums, video_nums, audio_nums = 0, 0, 0
                vision_start_indices = torch.argwhere(
                    current_input_ids == vision_start_token_id
                ).squeeze(1)
                if vision_start_indices.numel() > 0:
                    vision_tokens = current_input_ids[vision_start_indices + 1]
                    image_nums = (vision_tokens == image_token_id).sum()
                    video_nums = (
                        (vision_tokens == audio_start_token_id).sum()
                        if use_audio_in_video
                        else (vision_tokens == video_token_id).sum()
                    )
                audio_nums = torch.sum(current_input_ids == audio_start_token_id)
                input_tokens = current_input_ids.tolist()
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
                    ed_vision_start = (
                        input_tokens.index(vision_start_token_id, st)
                        if (
                            (
                                image_token_id in input_tokens
                                or video_token_id in input_tokens
                            )
                            and (remain_videos > 0 or remain_images > 0)
                        )
                        else len(input_tokens) + 1
                    )
                    ed_audio_start = (
                        input_tokens.index(audio_start_token_id, st)
                        if (audio_token_id in input_tokens and remain_audios > 0)
                        else len(input_tokens) + 1
                    )
                    min_ed = min(ed_vision_start, ed_audio_start)

                    text_len = min_ed - st
                    if text_len != 0:
                        llm_pos_ids_list.append(
                            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                        )
                        st_idx += text_len
                    # Audio in Video
                    if (
                        min_ed == ed_vision_start
                        and ed_vision_start + 1 == ed_audio_start
                    ):
                        bos_len, eos_len = 2, 2
                    else:
                        bos_len, eos_len = 1, 1
                    llm_pos_ids_list.append(
                        torch.arange(bos_len).view(1, -1).expand(3, -1) + st_idx
                    )
                    st_idx += bos_len
                    # Audio Only
                    if min_ed == ed_audio_start:
                        audio_len = MRotaryEmbedding._get_feat_extract_output_lengths(
                            audio_seqlens[audio_idx]
                        )
                        llm_pos_ids = (
                            torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        )
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + audio_len + eos_len)
                        audio_idx += 1
                        remain_audios -= 1

                    # Image Only
                    elif (
                        min_ed == ed_vision_start
                        and current_input_ids[ed_vision_start + 1] == image_token_id
                    ):
                        grid_t = image_grid_thw[image_idx][0]
                        grid_hs = image_grid_thw[:, 1]
                        grid_ws = image_grid_thw[:, 2]
                        t_index = (
                            torch.arange(grid_t) * 1 * position_id_per_seconds
                        ).float()
                        llm_pos_ids = MRotaryEmbedding._get_llm_pos_ids_for_vision(
                            st_idx,
                            image_idx,
                            spatial_merge_size,
                            t_index,
                            grid_hs,
                            grid_ws,
                            input_ids.device,
                        )
                        image_len = image_grid_thw[image_idx].prod() // (
                            spatial_merge_size**2
                        )
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + image_len + eos_len)
                        image_idx += 1
                        remain_images -= 1

                    # Video Only
                    elif (
                        min_ed == ed_vision_start
                        and current_input_ids[ed_vision_start + 1] == video_token_id
                    ):
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]
                        t_index = (
                            torch.arange(grid_t)
                            * second_per_grids[video_idx].cpu().float()
                            * position_id_per_seconds
                        ).float()
                        llm_pos_ids = MRotaryEmbedding._get_llm_pos_ids_for_vision(
                            st_idx,
                            video_idx,
                            spatial_merge_size,
                            t_index,
                            grid_hs,
                            grid_ws,
                            input_ids.device,
                        )
                        video_len = video_grid_thw[video_idx].prod() // (
                            spatial_merge_size**2
                        )
                        llm_pos_ids_list.append(llm_pos_ids)

                        st += int(text_len + bos_len + video_len + eos_len)
                        video_idx += 1
                        remain_videos -= 1

                    # Audio in Video
                    elif (
                        min_ed == ed_vision_start
                        and ed_vision_start + 1 == ed_audio_start
                    ):
                        audio_len = MRotaryEmbedding._get_feat_extract_output_lengths(
                            audio_seqlens[audio_idx]
                        )
                        audio_llm_pos_ids = (
                            torch.arange(audio_len).view(1, -1).expand(3, -1) + st_idx
                        )
                        grid_t = video_grid_thw[video_idx][0]
                        grid_hs = video_grid_thw[:, 1]
                        grid_ws = video_grid_thw[:, 2]

                        t_index = (
                            torch.arange(grid_t)
                            * second_per_grids[video_idx].cpu().float()
                            * position_id_per_seconds
                        ).float()
                        video_llm_pos_ids = (
                            MRotaryEmbedding._get_llm_pos_ids_for_vision(
                                st_idx,
                                video_idx,
                                spatial_merge_size,
                                t_index,
                                grid_hs,
                                grid_ws,
                                input_ids.device,
                            )
                        )
                        video_data_index, audio_data_index = 0, 0
                        while (
                            video_data_index < video_llm_pos_ids.shape[-1]
                            and audio_data_index < audio_llm_pos_ids.shape[-1]
                        ):
                            if (
                                video_llm_pos_ids[0][video_data_index]
                                <= audio_llm_pos_ids[0][audio_data_index]
                            ):
                                llm_pos_ids_list.append(
                                    video_llm_pos_ids[
                                        :, video_data_index : video_data_index + 1
                                    ]
                                )
                                video_data_index += 1
                            else:
                                llm_pos_ids_list.append(
                                    audio_llm_pos_ids[
                                        :, audio_data_index : audio_data_index + 1
                                    ]
                                )
                                audio_data_index += 1
                        if video_data_index < video_llm_pos_ids.shape[-1]:
                            llm_pos_ids_list.append(
                                video_llm_pos_ids[
                                    :, video_data_index : video_llm_pos_ids.shape[-1]
                                ]
                            )
                        if audio_data_index < audio_llm_pos_ids.shape[-1]:
                            llm_pos_ids_list.append(
                                audio_llm_pos_ids[
                                    :, audio_data_index : audio_llm_pos_ids.shape[-1]
                                ]
                            )
                        video_len = video_grid_thw[video_idx].prod() // (
                            spatial_merge_size**2
                        )

                        st += int(text_len + bos_len + audio_len + video_len + eos_len)

                        audio_idx += 1
                        video_idx += 1
                        remain_videos -= 1
                        remain_audios -= 1
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    llm_pos_ids_list.append(
                        torch.arange(eos_len).view(1, -1).expand(3, -1) + st_idx
                    )

                if st < len(input_tokens):
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )
                    text_len = len(input_tokens) - st
                    llm_pos_ids_list.append(
                        torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                    )

                llm_positions = torch.cat(
                    [item.float() for item in llm_pos_ids_list], dim=1
                ).reshape(3, -1)

                position_ids[..., i, :] = llm_positions.to(position_ids.device)
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(current_input_ids)
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)

            return position_ids, mrope_position_deltas
        else:
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

    # Adapted from https://github.com/vllm-project/vllm/blob/3779eb8c81449b924a23457fc77e45a0e6171178/vllm/model_executor/layers/rotary_embedding.py#L1120
    @staticmethod
    def get_rope_index_glm4v(
        input_ids: torch.Tensor,
        hf_config: Any,
        image_grid_thw: Union[list[list[int]], torch.Tensor],
        video_grid_thw: Union[list[list[int]], torch.Tensor],
        attention_mask: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Get mrope input positions and delta value for GLM4V."""
        image_token_id = hf_config.image_token_id
        video_start_token_id = hf_config.video_start_token_id
        video_end_token_id = hf_config.video_end_token_id
        spatial_merge_size = hf_config.vision_config.spatial_merge_size

        mrope_position_deltas = []
        if input_ids is not None and (
            image_grid_thw is not None or video_grid_thw is not None
        ):
            total_input_ids = input_ids
            if attention_mask is None:
                attention_mask = torch.ones_like(total_input_ids)
            position_ids = torch.ones(
                3,
                input_ids.shape[0],
                input_ids.shape[1],
                dtype=input_ids.dtype,
                device=input_ids.device,
            )
            image_index, video_index = 0, 0
            video_group_index = 0
            attention_mask = attention_mask.to(total_input_ids.device)
            for i, input_ids in enumerate(total_input_ids):
                input_ids = input_ids[attention_mask[i] == 1]
                input_tokens = input_ids.tolist()

                input_token_type = []
                video_check_flg = False
                for token in input_tokens:
                    if token == video_start_token_id:
                        video_check_flg = True
                    elif token == video_end_token_id:
                        video_check_flg = False

                    if token == image_token_id and not video_check_flg:
                        input_token_type.append("image")
                    elif token == image_token_id and video_check_flg:
                        input_token_type.append("video")
                    else:
                        input_token_type.append("text")

                input_type_group = []
                for key, group in itertools.groupby(
                    enumerate(input_token_type), lambda x: x[1]
                ):
                    group = list(group)
                    start_index = group[0][0]
                    end_index = group[-1][0] + 1
                    input_type_group.append((key, start_index, end_index))

                llm_pos_ids_list = []
                video_frame_num = 1
                for modality_type, start_idx, end_idx in input_type_group:
                    st_idx = (
                        llm_pos_ids_list[-1].max() + 1
                        if len(llm_pos_ids_list) > 0
                        else 0
                    )

                    if modality_type == "image":
                        t, h, w = (
                            image_grid_thw[image_index][0],
                            image_grid_thw[image_index][1],
                            image_grid_thw[image_index][2],
                        )
                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t.item(),
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )

                        t_index = (
                            torch.arange(llm_grid_t)
                            .view(-1, 1)
                            .expand(-1, llm_grid_h * llm_grid_w)
                            .flatten()
                        )
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
                            torch.stack([t_index, h_index, w_index]) + st_idx
                        )

                        image_index += 1
                        video_frame_num = 1

                    elif modality_type == "video":
                        t, h, w = (
                            video_frame_num,
                            video_grid_thw[video_index][1],
                            video_grid_thw[video_index][2],
                        )

                        llm_grid_t, llm_grid_h, llm_grid_w = (
                            t,
                            h.item() // spatial_merge_size,
                            w.item() // spatial_merge_size,
                        )

                        for t_idx in range(llm_grid_t):
                            t_index = (
                                torch.tensor(t_idx)
                                .view(-1, 1)
                                .expand(-1, llm_grid_h * llm_grid_w)
                                .flatten()
                            )

                            h_index = (
                                torch.arange(llm_grid_h)
                                .view(1, -1, 1)
                                .expand(1, -1, llm_grid_w)
                                .flatten()
                            )
                            w_index = (
                                torch.arange(llm_grid_w)
                                .view(1, 1, -1)
                                .expand(1, llm_grid_h, -1)
                                .flatten()
                            )
                            llm_pos_ids_list.append(
                                torch.stack([t_index, h_index, w_index]) + st_idx
                            )

                        video_group_index += 1

                        if video_group_index >= video_grid_thw[video_index][0]:
                            video_index += 1
                            video_group_index = 0

                        video_frame_num += 1

                    else:
                        text_len = end_idx - start_idx
                        llm_pos_ids_list.append(
                            torch.arange(text_len).view(1, -1).expand(3, -1) + st_idx
                        )

                        video_frame_num = 1

                llm_positions = torch.cat(llm_pos_ids_list, dim=1).reshape(3, -1)
                position_ids[..., i, attention_mask[i] == 1] = llm_positions.to(
                    position_ids.device
                )
                mrope_position_deltas.append(
                    llm_positions.max() + 1 - len(total_input_ids[i])
                )
            mrope_position_deltas = torch.tensor(
                mrope_position_deltas, device=input_ids.device
            ).unsqueeze(1)
            return position_ids, mrope_position_deltas
        else:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1) - 1
                position_ids.masked_fill_(attention_mask == 0, 1)
                position_ids = (
                    position_ids.unsqueeze(0)
                    .expand(3, -1, -1)
                    .to(attention_mask.device)
                )
                max_position_ids = position_ids.max(0, keepdim=False)[0].max(
                    -1, keepdim=True
                )[0]
                mrope_position_deltas = max_position_ids + 1 - attention_mask.shape[-1]
            else:
                position_ids = (
                    torch.arange(input_ids.shape[1], device=input_ids.device)
                    .view(1, 1, -1)
                    .expand(3, input_ids.shape[0], -1)
                )
                mrope_position_deltas = torch.zeros(
                    [input_ids.shape[0], 1],
                    device=input_ids.device,
                    dtype=input_ids.dtype,
                )

            return position_ids, mrope_position_deltas

    # For qwen3-omni
    @staticmethod
    def _get_feat_extract_output_lengths(input_lengths):
        """
        Computes the output length of the convolutional layers and the output length of the audio encoder
        """
        input_lengths_leave = input_lengths % 100
        feat_lengths = (input_lengths_leave - 1) // 2 + 1
        output_lengths = (
            ((feat_lengths - 1) // 2 + 1 - 1) // 2 + 1 + (input_lengths // 100) * 13
        )
        return output_lengths

    # For qwen3-omni
    @staticmethod
    def _get_llm_pos_ids_for_vision(
        st_idx, vision_idx, spatial_merge_size, t_index, grid_hs, grid_ws, device
    ):
        grid_h = grid_hs[vision_idx] // spatial_merge_size
        grid_w = grid_ws[vision_idx] // spatial_merge_size

        h_index = (
            torch.arange(grid_h, device=device)
            .view(1, -1, 1)
            .expand(len(t_index), -1, grid_w)
            .flatten()
        )
        w_index = (
            torch.arange(grid_w, device=device)
            .view(1, 1, -1)
            .expand(len(t_index), grid_h, -1)
            .flatten()
        )
        t_index = t_index.view(-1, 1).expand(-1, grid_h * grid_w).flatten()

        llm_pos_ids = torch.stack([t_index, h_index, w_index], dim=0) + st_idx
        return llm_pos_ids


class DualChunkRotaryEmbedding(CustomOp):
    """Rotary positional embedding for Dual Chunk Attention."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        chunk_size: int,
        local_size: int,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.chunk_size = chunk_size
        self.local_size = local_size
        self.dtype = dtype
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        (q_cache, qc_cache, k_cache, qc_no_clamp_cache, q_inter_cache) = (
            self._compute_cos_sin_cache()
        )

        self.register_buffer("cos_sin_q_cache", q_cache, persistent=False)
        self.register_buffer("cos_sin_qc_cache", qc_cache, persistent=False)
        self.register_buffer("cos_sin_k_cache", k_cache, persistent=False)
        self.register_buffer(
            "cos_sin_qc_no_clamp_cache", qc_no_clamp_cache, persistent=False
        )
        self.register_buffer("cos_sin_q_inter_cache", q_inter_cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): The HF implementation uses `torch.arange(...).float()`.
        # However, we use `torch.arange(..., dtype=torch.float)` instead to
        # avoid numerical issues with large base values (e.g., 10000000).
        # This may cause a slight numerical difference between the HF
        # implementation and ours.
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
        chunk_len = self.chunk_size - self.local_size
        q_t = torch.arange(chunk_len, dtype=torch.float)
        qc_t = (torch.arange(chunk_len, dtype=torch.float) + chunk_len).clamp(
            max=self.chunk_size
        )
        k_t = torch.arange(self.max_position_embeddings, dtype=torch.float) % chunk_len

        # count from chunk_len, no clamp(self.chunk_size) restriction
        qc_no_clamp_t = torch.arange(chunk_len, dtype=torch.float) + chunk_len
        # count from self.chunk_size for q_inter's rope
        q_inter_t = torch.arange(chunk_len, dtype=torch.float) + self.chunk_size

        q_freqs = torch.outer(q_t, inv_freq)
        qc_freqs = torch.outer(qc_t, inv_freq)
        k_freqs = torch.outer(k_t, inv_freq)
        qc_no_clamp_freqs = torch.outer(qc_no_clamp_t, inv_freq)
        q_inter_freqs = torch.outer(q_inter_t, inv_freq)

        q_cos = q_freqs.cos()
        q_sin = q_freqs.sin()
        qc_cos = qc_freqs.cos()
        qc_sin = qc_freqs.sin()
        k_cos = k_freqs.cos()
        k_sin = k_freqs.sin()

        qc_no_clamp_cos = qc_no_clamp_freqs.cos()
        qc_no_clamp_sin = qc_no_clamp_freqs.sin()
        q_inter_cos = q_inter_freqs.cos()
        q_inter_sin = q_inter_freqs.sin()

        q_cache = torch.cat((q_cos, q_sin), dim=-1).to(
            dtype=self.dtype, device=self.device
        )
        qc_cache = torch.cat((qc_cos, qc_sin), dim=-1).to(
            dtype=self.dtype, device=self.device
        )
        k_cache = torch.cat((k_cos, k_sin), dim=-1).to(
            dtype=self.dtype, device=self.device
        )
        qc_no_clamp_cache = torch.cat((qc_no_clamp_cos, qc_no_clamp_sin), dim=-1).to(
            dtype=self.dtype, device=self.device
        )
        q_inter_cache = torch.cat((q_inter_cos, q_inter_sin), dim=-1).to(
            dtype=self.dtype, device=self.device
        )
        return q_cache, qc_cache, k_cache, qc_no_clamp_cache, q_inter_cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        query = query.view(*query.shape[:-1], -1, self.head_size)
        key = key.view(*key.shape[:-1], -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        key_rot = key[..., : self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim :]
            key_pass = key[..., self.rotary_dim :]
        else:
            query_pass = None
            key_pass = None

        positions_with_offsets = (
            torch.add(positions, offsets) if offsets is not None else positions
        )
        key = self._apply_rotary_embedding(
            self.cos_sin_k_cache[positions_with_offsets], key_rot, key_pass
        )
        chunk_len = self.chunk_size - self.local_size
        query = self._apply_rotary_embedding(
            self.cos_sin_q_cache[positions_with_offsets % chunk_len],
            query_rot,
            query_pass,
        )
        query_succ = self._apply_rotary_embedding(
            self.cos_sin_qc_cache[positions_with_offsets % chunk_len],
            query_rot,
            query_pass,
        )
        query_inter = self._apply_rotary_embedding(
            self.cos_sin_qc_cache[chunk_len - 1].repeat(positions.shape[0], 1),
            query_rot,
            query_pass,
        )
        query_succ_critical = self._apply_rotary_embedding(
            self.cos_sin_qc_no_clamp_cache[positions_with_offsets % chunk_len],
            query_rot,
            query_pass,
        )
        query_inter_critical = self._apply_rotary_embedding(
            self.cos_sin_q_inter_cache[positions_with_offsets % chunk_len],
            query_rot,
            query_pass,
        )

        # merge query into one tensor to simplify the interfaces
        query = torch.cat(
            (
                query,
                query_succ,
                query_inter,
                query_succ_critical,
                query_inter_critical,
            ),
            dim=-1,
        )
        return query, key

    def _apply_rotary_embedding(self, cos_sin, hidden_rot, hidden_pass):
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
        hidden_rot = hidden_rot * cos + rotate_fn(hidden_rot) * sin

        if self.rotary_dim < self.head_size:
            hidden = torch.cat((hidden_rot, hidden_pass), dim=-1)
        else:
            hidden = hidden_rot
        return hidden.flatten(-2).squeeze(0)

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        s += f", chunk_size={self.chunk_size}, local_size={self.local_size}"
        return s


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
        # Transforms every value that is a list into a tuple for caching calls
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
                    mrope_interleaved=rope_scaling.get("mrope_interleaved", False),
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


def apply_rotary_pos_emb_native(
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


def apply_rotary_pos_emb_npu(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    unsqueeze_dim=1,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Ascend implementation equivalent to apply_rotary_pos_emb_native.

    Args:
        q: [num_tokens, num_heads, head_size]
        k: [num_tokens, num_kv_heads, head_size]
        cos: [num_tokens, head_size]
        sin: [num_tokens, head_size]
    """
    if (
        cos.dim() != 2
        or q.dim() != 3
        or q.shape[1] >= NPU_ROTARY_MUL_MAX_NUM_HEADS
        or q.shape[2] >= NPU_ROTARY_MUL_MAX_HEAD_SIZE
    ):
        # Note: num_heads and head_size of q must be less than 1000 and 896, respectively
        return apply_rotary_pos_emb_native(q, k, cos, sin, unsqueeze_dim)
    cos = cos.unsqueeze(unsqueeze_dim).unsqueeze(0)
    sin = sin.unsqueeze(unsqueeze_dim).unsqueeze(0)
    q = q.unsqueeze(0)
    k = k.unsqueeze(0)
    q_embed = torch_npu.npu_rotary_mul(q, cos, sin)
    k_embed = torch_npu.npu_rotary_mul(k, cos, sin)
    q_embed = q_embed.squeeze(0)
    k_embed = k_embed.squeeze(0)
    return q_embed, k_embed


if _is_npu:
    apply_rotary_pos_emb = apply_rotary_pos_emb_npu
else:
    apply_rotary_pos_emb = apply_rotary_pos_emb_native


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
