# Adapted from https://raw.githubusercontent.com/vllm-project/vllm/refs/tags/v0.6.6.post1/vllm/model_executor/layers/rotary_embedding.py

"""Rotary Positional Embeddings."""
import math
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn

from sglang.srt.custom_op import CustomOp
from sglang.srt.utils import is_cuda_available

_is_cuda_available = is_cuda_available()
if _is_cuda_available:
    from sgl_kernel import apply_rope_with_cos_sin_cache_inplace
else:
    from vllm import _custom_ops as ops

import triton
import triton.language as tl

from sglang.srt.utils import libentry


# From FlagGems
@libentry()
@triton.jit
def apply_rotary_pos_emb_kernel(
    oq_ptr,
    ok_ptr,
    q_ptr,  # (n_tokens, q_heads, head_dim)
    k_ptr,  # (n_tokens, k_heads, head_dim)
    cos_ptr,  # (max_seq_len, dim // 2)
    sin_ptr,  # (max_seq_len, dim // 2)
    pos_ptr,  # (n_tokens, )
    q_stride_s,
    q_stride_h,
    q_stride_d,
    k_stride_s,
    k_stride_h,
    k_stride_d,
    oq_stride_s,
    oq_stride_h,
    oq_stride_d,
    ok_stride_s,
    ok_stride_h,
    ok_stride_d,
    p_stride_s,
    cos_stride_s,
    sin_stride_s,
    seq_len,
    NUM_Q_HEADS: tl.constexpr,
    NUM_K_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    PADDED_HEAD_DIM: tl.constexpr,
    ROTARY_INTERLEAVED: tl.constexpr,
    MAX_POSITION_EMBEDDINGS: tl.constexpr,
):
    s_id = tl.program_id(0)

    if pos_ptr is None:
        pos_id = s_id % seq_len
    else:
        pos_ptr += s_id * p_stride_s
        pos_id = tl.load(pos_ptr)
    cos_ptr += pos_id * cos_stride_s
    sin_ptr += pos_id * sin_stride_s

    # note: set TRITON_DEBUG=1 to enable this check
    tl.device_assert(pos_id < MAX_POSITION_EMBEDDINGS, "position id out of bound")

    ordered_block = tl.arange(0, PADDED_HEAD_DIM)
    mask = ordered_block < HEAD_DIM
    if ROTARY_INTERLEAVED:
        odd_mask = ordered_block % 2 == 0
        rotated_block = tl.where(odd_mask, ordered_block + 1, ordered_block - 1)
        sin_cos_block = ordered_block // 2
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(odd_mask, -sin, sin)
    else:
        rotated_block = (ordered_block + HEAD_DIM // 2) % HEAD_DIM
        sin_cos_block = ordered_block % (HEAD_DIM // 2)
        cos = tl.load(cos_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.load(sin_ptr + sin_cos_block, mask=mask, other=0.0).to(tl.float32)
        sin = tl.where(rotated_block < HEAD_DIM // 2, sin, -sin)

    oq_ptr += s_id * oq_stride_s
    q_ptr += s_id * q_stride_s

    for off_h in range(0, NUM_Q_HEADS):
        ordered_cols = off_h * q_stride_h + (ordered_block * q_stride_d)
        rotated_cols = off_h * q_stride_h + (rotated_block * q_stride_d)
        output_offs = off_h * oq_stride_h + (ordered_block * oq_stride_d)

        q = tl.load(q_ptr + ordered_cols, mask=mask, other=0.0)
        rotated_q = tl.load(q_ptr + rotated_cols, mask=mask, other=0.0)
        y = q * cos + rotated_q * sin
        tl.store(oq_ptr + output_offs, y, mask=mask)

    ok_ptr += s_id * ok_stride_s
    k_ptr += s_id * k_stride_s

    for off_h in range(0, NUM_K_HEADS):
        ordered_cols = off_h * k_stride_h + (ordered_block * k_stride_d)
        rotated_cols = off_h * k_stride_h + (rotated_block * k_stride_d)
        output_offs = off_h * ok_stride_h + (ordered_block * ok_stride_d)

        k = tl.load(k_ptr + ordered_cols, mask=mask, other=0.0)
        rotated_k = tl.load(k_ptr + rotated_cols, mask=mask, other=0.0)
        y = k * cos + rotated_k * sin
        tl.store(ok_ptr + output_offs, y, mask=mask)


# Integrated from FlagGems
def _apply_rotary_pos_emb(
    q,
    k,
    cos,
    sin,
    position_ids: Optional[torch.IntTensor] = None,
    rotary_interleaved: bool = False,
):
    """
    Apply rotary position embedding to q and k
    Args:
        q: (*, q_heads, head_dim)
        k: (*, k_heads, head_dim)
        cos: (max_seq_len, head_dim // 2)
        sin: (max_seq_len, head_dim // 2)
        position_ids: (*, ), optional, position ids for each token
        rotary_interleaved: whether the head_dim is rotated in an interleaved way
    Returns:
        q_embed: (*, q_heads, head_dim)
        k_embed: (*, k_heads, head_dim)
    """
    assert (
        k.shape[-1] == q.shape[-1]  # 128
    ), f"q and k must have the same last dimension, got {q.shape} and {k.shape}"
    assert (
        cos.shape[-1] == sin.shape[-1]  # 128 / 2 = 64
    ), f"cos and sin must have the same last dimension, got {cos.shape} and {sin.shape}"
    assert (
        cos.shape[-1] * 2 == q.shape[-1]  # 128
    ), f"cos/sin dim must be half of q/k dim, got {cos.shape} and {q.shape}"
    assert cos.stride(-1) == 1, "cos must be contiguous at the last dimension"
    assert sin.stride(-1) == 1, "sin must be contiguous at the last dimension"

    q_shape = q.shape
    k_shape = k.shape

    assert (
        q.shape[:-2] == k.shape[:-2]
    ), f"q and k must have the same length, got {q.shape[:-2]} and {k.shape[:-2]}"
    if position_ids is None:
        assert (
            len(q.shape) == 4
        ), f"q must have 4 dimensions if position_ids is not provided, got {q.shape}"
        seq_len = q.shape[-3]
    else:
        assert (
            position_ids.shape == q.shape[:-2]  # num_head 32
        ), f"position_ids must have the same length as q, got {position_ids.shape} and {q.shape[:-2]}"

        position_ids = position_ids.view(-1)
        seq_len = None

    q = q.view(-1, q.shape[-2], q.shape[-1])
    k = k.view(-1, k.shape[-2], k.shape[-1])

    q_embed = torch.empty_like(q)
    k_embed = torch.empty_like(k)

    n_tokens, q_heads, head_dim = q.shape

    # The block size must be the next power of two, sometimes we need to pad it.
    padded_head_dim = max(triton.next_power_of_2(head_dim), 16)

    grid = (n_tokens,)

    apply_rotary_pos_emb_kernel[grid](
        q_embed,
        k_embed,
        q,
        k,
        cos,
        sin,
        position_ids,
        q.stride(0),
        q.stride(1),
        q.stride(2),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        q_embed.stride(0),
        q_embed.stride(1),
        q_embed.stride(2),
        k_embed.stride(0),
        k_embed.stride(1),
        k_embed.stride(2),
        position_ids.stride(0) if position_ids is not None else 0,
        cos.stride(0),
        sin.stride(0),
        seq_len,
        q.shape[-2],
        k.shape[-2],
        head_dim,
        padded_head_dim,
        rotary_interleaved,
        MAX_POSITION_EMBEDDINGS=cos.shape[0],
    )

    q_embed = q_embed.view(q_shape)
    k_embed = k_embed.view(k_shape)
    return q_embed, k_embed


def _rotary_embedding(
    positions: torch.Tensor,
    query: torch.Tensor,
    key: torch.Tensor,
    head_size: int,
    cos_sin_cache: torch.Tensor,
    is_neox: bool,
) -> None:
    cos_cache, sin_cache = torch.chunk(cos_sin_cache, chunks=2, dim=-1)
    reshaped_query = query.reshape(-1, query.shape[1] // head_size, head_size)
    reshaped_key = key.reshape(-1, key.shape[1] // head_size, head_size)
    reshaped_query, reshaped_key = _apply_rotary_pos_emb(
        reshaped_query, reshaped_key, cos_cache, sin_cache, positions, False
    )
    query.copy_(reshaped_query.reshape(-1, query.shape[1]))
    key.copy_(reshaped_key.reshape(-1, key.shape[1]))


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
            ops.rotary_embedding(
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

    def forward_triton(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if offsets is not None:
            raise Exception(
                "Terminating the code as batched_rotary_embedding is not supported yet."
            )
        else:
            self.cos_sin_cache = self.cos_sin_cache.to(query.device, dtype=query.dtype)
            _rotary_embedding(
                positions,
                query,
                key,
                self.head_size,
                self.cos_sin_cache,
                self.is_neox_style,
            )
        return query, key


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

    def forward(
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
        tokens_per_second: Optional[int] = None,
    ) -> Tuple[List[List[int]], int]:
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

        return llm_positions.tolist(), mrope_position_delta

    @staticmethod
    def get_next_input_positions(
        mrope_position_delta: int,
        context_len: int,
        seq_len: int,
    ) -> List[List[int]]:
        return [
            list(
                range(
                    context_len + mrope_position_delta, seq_len + mrope_position_delta
                )
            )
            for _ in range(3)
        ]


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
