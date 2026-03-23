"""RoPE scaling variants: Phi3LongRoPE, FourierRoPE, DeepseekScaling, Llama3,
Llama4Vision, DynamicNTK, DynamicNTKAlpha, DualChunkRotaryEmbedding."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from sglang.srt.layers.rotary_embedding.base import RotaryEmbedding
from sglang.srt.layers.rotary_embedding.utils import (
    apply_rotary_pos_emb_native,
    rotate_gptj,
    rotate_neox,
)
from sglang.srt.layers.rotary_embedding.yarn import (
    yarn_find_correction_range,
    yarn_get_mscale,
    yarn_linear_ramp_mask,
)
from sglang.srt.layers.utils import MultiPlatformOp
from sglang.srt.utils import cpu_has_amx_support, get_device, is_cuda, is_hip, is_npu

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_npu = is_npu()
_is_cpu_amx_available = cpu_has_amx_support()

if _is_npu:
    import torch_npu


class Phi3LongRoPEScaledRotaryEmbedding(nn.Module):
    """Phi3 family of models scaled rotary embedding."""

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
        query = query.unflatten(1, (-1, self.head_size))
        key = key.unflatten(1, (-1, self.head_size))

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
        query_rot = query_rot * cos + rotate_neox(query_rot) * sin
        query = torch.cat((query_rot, query_pass), dim=-1)

        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = key_rot * cos + rotate_neox(key_rot) * sin
        key = torch.cat((key_rot, key_pass), dim=-1)

        return query.flatten(-2), key.flatten(-2)


class FourierRotaryEmbedding(nn.Module):
    """Fourier RotaryEmbedding extended."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        num_kv_heads: int,
        *,
        fope_init_factor: float = 0.1,
        fope_sep_head: bool = True,
        num_inv_freq: int = None,
        device: Optional[str] = "cuda",
    ) -> None:
        self.fope_init_factor = fope_init_factor
        self.fope_sep_head = fope_sep_head
        self.num_inv_freq = num_inv_freq
        self.num_kv_heads = num_kv_heads
        self.device = device

        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        self.inv_freq: torch.Tensor
        self.register_buffer(
            "inv_freq", self._compute_inv_freq(self.base), persistent=False
        )
        self.input_dim = self.inv_freq.shape[-1]
        self.output_dim = self.inv_freq.shape[-1]
        self.cos_coef = nn.Parameter(
            torch.empty(
                self.num_kv_heads, self.input_dim, self.output_dim, dtype=torch.float32
            ),
            requires_grad=False,
        )
        self.sin_coef = nn.Parameter(
            torch.empty(
                self.num_kv_heads, self.input_dim, self.output_dim, dtype=torch.float32
            ),
            requires_grad=False,
        )
        self.cos_sin_cache: torch.Tensor
        self.register_buffer(
            "cos_sin_cache", self._compute_cos_sin_cache(), persistent=False
        )
        self.update_buffer = False

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.int64).to(
                    device=self.device, dtype=torch.float
                )
                / self.rotary_dim
            )
        )
        assert (
            inv_freq[:-1] > inv_freq[1:]
        ).all(), "Expected inv_freq to be in decreasing order"
        inv_freq_idx_selected = torch.ones_like(inv_freq, dtype=torch.bool)
        if self.num_inv_freq is not None:
            inv_freq_idx_selected[self.num_inv_freq :] = False
        else:
            inv_freq_idx_selected = inv_freq > (
                2.0 * torch.pi / self.max_position_embeddings
            )
        inv_freq = inv_freq[inv_freq_idx_selected]
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        t = torch.arange(
            self.max_position_embeddings, dtype=torch.float, device=self.device
        )
        freqs = torch.einsum("i,j -> ij", t, self.inv_freq)
        if self.fope_sep_head:
            pos_cos = freqs.cos().unsqueeze(0).expand(self.num_kv_heads, -1, -1)
            pos_sin = freqs.sin().unsqueeze(0).expand(self.num_kv_heads, -1, -1)
        else:
            pos_cos = freqs.cos()
            pos_sin = freqs.sin()
        if self.fope_sep_head:
            sin = torch.einsum("htD, hDd -> thd", pos_sin, self.sin_coef.float())
            cos = torch.einsum("htD, hDd -> thd", pos_cos, self.cos_coef.float())
        else:
            sin = torch.einsum("tD, Dd -> td", pos_sin, self.sin_coef.float())
            cos = torch.einsum("tD, Dd -> td", pos_cos, self.cos_coef.float())
        sin = F.pad(
            input=sin,
            pad=(0, self.head_size // 2 - sin.size(-1)),
            mode="constant",
            value=1,
        )
        cos = F.pad(
            input=cos,
            pad=(0, self.head_size // 2 - cos.size(-1)),
            mode="constant",
            value=1,
        )
        sin = torch.cat((sin, sin), dim=-1)
        cos = torch.cat((cos, cos), dim=-1)
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if not self.update_buffer:
            self.cos_sin_cache = self._compute_cos_sin_cache()
            self.update_buffer = True
        query = query.unflatten(-1, (-1, self.head_size))
        key = key.unflatten(-1, (-1, self.head_size))
        positions_with_offsets = (
            torch.add(positions, offsets) if offsets is not None else positions
        )
        cos_sin = torch.index_select(self.cos_sin_cache, 0, positions_with_offsets).to(
            dtype=query.dtype
        )
        cos, sin = cos_sin.chunk(2, dim=-1)
        assert (
            query.dim() == key.dim() == 3
        ), "Expected query key (seq_len, heads, head_dim)"
        assert cos.dim() <= 3 and sin.dim() <= 3
        need_reshape = False
        if cos.dim() == 3:
            need_reshape = True
            query_shape = query.shape
            key_shape = key.shape
            cos = cos.flatten(0, 1)
            sin = sin.flatten(0, 1)
            seq_len = cos.size(0)
            query = query.reshape(seq_len, -1, query.size(-1))
            key = key.reshape(seq_len, -1, key.size(-1))
        query, key = apply_rotary_pos_emb_native(query, key, cos, sin)
        if need_reshape:
            query = query.reshape(query_shape)
            key = key.reshape(key_shape)
        return query.flatten(-2), key.flatten(-2)

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        s += f", fope_init_factor={self.fope_init_factor}, fope_sep_head={self.fope_sep_head}"
        s += f", num_inv_freq={self.num_inv_freq}, num_kv_heads={self.num_kv_heads}"
        return s


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
        device: Optional[str] = None,
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
        self.cos_cached_total = None
        self.sin_cached_total = None
        self.cos_cached = None
        self.sin_cached = None
        self.device = device if device is not None else get_device()
        super().__init__(
            head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
        )
        if _is_hip:
            self._forward_method = self.forward_native

    def _compute_inv_freq(self, scaling_factor: float) -> torch.Tensor:
        pos_freqs = self.base ** (
            torch.arange(0, self.rotary_dim, 2, dtype=torch.float, device=self.device)
            / self.rotary_dim
        )
        inv_freq_extrapolation = 1.0 / pos_freqs
        inv_freq_interpolation = 1.0 / (scaling_factor * pos_freqs)
        low, high = yarn_find_correction_range(
            self.beta_fast,
            self.beta_slow,
            self.rotary_dim,
            self.base,
            self.max_position_embeddings,
        )
        inv_freq_mask = (
            1
            - yarn_linear_ramp_mask(
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
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached_total = torch.cos(emb) * self.mscale
        self.sin_cached_total = torch.sin(emb) * self.mscale
        return cache

    def get_cos_cached_total(self):
        return self.cos_cached_total

    def get_sin_cached_total(self):
        return self.sin_cached_total

    def get_cos_sin_cache(
        self, positions, dtype, offsets: Optional[torch.Tensor] = None
    ):
        self.cos_cached = (
            self.cos_cached_total[
                torch.add(positions, offsets) if offsets is not None else positions
            ]
            .unsqueeze(-2)
            .unsqueeze(-2)
            .to(dtype)
        )
        self.sin_cached = (
            self.sin_cached_total[
                torch.add(positions, offsets) if offsets is not None else positions
            ]
            .unsqueeze(-2)
            .unsqueeze(-2)
            .to(dtype)
        )
        cos = self.cos_cached.to(positions.device)
        sin = self.sin_cached.to(positions.device)
        return cos, sin

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
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
        rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
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
        cos, sin = self.get_cos_sin_cache(positions, query.dtype, offsets)
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


class DualChunkRotaryEmbedding(MultiPlatformOp):
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
        q_cache, qc_cache, k_cache, qc_no_clamp_cache, q_inter_cache = (
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
        inv_freq = 1.0 / (
            base
            ** (
                torch.arange(0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim
            )
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        chunk_len = self.chunk_size - self.local_size
        q_t = torch.arange(chunk_len, dtype=torch.float)
        qc_t = (torch.arange(chunk_len, dtype=torch.float) + chunk_len).clamp(
            max=self.chunk_size
        )
        k_t = torch.arange(self.max_position_embeddings, dtype=torch.float) % chunk_len
        qc_no_clamp_t = torch.arange(chunk_len, dtype=torch.float) + chunk_len
        q_inter_t = torch.arange(chunk_len, dtype=torch.float) + self.chunk_size

        q_freqs = torch.outer(q_t, inv_freq)
        qc_freqs = torch.outer(qc_t, inv_freq)
        k_freqs = torch.outer(k_t, inv_freq)
        qc_no_clamp_freqs = torch.outer(qc_no_clamp_t, inv_freq)
        q_inter_freqs = torch.outer(q_inter_t, inv_freq)

        q_cache = torch.cat((q_freqs.cos(), q_freqs.sin()), dim=-1).to(
            dtype=self.dtype, device=self.device
        )
        qc_cache = torch.cat((qc_freqs.cos(), qc_freqs.sin()), dim=-1).to(
            dtype=self.dtype, device=self.device
        )
        k_cache = torch.cat((k_freqs.cos(), k_freqs.sin()), dim=-1).to(
            dtype=self.dtype, device=self.device
        )
        qc_no_clamp_cache = torch.cat(
            (qc_no_clamp_freqs.cos(), qc_no_clamp_freqs.sin()), dim=-1
        ).to(dtype=self.dtype, device=self.device)
        q_inter_cache = torch.cat(
            (q_inter_freqs.cos(), q_inter_freqs.sin()), dim=-1
        ).to(dtype=self.dtype, device=self.device)
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
        query = torch.cat(
            (query, query_succ, query_inter, query_succ_critical, query_inter_critical),
            dim=-1,
        )
        return query, key

    def _apply_rotary_embedding(self, cos_sin, hidden_rot, hidden_pass):
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.is_neox_style:
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
        rotate_fn = rotate_neox if self.is_neox_style else rotate_gptj
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
