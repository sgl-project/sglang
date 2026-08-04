"""RotaryEmbedding base class and LinearScalingRotaryEmbedding variant."""

from typing import Optional, Tuple

import torch

from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp

from .utils import (
    _apply_rotary_emb,
    _apply_rotary_emb_complex,
    apply_flashinfer_rope_qk_inplace,
)


@CustomOp.register("rotary_embedding")
class RotaryEmbedding(CustomOp):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: Optional[int] = 4096,
        base: Optional[int | float] = 10000,
        is_neox_style: bool = False,
        is_interleaved: bool = True,
        dtype: Optional[torch.dtype] = torch.float16,
        use_precomputed_cache: bool = False,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.interleaved = is_interleaved
        self.dtype = dtype

        if use_precomputed_cache:
            cache = self._compute_cos_sin_cache()
            cache = cache.to(dtype)
            self.cos_sin_cache: torch.Tensor
            self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: int | float) -> torch.Tensor:
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

    def forward_npu(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        complex_freqs: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        batch_size, seq_len, num_heads, head_dim = query.shape
        total_tokens = batch_size * seq_len
        if self.interleaved:
            can_use_complex = (
                complex_freqs is not None
                and query.dim() == 4
                and key.dim() == 4
                and complex_freqs.dim() == 3
                and self.is_neox_style == False
            )
            if can_use_complex:
                return (
                    _apply_rotary_emb_complex(query, complex_freqs),
                    _apply_rotary_emb_complex(key, complex_freqs),
                )
            if self.is_neox_style:
                raise ValueError("Requested interleaved=True, but neox_style=True.")

            if complex_freqs is None:
                if cos is None or sin is None:
                    raise ValueError("Freqs are none for interleaved form.")
                q_flat = query.reshape(total_tokens, num_heads, head_dim)
                k_flat = key.reshape(total_tokens, num_heads, head_dim)
                q_rot = _apply_rotary_emb(
                    q_flat, cos, sin, is_neox_style=self.is_neox_style, interleaved=True
                )
                k_rot = _apply_rotary_emb(
                    k_flat, cos, sin, is_neox_style=self.is_neox_style, interleaved=True
                )
                return q_rot.view(batch_size, seq_len, num_heads, head_dim), k_rot.view(
                    batch_size, seq_len, num_heads, head_dim
                )
        if cos is not None and sin is not None:
            q_flat = query.reshape(total_tokens, num_heads, head_dim)
            k_flat = key.reshape(total_tokens, num_heads, head_dim)
            q_rot = _apply_rotary_emb(
                q_flat, cos, sin, is_neox_style=self.is_neox_style
            )
            k_rot = _apply_rotary_emb(
                k_flat, cos, sin, is_neox_style=self.is_neox_style
            )
            return q_rot.view(batch_size, seq_len, num_heads, head_dim), k_rot.view(
                batch_size, seq_len, num_heads, head_dim
            )
        if cos_sin_cache is not None:
            return self.forward_native(
                query=query,
                key=key,
                positions=positions,
                position_offset=position_offset,
                cos=cos,
                sin=sin,
                complex_freqs=complex_freqs,
                cos_sin_cache=cos_sin_cache,
                **kwargs,
            )

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        complex_freqs: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if query.dim() != 4 or key.dim() != 4:
            raise ValueError(
                f"query and key must be [batch_size, seq_len, num_heads, head_dim],"
                f"got query: {tuple(query.shape)}, key: {tuple(key.shape)}"
            )

        if cos_sin_cache is None:
            return self.forward_native(
                query=query,
                key=key,
                positions=positions,
                position_offset=position_offset,
                cos=cos,
                sin=sin,
                complex_freqs=complex_freqs,
                cos_sin_cache=cos_sin_cache,
                **kwargs,
            )

        batch_size, seq_len, _, head_dim = query.shape

        if positions is None:
            pos_1d = torch.arange(
                position_offset,
                position_offset + seq_len,
                device=query.device,
                dtype=torch.int64,
            )
            positions = pos_1d if batch_size == 1 else pos_1d.repeat(batch_size)
        else:
            positions = positions.to(device=query.device, dtype=torch.long)

        return apply_flashinfer_rope_qk_inplace(
            q=query,
            k=key,
            cos_sin_cache=cos_sin_cache,
            head_size=head_dim,
            is_neox=self.is_neox_style,
            positions=positions,
        )

    def forward_xpu(self, *args, **kwargs):
        return self.forward_native(*args, **kwargs)

    def forward_native(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        complex_freqs: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""

        if query.dim() != 4 or key.dim() != 4:
            raise ValueError(
                f"query and key must be [batch_size, seq_len, num_heads, head_dim],"
                f"got query: {tuple(query.shape)}, key: {tuple(key.shape)}"
            )
        batch_size, seq_len, num_heads, head_dim = query.shape
        total_tokens = batch_size * seq_len

        if positions is None:
            pos_1d = torch.arange(
                position_offset,
                position_offset + seq_len,
                device=query.device,
                dtype=torch.int64,
            )
            positions = pos_1d if batch_size == 1 else pos_1d.repeat(batch_size)
        else:
            if positions.dim() != 1 or positions.numel() != total_tokens:
                raise ValueError(
                    f"positions must be 1D of length {total_tokens}, got shape={tuple(positions.shape)}"
                )
            positions = positions.to(device=query.device, dtype=torch.long)

        if self.interleaved:
            if complex_freqs is not None:
                return (
                    _apply_rotary_emb_complex(query, complex_freqs),
                    _apply_rotary_emb_complex(key, complex_freqs),
                )
            if cos is not None and sin is not None:
                freqs = torch.complex(cos.float(), sin.float())
                if freqs.dim() == 2:
                    freqs = freqs.unsqueeze(-2)
                return (
                    _apply_rotary_emb_complex(query, freqs),
                    _apply_rotary_emb_complex(key, freqs),
                )

            if cos_sin_cache is not None:
                return apply_flashinfer_rope_qk_inplace(
                    q=query,
                    k=key,
                    cos_sin_cache=cos_sin_cache,
                    head_size=head_dim,
                    is_neox=self.is_neox_style,
                    positions=positions,
                )

            raise ValueError(
                "No valid inputs (complex_freqs, cos/sin, or cos_sin_cache) for interleaved RoPE."
            )

        else:
            if cos is not None and sin is not None:
                q_flat = query.reshape(total_tokens, num_heads, head_dim)
                k_flat = key.reshape(total_tokens, num_heads, head_dim)
                q_rot = _apply_rotary_emb(
                    q_flat, cos, sin, is_neox_style=self.is_neox_style
                )
                k_rot = _apply_rotary_emb(
                    k_flat, cos, sin, is_neox_style=self.is_neox_style
                )
                return q_rot.view(batch_size, seq_len, num_heads, head_dim), k_rot.view(
                    batch_size, seq_len, num_heads, head_dim
                )

            if cos_sin_cache is not None:
                return apply_flashinfer_rope_qk_inplace(
                    q=query,
                    k=key,
                    cos_sin_cache=cos_sin_cache,
                    head_size=head_dim,
                    is_neox=self.is_neox_style,
                    positions=positions,
                )

            raise ValueError(
                "No valid inputs (cos/sin or cos_sin_cache) for NeoX RoPE."
            )

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        return s


class LinearScalingRotaryEmbedding(RotaryEmbedding):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int | float,
        is_neox_style: bool,
        dtype: torch.dtype,
        scaling_factor: float,
    ) -> None:
        self.scaling_factor = float(scaling_factor)
        super().__init__(
            head_size=head_size,
            rotary_dim=rotary_dim,
            max_position_embeddings=max_position_embeddings,
            base=base,
            is_neox_style=is_neox_style,
            dtype=dtype,
        )

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)
        t = t / self.scaling_factor
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache
