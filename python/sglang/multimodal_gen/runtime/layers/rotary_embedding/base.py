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
        dtype: Optional[torch.dtype] = torch.float16,
        use_precomputed_cache: Optional[bool] = True,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
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
        positions: Optional[torch.Tensor] = None,
        query: Optional[torch.Tensor] = None,
        key: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        complex_freqs: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if hasattr(self, "cos_sin_cache") or query.dim() == 3 or key.dim() == 3:
            return self.forward_native(
                query=query,
                key=key,
                positions=positions,
                position_offset=position_offset,
                cos=cos,
                sin=sin,
                complex_freqs=complex_freqs,
                cos_sin_cache=cos_sin_cache,
                offsets=offsets,
                **kwargs,
            )

        if query.dim() != 4 or key.dim() != 4:
            raise ValueError(
                f"query and key must be [batch_size, seq_len, num_heads, head_dim],"
                f"got query: {tuple(query.shape)}, key: {tuple(key.shape)}"
            )

        batch_size, seq_len, num_heads, head_dim = query.shape
        total_tokens = batch_size * seq_len

        can_use_complex = (
            complex_freqs is not None
            and complex_freqs.dim() == 3
            and self.is_neox_style == False
        )
        if can_use_complex:
            return (
                _apply_rotary_emb_complex(query, complex_freqs),
                _apply_rotary_emb_complex(key, complex_freqs),
            )

        if cos is not None and sin is not None:
            q_flat = query.reshape(total_tokens, num_heads, self.head_size)
            q_rot = q_flat[..., : self.rotary_dim]
            q_pass = q_flat[..., self.rotary_dim :]

            k_flat = key.reshape(total_tokens, num_heads, self.head_size)
            k_rot = k_flat[..., : self.rotary_dim]
            k_pass = k_flat[..., self.rotary_dim :]

            q_rotated = _apply_rotary_emb(
                q_rot,
                cos,
                sin,
                is_neox_style=self.is_neox_style,
                interleaved=not self.is_neox_style,
            )
            q = torch.cat((q_rotated, q_pass), dim=-1)
            k_rotated = _apply_rotary_emb(
                k_rot,
                cos,
                sin,
                is_neox_style=self.is_neox_style,
                interleaved=not self.is_neox_style,
            )
            k = torch.cat((k_rotated, k_pass), dim=-1)
            return q.view(batch_size, seq_len, num_heads, head_dim), k.view(
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
                offsets=offsets,
                **kwargs,
            )

    def forward_cuda(
        self,
        positions: Optional[torch.Tensor] = None,
        query: Optional[torch.Tensor] = None,
        key: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        complex_freqs: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if cos_sin_cache is None and cos is not None and sin is not None:
            cos_sin_cache = torch.cat(
                [
                    cos.to(dtype=torch.float32).contiguous(),
                    sin.to(dtype=torch.float32).contiguous(),
                ],
                dim=-1,
            )
        can_use_cuda = cos_sin_cache is not None and not hasattr(self, "cos_sin_cache")

        if not can_use_cuda:
            return self.forward_native(
                query=query,
                key=key,
                positions=positions,
                position_offset=position_offset,
                cos=cos,
                sin=sin,
                complex_freqs=complex_freqs,
                cos_sin_cache=cos_sin_cache,
                offsets=offsets,
                **kwargs,
            )

        if query.dim() != 4 or key.dim() != 4:
            raise ValueError(
                f"query and key must be [batch_size, seq_len, num_heads, head_dim],"
                f"got query: {tuple(query.shape)}, key: {tuple(key.shape)}"
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
        positions: Optional[torch.Tensor] = None,
        query: Optional[torch.Tensor] = None,
        key: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        complex_freqs: Optional[torch.Tensor] = None,
        cos_sin_cache: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""

        if hasattr(self, "cos_sin_cache"):
            if offsets is not None:
                positions = positions + offsets
            positions = positions.flatten()
            num_tokens = positions.shape[0]
            cos_sin = self.cos_sin_cache.index_select(0, positions)
            cos, sin = cos_sin.chunk(2, dim=-1)
        else:
            num_tokens = query.shape[:-2].numel()
        if cos is not None and sin is not None:
            q_shape = query.shape
            q_flat = query.reshape(num_tokens, -1, self.head_size)
            q_rot = q_flat[..., : self.rotary_dim]
            q_pass = q_flat[..., self.rotary_dim :]

            k_shape = key.shape
            k_flat = key.reshape(num_tokens, -1, self.head_size)
            k_rot = k_flat[..., : self.rotary_dim]
            k_pass = k_flat[..., self.rotary_dim :]

            q_rotated = _apply_rotary_emb(
                q_rot,
                cos,
                sin,
                is_neox_style=self.is_neox_style,
                interleaved=not self.is_neox_style,
            )
            q = torch.cat((q_rotated, q_pass), dim=-1).reshape(q_shape)
            k_rotated = _apply_rotary_emb(
                k_rot,
                cos,
                sin,
                is_neox_style=self.is_neox_style,
                interleaved=not self.is_neox_style,
            )
            k = torch.cat((k_rotated, k_pass), dim=-1).reshape(k_shape)
            return q, k

        if query.dim() != 4 or key.dim() != 4:
            raise ValueError(
                f"query and key must be [batch_size, seq_len, num_heads, head_dim],"
                f"got query: {tuple(query.shape)}, key: {tuple(key.shape)}"
            )

        if complex_freqs is not None:
            return (
                _apply_rotary_emb_complex(query, complex_freqs),
                _apply_rotary_emb_complex(key, complex_freqs),
            )

        if cos_sin_cache is not None:
            batch_size, seq_len, _, _ = query.shape
            num_tokens = batch_size * seq_len

            if positions is None:
                pos_1d = torch.arange(
                    position_offset,
                    position_offset + seq_len,
                    device=query.device,
                    dtype=torch.int64,
                )
                positions = pos_1d if batch_size == 1 else pos_1d.repeat(batch_size)
            else:
                if positions.dim() != 1 or positions.numel() != num_tokens:
                    raise ValueError(
                        f"positions must be 1D of length {num_tokens}, got shape={tuple(positions.shape)}"
                    )
                positions = positions.to(device=query.device, dtype=torch.long)

            return apply_flashinfer_rope_qk_inplace(
                q=query,
                k=key,
                cos_sin_cache=cos_sin_cache,
                head_size=self.head_size,
                is_neox=self.is_neox_style,
                positions=positions,
            )

        raise ValueError(
            "No valid inputs (complex_freqs, cos/sin, or cos_sin_cache) for interleaved RoPE."
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
