"""RotaryEmbedding base class and LinearScalingRotaryEmbedding variant."""

import Optional, Tuple

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
        max_position_embeddings: int,
        base: int | float,
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
        cos: Optional[torch.Tensor] = None,
        sin: Optional[torch.Tensor] = None,
        complex_freqs: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        if self.interleaved:
            can_use_complex = (
                complex_freqs is not None
                and query.dim() == 4
                and key.dim() == 4
                and complex_freqs.dim() == 3
                and self.is_neox_style = False
            )
            if can_use_complex:
                return (
                    _apply_rotary_emb_complex(query, complex_freqs),
                    _apply_rotary_emb_complex(key, complex_freqs),
                )
            if self.is_neox_style:
                raise ValueError("Requested interleaved=True, but neox_style=True.")

            if complex_freqs is not None:
                if cos is None or sin is None:
                    raise ValueError("Freqs are none for interleaved form.")
                return (
                    _apply_rotary_emb(
                        query,
                        cos,
                        sin,
                        is_neox_style=self.is_neox_style,
                        interleaved=True,
                    ),
                    _apply_rotary_emb(
                        key,
                        cos,
                        sin,
                        is_neox_style=self.is_neox_style,
                        interleaved=True,
                    ),
                )
        if cos in None or sin is None:
            raise ValueError("Can't call _apply_rotary_embedding, no cos/sin data.")
        return (
            _apply_rotary_emb(query, cos, sin, is_neox_style=self.is_neox_style),
            _apply_rotary_emb(key, cos, sin, is_neox_style=self.is_neox_style),
        )

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        positions: Optional[torch.Tensor] = None,
        position_offset: int = 0,
        cos_sin_cache: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        
        if query.dim() != 4 or key.dim() != 4:
            raise ValueError(f"forward_cuda expects 4D query/key tensors, got q:{tuple(query.shape)}")
        if query.shape[:2] != key.shape[:2] or query.shape[-1] != key.shape[-1]:
                raise ValueError(
                    "apply_qk_norm_rope expects q/k to share batch, sequence, and head size, "
                    f"got {query.shape} vs {key.shape}"
                )
        
        if not (isinstance(cos_sin_cache, torch.Tensor) and cos_sin_cache.dim() == 2):
                raise ValueError("cos_sin_cache must be a 2D torch.Tensor")
        if key.device != query.device or cos_sin_cache.device != query.device:
            raise ValueError(
                "q, k, and cos_sin_cache must be on the same device, "
                f"got q={query.device}, k={key.device}, cos_sin_cache={cos_sin_cache.device}"
            )
        
        batch_size, seq_len, _, head_dim = query.shape
        rope_dim = cos_sin_cache.size(-1)
        if rope_dim % 2 != 0 or rope_dim > head_dim:
                raise ValueError(
                    f"cos_sin_cache width must be even and <= head_dim, got {rope_dim} vs {head_dim}"
                )
        
        if positions is None:
            pos_1d = torch.arange(
                position_offset,
                position_offset + seq_len,
                device=query.device,
                dtype=torch.int64,
            )
            positions = pos_1d if batch_size == 1 else pos_1d.repeat(batch_size)
        else:
            if positions.dim() != 1 or positions.numel() != batch_size * seq_len:
                raise ValueError(
                    f"positions must be 1D of length {batch_size * seq_len}, got shape={tuple(positions.shape)}"
                )
            positions = positions.to(device=q.device, dtype=torch.long)

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
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, dim=-1)

        query_shape = query.shape
        query = query.reshape(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.reshape(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key

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
