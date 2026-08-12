"""RotaryEmbedding base class and LinearScalingRotaryEmbedding variant."""

from typing import Optional, Tuple

import torch

from sglang.multimodal_gen.runtime.layers.custom_op import CustomOp
from sglang.multimodal_gen.runtime.platforms import current_platform

from .utils import (
    _apply_rotary_emb,
    _apply_rotary_emb_complex,
    apply_flashinfer_rope_qk_inplace,
)

if current_platform.is_npu():
    import torch_npu

    from sglang.kernels.ops.diffusion.triton.npu_fallback import (
        NPU_ROTARY_MUL_MAX_HEAD_SIZE,
        NPU_ROTARY_MUL_MAX_NUM_HEADS,
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
        self.use_precomputed_cache = use_precomputed_cache

        # Static per-instance gates: fixed for the lifetime of this object,
        # so compute them once here instead of re-deriving them from
        # is_neox_style/rotary_dim/head_size on every forward call.
        self._is_full_rotation = rotary_dim == head_size
        self._can_use_complex_style = not is_neox_style
        self._npu_rotary_mul_style_ok = (
            current_platform.is_npu()
            and is_neox_style
            and rotary_dim < NPU_ROTARY_MUL_MAX_HEAD_SIZE
        )

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

    def _combine_rotated_and_pass(
        self,
        q_rotated: torch.Tensor,
        k_rotated: torch.Tensor,
        q_pass: torch.Tensor,
        k_pass: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Reattach the untouched tail (rotary_dim < head_size) if present.

        When rotary_dim == head_size, q_pass/k_pass are always empty and
        torch.cat against an empty tensor still allocates + copies the
        whole (large) q_rotated/k_rotated — skip it in that (common) case.
        """
        if self._is_full_rotation:
            return q_rotated, k_rotated
        return (
            torch.cat((q_rotated, q_pass), dim=-1),
            torch.cat((k_rotated, k_pass), dim=-1),
        )

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

        if self.use_precomputed_cache or query.dim() == 3 or key.dim() == 3:
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

        seq_len = query.shape[1]

        can_use_complex = (
            complex_freqs is not None
            and complex_freqs.dim() == 3
            and self._can_use_complex_style
        )
        if can_use_complex:
            return (
                _apply_rotary_emb_complex(query, complex_freqs),
                _apply_rotary_emb_complex(key, complex_freqs),
            )

        can_derive_complex_from_cos_sin = (
            complex_freqs is None
            and cos is not None
            and sin is not None
            and self._can_use_complex_style
            and self._is_full_rotation
            and cos.shape[0] == seq_len
        )
        if can_derive_complex_from_cos_sin:
            # Interleaved (non-neox) rotation has no fused NPU kernel (always
            # falls back to several elementwise ops in apply_rotary_embedding);
            # the complex-multiply form is mathematically equivalent for this
            # pairing convention and needs fewer kernel launches.
            derived_complex_freqs = torch.complex(
                cos.to(torch.float32), sin.to(torch.float32)
            ).unsqueeze(-2)
            return (
                _apply_rotary_emb_complex(query, derived_complex_freqs),
                _apply_rotary_emb_complex(key, derived_complex_freqs),
            )

        if cos is not None and sin is not None:
            num_heads = query.shape[2]

            can_use_npu_rotary_mul = (
                self._npu_rotary_mul_style_ok
                and cos.shape[0] == seq_len
                and num_heads < NPU_ROTARY_MUL_MAX_NUM_HEADS
            )
            if can_use_npu_rotary_mul:
                # torch_npu.npu_rotary_mul, called directly on the natural
                # BSND [batch, seq, heads, rotary_dim] layout instead of
                # flattening to [batch*seq, heads, rotary_dim]: cos/sin get
                # a batch dim of 1 and a heads dim of 1 and broadcast
                # against query/key's real batch and heads dims — the
                # documented "1S1D" pattern for BSND inputs — so no
                # expand+reshape copy of cos/sin across the batch is
                # needed (the previous version paid for that copy every
                # call). heads/rotary_dim must stay under the same size
                # limits apply_rotary_embedding_native's own gate uses;
                # replicated here because if we called through with
                # interleaved=False and that gate failed on the other
                # side, apply_rotary_embedding_native would silently fall
                # back to its always-interleaved generic path — wrong for
                # neox-style data.
                #
                # EXPERIMENTAL: this batch/heads broadcast behavior is not
                # independently verified on real hardware beyond the
                # already-profiled single-batch case — validate
                # numerically against the previous implementation (git
                # history) before trusting it for real generations.
                q_rot = query[..., : self.rotary_dim]
                q_pass = query[..., self.rotary_dim :]
                k_rot = key[..., : self.rotary_dim]
                k_pass = key[..., self.rotary_dim :]

                cos_prepared = cos.reshape(1, seq_len, 1, -1).to(query.dtype)
                sin_prepared = sin.reshape(1, seq_len, 1, -1).to(query.dtype)
                if cos_prepared.size(-1) * 2 == self.rotary_dim:
                    cos_prepared = torch.cat(
                        (cos_prepared, cos_prepared), dim=-1
                    )
                    sin_prepared = torch.cat(
                        (sin_prepared, sin_prepared), dim=-1
                    )

                q_rotated = torch_npu.npu_rotary_mul(
                    q_rot, cos_prepared, sin_prepared
                )
                k_rotated = torch_npu.npu_rotary_mul(
                    k_rot, cos_prepared, sin_prepared
                )
                return self._combine_rotated_and_pass(
                    q_rotated, k_rotated, q_pass, k_pass
                )

            # Keep batch and sequence as separate axes (don't flatten to
            # [batch*seq, ...]) so that cos/sin — shaped [seq_len,
            # rotary_dim // 2] and shared across the batch — broadcast
            # correctly for batch_size > 1. self.use_precomputed_cache is
            # already False here (checked at the top of this method), so
            # query/key are guaranteed to be the caller's original 4D
            # [batch, seq, heads, head_dim] tensors, not the legacy
            # positions-indexed [num_tokens, hidden] layout.
            q_rot = query[..., : self.rotary_dim]
            q_pass = query[..., self.rotary_dim :]

            k_rot = key[..., : self.rotary_dim]
            k_pass = key[..., self.rotary_dim :]

            q_rotated = _apply_rotary_emb(
                q_rot,
                cos,
                sin,
                is_neox_style=self.is_neox_style,
                interleaved=not self.is_neox_style,
            )
            k_rotated = _apply_rotary_emb(
                k_rot,
                cos,
                sin,
                is_neox_style=self.is_neox_style,
                interleaved=not self.is_neox_style,
            )
            return self._combine_rotated_and_pass(
                q_rotated, k_rotated, q_pass, k_pass
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

        raise ValueError(
            "No valid inputs (complex_freqs, cos/sin, or cos_sin_cache) for interleaved RoPE."
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

        can_use_cuda = (
            (cos_sin_cache is not None or cos is not None and sin is not None)
            and not self.use_precomputed_cache
            and query.dim() == 4
            and key.dim() == 4
        )

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
        if cos_sin_cache is None:
            cos_sin_cache = torch.cat(
                [
                    cos.to(dtype=torch.float32).contiguous(),
                    sin.to(dtype=torch.float32).contiguous(),
                ],
                dim=-1,
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

        use_precomputed_cache = self.use_precomputed_cache
        if use_precomputed_cache:
            if offsets is not None:
                positions = positions + offsets
            positions = positions.flatten()
            num_tokens = positions.shape[0]
            cos_sin = self.cos_sin_cache.index_select(0, positions)
            cos, sin = cos_sin.chunk(2, dim=-1)

        can_derive_complex_from_cos_sin = (
            not use_precomputed_cache
            and complex_freqs is None
            and cos is not None
            and sin is not None
            and self._can_use_complex_style
            and query.dim() == 4
            and key.dim() == 4
            and self._is_full_rotation
            and cos.shape[0] == query.shape[1]
        )
        if can_derive_complex_from_cos_sin:
            # Interleaved (non-neox) rotation has no fused kernel on native
            # backends (CPU/MPS/etc. always fall back to several elementwise
            # ops in apply_rotary_embedding); the complex-multiply form is
            # mathematically equivalent for this pairing convention and needs
            # fewer kernel launches. CUDA already has a fused Triton kernel
            # for this case and keeps using the cos/sin path below.
            derived_complex_freqs = torch.complex(
                cos.to(torch.float32), sin.to(torch.float32)
            ).unsqueeze(-2)
            return (
                _apply_rotary_emb_complex(query, derived_complex_freqs),
                _apply_rotary_emb_complex(key, derived_complex_freqs),
            )

        if cos is not None and sin is not None:
            if use_precomputed_cache:
                # Legacy positions-indexed callers (e.g. llama/qwen3/gemma2/
                # gemma3 via get_rope()) pass query/key as 3D [batch, seq,
                # hidden], and cos/sin above were already index_select'd
                # per-token via positions.flatten(), so they match
                # num_tokens = batch*seq row-for-row. Flattening here is
                # required to split the hidden dim into heads.
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
                k_rotated = _apply_rotary_emb(
                    k_rot,
                    cos,
                    sin,
                    is_neox_style=self.is_neox_style,
                    interleaved=not self.is_neox_style,
                )
                q, k = self._combine_rotated_and_pass(
                    q_rotated, k_rotated, q_pass, k_pass
                )
                return q.reshape(q_shape), k.reshape(k_shape)

            # Direct DiT-style call: keep batch and sequence as separate
            # axes (don't flatten to [batch*seq, ...]) so that cos/sin —
            # shaped [seq_len, rotary_dim // 2] and shared across the
            # batch — broadcast correctly for batch_size > 1.
            q_rot = query[..., : self.rotary_dim]
            q_pass = query[..., self.rotary_dim :]
            k_rot = key[..., : self.rotary_dim]
            k_pass = key[..., self.rotary_dim :]

            q_rotated = _apply_rotary_emb(
                q_rot,
                cos,
                sin,
                is_neox_style=self.is_neox_style,
                interleaved=not self.is_neox_style,
            )
            k_rotated = _apply_rotary_emb(
                k_rot,
                cos,
                sin,
                is_neox_style=self.is_neox_style,
                interleaved=not self.is_neox_style,
            )
            return self._combine_rotated_and_pass(
                q_rotated, k_rotated, q_pass, k_pass
            )

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
