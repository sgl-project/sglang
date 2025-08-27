from typing import Any, Dict, List, Optional, Tuple, Union

import pytest
import torch
<<<<<<< HEAD
from sgl_kernel import apply_rope_with_cos_sin_cache_inplace


# vLLM torch native
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


class RotaryEmbedding(torch.nn.Module):
    # Reference: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py
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
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
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

        # Modification: float32 is required for the rotary embedding to work correctly
        query = query.to(torch.float32)
        key = key.to(torch.float32)

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

        # Modification: convert to the correct dtype
        query = query.to(self.dtype)
        key = key.to(self.dtype)
        return query, key


class FlashInferRotaryEmbedding(RotaryEmbedding):
    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            head_size=self.head_size,
            cos_sin_cache=self.cos_sin_cache,
            is_neox=self.is_neox_style,
        )

        return query, key


@pytest.mark.parametrize(
    "head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, device, batch_size, seq_len, num_q_heads, num_kv_heads",
    [
        (64, 64, 32, 8000, True, torch.bfloat16, "cuda", 32, 32, 1, 1),
        (256, 128, 4096, 10000, True, torch.bfloat16, "cuda", 2, 512, 4, 2),
        (512, 128, 311, 10000, True, torch.bfloat16, "cuda", 3, 39, 4, 2),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cuda", 2, 512, 32, 8),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cuda", 2, 512, 16, 4),
        (512, 128, 311, 10000, False, torch.bfloat16, "cuda", 3, 39, 4, 2),
=======
from sgl_kernel import FusedSetKVBufferArg, apply_rope_with_cos_sin_cache_inplace
from sgl_kernel.testing.rotary_embedding import (
    FlashInferRotaryEmbedding,
    MHATokenToKVPool,
    RotaryEmbedding,
    create_inputs,
)


@pytest.mark.parametrize(
    "head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, device, batch_size, seq_len, num_q_heads, num_kv_heads, save_kv_cache",
    [
        # GPT-OSS cases
        *[
            (
                64,
                64,
                4096,
                8000,
                True,
                torch.bfloat16,
                "cuda",
                batch_size,
                seq_len,
                64,
                8,
                save_kv_cache,
            )
            for batch_size, seq_len in (
                (1, 1),
                (32, 1),
                (128, 1),
                (512, 1),
                (2, 512),
                (4, 4096),
            )
            for save_kv_cache in (False, True)
        ],
        # Other cases
        (64, 64, 32, 8000, True, torch.bfloat16, "cuda", 32, 32, 1, 1, False),
        (256, 128, 4096, 10000, True, torch.bfloat16, "cuda", 2, 512, 4, 2, False),
        (512, 128, 311, 10000, True, torch.bfloat16, "cuda", 3, 39, 4, 2, False),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cuda", 2, 512, 32, 8, False),
        (128, 128, 2048, 10000, False, torch.bfloat16, "cuda", 2, 512, 16, 4, False),
        (512, 128, 311, 10000, False, torch.bfloat16, "cuda", 3, 39, 4, 2, False),
>>>>>>> origin/main
    ],
)
def test_correctness(
    head_size: int,
    rotary_dim: int,
    max_position_embeddings: int,
    base: int,
    is_neox_style: bool,
    dtype: torch.dtype,
    device: str,
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_kv_heads: int,
<<<<<<< HEAD
):
    rope_ref = RotaryEmbedding(
        head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
    ).to(device)
    rope_flashinfer = FlashInferRotaryEmbedding(
        head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype
    ).to(device)

    pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)
    query = torch.randn(
        batch_size * seq_len, num_q_heads * head_size, dtype=dtype, device=device
    )
    key = torch.randn(
        batch_size * seq_len, num_kv_heads * head_size, dtype=dtype, device=device
    )

    query_ref, key_ref = query.clone(), key.clone()
    query_flashinfer, key_flashinfer = query.clone(), key.clone()

    query_ref_out, key_ref_out = rope_ref.forward_native(pos_ids, query_ref, key_ref)
    query_flashinfer_out, key_flashinfer_out = rope_flashinfer.forward_cuda(
        pos_ids, query_flashinfer, key_flashinfer
=======
    save_kv_cache: bool,
):
    config = dict(
        head_size=head_size,
        rotary_dim=rotary_dim,
        max_position_embeddings=max_position_embeddings,
        base=base,
        is_neox_style=is_neox_style,
        dtype=dtype,
    )

    rope_ref = RotaryEmbedding(**config).to(device)
    rope_flashinfer = FlashInferRotaryEmbedding(**config).to(device)

    inputs = create_inputs(
        head_size=head_size,
        batch_size=batch_size,
        seq_len=seq_len,
        device=device,
        dtype=dtype,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
    )

    if save_kv_cache:
        pool_ref = MHATokenToKVPool(head_num=num_kv_heads, head_dim=head_size)
        pool_flashinfer = MHATokenToKVPool(head_num=num_kv_heads, head_dim=head_size)

    query_ref, key_ref = inputs["query"].clone(), inputs["key"].clone()
    query_flashinfer, key_flashinfer = inputs["query"].clone(), inputs["key"].clone()

    query_ref_out, key_ref_out = rope_ref.forward_native(
        inputs["pos_ids"], query_ref, key_ref
    )
    if save_kv_cache:
        pool_ref.set_kv_buffer(
            loc=inputs["out_cache_loc"],
            cache_k=key_ref_out.view(-1, num_kv_heads, head_size),
            cache_v=inputs["value"].view(-1, num_kv_heads, head_size),
        )

    query_flashinfer_out, key_flashinfer_out = rope_flashinfer.forward_cuda(
        inputs["pos_ids"],
        query_flashinfer,
        key_flashinfer,
        fused_set_kv_buffer_arg=(
            FusedSetKVBufferArg(
                value=inputs["value"],
                k_buffer=pool_flashinfer.k_buffer[0].view(-1, num_kv_heads * head_size),
                v_buffer=pool_flashinfer.v_buffer[0].view(-1, num_kv_heads * head_size),
                k_scale=None,
                v_scale=None,
                cache_loc=inputs["out_cache_loc"],
            )
            if save_kv_cache
            else None
        ),
>>>>>>> origin/main
    )

    torch.testing.assert_close(
        query_ref_out, query_flashinfer_out, atol=1e-2, rtol=1e-2
    )
    torch.testing.assert_close(key_ref_out, key_flashinfer_out, atol=1e-2, rtol=1e-2)
<<<<<<< HEAD
=======
    if save_kv_cache:
        for field in ["k_buffer", "v_buffer"]:
            x_ref = getattr(pool_ref, field)[0]
            x_flashinfer = getattr(pool_flashinfer, field)[0]
            torch.testing.assert_close(x_ref, x_flashinfer, atol=1e-2, rtol=1e-2)
            nonzero_ref = x_ref != 0
            nonzero_flashinfer = x_ref != 0
            assert torch.all(nonzero_ref == nonzero_flashinfer)
>>>>>>> origin/main


if __name__ == "__main__":
    pytest.main([__file__])
