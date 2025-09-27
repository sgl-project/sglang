from typing import Tuple, Union

import pytest
import torch
from sgl_kernel import rotary_embedding


# torch native
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
    print(f"perform in {cos.dtype=}")
    q_embed = q_embed.to(orig_q_dtype)
    k_embed = k_embed.to(orig_k_dtype)

    return q_embed, k_embed


class RotaryEmbedding(torch.nn.Module):
    # Reference: https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/rotary_embedding.py
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        dtype: torch.dtype,
        is_neox_style: bool = False,
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
        cos: torch.Tensor,
        sin: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        query, key = apply_rotary_pos_emb(query, key, cos, sin)

        query = query.to(self.dtype)
        key = key.to(self.dtype)
        return query, key

    def forward_kernel_inplace(
        self,
        cos: torch.Tensor,
        sin: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """A PyTorch-native implementation of forward()."""
        cos, sin = cos.float(), sin.float()
        query, key = query.float(), key.float()
        print(f"kernel: perform in {cos.dtype=}")
        rotary_embedding(
            cos,
            sin,
            query,
            key,
            self.head_size,
            self.is_neox_style,
        )
        query = query.to(self.dtype)
        key = key.to(self.dtype)
        return query, key


@pytest.mark.benchmark(group="rotary_embedding")
@pytest.mark.parametrize(
    "head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, device, batch_size, seq_len, num_q_heads, num_kv_heads",
    [
        (80, 80, 1e6, 1e6, False, torch.bfloat16, "cuda", 32, 32, 16, 16),
        (320, 230, 1e6, 1e6, False, torch.bfloat16, "cuda", 32, 32, 16, 16),
        (80, 80, 1e6, 1e6, True, torch.bfloat16, "cuda", 32, 32, 16, 16),
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
):
    rope_ref = RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        dtype=dtype,
        is_neox_style=is_neox_style,
    ).to(device)

    query = torch.randn(
        batch_size * seq_len, num_q_heads, head_size, dtype=dtype, device=device
    )

    key = torch.randn(
        batch_size * seq_len, num_kv_heads, head_size, dtype=dtype, device=device
    )

    cos = torch.randn(batch_size * seq_len, head_size, dtype=dtype, device=device)
    sin = torch.randn(batch_size * seq_len, head_size, dtype=dtype, device=device)

    query_native_out, key_native_out = rope_ref.forward_native(
        cos, sin, query.clone(), key.clone()
    )

    # in-place
    query_kernel_out, key_kernel_out = rope_ref.forward_kernel_inplace(
        cos, sin, query, key
    )

    torch.testing.assert_close(query_native_out, query_kernel_out, atol=1e-3, rtol=1e-3)
    torch.testing.assert_close(key_native_out, key_kernel_out, atol=1e-3, rtol=1e-3)


@pytest.mark.benchmark(group="rotary_embedding")
@pytest.mark.parametrize(
    "head_size, rotary_dim, max_position_embeddings, base, is_neox_style, dtype, device, batch_size, seq_len, num_q_heads, num_kv_heads",
    [
        (80, 80, 1e6, 1e6, False, torch.bfloat16, "cuda", 1, 8840, 16, 16),
        (80, 80, 1e6, 1e6, False, torch.bfloat16, "cuda", 1, 4000, 16, 16),
        (80, 80, 1e6, 1e6, True, torch.bfloat16, "cuda", 8, 8840, 16, 16),
    ],
)
def test_rotary_embedding_benchmark(
    benchmark,
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
):
    rope_ref = RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position_embeddings,
        base,
        dtype=dtype,
        is_neox_style=is_neox_style,
    ).to(device)
    query = torch.randn(
        batch_size * seq_len, num_q_heads, head_size, dtype=dtype, device=device
    )
    key = torch.randn(
        batch_size * seq_len, num_kv_heads, head_size, dtype=dtype, device=device
    )
    cos = torch.randn(batch_size * seq_len, head_size, dtype=dtype, device=device)
    sin = torch.randn(batch_size * seq_len, head_size, dtype=dtype, device=device)

    def run_kernel():
        rope_ref.forward_kernel_inplace(
            cos,
            sin,
            query,
            key,
        )
        torch.cuda.synchronize()

    benchmark.pedantic(run_kernel, rounds=20000, warmup_rounds=5)


if __name__ == "__main__":
    pytest.main([__file__, "--capture=no"])
