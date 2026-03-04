from typing import Optional, Tuple, Union

import torch
from sgl_kernel import FusedSetKVBufferArg, apply_rope_with_cos_sin_cache_inplace


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
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        apply_rope_with_cos_sin_cache_inplace(
            positions=positions,
            query=query,
            key=key,
            fused_set_kv_buffer_arg=fused_set_kv_buffer_arg,
            head_size=self.head_size,
            cos_sin_cache=self.cos_sin_cache,
            is_neox=self.is_neox_style,
        )

        return query, key


class SglKernelRotaryEmbedding(RotaryEmbedding):
    def forward_cuda(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
        fused_set_kv_buffer_arg: Optional[FusedSetKVBufferArg] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert (
            fused_set_kv_buffer_arg is None
        ), "fused_set_kv_buffer_arg is not supported for sgl-kernel implementation"
        if self.cos_sin_cache.dtype != query.dtype:
            self.cos_sin_cache = self.cos_sin_cache.to(query.dtype)
        torch.ops.sgl_kernel.rotary_embedding(
            positions,
            query,
            key,
            self.head_size,
            self.cos_sin_cache,
            self.is_neox_style,
        )
        return query, key


class MHATokenToKVPool:
    KV_POOL_SIZE = 16384

    def __init__(
        self,
        head_num: int,
        head_dim: int,
    ):
        self.head_num = head_num
        self.head_dim = head_dim
        self.size = MHATokenToKVPool.KV_POOL_SIZE
        self.page_size = 1
        self.store_dtype = torch.bfloat16
        self.device = "cuda"
        self.layer_num = 1
        self.start_layer = 0
        self._create_buffers()

    def _create_buffers(self):
        self.k_buffer = [
            torch.zeros(
                (self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]
        self.v_buffer = [
            torch.zeros(
                (self.size + self.page_size, self.head_num, self.head_dim),
                dtype=self.store_dtype,
                device=self.device,
            )
            for _ in range(self.layer_num)
        ]

    def set_kv_buffer(
        self,
        loc: torch.Tensor,
        cache_k: torch.Tensor,
        cache_v: torch.Tensor,
    ):
        layer_id = 0
        self.k_buffer[layer_id - self.start_layer][loc] = cache_k
        self.v_buffer[layer_id - self.start_layer][loc] = cache_v


def create_inputs(
    head_size: int,
    batch_size: int,
    seq_len: int,
    device,
    dtype: torch.dtype,
    num_q_heads: int,
    num_kv_heads: int,
):
    pos_ids = torch.arange(seq_len, device=device).repeat(batch_size)
    query = torch.randn(
        batch_size * seq_len, num_q_heads * head_size, dtype=dtype, device=device
    )
    key = torch.randn(
        batch_size * seq_len, num_kv_heads * head_size, dtype=dtype, device=device
    )
    value = torch.randn(
        batch_size * seq_len, num_kv_heads * head_size, dtype=dtype, device=device
    )
    out_cache_loc = torch.randperm(
        MHATokenToKVPool.KV_POOL_SIZE, dtype=torch.int64, device=device
    )[: batch_size * seq_len].clone()

    return dict(
        pos_ids=pos_ids, query=query, key=key, value=value, out_cache_loc=out_cache_loc
    )
