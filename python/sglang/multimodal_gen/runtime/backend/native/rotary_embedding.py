from typing import Optional, Tuple, Union

import torch

from sglang.multimodal_gen.runtime.layers.triton_ops import (
    apply_rotary_embedding,
    apply_rotary_embedding_qk,
)
from sglang.multimodal_gen.runtime.platforms import current_platform

try:
    from flashinfer.rope import apply_rope_with_cos_sin_cache_inplace
except ImportError:
    apply_rope_with_cos_sin_cache_inplace = None

_is_flashinfer_available = (
    current_platform.is_cuda() and apply_rope_with_cos_sin_cache_inplace is not None
)


def _rope_impl_naive(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    cos = cos.float().unsqueeze(1)
    sin = sin.float().unsqueeze(1)

    def _rope(x):
        if is_neox_style:
            x1, x2 = torch.chunk(x, 2, dim=-1)
        else:
            x1, x2 = x[..., 0::2], x[..., 1::2]

        o1 = (x1.float() * cos - x2.float() * sin).to(dtype=x.dtype)
        o2 = (x2.float() * cos + x1.float() * sin).to(dtype=x.dtype)

        if is_neox_style:
            return torch.cat((o1, o2), dim=-1)
        else:
            return torch.stack((o1, o2), dim=-1).flatten(-2)

    q_out = _rope(q)
    if k is not None:
        k_out = _rope(k)
        return q_out, k_out
    return q_out


def _rope_impl_triton(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
    cos, sin = cos.float(), sin.float()

    if k is not None:
        return apply_rotary_embedding_qk(q, k, cos, sin, is_neox_style)
    else:
        return apply_rotary_embedding(q, cos, sin, is_neox_style)


def _rope_impl_flashinfer(
    q: torch.Tensor,
    k: Optional[torch.Tensor],
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:

    q = q.contiguous()
    if k is not None:
        _k = k.contiguous()
    else:
        _k = torch.empty_like(q)

    cos_sin_cache = torch.cat([cos, sin], dim=-1).float()

    if q.dim() == 3:
        seq_len, num_q_heads, head_dim = q.shape
        bsz = 1
    else:
        bsz, seq_len, num_q_heads, head_dim = q.shape

    num_kv_heads = _k.size(-2)

    pos_1d = torch.arange(seq_len, device=q.device, dtype=torch.long)
    positions = pos_1d if bsz == 1 else pos_1d.repeat(bsz)

    apply_rope_with_cos_sin_cache_inplace(
        positions=positions,
        query=q.view(-1, num_q_heads * head_dim),
        key=_k.view(-1, num_kv_heads * head_dim),
        head_size=head_dim,
        cos_sin_cache=cos_sin_cache,
        is_neox=is_neox_style,
    )

    if k is not None:
        return q, _k
    return q


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = False,
) -> torch.Tensor:
    """
    Args:
        x: [batch_size, seq_len, num_heads, head_dim] or [seq_len, num_heads, head_dim]
        cos: [seq_len, head_dim // 2]
        sin: [seq_len, head_dim // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    if _is_flashinfer_available and x.dtype in {torch.bfloat16, torch.float16}:
        return _rope_impl_flashinfer(x, None, cos, sin, is_neox_style)
    else:
        try:
            return _rope_impl_triton(x, None, cos, sin, is_neox_style)
        except Exception:
            return _rope_impl_naive(x, None, cos, sin, is_neox_style)


# TODO: create custom_op for this
def _apply_rotary_emb_qk(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Args:
        q: [batch_size, seq_len, num_heads, head_dim] or [seq_len, num_heads, head_dim]
        k: [batch_size, seq_len, num_heads, head_dim] or [seq_len, num_heads, head_dim]
        cos: [seq_len, head_dim // 2]
        sin: [seq_len, head_dim // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    if (
        _is_flashinfer_available
        and q.dtype in {torch.bfloat16, torch.float16}
        and k.dtype in {torch.bfloat16, torch.float16}
    ):
        return _rope_impl_flashinfer(q, k, cos, sin, is_neox_style)
    else:
        try:
            return _rope_impl_triton(q, k, cos, sin, is_neox_style)
        except Exception:
            return _rope_impl_naive(q, k, cos, sin, is_neox_style)


def rotary_embedding(
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
