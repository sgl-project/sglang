import torch

from sglang.multimodal_gen.runtime.layers.triton_ops import apply_rotary_embedding


def _apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
    interleaved: bool = False,
) -> torch.Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size] or [num_tokens, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    # cos = cos.unsqueeze(-2).to(x.dtype)
    # sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        cos = cos.unsqueeze(-2)
        sin = sin.unsqueeze(-2)
        x1, x2 = torch.chunk(x, 2, dim=-1)
        o1 = (x1.float() * cos - x2.float() * sin).type_as(x)
        o2 = (x2.float() * cos + x1.float() * sin).type_as(x)
        return torch.cat((o1, o2), dim=-1)
    else:
        return apply_rotary_embedding(x, cos, sin, interleaved)


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
