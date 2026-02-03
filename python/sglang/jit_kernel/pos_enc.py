from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from sglang.jit_kernel.utils import cache_once, load_jit

if TYPE_CHECKING:
    from tvm_ffi.module import Module


@cache_once
def _jit_rotary_embedding_module() -> Module:
    return load_jit(
        "rotary_embedding",
        cuda_files=["elementwise/pos_enc.cuh"],
        cuda_wrappers=[("rotary_embedding", "RotaryEmbeddingKernel::run")],
        extra_ldflags=[
            "-L/usr/local/lib/python3.12/dist-packages/torch/lib",
            "-lc10",
            "-ltorch",
        ],
        extra_include_paths=[
            "/usr/local/lib/python3.12/dist-packages/torch/include",
        ],
    )


def rotary_embedding(
    positions: torch.Tensor,  # [batch_size, seq_len] or [num_tokens]
    query: torch.Tensor,  # [batch_size, seq_len, num_heads * head_size] or
    # [num_tokens, num_heads * head_size] or
    # [batch_size, seq_len, num_heads, head_size] or
    # [num_tokens, num_heads, head_size]
    key: torch.Tensor,  # [batch_size, seq_len, num_kv_heads * head_size] or
    # [num_tokens, num_kv_heads * head_size] or
    # [batch_size, seq_len, num_heads, head_size] or
    # [num_tokens, num_heads, head_size]
    head_size: int,
    cos_sin_cache: torch.Tensor,  # [max_position, rot_dim]
    is_neox: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Apply rotary embedding to query and key tensors.

    Args:
        positions: Position indices of shape [num_tokens] or [batch_size, seq_len]
        query: Query tensor of shape [num_tokens, num_heads, head_size] or [num_tokens, num_heads * head_size]
        key: Key tensor of shape [num_tokens, num_kv_heads, head_size] or [num_tokens, num_kv_heads * head_size]
        cos_sin_cache: Cosine and sine cache of shape [max_position, rot_dim]
        is_neox: Whether to use GPT-NeoX style rotary embedding (True) or GPT-J style (False)
    """
    module = _jit_rotary_embedding_module()
    module.rotary_embedding(positions, query, key, head_size, cos_sin_cache, is_neox)
    return query, key
