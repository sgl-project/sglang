from typing import Tuple

import torch

block_sparse_attn = None
import torch

major, minor = torch.cuda.get_device_capability(0)
if major == 9 and minor == 0:  # check if H100
    from vsa.block_sparse_wrapper import block_sparse_attn_SM90
    from vsa_cuda import block_sparse_bwd, block_sparse_fwd

    block_sparse_attn = block_sparse_attn_SM90
else:
    from vsa.block_sparse_wrapper import block_sparse_attn_triton

    block_sparse_fwd = None
    block_sparse_bwd = None
    block_sparse_attn = block_sparse_attn_triton

BLOCK_M = 64
BLOCK_N = 64


def torch_attention(q, k, v) -> Tuple[torch.Tensor, torch.Tensor]:
    QK = torch.matmul(q, k.transpose(-2, -1))
    QK /= q.size(-1) ** 0.5

    # Causal mask removed since causal is always false

    QK = torch.nn.functional.softmax(QK, dim=-1)
    output = torch.matmul(QK, v)
    return output, QK


def video_sparse_attn(
    q, k, v, variable_block_sizes, topk, block_size, compress_attn_weight=None
):
    """
    q: [batch_size, num_heads, seq_len, head_dim]
    k: [batch_size, num_heads, seq_len, head_dim]
    v: [batch_size, num_heads, seq_len, head_dim]
    topk: int
    block_size: int or tuple of 3 ints
    video_shape: tuple of (T, H, W)
    compress_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    select_attn_weight: [batch_size, num_heads, seq_len, head_dim]
    NOTE: We assume q, k, v is zero padded!!
    V1 of sparse attention. Include compress attn and sparse attn branch, use average pooling to compress.
    Assume q, k, v is flattened in this way: [batch_size, num_heads, T//block_size[0], H//block_size[1], W//block_size[2], block_size[0], block_size[1], block_size[2]]
    """

    if isinstance(block_size, int):
        block_size = (block_size, block_size, block_size)

    block_elements = block_size[0] * block_size[1] * block_size[2]
    assert block_elements == 64
    assert q.shape[2] % block_elements == 0
    batch_size, num_heads, seq_len, head_dim = q.shape
    # compress attn
    q_compress = (
        q.view(
            batch_size, num_heads, seq_len // block_elements, block_elements, head_dim
        )
        .float()
        .sum(dim=3)
        / variable_block_sizes.view(1, 1, -1, 1)
    ).to(q.dtype)
    k_compress = (
        k.view(
            batch_size, num_heads, seq_len // block_elements, block_elements, head_dim
        )
        .float()
        .sum(dim=3)
        / variable_block_sizes.view(1, 1, -1, 1)
    ).to(k.dtype)
    v_compress = (
        v.view(
            batch_size, num_heads, seq_len // block_elements, block_elements, head_dim
        )
        .float()
        .sum(dim=3)
        / variable_block_sizes.view(1, 1, -1, 1)
    ).to(v.dtype)

    output_compress, block_attn_score = torch_attention(
        q_compress, k_compress, v_compress
    )

    output_compress = output_compress.view(
        batch_size, num_heads, seq_len // block_elements, 1, head_dim
    )
    output_compress = output_compress.repeat(1, 1, 1, block_elements, 1).view(
        batch_size, num_heads, seq_len, head_dim
    )

    topK_indices = torch.topk(block_attn_score, topk, dim=-1).indices
    block_mask = torch.zeros_like(block_attn_score, dtype=torch.bool).scatter_(
        -1, topK_indices, True
    )
    output_select, _ = block_sparse_attn(q, k, v, block_mask, variable_block_sizes)

    if compress_attn_weight is not None:
        final_output = output_compress * compress_attn_weight + output_select
    else:
        final_output = output_compress + output_select
    return final_output
