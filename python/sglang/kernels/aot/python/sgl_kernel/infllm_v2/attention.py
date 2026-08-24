"""InfLLM-V2 sparse FlashAttention public API.

Ported (drop-in) from ``3rdparty/infllmv2_cuda_impl/infllm_v2/infllmv2_sparse_attention.py``.
The CUDA backend now lives in the standalone ``infllm_ops`` extension.
"""

import torch
from sgl_kernel.infllm_v2._loader import load_infllm_ops


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def infllmv2_attn_stage1(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    cu_seqlens_v,
    max_seqlen_q,
    max_seqlen_k,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=True,
    block_table=None,
):
    """Neighborhood Sparse Attention (NSA) Stage 1 with varlen support.

    Drop-in replacement for ``infllm_v2.infllmv2_attn_stage1``. Returns the
    attention-score matrix with the NSA sparsity pattern, shape
    ``(num_heads_k, total_q, max_seqlen_k)``.
    """
    infllm_ops = load_infllm_ops()
    if softmax_scale is None:
        softmax_scale = q.shape[-1] ** (-0.5)

    q, k, v = [maybe_contiguous(x) for x in (q, k, v)]

    total_q, nheads, head_dim = q.shape
    nheads_k = k.shape[1]
    nheads_per_group = nheads // nheads_k

    q = q.reshape(total_q, nheads_k, nheads_per_group, head_dim)
    q = (
        q.transpose(1, 2)
        .reshape(total_q * nheads_per_group, nheads_k, head_dim)
        .contiguous()
    )

    result = infllm_ops.varlen_fwd_stage1(
        q,
        k,
        v,
        None,
        cu_seqlens_q,
        cu_seqlens_k,
        cu_seqlens_v,
        None,
        None,
        block_table,
        alibi_slopes,
        max_seqlen_q,
        max_seqlen_k,
        dropout_p,
        softmax_scale,
        True,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        True,
        None,
    )

    return result[0]
