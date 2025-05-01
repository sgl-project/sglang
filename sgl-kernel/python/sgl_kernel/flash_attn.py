from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

try:
    from sgl_kernel import flash_ops
except:
    raise ImportError("Can not import sgl_kernel. Please check your installation.")


def is_fa3_supported(device=None) -> bool:
    #  There some fa3 FYI
    #  FA3 can fail without a enough shared memory for a some shapes, such as higher
    #  hidden_dim or some special cases.
    #  Right now, fa3 is supported for sm80/sm87 and sm86/sm89. The main different
    #  Between sm80/sm87 and sm86/sm89 is the shared memory size. you can follow the link below for more information
    #  https://docs.nvidia.com/cuda/cuda-c-programming-guide/#shared-memory-8-x
    #  And for sgl-kernel right now, we can build fa3 on sm80/sm86/sm89/sm90a.
    #  Thats mean if you use A100/A*0/L20/L40/L40s/4090 you can use fa3.
    return (
        torch.cuda.get_device_capability(device)[0] == 9
        or torch.cuda.get_device_capability(device)[0] == 8
    ) and (torch.version.cuda >= "12.3")


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


def flash_attn_with_kvcache(
    q,
    k_cache,
    v_cache,
    k=None,
    v=None,
    qv=None,
    rotary_cos=None,
    rotary_sin=None,
    cache_seqlens: Optional[Union[(int, torch.Tensor)]] = None,
    cache_batch_idx: Optional[torch.Tensor] = None,
    cache_leftpad: Optional[torch.Tensor] = None,
    page_table: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k_new: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    rotary_seqlens: Optional[torch.Tensor] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),  # -1 means infinite context window
    softcap=0.0,  # 0.0 means deactivated
    rotary_interleaved=True,
    scheduler_metadata=None,
    num_splits=0,  # Can be tuned for speed
    pack_gqa=None,  # Can be tuned for speed
    sm_margin=0,  # Can be tuned if some SMs are used for communication
    return_softmax_lse=False,
):
    r"""
    Compute Flash Attention with optional incremental key/value cache update, rotary embedding,
    sliding window local attention, and support for multi-query/grouped-query attention.

    This function updates `k_cache` and `v_cache` *inplace* with new values from `k` and `v`
    when provided. This is useful for incremental decoding: you can pass cached keys/values
    from the previous step, update them with the new keys/values for the current step, and perform
    attention using the updated cache, all in a single kernel call.

    The cache should be pre-allocated with sufficient space for the maximum sequence length, and
    `cache_seqlens` should track the current valid length for each batch element.

    Rotary embedding is applied to keys and queries if `rotary_cos` and `rotary_sin` are provided,
    using the appropriate indices depending on whether attention is causal or local.

    Supports multi-query/grouped-query attention (MQA/GQA) by allowing KV heads to be fewer than query heads.
    The number of query heads must be divisible by the number of KV heads.

    If `causal=True`, the causal mask aligns to the bottom-right of the attention matrix.
    If `window_size` is set (not `(-1, -1)`), implements sliding window local attention.

    Backward pass is **not supported**.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape (batch_size, seqlen, nheads, headdim).
    k_cache : torch.Tensor
        Key cache tensor. Shape:
            - (batch_size_cache, seqlen_cache, nheads_k, headdim) if no page_table,
            - (num_blocks, page_block_size, nheads_k, headdim) if page_table is used
              (paged KV cache; page_block_size must be a multiple of 256).
    v_cache : torch.Tensor
        Value cache tensor. Shape matches `k_cache`, but the last dimension is `headdim_v`.
    k : torch.Tensor, optional
        New key tensor to insert into the cache, shape (batch_size, seqlen_new, nheads_k, headdim).
        Concatenated to cache at indices specified by `cache_seqlens`.
    v : torch.Tensor, optional
        New value tensor, similar to `k`.
    qv : torch.Tensor, optional
        Optional query-value tensor, shape (batch_size, seqlen, nheads, headdim_v).
    rotary_cos : torch.Tensor, optional
        Rotary embedding cosine component, shape (seqlen_ro, rotary_dim / 2).
        Applied to `k` and `q` if provided (only if `k` and `v` are present).
        `rotary_dim` must be divisible by 16.
    rotary_sin : torch.Tensor, optional
        Rotary embedding sine component, same shape as `rotary_cos`.
    cache_seqlens : int or torch.Tensor
        The current sequence lengths of the KV cache (scalar or (batch_size,)).
    cache_batch_idx : torch.Tensor, optional
        Indices for batch lookup into the KV cache, shape (batch_size,). Default is [0, ..., batch_size-1].
        If not unique, and `k`/`v` are provided, updated cache values may come from any duplicate index.
    cache_leftpad : torch.Tensor, optional
        Left padding offset for the KV cache, shape (batch_size,). If None, assume 0.
    page_table : torch.Tensor, optional
        Mapping for paged KV cache, shape (batch_size, max_num_blocks_per_seq), dtype torch.int32.
    softmax_scale : float, optional
        Scaling factor for QK^T before softmax. Defaults to 1 / sqrt(headdim).
    causal : bool, default = False
        Whether to apply a causal mask (for autoregressive models).
    window_size : Tuple[int, int], default = (-1, -1)
        If not (-1, -1), implements sliding window local attention with the given (left, right) window.
    softcap : float, default = 0.0
        If > 0, activates softcapping attention.
    rotary_interleaved : bool, default = False
        If True, rotary embedding interleaves pairs of dimensions (0&1, 2&3, ...). If False, uses GPT-NeoX style.
    num_splits : int, default = 0
        If > 1, splits the key/value into this many chunks along the sequence. If 0, uses heuristic.
    return_softmax_lse : bool, default = False
        Whether to also return the logsumexp of attention scores.

    Returns
    -------
    out : torch.Tensor
        Attention output, shape (batch_size, seqlen, nheads, headdim).
    softmax_lse : torch.Tensor, optional
        If `return_softmax_lse` is True, also returns (batch_size, nheads, seqlen) tensor of
        logsumexp per row of the attention logits before softmax.

    Examples
    --------
    See `tests/test_flash_attn.py::test_flash_attn_kvcache` for usage examples.

    Notes
    -----
    - The cache (`k_cache`, `v_cache`) must be large enough to accommodate new keys/values from `k`/`v`.
    - Sliding window local attention restricts queries to attend to keys in a specified window.
    - Causal masking aligns to the bottom right of the attention matrix. Rows with all zeros in the mask
      result in output rows of all zeros.
    - Multi-query/grouped-query attention is supported by using fewer KV heads than query heads, as long as
      the number of query heads is divisible by the number of KV heads.
    - Backward (gradient) computation is **not supported**.
    """
    if not is_fa3_supported():
        raise NotImplementedError(
            "flash_attn at sgl-kernel is only supported on sm90 and cu123 above"
        )
    assert k_cache.stride(-1) == 1, "k_cache must have contiguous last dimension"
    assert v_cache.stride(-1) == 1, "v_cache must have contiguous last dimension"
    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (
            -0.5
        )
    if cache_seqlens is not None and isinstance(cache_seqlens, int):
        cache_seqlens = torch.full(
            (k_cache.shape[0],), cache_seqlens, dtype=torch.int32, device=k_cache.device
        )
        cache_seqlens = maybe_contiguous(cache_seqlens)

    q, k_cache, k, v = [maybe_contiguous(x) for x in (q, k_cache, k, v)]
    v_cache = (
        v_cache.contiguous()
        if v_cache.stride(-1) != 1 and v_cache.stride(-3) != 1
        else v_cache
    )
    cu_seqlens_q, cu_seqlens_k_new = [
        maybe_contiguous(x) for x in (cu_seqlens_q, cu_seqlens_k_new)
    ]
    page_table, cache_batch_idx, cache_leftpad = [
        maybe_contiguous(x) for x in (page_table, cache_batch_idx, cache_leftpad)
    ]
    rotary_cos, rotary_sin = [maybe_contiguous(x) for x in (rotary_cos, rotary_sin)]
    rotary_seqlens = maybe_contiguous(rotary_seqlens)

    out, softmax_lse, *rest = torch.ops.sgl_kernel.fwd.default(
        q,
        k_cache,
        v_cache,
        k,
        v,
        qv,
        None,  # out
        cu_seqlens_q,
        None,  # cu_seqlens_k
        cu_seqlens_k_new,
        None,  # seqused_q
        cache_seqlens,
        max_seqlen_q,
        None,  # max_seqlen_k
        page_table,
        cache_batch_idx,
        cache_leftpad,
        rotary_cos,
        rotary_sin,
        rotary_seqlens,
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        rotary_interleaved,
        scheduler_metadata,
        num_splits,
        pack_gqa,
        sm_margin,
    )
    # return (out, softmax_lse) if return_softmax_lse else out
    return (out, softmax_lse, *rest) if return_softmax_lse else out


def flash_attn_varlen_func(
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    max_seqlen_q,
    max_seqlen_k,
    seqused_q=None,
    seqused_k=None,
    softmax_scale=None,
    causal=False,
    qv=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    window_size=(-1, -1),
    softcap=0.0,
    num_splits=1,
    pack_gqa=None,
    sm_margin=0,
    return_softmax_lse=False,
):
    r"""
    Compute Flash Attention for variable-length sequences, supporting efficient attention
    across batches with sequences of different lengths.

    Parameters
    ----------
    q : torch.Tensor
        Query tensor of shape (total_q, nheads, headdim), where `total_q` is the sum of all query lengths
        across the batch.
    k : torch.Tensor
        Key tensor of shape (total_k, nheads_k, headdim).
    v : torch.Tensor
        Value tensor of shape (total_k, nheads_k, headdim_v).
    cu_seqlens_q : torch.Tensor
        Cumulative sequence lengths for queries. 1D tensor of shape (batch_size_q + 1,).
        Defines the start and end indices for each query sequence in the batch.
    cu_seqlens_k : torch.Tensor
        Cumulative sequence lengths for keys/values. 1D tensor of shape (batch_size_k + 1,).
        Defines the start and end indices for each key/value sequence in the batch.
    max_seqlen_q : int
        Maximum sequence length for queries in the batch.
    max_seqlen_k : int
        Maximum sequence length for keys/values in the batch.
    seqused_q : torch.Tensor, optional
        Optional 1D tensor indicating which query tokens are used (mask), shape (total_q,).
        If None, all queries are used.
    seqused_k : torch.Tensor, optional
        Optional 1D tensor indicating which key tokens are used (mask), shape (total_k,).
        If None, all keys are used.
    softmax_scale : float, optional
        Scaling factor for QK^T before applying softmax. Defaults to 1 / sqrt(headdim) if None.
    causal : bool, default = False
        Whether to apply a causal mask (for autoregressive attention).
    qv : torch.Tensor, optional
        Optional query-value tensor for attention output, shape (total_q, nheads, headdim_v).
    q_descale : torch.Tensor, optional
        Optional tensor to rescale queries before attention.
    k_descale : torch.Tensor, optional
        Optional tensor to rescale keys before attention.
    v_descale : torch.Tensor, optional
        Optional tensor to rescale values before attention.
    window_size : Tuple[int, int], default = (-1, -1)
        If not (-1, -1), applies sliding window local attention with the given (left, right) window.
    softcap : float, default = 0.0
        If > 0, activates softcapping for attention scores.
    num_splits : int, default = 1
        If > 1, splits the key/value tensors into chunks for processing. Useful for very long sequences.
    pack_gqa : torch.Tensor, optional
        Optional tensor for packed grouped-query attention (GQA) management.
    sm_margin : int, default = 0
        Softmax margin for numerical stability.
    return_softmax_lse : bool, default = False
        Whether to return the logsumexp of the attention scores for each query.

    Returns
    -------
    out : torch.Tensor
        Attention output tensor, shape (total_q, nheads, headdim_v).
    softmax_lse : torch.Tensor, optional
        If `return_softmax_lse` is True, returns a tensor of shape (batch_size_q, nheads, max_seqlen_q)
        containing the logsumexp of attention scores for each output query row.

    Notes
    -----
    - This function supports variable-length sequences within a batch using cumulative sequence length arrays.
    - Sliding window local attention can be enabled via `window_size`.
    - Supports both standard and grouped-query attention (GQA).
    - Efficient for distributed or packed inference scenarios.
    - For usage examples, see relevant unit tests or documentation.

    """
    if not is_fa3_supported():
        raise NotImplementedError(
            "flash_attn at sgl-kernel is only supported on sm90 and above"
        )

    if softmax_scale is None:
        softmax_scale = (q.shape[-1] + (qv.shape[-1] if qv is not None else 0)) ** (
            -0.5
        )

    out, softmax_lse, *rest = torch.ops.sgl_kernel.fwd.default(
        q,
        k,
        v,
        None,  # k_new
        None,  # v_new
        qv,  # qv
        None,  # out
        cu_seqlens_q,
        cu_seqlens_k,
        None,  # cu_seqlens_k_new
        seqused_q,
        seqused_k,
        max_seqlen_q,
        max_seqlen_k,
        None,  # page_table,
        None,  # kv_batch_idx
        None,  # leftpad_k
        None,  # rotary cos
        None,  # rotary sin
        None,  # seqlens_rotary
        q_descale,
        k_descale,
        v_descale,
        softmax_scale,
        causal,
        window_size[0],
        window_size[1],
        softcap,
        is_rotary_interleaved=False,
        scheduler_metadata=None,
        num_splits=num_splits,
        pack_gqa=pack_gqa,
        sm_margin=sm_margin,
    )

    return (out, softmax_lse, *rest) if return_softmax_lse else out
