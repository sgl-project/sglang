import torch


def fa3_token_sparse_attention(
    q: torch.Tensor,
    gathered_k: torch.Tensor,
    gathered_v: torch.Tensor,
    k_cur: torch.Tensor,
    v_cur: torch.Tensor,
    cache_seqlens: torch.Tensor,
    scaling: float,
) -> torch.Tensor:
    """Sparse decode attention over gathered TopK KV via FlashAttention-3, GQA-native.

    The TopK history tokens are gathered per ``(batch, kv_head)`` into a dense
    cache of width ``cache_len = topk + 1`` laid out as ``[batch, num_kv_heads,
    cache_len, head_dim]``. Valid history is front-packed in ``[0,
    cache_seqlens)`` and the trailing slot is scratch for FA3's in-place
    current-token append. We run FA3 with the full query-head count and the
    kv-head count and let the kernel broadcast each kv-head over its GQA group
    -- matching the dense fa3 decode backend's GQA call -- so the gather and
    attention are done once per kv-head instead of once per query head. The
    current decode token is passed via ``k``/``v`` so the kernel appends it at
    ``cache_seqlens`` and attends ``[0, cache_seqlens]``. Everything stays bf16.

    Args:
        q: ``[batch, num_heads, head_dim]`` current decode query (all q-heads).
        gathered_k: ``[batch, num_kv_heads, topk + 1, head_dim]`` gathered
            history K; the trailing slot is overwritten in-place with the
            current token.
        gathered_v: ``[batch, num_kv_heads, topk + 1, head_dim]`` gathered
            history V.
        k_cur: ``[batch, num_kv_heads, head_dim]`` current-token K (per kv-head).
        v_cur: ``[batch, num_kv_heads, head_dim]`` current-token V (per kv-head).
        cache_seqlens: ``[batch]`` int32 count of valid front-packed history
            tokens. Identical across kv-heads for a batch row (``min(seq_len,
            topk)``), so FA3's per-batch ``cache_seqlens`` applies directly.
        scaling: softmax scale (``1 / sqrt(head_dim)`` for GLM).

    Returns:
        ``[batch, num_heads * head_dim]`` attention output in ``q``'s dtype.
    """
    from sglang.jit_kernel.flash_attention import flash_attn_with_kvcache

    batch, num_heads, head_dim = q.shape
    num_kv_heads = gathered_k.shape[1]
    cache_len = gathered_k.shape[2]
    assert gathered_k.shape[:2] == (batch, num_kv_heads)
    assert gathered_v.shape[1:3] == (num_kv_heads, cache_len)
    assert num_heads % num_kv_heads == 0

    kv_dtype = gathered_k.dtype

    # q -> (batch, seqlen=1, num_heads, head_dim)
    q_fa = q.reshape(batch, 1, num_heads, head_dim)
    if q_fa.dtype != kv_dtype:
        q_fa = q_fa.to(kv_dtype)

    # gathered K/V -> (batch, seqlen_cache, num_kv_heads, head_dim) via a
    # middle-axis transpose. head_dim stays last+contiguous so FA3's
    # stride(-1)==1 assert holds without a copy.
    k_cache = gathered_k.transpose(1, 2)
    v_cache = gathered_v.transpose(1, 2)

    # current-token K/V -> (batch, seqlen_new=1, num_kv_heads, head_dim)
    k_new = k_cur.reshape(batch, 1, num_kv_heads, head_dim)
    v_new = v_cur.reshape(batch, 1, num_kv_heads, head_dim)
    if k_new.dtype != kv_dtype:
        k_new = k_new.to(kv_dtype)
    if v_new.dtype != kv_dtype:
        v_new = v_new.to(kv_dtype)

    seqlens = cache_seqlens.reshape(batch).to(torch.int32)

    out = flash_attn_with_kvcache(
        q=q_fa,
        k_cache=k_cache,
        v_cache=v_cache,
        k=k_new,
        v=v_new,
        cache_seqlens=seqlens,
        softmax_scale=scaling,
        causal=True,
    )
    # out: (batch, 1, num_heads, head_dim)
    return out.reshape(batch, num_heads * head_dim).to(q.dtype)
