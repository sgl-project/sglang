"""Sparse FlashMLA BF16 layout preparation shared by serving and TOP trainers."""


def flashmla_sparse_bf16(
    q_all,
    kv_cache,
    v_head_dim,
    page_table_1,
    sm_scale,
    *,
    device_sm_major,
    topk_length=None,
    attn_sink=None,
    sink_pad_cache=None,
):
    """Preserve hardware head padding and per-row valid-index lengths."""
    from sglang.kernels.ops.attention.flash_mla_interface import (
        flash_mla_sparse_fwd,
    )

    # FlashMLA sparse kernel requires num_heads to be a multiple of 64 (Hopper) or 128 (Blackwell)
    # When using TP, num_heads might be smaller (e.g., 256//8=32)
    num_tokens, num_heads, head_dim = q_all.shape

    # Determine required padding based on GPU architecture (use cached value)
    required_padding = 128 if device_sm_major >= 10 else 64

    need_padding = num_heads % required_padding != 0

    if need_padding:
        assert required_padding % num_heads == 0, (
            f"num_heads {num_heads} cannot be padded to {required_padding}. "
            f"TP size may be too large for this model."
        )

        # Pad q to required size
        q_padded = q_all.new_zeros((num_tokens, required_padding, head_dim))
        q_padded[:, :num_heads, :] = q_all
        q_input = q_padded
    else:
        q_input = q_all

    sink_input = attn_sink
    if need_padding and attn_sink is not None:
        if sink_pad_cache is None:
            sink_pad_cache = {}
        key = (attn_sink.data_ptr(), required_padding)
        sink_input = sink_pad_cache.get(key)
        if sink_input is None:
            sink_input = attn_sink.new_zeros(required_padding)
            sink_pad_cache[key] = sink_input
        sink_input[:num_heads].copy_(attn_sink)

    # indices shape must be (s_q, h_kv=1, topk), keep h_kv=1 unchanged
    indices_input = page_table_1.unsqueeze(1)

    # topk_length is the per-row count of valid indices
    # (`dsa_cache_seqlens_int32` = seqlens clipped to `index_topk`). Rows
    # whose context is shorter than `index_topk` have their indices
    # tail-padded with -1; passing the valid length lets the kernel skip
    # the padded tail instead of scanning the full topk width. The output
    # is unchanged: the kernel masks -1 indices either way.
    if topk_length is not None and topk_length.shape[0] != num_tokens:
        # Metadata rows are expected to match q rows (the DP/CP padding
        # helpers keep them aligned); fall back to full-width compute if
        # they ever diverge.
        topk_length = None

    o, _, _ = flash_mla_sparse_fwd(
        q=q_input,
        kv=kv_cache,
        indices=indices_input,
        sm_scale=sm_scale,
        d_v=v_head_dim,
        attn_sink=sink_input,
        topk_length=topk_length,
    )

    # Trim output back to original num_heads if we padded
    if need_padding:
        o = o[:, :num_heads, :]

    return o
