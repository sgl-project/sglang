"""CANN BF16 sparse attention and two-level indexer for DeepSeek-V4.1 on 910B."""

import torch


def sparse_mla_args(metadata, lengths, ratio, window_size):
    return dict(
        cu_seqlens_q=metadata.actual_seq_lengths_q_pa,
        seqused_ori_kv=lengths,
        seqused_cmp_kv=lengths // ratio if ratio else None,
        cmp_residual_kv=lengths.remainder(ratio) if ratio > 1 else None,
        cmp_ratio=max(1, ratio),
        ori_mask_mode=4,
        cmp_mask_mode=3 if ratio else 0,
        ori_win_left=window_size - 1,
        ori_win_right=0,
        layout_q="TND",
        layout_kv="PA_BBND",
    )


def attention_metadata(backend, fm, lengths, batch_size):
    from sgl_kernel_npu.operators.attention import sparse_flash_mla_metadata

    result = {}
    for ratio in (0, *backend._dsv41_low_ratios()):
        key = f"c{ratio}a_metadata" if ratio else "win_metadata"
        metadata = sparse_flash_mla_metadata(
            backend._dsv4_q_head_num,
            1,
            backend._dsv4_head_dim,
            batch_size=batch_size,
            has_ori_kv=True,
            has_cmp_kv=bool(ratio),
            cmp_topk=backend._dsv4_index_topk if ratio else 0,
            **sparse_mla_args(fm, lengths, ratio, backend._dsv4_sliding_window_size),
        )
        previous = fm.kernel_metadata.get(key)
        if previous is not None and previous.shape == metadata.shape:
            previous.copy_(metadata)
            metadata = previous
        result[key] = metadata
    return result


def attend(backend, query, layer, forward_batch, sink, ratio):
    from sgl_kernel_npu.operators.attention import sparse_flash_mla

    fm = backend.forward_metadata
    pool = backend.token_to_kv_pool
    lengths = backend._dsv41_qli_kv_lens(forward_batch).to(torch.int32)
    kwargs = sparse_mla_args(fm, lengths, ratio, backend._dsv4_sliding_window_size)
    if ratio:
        indices = getattr(fm, f"c{ratio}_topk_indices")
        kwargs.update(
            cmp_kv=pool.get_compress_buffer(layer.layer_id),
            cmp_block_table=getattr(fm, f"c{ratio}_page_table").int(),
            cmp_sparse_indices=indices.reshape(-1, 1, indices.shape[-1]).int(),
        )
        if kwargs["cmp_kv"].dtype == torch.uint8:
            from sglang.srt.hardware_backend.npu.dsv4.dsv41_int8_kv import (
                materialize_kv_pages,
            )
            from sglang.srt.model_executor.runner_utils.capture_mode import (
                get_is_capture_mode,
            )

            kwargs["cmp_kv"], kwargs["cmp_block_table"] = materialize_kv_pages(
                kwargs["cmp_kv"],
                kwargs["cmp_block_table"],
                kwargs["seqused_cmp_kv"],
                fixed_shape=get_is_capture_mode(),
            )
    metadata = fm.kernel_metadata[f"c{ratio}a_metadata" if ratio else "win_metadata"]
    output, _ = sparse_flash_mla(
        query,
        ori_kv=pool.get_swa_buffer(layer.layer_id),
        ori_block_table=fm.swa_page_table.int(),
        sinks=None if sink is None else sink.reshape(-1).float(),
        metadata=metadata,
        softmax_scale=layer.scaling,
        **kwargs,
    )
    return output


def indexer_args(backend, forward_batch, ratio):
    lengths = backend._dsv41_qli_kv_lens(forward_batch).to(torch.int32)
    fm = backend.forward_metadata
    return dict(
        cu_seqlens_q=fm.actual_seq_lengths_q_pa,
        seqused_k=lengths // ratio,
        cmp_residual_k=lengths.remainder(ratio) if ratio > 1 else None,
        block_table=getattr(fm, f"c{ratio}_index_page_table").int(),
        layout_q="TND",
        layout_k="PA_BBND",
        mask_mode=3,
        cmp_ratio=ratio,
    )


def index_topk(
    backend,
    layer,
    forward_batch,
    query,
    keys,
    weights,
    *,
    publish=False,
    candidates=None,
):
    from sgl_kernel_npu.operators.indexer import (
        lightning_indexer_v2,
        sparse_lightning_indexer,
    )

    indexer = layer.indexer
    kwargs = indexer_args(backend, forward_batch, layer.compress_ratio)
    topk = backend._dsv4_index_topk
    if candidates is not None:
        indices, _ = sparse_lightning_indexer(
            query,
            keys,
            weights,
            topk,
            candidates,
            candidate_block_size=indexer.candidate_block_size,
            **kwargs,
        )
        return indices.reshape(query.shape[0], topk), None
    if publish:
        indices, _, blocks, lengths = lightning_indexer_v2(
            query,
            keys,
            weights,
            topk,
            candidate_topk_blocks=indexer.candidate_topk_blocks,
            candidate_block_size=indexer.candidate_block_size,
            **kwargs,
        )
        return indices.reshape(query.shape[0], topk), (blocks, lengths)
    indices, _ = lightning_indexer_v2(query, keys, weights, topk, **kwargs)
    return indices.reshape(query.shape[0], topk), None
