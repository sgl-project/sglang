"""CPU checks for QSA's zigzag-only context parallel entrypoints."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend
from sglang.srt.layers.attention.qsa import qsa_indexer
from sglang.srt.layers.attention.qsa.mqa import _scoring_dtype, torch_qsa_mqa_prefill
from sglang.srt.layers.attention.qsa.qsa_indexer import QSAIndexer
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.layers.cp import base as cp_base
from sglang.srt.layers.cp import zigzag
from sglang.srt.layers.cp.base import ContextParallelStrategyKind
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize(
    "strategy",
    [
        None,
        SimpleNamespace(name="interleave", kind=ContextParallelStrategyKind.INTERLEAVE),
    ],
)
@pytest.mark.parametrize("entrypoint", ["indexer", "attention"])
def test_qsa_cp_rejects_unsupported_strategy_before_metadata_or_collectives(
    monkeypatch, strategy, entrypoint
):
    module = qsa_indexer if entrypoint == "indexer" else qwen_sparse_attn_backend
    monkeypatch.setattr(module, "get_cp_strategy", lambda: strategy)
    gather = Mock(side_effect=AssertionError("unsupported CP must not communicate"))
    monkeypatch.setattr(module, "cp_materialize_global_token_order", gather)
    # None metadata and tensors ensure rejection precedes zigzag-specific access.
    with pytest.raises(
        NotImplementedError, match="QSA prefill CP only supports the zigzag strategy"
    ):
        if entrypoint == "indexer":
            QSAIndexer.forward_cuda_cp(None, None, None, None, None, None)
        else:
            QwenSparseAttnBackend._forward_extend_cp(
                None, None, None, None, None, None, None, False
            )
    gather.assert_not_called()


def test_cuda_indexer_forwards_global_rope_to_cp_path():
    hidden, local_rope, metadata, global_rope = (object() for _ in range(4))
    batch = SimpleNamespace(forward_mode=ForwardMode.EXTEND)
    expected = object()
    indexer = SimpleNamespace(forward_cuda_cp=Mock(return_value=expected))
    indexer._forward_impl = lambda *args: QSAIndexer._forward_impl(indexer, *args)

    actual = QSAIndexer.forward_cuda(
        indexer,
        hidden,
        local_rope,
        batch,
        metadata,
        cp_global_rope_positions=global_rope,
    )

    assert actual is expected
    indexer.forward_cuda_cp.assert_called_once_with(
        hidden, local_rope, batch, metadata, global_rope
    )


@pytest.mark.parametrize("compressed_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_cp_indexer_matches_compressed_dtype_without_storing_local_keys(
    monkeypatch, compressed_dtype
):
    local_rows = torch.tensor([0, 1, 6, 7])
    hidden = (torch.arange(32).reshape(8, 4) / 13).to(torch.bfloat16)
    positions = torch.arange(8)
    compressed_keys = hidden.reshape(4, 2, 4).mean(1).unsqueeze(1).to(compressed_dtype)
    row_starts = torch.zeros(4, dtype=torch.int32)
    row_ends = (positions[local_rows] + 1).to(torch.int32) // 2
    pool = SimpleNamespace(qsa_compressed_dtype=compressed_dtype)
    metadata = SimpleNamespace(
        token_to_kv_pool=pool,
        pending_ring_slots=torch.arange(8),
        get_token_to_batch_idx=lambda: torch.zeros(8, dtype=torch.int32),
        get_prefill_mqa_inputs=lambda *args, **kwargs: (
            compressed_keys,
            row_starts,
            row_ends,
            torch.tensor([8], dtype=torch.int32),
        ),
    )
    batch = SimpleNamespace(
        positions=positions,
        attn_cp_metadata=SimpleNamespace(total_q_prev_tokens=2, total_q_next_tokens=2),
    )
    monkeypatch.setattr(
        qsa_indexer,
        "get_cp_strategy",
        lambda: SimpleNamespace(kind=ContextParallelStrategyKind.ZIGZAG),
    )
    for name in ("cp_shard_position_ids", "cp_shard_hidden_states"):
        monkeypatch.setattr(qsa_indexer, name, lambda value, batch: value[local_rows])

    def gather_raw_keys(local, batch):
        torch.testing.assert_close(local, hidden[local_rows])
        return hidden

    monkeypatch.setattr(
        qsa_indexer, "cp_materialize_global_token_order", gather_raw_keys
    )

    def score_queries(q, k, starts, ends, query_positions, sequence_lengths):
        # Use the same operand check as GPU scoring; the CPU reference otherwise
        # accepts mixed dtypes and would conceal the FP8 compatibility failure.
        assert _scoring_dtype(q, k) == compressed_dtype
        return torch_qsa_mqa_prefill(q, k, starts, ends)

    indexer = SimpleNamespace(
        layer_id=0,
        index_n_heads=1,
        index_kv_heads=1,
        index_head_dim=4,
        index_qk_proj=lambda value: (torch.cat([value, value], dim=-1), None),
        q_layernorm=lambda value: value,
        apply_rope=lambda positions, value: value,
        _use_fused_prep=Mock(
            side_effect=AssertionError("local keys must not be stored")
        ),
        update_key_state_and_compress=Mock(),
        select_prefill_tokens=score_queries,
    )
    indexer.project_qk = lambda *args, **kwargs: QSAIndexer.project_qk(
        indexer, *args, **kwargs
    )
    actual = QSAIndexer.forward_cuda_cp(
        indexer, hidden[local_rows], positions[local_rows], batch, metadata, positions
    )
    expected = torch_qsa_mqa_prefill(
        hidden[local_rows].unsqueeze(1).to(compressed_dtype),
        compressed_keys,
        row_starts,
        row_ends,
    )
    torch.testing.assert_close(actual, expected)
    indexer._use_fused_prep.assert_not_called()
    update = indexer.update_key_state_and_compress.call_args
    torch.testing.assert_close(update.args[0], hidden.unsqueeze(1))
    assert update.kwargs["state_stored"] is False


def _cpu_sparse_chunk_attention(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
    """CPU oracle for the chunk kernel's packed K/V and causal conventions."""
    output = torch.empty_like(q)
    for entry in range(kv_lens.numel()):
        start, end = int(cu_q[entry]), int(cu_q[entry + 1])
        for row in range(start, end):
            visible = int(kv_lens[entry]) - (end - start) + row - start + 1
            selected = indices[row]
            selected = selected[selected >= 0].long()
            assert (selected < visible).all()
            selected = selected + int(cu_k[entry])
            keys = k.index_select(0, selected).repeat_interleave(
                q.shape[1] // k.shape[1], dim=1
            )
            values = v.index_select(0, selected).repeat_interleave(
                q.shape[1] // v.shape[1], dim=1
            )
            scores = torch.einsum("hd,nhd->hn", q[row], keys) * scale
            output[row] = torch.einsum("hn,nhd->hd", scores.softmax(-1), values)
    return output


@pytest.mark.parametrize("prefix_lens", [(4, 8), (0, 4), (0, 0)])
@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("save_kv_cache", [True, False])
@pytest.mark.parametrize(
    "query_dtype,cache_dtype",
    [(torch.float32, torch.float32), (torch.bfloat16, torch.float8_e4m3fn)],
)
def test_qsa_cp_cached_prefix_matches_full_context_attention(
    monkeypatch, prefix_lens, rank, save_kv_cache, query_dtype, cache_dtype
):
    # Unequal request lengths exercise both zigzag halves and uneven rank sizes.
    extend_lens = [8, 5]
    seq_lens = [prefix + length for prefix, length in zip(prefix_lens, extend_lens)]
    monkeypatch.setattr(
        cp_base, "get_parallel", lambda: SimpleNamespace(attn_cp_rank=rank)
    )
    monkeypatch.setattr(zigzag, "get_device", lambda: SimpleNamespace(device="cpu"))
    strategy = zigzag.ZigzagCPStrategy(cp_size=2)
    meta = strategy.build_metadata(sum(extend_lens), seq_lens, extend_lens)
    meta.per_rank_logical_token = meta.per_rank_actual_token
    meta.per_rank_actual_token = [8, 8]
    meta.max_rank_len = [8, 8]
    monkeypatch.setattr(qwen_sparse_attn_backend, "get_cp_strategy", lambda: strategy)

    generator = torch.Generator().manual_seed(42)
    keys = [
        torch.randn(length, 1, 4, generator=generator).to(query_dtype)
        for length in seq_lens
    ]
    values = [
        torch.randn(length, 1, 4, generator=generator).to(query_dtype)
        for length in seq_lens
    ]
    # Cached prefixes have already incurred the KV pool's storage rounding.
    for key, value, prefix in zip(keys, values, prefix_lens):
        key[:prefix] = key[:prefix].to(cache_dtype).to(query_dtype)
        value[:prefix] = value[:prefix].to(cache_dtype).to(query_dtype)
    queries = torch.randn(sum(extend_lens), 2, 4, generator=generator).to(query_dtype)
    new_k = torch.cat([key[prefix:] for key, prefix in zip(keys, prefix_lens)])
    new_v = torch.cat([value[prefix:] for value, prefix in zip(values, prefix_lens)])

    # Request IDs and physical token slots deliberately differ from packed order.
    req_indices = [4, 1]
    req_to_token = torch.zeros(5, max(seq_lens), dtype=torch.int32)
    slots = torch.randperm(80, generator=generator)[: sum(seq_lens)] + 1
    k_cache = torch.full((81, 1, 4), float("nan"), dtype=cache_dtype)
    v_cache = torch.full_like(k_cache, float("nan"))
    out_cache_locs = []
    offset = 0
    for i, (length, prefix) in enumerate(zip(seq_lens, prefix_lens)):
        request_slots = slots[offset : offset + length]
        offset += length
        req_to_token[req_indices[i], :length] = request_slots.to(torch.int32)
        k_cache[request_slots[:prefix]] = keys[i][:prefix].to(cache_dtype)
        v_cache[request_slots[:prefix]] = values[i][:prefix].to(cache_dtype)
        out_cache_locs.append(request_slots[prefix:])
    out_cache_loc = torch.cat(out_cache_locs)
    initial_k, initial_v = k_cache.clone(), v_cache.clone()

    def store_kv(layer, locations, k, v):
        k_cache[locations.long()] = k.to(cache_dtype)
        v_cache[locations.long()] = v.to(cache_dtype)

    pool = SimpleNamespace(
        set_kv_buffer=Mock(side_effect=store_kv),
        get_key_buffer=Mock(return_value=k_cache),
        get_value_buffer=Mock(return_value=v_cache),
    )
    if not any(prefix_lens):
        pool.get_key_buffer.side_effect = AssertionError(
            "zero prefix needs no cache read"
        )
        pool.get_value_buffer.side_effect = AssertionError(
            "zero prefix needs no cache read"
        )
    backend = SimpleNamespace(
        token_to_kv_pool=pool,
        req_to_token_pool=SimpleNamespace(req_to_token=req_to_token),
        _pad_extend_output=QwenSparseAttnBackend._pad_extend_output,
    )
    batch = SimpleNamespace(
        attn_cp_metadata=meta,
        seq_lens_cpu=seq_lens,
        extend_seq_lens_cpu=extend_lens,
        extend_prefix_lens_cpu=list(prefix_lens),
        req_pool_indices=torch.tensor(req_indices),
        out_cache_loc=out_cache_loc,
    )
    local_q = strategy.shard_hidden_states(queries, batch)
    local_k = strategy.shard_hidden_states(new_k, batch)
    local_v = strategy.shard_hidden_states(new_v, batch)
    num_local = meta.total_q_prev_tokens + meta.total_q_next_tokens
    local_rows = strategy.shard_hidden_states(torch.arange(sum(extend_lens)), batch)[
        :num_local
    ]
    logical_positions = torch.cat(
        [torch.arange(prefix, length) for prefix, length in zip(prefix_lens, seq_lens)]
    )
    local_positions = logical_positions.index_select(0, local_rows)

    def gather_new_kv(local_kv, forward_batch):
        assert forward_batch is batch
        torch.testing.assert_close(
            local_kv, torch.cat([local_k.flatten(1), local_v.flatten(1)], dim=-1)
        )
        return torch.cat([new_k.flatten(1), new_v.flatten(1)], dim=-1)

    monkeypatch.setattr(
        qwen_sparse_attn_backend, "cp_materialize_global_token_order", gather_new_kv
    )

    def sparse_chunk_attention(q, k, v, indices, cu_q, cu_k, kv_lens, scale):
        # The last local query in each block identifies its absolute causal end.
        expected_ends = local_positions[cu_q[1:].long() - 1] + 1
        torch.testing.assert_close(kv_lens.long(), expected_ends)
        assert k.dtype == v.dtype == q.dtype
        return _cpu_sparse_chunk_attention(q, k, v, indices, cu_q, cu_k, kv_lens, scale)

    monkeypatch.setattr(
        qwen_sparse_attn_backend,
        "sparse_gqa_fwd_interface_triton_ck",
        sparse_chunk_attention,
    )
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=2,
        tp_k_head_num=1,
        tp_v_head_num=1,
        head_dim=4,
        qk_head_dim=4,
        v_head_dim=4,
        scaling=0.5,
    )
    # The indexer supplies causal logical positions and marks other slots invalid.
    candidates = torch.arange(max(seq_lens), dtype=torch.int32).expand(num_local, -1)
    topk = torch.where(candidates <= local_positions[:, None], candidates, -1)
    actual = QwenSparseAttnBackend._forward_extend_cp(
        backend, local_q, local_k, local_v, layer, batch, topk, save_kv_cache
    )

    expected = []
    row = 0
    for i, (length, prefix) in enumerate(zip(extend_lens, prefix_lens)):
        for position in range(prefix, prefix + length):
            visible_keys = keys[i][: position + 1, 0]
            visible_values = values[i][: position + 1, 0]
            scores = queries[row] @ visible_keys.T * layer.scaling
            expected.append(scores.softmax(-1) @ visible_values)
            row += 1
    expected = torch.stack(expected).index_select(0, local_rows).flatten(1)
    torch.testing.assert_close(actual[:num_local], expected)
    assert actual.shape == (8, 8)
    torch.testing.assert_close(actual[num_local:], torch.zeros_like(actual[num_local:]))
    if save_kv_cache:
        pool.set_kv_buffer.assert_called_once()
        initial_k[out_cache_loc] = new_k.to(cache_dtype)
        initial_v[out_cache_loc] = new_v.to(cache_dtype)
    else:
        pool.set_kv_buffer.assert_not_called()
    torch.testing.assert_close(k_cache.float(), initial_k.float(), equal_nan=True)
    torch.testing.assert_close(v_cache.float(), initial_v.float(), equal_nan=True)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
