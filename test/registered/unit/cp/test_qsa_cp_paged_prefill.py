"""Exercise QSA CP cached-prefix attention with the CUDA paged kernel."""

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.layers.attention import qwen_sparse_attn_backend
from sglang.srt.layers.attention.qwen_sparse_attn_backend import QwenSparseAttnBackend
from sglang.srt.layers.cp import base as cp_base
from sglang.srt.layers.cp import zigzag
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.mem_cache_utils import routed

register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("sharded", [False, True])
@pytest.mark.parametrize("cache_dtype", [torch.bfloat16, torch.float8_e4m3fn])
def test_qsa_cp_paged_prefix_matches_full_context(
    monkeypatch, rank, sharded, cache_dtype
):
    device = "cuda"
    prefix_lens, extend_lens = [5, 9], [8, 5]
    seq_lens = [prefix + extend for prefix, extend in zip(prefix_lens, extend_lens)]
    monkeypatch.setattr(
        cp_base, "get_parallel", lambda: SimpleNamespace(attn_cp_rank=rank)
    )
    monkeypatch.setattr(zigzag, "get_device", lambda: SimpleNamespace(device=device))
    strategy = zigzag.ZigzagCPStrategy(cp_size=2)
    meta = strategy.build_metadata(sum(extend_lens), seq_lens, extend_lens)
    meta.per_rank_logical_token = meta.per_rank_actual_token
    meta.per_rank_actual_token = [8, 8]
    meta.max_rank_len = [8, 8]
    monkeypatch.setattr(qwen_sparse_attn_backend, "get_cp_strategy", lambda: strategy)

    generator = torch.Generator(device=device).manual_seed(42)
    queries = torch.randn(
        sum(extend_lens),
        4,
        64,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    keys, values = [
        [
            torch.randn(length, 2, 64, generator=generator, device=device).to(
                torch.bfloat16
            )
            for length in seq_lens
        ]
        for _ in range(2)
    ]
    new_k = torch.cat([key[prefix:] for key, prefix in zip(keys, prefix_lens)])
    new_v = torch.cat([value[prefix:] for value, prefix in zip(values, prefix_lens)])

    # Logical slots and scratch rows are disjoint when sharded. Keep the wrong
    # slots in bounds and zero-filled so a regression fails numerically without
    # poisoning the CUDA context with an illegal memory access.
    capacity = 64
    logical_offset = capacity if sharded else 0
    slots = torch.randperm(capacity - 1, generator=generator, device=device) + 1
    slot_table = torch.zeros((2, max(seq_lens)), dtype=torch.int32, device=device)
    k_cache = torch.zeros((2 * capacity, 2, 64), dtype=cache_dtype, device=device)
    v_cache = torch.zeros_like(k_cache)
    out_cache_locs = []
    offset = 0
    for row, (length, prefix) in enumerate(zip(seq_lens, prefix_lens)):
        request_slots = slots[offset : offset + length]
        offset += length
        slot_table[row, :length] = request_slots.to(torch.int32)
        k_cache[request_slots[:prefix]] = keys[row][:prefix].to(cache_dtype)
        v_cache[request_slots[:prefix]] = values[row][:prefix].to(cache_dtype)
        out_cache_locs.append(request_slots[prefix:] + logical_offset)

    def store_kv(layer, locations, k, v):
        scratch_rows = locations.long() - logical_offset
        k_cache[scratch_rows] = k.to(cache_dtype)
        v_cache[scratch_rows] = v.to(cache_dtype)

    pool = SimpleNamespace(
        set_kv_buffer=Mock(side_effect=store_kv),
        get_key_buffer=lambda _: k_cache,
        get_value_buffer=lambda _: v_cache,
    )
    backend = SimpleNamespace(
        token_to_kv_pool=routed(pool),
        kv_layers=pool.layer_router,
        _kv_shard_pool=object() if sharded else None,
        _pad_extend_output=QwenSparseAttnBackend._pad_extend_output,
        _resolve_metadata=lambda _: SimpleNamespace(
            token_slot_table=slot_table + logical_offset,
            kv_shard_slot_table=slot_table if sharded else None,
        ),
        _cp_kv_pending=None,
    )
    batch = SimpleNamespace(
        attn_cp_metadata=meta,
        seq_lens_cpu=seq_lens,
        extend_seq_lens_cpu=extend_lens,
        out_cache_loc=torch.cat(out_cache_locs),
    )
    local_q, local_k, local_v = [
        strategy.shard_hidden_states(tensor, batch)
        for tensor in (queries, new_k, new_v)
    ]
    num_local = meta.total_q_prev_tokens + meta.total_q_next_tokens
    local_rows = strategy.shard_hidden_states(
        torch.arange(sum(extend_lens), device=device), batch
    )[:num_local]
    positions = torch.cat(
        [
            torch.arange(prefix, length, device=device)
            for prefix, length in zip(prefix_lens, seq_lens)
        ]
    )[local_rows]
    candidates = torch.arange(max(seq_lens), dtype=torch.int32, device=device)
    topk = torch.where(candidates <= positions[:, None], candidates, -1)
    monkeypatch.setattr(
        qwen_sparse_attn_backend,
        "_gather_cp_kv",
        lambda *args: (new_k, new_v),
    )
    paged_attention = Mock(
        wraps=qwen_sparse_attn_backend.sparse_gqa_paged_prefill_triton
    )
    monkeypatch.setattr(
        qwen_sparse_attn_backend, "sparse_gqa_paged_prefill_triton", paged_attention
    )
    layer = SimpleNamespace(layer_id=0, tp_q_head_num=4, head_dim=64, scaling=64**-0.5)
    actual = QwenSparseAttnBackend._forward_extend_cp(
        backend, local_q, local_k, local_v, layer, batch, topk, True
    )
    paged_attention.assert_called_once()
    pool.set_kv_buffer.assert_called_once()

    # Independent dense causal reference, including the storage rounding of
    # both cached prefixes and the current chunk before the paged kernel reads.
    expected = []
    query_row = 0
    for row, prefix in enumerate(prefix_lens):
        key = keys[row].to(cache_dtype).float().repeat_interleave(2, dim=1)
        value = values[row].to(cache_dtype).float().repeat_interleave(2, dim=1)
        for position in range(prefix, seq_lens[row]):
            scores = (
                torch.einsum(
                    "hd,nhd->hn", queries[query_row].float(), key[: position + 1]
                )
                * layer.scaling
            )
            expected.append(
                torch.einsum("hn,nhd->hd", scores.softmax(-1), value[: position + 1])
            )
            query_row += 1
    expected = torch.stack(expected)[local_rows].flatten(1)
    torch.testing.assert_close(
        actual[:num_local].float(), expected, atol=2e-2, rtol=2e-2
    )
    assert actual.shape == (8, 4 * 64)
    torch.testing.assert_close(actual[num_local:], torch.zeros_like(actual[num_local:]))


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
