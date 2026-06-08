"""Round-trip test for DeepSeek V4 request-level CPU KV copy.

This test exercises DeepSeekV4TokenToKVPool.get_cpu_copy/load_cpu_copy without
starting an SGLang server or loading a model. It constructs minimal pool objects
via __new__ and fills their backing tensors with deterministic byte patterns.

Run on an environment with SGLang Python dependencies installed:
    PYTHONPATH=python python -m pytest test/manual/dsv4/test_dsv4_cpu_copy_roundtrip.py -v

The test intentionally assumes HiSparse is disabled: c4_kv_pool is a normal
DeepSeekV4SingleKVPool-like object, not HiSparseC4DevicePool.
"""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.deepseek_v4_compress_state import KVAndScore
from sglang.srt.mem_cache.deepseek_v4_memory_pool import (
    DeepSeekV4IndexerPool,
    DeepSeekV4SingleKVPool,
    DeepSeekV4TokenToKVPool,
)


def _make_single_pool(
    *, num_layers: int, num_pages: int, page_size: int, row_bytes: int
):
    pool = DeepSeekV4SingleKVPool.__new__(DeepSeekV4SingleKVPool)
    pool.page_size = page_size
    pool.layer_num = num_layers
    pool.cpu_offloading_chunk_size = 17  # deliberately not page-aligned
    pool.kv_buffer = [
        torch.zeros((num_pages, row_bytes), dtype=torch.uint8, device="cpu")
        for _ in range(num_layers)
    ]
    return pool


def _make_indexer_pool(
    *, num_layers: int, num_pages: int, page_size: int, row_bytes: int
):
    pool = DeepSeekV4IndexerPool.__new__(DeepSeekV4IndexerPool)
    pool.page_size = page_size
    pool.layer_num = num_layers
    pool.cpu_offloading_chunk_size = 19
    pool.index_k_with_scale_buffer = [
        torch.zeros((num_pages, row_bytes), dtype=torch.uint8, device="cpu")
        for _ in range(num_layers)
    ]
    return pool


def _make_state_pool(
    *, num_rows: int, row_dim: int, ring_size: int, swa_page_size: int
):
    class _StatePool:
        pass

    pool = _StatePool()
    pool.ring_size = ring_size
    pool.swa_page_size = swa_page_size
    pool.kv_score_buffer = KVAndScore(
        torch.zeros((num_rows, row_dim), dtype=torch.float32, device="cpu")
    )

    def translate_from_swa_loc_to_state_loc(swa_loc: torch.Tensor) -> torch.Tensor:
        swa_pages = swa_loc // pool.swa_page_size
        state_loc = swa_pages * pool.ring_size + (swa_loc % pool.ring_size)
        minus_one = torch.tensor(-1, dtype=state_loc.dtype, device=state_loc.device)
        return torch.where(swa_loc < 0, minus_one, state_loc)

    pool.translate_from_swa_loc_to_state_loc = translate_from_swa_loc_to_state_loc
    return pool


def _fill_page_pool(pool, base: int):
    for layer_id, buf in enumerate(pool.kv_buffer):
        values = torch.arange(buf.numel(), dtype=torch.int64).reshape(buf.shape)
        buf.copy_(((values + base + layer_id * 37) % 251).to(torch.uint8))


def _fill_indexer_pool(pool, base: int):
    for layer_id, buf in enumerate(pool.index_k_with_scale_buffer):
        values = torch.arange(buf.numel(), dtype=torch.int64).reshape(buf.shape)
        buf.copy_(((values + base + layer_id * 41) % 251).to(torch.uint8))


def _fill_state_pool(pool, base: float):
    values = torch.arange(pool.kv_score_buffer.kv_score.numel(), dtype=torch.float32)
    pool.kv_score_buffer.kv_score.copy_(
        values.reshape(pool.kv_score_buffer.kv_score.shape) + base
    )


def _page_indices(indices: torch.Tensor, page_size: int) -> torch.Tensor:
    return torch.unique_consecutive(indices // page_size)


def _snapshot_pages(pool, indices):
    pages = _page_indices(indices, pool.page_size)
    return [buf[pages].clone() for buf in pool.kv_buffer]


def _snapshot_indexer_pages(pool, indices):
    pages = _page_indices(indices, pool.page_size)
    return [buf[pages].clone() for buf in pool.index_k_with_scale_buffer]


def _assert_pages_equal(dst_pool, dst_indices, expected_by_layer, *, label: str):
    dst_pages = _page_indices(dst_indices, dst_pool.page_size)
    for layer_id, expected in enumerate(expected_by_layer):
        actual = dst_pool.kv_buffer[layer_id][dst_pages]
        assert torch.equal(actual, expected), f"{label} layer {layer_id} mismatch"


def _assert_indexer_pages_equal(
    dst_pool, dst_indices, expected_by_layer, *, label: str
):
    dst_pages = _page_indices(dst_indices, dst_pool.page_size)
    for layer_id, expected in enumerate(expected_by_layer):
        actual = dst_pool.index_k_with_scale_buffer[layer_id][dst_pages]
        assert torch.equal(actual, expected), f"{label} layer {layer_id} mismatch"


def test_dsv4_cpu_copy_roundtrip_non_hisparse():
    # Full-token locations before and after retract. The new allocation uses
    # different pages, so this catches accidental in-place assumptions.
    old_full = torch.arange(1, 257, dtype=torch.int64)
    new_full = torch.arange(513, 769, dtype=torch.int64)

    c4_old = old_full[(old_full + 1) % 4 == 0] // 4
    c4_new = new_full[(new_full + 1) % 4 == 0] // 4
    c128_old = old_full[(old_full + 1) % 128 == 0] // 128
    c128_new = new_full[(new_full + 1) % 128 == 0] // 128

    # Tail-only SWA mapping: only the last 64 full tokens have SWA slots.
    old_swa = torch.arange(17, 81, dtype=torch.int64)
    new_swa = torch.arange(217, 281, dtype=torch.int64)
    old_mapping = torch.zeros(900, dtype=torch.int64)
    new_mapping = torch.zeros(900, dtype=torch.int64)
    old_mapping[old_full[-64:]] = old_swa
    new_mapping[new_full[-64:]] = new_swa

    token_pool = DeepSeekV4TokenToKVPool.__new__(DeepSeekV4TokenToKVPool)
    token_pool.swa_kv_pool = _make_single_pool(
        num_layers=2, num_pages=128, page_size=8, row_bytes=13
    )
    token_pool.c4_kv_pool = _make_single_pool(
        num_layers=3, num_pages=256, page_size=4, row_bytes=11
    )
    token_pool.c128_kv_pool = _make_single_pool(
        num_layers=2, num_pages=32, page_size=1, row_bytes=7
    )
    token_pool.c4_indexer_kv_pool = _make_indexer_pool(
        num_layers=3, num_pages=256, page_size=4, row_bytes=17
    )
    token_pool.compress_state_pools = [
        _make_state_pool(num_rows=512, row_dim=10, ring_size=8, swa_page_size=8),
        None,
        _make_state_pool(num_rows=512, row_dim=12, ring_size=8, swa_page_size=8),
    ]
    token_pool.indexer_compress_state_pools = [
        _make_state_pool(num_rows=512, row_dim=14, ring_size=8, swa_page_size=8),
        None,
        None,
    ]
    token_pool.full_to_swa_index_mapping = old_mapping

    _fill_page_pool(token_pool.swa_kv_pool, 3)
    _fill_page_pool(token_pool.c4_kv_pool, 11)
    _fill_page_pool(token_pool.c128_kv_pool, 19)
    _fill_indexer_pool(token_pool.c4_indexer_kv_pool, 29)
    for i, state_pool in enumerate(token_pool.compress_state_pools):
        if state_pool is not None:
            _fill_state_pool(state_pool, 1000.0 + i * 100)
    for i, state_pool in enumerate(token_pool.indexer_compress_state_pools):
        if state_pool is not None:
            _fill_state_pool(state_pool, 2000.0 + i * 100)

    expected_swa = _snapshot_pages(token_pool.swa_kv_pool, old_swa)
    expected_c4 = _snapshot_pages(token_pool.c4_kv_pool, c4_old)
    expected_c128 = _snapshot_pages(token_pool.c128_kv_pool, c128_old)
    expected_indexer = _snapshot_indexer_pages(token_pool.c4_indexer_kv_pool, c4_old)

    expected_attention_state = []
    expected_indexer_state = []
    for state_pool in token_pool.compress_state_pools:
        if state_pool is None:
            expected_attention_state.append(None)
        else:
            loc = state_pool.translate_from_swa_loc_to_state_loc(old_swa)
            expected_attention_state.append(
                state_pool.kv_score_buffer.kv_score[loc].clone()
            )
    for state_pool in token_pool.indexer_compress_state_pools:
        if state_pool is None:
            expected_indexer_state.append(None)
        else:
            loc = state_pool.translate_from_swa_loc_to_state_loc(old_swa)
            expected_indexer_state.append(
                state_pool.kv_score_buffer.kv_score[loc].clone()
            )

    cpu_copy = token_pool.get_cpu_copy(old_full)

    # Destroy device-side data to prove load_cpu_copy really restores it.
    for pool in [
        token_pool.swa_kv_pool,
        token_pool.c4_kv_pool,
        token_pool.c128_kv_pool,
    ]:
        for buf in pool.kv_buffer:
            buf.fill_(0)
    for buf in token_pool.c4_indexer_kv_pool.index_k_with_scale_buffer:
        buf.fill_(0)
    for state_pool in (
        token_pool.compress_state_pools + token_pool.indexer_compress_state_pools
    ):
        if state_pool is not None:
            state_pool.kv_score_buffer.kv_score.fill_(-999.0)

    token_pool.full_to_swa_index_mapping = new_mapping
    token_pool.load_cpu_copy(cpu_copy, new_full)

    _assert_pages_equal(token_pool.swa_kv_pool, new_swa, expected_swa, label="swa")
    _assert_pages_equal(token_pool.c4_kv_pool, c4_new, expected_c4, label="c4")
    _assert_pages_equal(token_pool.c128_kv_pool, c128_new, expected_c128, label="c128")
    _assert_indexer_pages_equal(
        token_pool.c4_indexer_kv_pool, c4_new, expected_indexer, label="c4_indexer"
    )

    for state_pool, expected in zip(
        token_pool.compress_state_pools, expected_attention_state
    ):
        if state_pool is None:
            assert expected is None
            continue
        loc = state_pool.translate_from_swa_loc_to_state_loc(new_swa)
        assert torch.equal(state_pool.kv_score_buffer.kv_score[loc], expected)

    for state_pool, expected in zip(
        token_pool.indexer_compress_state_pools, expected_indexer_state
    ):
        if state_pool is None:
            assert expected is None
            continue
        loc = state_pool.translate_from_swa_loc_to_state_loc(new_swa)
        assert torch.equal(state_pool.kv_score_buffer.kv_score[loc], expected)


if __name__ == "__main__":
    test_dsv4_cpu_copy_roundtrip_non_hisparse()
    print("dsv4 cpu copy roundtrip ok")
