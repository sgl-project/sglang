"""Unit tests for per-item DataEmbeddingFunc results in the chunked mm path.

A DataEmbeddingFunc may return either one combined [tokens, hidden] tensor or
one tensor per item (see mm_schedule.DataEmbeddingFunc). These tests assert the
two forms produce bitwise-identical chunked-prefill embeddings, and that the
per-item form yields cache entries that own their storage (a torch.split view
of the combined tensor pins the whole concatenated buffer).

CPU-only: exercises mm_schedule internals directly, no engine or GPU.
"""

import logging
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.fixture(autouse=True)
def publish_config_and_parallel_state():
    """Applied to every test in this module (``autouse``), named by none of them.

    The embedding path reads the config namespaces and the attention-TP rank —
    process state a served engine establishes at startup. Without this the
    accessors raise instead of answering.
    """
    override = get_context().override_server_args(tp_size=1)
    override.install()
    try:
        with get_parallel().override(attn_tp_rank=0, attn_tp_size=1, tp_size=1):
            yield
    finally:
        override.restore()


HIDDEN = 16


@pytest.fixture(autouse=True)
def single_process_runtime_context():
    # These mm_utils unit tests exercise cache-hit paths that acknowledge
    # deferred CUDA IPC through runtime_context. They do not start an engine, so
    # pin the runtime topology to a single-process CPU setup.
    server_args_override = get_context().override_server_args(tp_size=1)
    server_args_override.install()
    try:
        with get_parallel().override(attn_tp_rank=0, attn_tp_size=1):
            yield
    finally:
        server_args_override.restore()


# Three items with text gaps between their placeholder runs; offsets are
# (start, end) inclusive, mirroring processor output.
ITEM_OFFSETS = [(2, 5), (9, 14), (20, 24)]
TOTAL_LEN = 30

# Chunk windows (prefix_len, extend_len) covering the sequence, sized so item
# boundaries fall both inside and across chunks.
CHUNKS = [(0, 8), (8, 8), (16, 8), (24, 6)]

_CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _skip_cuda_ipc_acknowledgement(monkeypatch):
    """Keep CPU embedding tests independent of tensor-parallel runtime state."""
    monkeypatch.setattr(
        mm_schedule, "_acknowledge_deferred_cuda_ipc_cache_hits", lambda _items: None
    )


def _num_tokens(item: MultimodalDataItem) -> int:
    start, end = item.offsets[0]
    return end - start + 1


def _item_embedding(item: MultimodalDataItem) -> torch.Tensor:
    gen = torch.Generator().manual_seed(item.hash)
    return torch.randn(_num_tokens(item), HIDDEN, generator=gen)


def _encoder_tensor(items):
    return torch.cat([_item_embedding(item) for item in items], dim=0)


def _encoder_list(items):
    return [_item_embedding(item) for item in items]


def _make_items():
    return [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            hash=1000 + i,
            feature=torch.zeros(1),
            offsets=[offset],
        )
        for i, offset in enumerate(ITEM_OFFSETS)
    ]


def _run_by_item_chunks(encoder):
    mm_schedule.init_mm_embedding_cache(1 << 30)
    items = _make_items()
    return [
        mm_schedule._get_chunked_embedding_by_item(
            encoder, items, ITEM_OFFSETS, prefix_len, extend_len, _CPU
        )
        for prefix_len, extend_len in CHUNKS
    ]


def _run_full_chunks(encoder):
    mm_schedule.init_mm_embedding_cache(1 << 30)
    items = _make_items()
    input_ids = torch.zeros(TOTAL_LEN, dtype=torch.long)
    outs = []
    for prefix_len, extend_len in CHUNKS:
        chunk, _ = mm_schedule._get_chunked_embedding_full(
            encoder, items, ITEM_OFFSETS, prefix_len, extend_len, input_ids, _CPU
        )
        outs.append(chunk)
    return outs


def _assert_chunks_equal(chunks_a, chunks_b):
    assert len(chunks_a) == len(chunks_b)
    for a, b in zip(chunks_a, chunks_b):
        if a is None or b is None:
            assert a is None and b is None
            continue
        assert a.shape == b.shape
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_by_item_list_matches_tensor():
    _assert_chunks_equal(
        _run_by_item_chunks(_encoder_tensor), _run_by_item_chunks(_encoder_list)
    )


def test_full_list_matches_tensor():
    _assert_chunks_equal(
        _run_full_chunks(_encoder_tensor), _run_full_chunks(_encoder_list)
    )


def test_full_matches_by_item():
    # The two chunked strategies agree with each other for single-offset items.
    _assert_chunks_equal(
        _run_full_chunks(_encoder_tensor), _run_by_item_chunks(_encoder_list)
    )


def test_list_cache_entries_own_storage():
    mm_schedule.init_mm_embedding_cache(1 << 30)
    items = _make_items()
    mm_schedule._get_chunked_embedding_by_item(
        _encoder_list, items, ITEM_OFFSETS, 0, TOTAL_LEN, _CPU
    )
    for item in items:
        emb = mm_schedule.embedding_cache.get_single(item.hash).embedding
        own_bytes = emb.numel() * emb.element_size()
        assert emb.untyped_storage().nbytes() == own_bytes


def test_tensor_cache_entries_share_storage():
    # Documents the motivation for the per-item form: split views of the
    # combined tensor keep the whole concatenated buffer alive.
    mm_schedule.init_mm_embedding_cache(1 << 30)
    items = _make_items()
    mm_schedule._get_chunked_embedding_by_item(
        _encoder_tensor, items, ITEM_OFFSETS, 0, TOTAL_LEN, _CPU
    )
    total_tokens = sum(_num_tokens(item) for item in items)
    for item in items:
        emb = mm_schedule.embedding_cache.get_single(item.hash).embedding
        assert (
            emb.untyped_storage().nbytes() == total_tokens * HIDDEN * emb.element_size()
        )


def test_by_item_mismatched_cache_entry_is_reencoded():
    mm_schedule.init_mm_embedding_cache(1 << 30)
    items = _make_items()
    first_item = items[0]
    mm_schedule.embedding_cache.set(
        first_item.hash,
        mm_schedule.EmbeddingResult(embedding=torch.zeros(1, HIDDEN)),
    )
    encoder = Mock(side_effect=_encoder_list)

    chunk = mm_schedule._get_chunked_embedding_by_item(
        encoder, items, ITEM_OFFSETS, 0, TOTAL_LEN, _CPU
    )

    assert chunk.shape == (sum(_num_tokens(item) for item in items), HIDDEN)
    encoder.assert_called_once()
    assert mm_schedule.embedding_cache.get_single(first_item.hash).embedding.shape[
        0
    ] == _num_tokens(first_item)


def test_batched_mismatched_cache_entry_is_reencoded():
    mm_schedule.init_mm_embedding_cache(1 << 30)
    items = _make_items()
    first_item = items[0]
    mm_schedule.embedding_cache.set(
        first_item.hash,
        mm_schedule.EmbeddingResult(embedding=torch.zeros(1, HIDDEN)),
    )
    request = mm_schedule.PerImageRequestInfo(
        req_idx=0,
        items=items,
        items_offset=ITEM_OFFSETS,
        extend_prefix_len=0,
        extend_seq_len=TOTAL_LEN,
    )
    encoder = Mock(side_effect=_encoder_list)

    embeddings = mm_schedule._batch_encode_per_image_misses(encoder, [request], _CPU)

    assert embeddings[(first_item.hash, _num_tokens(first_item))].shape == (
        _num_tokens(first_item),
        HIDDEN,
    )
    encoder.assert_called_once()


def test_batched_colliding_hashes_with_different_lengths_are_not_deduplicated():
    mm_schedule.init_mm_embedding_cache(1 << 30)
    items = _make_items()
    # Simulate the compact-hash collision that motivated the cache guard.
    items[1].hash = items[0].hash
    requests = [
        mm_schedule.PerImageRequestInfo(
            req_idx=0,
            items=items[:2],
            items_offset=ITEM_OFFSETS[:2],
            extend_prefix_len=0,
            extend_seq_len=TOTAL_LEN,
        )
    ]
    encoder = Mock(side_effect=_encoder_list)

    embeddings = mm_schedule._batch_encode_per_image_misses(encoder, requests, _CPU)

    first_key = (items[0].hash, _num_tokens(items[0]))
    second_key = (items[1].hash, _num_tokens(items[1]))
    assert embeddings[first_key].shape == (_num_tokens(items[0]), HIDDEN)
    assert embeddings[second_key].shape == (_num_tokens(items[1]), HIDDEN)
    encoder.assert_called_once()


def test_full_mismatched_cache_entry_is_reencoded(caplog):
    mm_schedule.init_mm_embedding_cache(1 << 30)
    items = _make_items()
    combined_hash = mm_schedule.MultiModalStaticCache.combine_hashes(
        [item.hash for item in items]
    )
    mm_schedule.embedding_cache.set(
        combined_hash,
        mm_schedule.EmbeddingResult(embedding=torch.zeros(1, HIDDEN)),
    )
    input_ids = torch.zeros(TOTAL_LEN, dtype=torch.long)
    encoder = Mock(side_effect=_encoder_tensor)

    with caplog.at_level(logging.WARNING, logger=mm_schedule.logger.name):
        chunk, _ = mm_schedule._get_chunked_embedding_full(
            encoder, items, ITEM_OFFSETS, 0, TOTAL_LEN, input_ids, _CPU
        )

    assert chunk.shape == (sum(_num_tokens(item) for item in items), HIDDEN)
    encoder.assert_called_once()
    assert "Discarding cached multimodal embedding" in caplog.text
    assert "expected_tokens=15" in caplog.text
    assert "cached_tokens=1" in caplog.text


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
