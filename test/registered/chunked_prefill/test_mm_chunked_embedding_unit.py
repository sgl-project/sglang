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
from sglang.srt.multimodal.transport.cuda_ipc import (
    BORROW_CUDA_IPC_FEATURE_KEY,
    DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY,
    CudaIpcTensorTransportProxy,
)
from sglang.srt.runtime_context import get_context, get_parallel
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=12, suite="base-a-test-cpu")


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
        torch.cat(
            mm_schedule._get_chunked_embedding_by_item(
                encoder, items, prefix_len, extend_len, _CPU
            ),
            dim=0,
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
    mm_schedule._get_chunked_embedding_by_item(_encoder_list, items, 0, TOTAL_LEN, _CPU)
    for item in items:
        emb = mm_schedule.embedding_cache.get_single(item.hash).embedding
        own_bytes = emb.numel() * emb.element_size()
        assert emb.untyped_storage().nbytes() == own_bytes


@pytest.mark.parametrize("per_request", [False, True])
def test_tensor_cache_entries_own_storage(per_request):
    mm_schedule.init_mm_embedding_cache(1 << 30)
    items = _make_items()
    if per_request:
        chunks = mm_schedule._get_chunked_embedding_by_item(
            _encoder_tensor, items, 0, TOTAL_LEN, _CPU
        )
    else:
        request = mm_schedule.PerImageRequestInfo(
            req_idx=0,
            items=items,
            items_offset=ITEM_OFFSETS,
            extend_prefix_len=0,
            extend_seq_len=TOTAL_LEN,
        )
        embeddings = mm_schedule._batch_encode_per_image_misses(
            _encoder_tensor, [request], _CPU
        )
        chunks = mm_schedule._assemble_per_image_chunk(
            request.overlapping, embeddings, 0, TOTAL_LEN
        )
    for item, chunk in zip(items, chunks):
        emb = mm_schedule.embedding_cache.get_single(item.hash).embedding
        assert emb.untyped_storage().nbytes() == emb.numel() * emb.element_size()
        assert chunk.untyped_storage().data_ptr() == emb.untyped_storage().data_ptr()


@pytest.mark.parametrize("per_request", [False, True])
@pytest.mark.parametrize("cache_bytes", [0, 6 * HIDDEN * 4])
def test_tensor_encoder_chunks_survive_cache_pressure(
    monkeypatch, per_request, cache_bytes
):
    monkeypatch.setattr(mm_schedule, "_is_hip", per_request)
    mm_schedule.init_mm_embedding_cache(cache_bytes)
    items = _make_items()
    chunk, _ = mm_schedule._get_chunked_prefill_embedding(
        _encoder_tensor,
        items,
        [0, len(items)],
        [0],
        [TOTAL_LEN],
        [ITEM_OFFSETS],
        torch.zeros(TOTAL_LEN, dtype=torch.long),
    )
    assert torch.equal(chunk, _encoder_tensor(items))
    assert mm_schedule.embedding_cache.current_size <= cache_bytes


def test_by_item_mismatched_cache_entry_is_reencoded():
    mm_schedule.init_mm_embedding_cache(1 << 30)
    items = _make_items()
    first_item = items[0]
    mm_schedule.embedding_cache.set(
        first_item.hash,
        mm_schedule.EmbeddingResult(embedding=torch.zeros(1, HIDDEN)),
    )
    encoder = Mock(side_effect=_encoder_list)

    chunks = mm_schedule._get_chunked_embedding_by_item(
        encoder, items, 0, TOTAL_LEN, _CPU
    )

    assert sum(chunk.shape[0] for chunk in chunks) == sum(
        _num_tokens(item) for item in items
    )
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


def test_full_deferred_ipc_item_is_marked_for_borrow():
    mm_schedule.init_mm_embedding_cache(1 << 30)
    proxy = object.__new__(CudaIpcTensorTransportProxy)
    item = MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=1000,
        pad_value=1000,
        feature=proxy,
        offsets=[ITEM_OFFSETS[0]],
        model_specific_data={DEFER_CUDA_IPC_FEATURE_RECONSTRUCTION_KEY: True},
    )
    request = mm_schedule.PerImageRequestInfo(
        req_idx=0,
        items=[item],
        items_offset=[ITEM_OFFSETS[0]],
        extend_prefix_len=0,
        extend_seq_len=TOTAL_LEN,
    )

    mm_schedule._batch_encode_per_image_misses(_encoder_list, [request], _CPU)

    assert item.model_specific_data[BORROW_CUDA_IPC_FEATURE_KEY]


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


@pytest.mark.parametrize("per_request", [False, True])
def test_multi_offset_audio_item_is_chunked_and_cached_per_item(
    monkeypatch, per_request
):
    monkeypatch.setattr(mm_schedule, "_is_hip", per_request)
    mm_schedule.init_mm_embedding_cache(1 << 30)
    history_offsets = [(2, 3), (6, 7), (10, 11)]
    new_offsets = [(14, 15), (18, 19)]
    input_ids = torch.zeros(22, dtype=torch.long)

    def audio_item(hash_, offsets):
        return MultimodalDataItem(
            modality=Modality.AUDIO, hash=hash_, feature=torch.zeros(1), offsets=offsets
        )

    def audio_embedding(item):
        n = sum(end - start + 1 for start, end in item.offsets)
        return torch.randn(
            n, HIDDEN, generator=torch.Generator().manual_seed(item.hash)
        )

    def run(items, prefix_len, extend_len):
        chunk, _ = mm_schedule._get_chunked_prefill_embedding(
            encoder,
            items,
            [0, len(items)],
            [prefix_len],
            [extend_len],
            [[offset for item in items for offset in item.offsets]],
            input_ids,
        )
        return chunk

    encoder = Mock(side_effect=lambda items: [audio_embedding(i) for i in items])
    history = audio_item(7, history_offsets)
    expected = audio_embedding(history)

    # Turn 1, two prefill chunks split inside the item: tokens {2,3,6} | {7,10,11}.
    torch.testing.assert_close(run([history], 0, 7), expected[:3], rtol=0, atol=0)
    torch.testing.assert_close(run([history], 7, 6), expected[3:], rtol=0, atol=0)
    encoder.assert_called_once()

    # Turn 2 replays the history item (same hash); only the new item is encoded.
    new = audio_item(8, new_offsets)
    chunk = run([audio_item(7, history_offsets), new], 0, 22)
    torch.testing.assert_close(
        chunk, torch.cat([expected, audio_embedding(new)]), rtol=0, atol=0
    )
    assert encoder.call_count == 2
    assert encoder.call_args.args[0] == [new]


@pytest.mark.parametrize("per_request", [False, True])
@pytest.mark.parametrize("num_requests", [1, 2])
@pytest.mark.parametrize("num_items", [1, 2])
def test_cached_audio_items_concatenate_once_in_request_order(
    monkeypatch, per_request, num_requests, num_items
):
    monkeypatch.setattr(mm_schedule, "_is_hip", per_request)
    mm_schedule.init_mm_embedding_cache(1 << 30)
    items = []
    expected_slices = []
    offsets_per_item = [[(2, 3), (6, 7)], [(10, 11), (14, 15)]][:num_items]
    for req_idx in range(num_requests):
        for item_idx, offsets in enumerate(offsets_per_item):
            item = MultimodalDataItem(
                modality=Modality.AUDIO,
                hash=100 + 2 * req_idx + item_idx,
                offsets=offsets,
            )
            emb = torch.full((4, HIDDEN), item.hash, dtype=torch.bfloat16)
            mm_schedule.embedding_cache.set(
                item.hash, mm_schedule.EmbeddingResult(embedding=emb)
            )
            items.append(item)
            expected_slices.append(emb[1:] if item_idx == 0 else emb[:3])
    expected = torch.cat(expected_slices)
    cat = Mock(wraps=torch.cat)
    concat = Mock(wraps=torch.concat)
    monkeypatch.setattr(torch, "cat", cat)
    monkeypatch.setattr(torch, "concat", concat)
    encoder = Mock(side_effect=AssertionError("cached items should not be reencoded"))
    input_ids = torch.zeros(12 * num_requests, dtype=torch.long)

    chunk, returned_input_ids = mm_schedule._get_chunked_prefill_embedding(
        encoder,
        items,
        list(range(0, num_items * num_requests + 1, num_items)),
        [3] * num_requests,
        [12] * num_requests,
        [[offset for item_offsets in offsets_per_item for offset in item_offsets]]
        * num_requests,
        input_ids,
    )

    assert torch.equal(chunk, expected)
    assert returned_input_ids is input_ids
    encoder.assert_not_called()
    assert cat.call_count + concat.call_count == (num_items * num_requests > 1)


@pytest.mark.parametrize(
    "prefix, length, rows", [(3, 12, [1, 2, 3, 4, 5, 6]), (0, 0, []), (16, 4, [])]
)
def test_precomputed_audio_slices_concatenate_once(monkeypatch, prefix, length, rows):
    embeddings = torch.arange(8 * HIDDEN, dtype=torch.float32).reshape(8, HIDDEN)
    offsets = [[(2, 3), (6, 7)], [(10, 11), (14, 15)]]
    items = [
        MultimodalDataItem(
            modality=Modality.AUDIO,
            offsets=item_offsets,
            precomputed_embeddings=emb.reshape(2, 2, HIDDEN),
        )
        for item_offsets, emb in zip(offsets, embeddings.split(4))
    ]
    expected = torch.cat([embeddings[rows], embeddings[rows]])
    concat = Mock(wraps=torch.concat)
    monkeypatch.setattr(torch, "concat", concat)
    result = mm_schedule._get_precomputed_embedding(
        items * 2,
        [0, 2, 4],
        [prefix] * 2,
        [length] * 2,
        [[offset for item_offsets in offsets for offset in item_offsets]] * 2,
    )
    assert torch.equal(result, expected)
    concat.assert_called_once()
    singleton = mm_schedule._get_precomputed_embedding(
        items[:1],
        [0, 1],
        [prefix],
        [length],
        [offsets[0]],
    )
    assert torch.equal(singleton, embeddings[[row for row in rows if row < 4]])
    assert (
        singleton.untyped_storage().data_ptr()
        == embeddings.untyped_storage().data_ptr()
    )
    concat.assert_called_once()


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
