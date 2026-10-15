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
from sglang.srt.managers.schedule_batch import (
    PREFILL_ROWS_KEY,
    Modality,
    MultimodalDataItem,
)
from sglang.srt.multimodal.evs import EVSEmbeddingResult
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
    # Exercise the embedding path without starting an engine.
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


# One video: four 4-token frames with separators; the chunks split frames 2 and 4.
VIDEO_RUNS = [(2, 5), (7, 10), (12, 15), (17, 20)]
VIDEO_LEN = 24
VIDEO_CHUNKS = [(0, 9), (9, 9), (18, 6)]
VIDEO_TOKENS = sum(end - start + 1 for start, end in VIDEO_RUNS)


def _video_item(hash_=77):
    return MultimodalDataItem(
        modality=Modality.VIDEO,
        hash=hash_,
        feature=torch.zeros(1),
        offsets=list(VIDEO_RUNS),
    )


def _video_embedding(item):
    return torch.randn(
        VIDEO_TOKENS, HIDDEN, generator=torch.Generator().manual_seed(item.hash)
    )


def _video_encoder():
    return Mock(
        side_effect=lambda items: torch.cat([_video_embedding(i) for i in items])
    )


def _prefill_chunk(encoder, item, prefix_len, extend_len):
    chunk, _ = mm_schedule._get_chunked_prefill_embedding(
        encoder,
        [item],
        [0, 1],
        [prefix_len],
        [extend_len],
        [item.offsets],
        torch.zeros(VIDEO_LEN, dtype=torch.long),
    )
    return chunk


def test_multi_run_item_is_encoded_once_per_request():
    mm_schedule.init_mm_embedding_cache(0)
    item = _video_item()
    encoder = _video_encoder()

    chunks = [
        _prefill_chunk(encoder, item, prefix_len, extend_len)
        for prefix_len, extend_len in VIDEO_CHUNKS
    ]

    encoder.assert_called_once()
    assert [chunk.shape[0] for chunk in chunks] == [6, 7, 3]
    assert torch.equal(torch.cat(chunks), _video_embedding(item))
    assert PREFILL_ROWS_KEY not in item.model_specific_data


def test_prefill_embedding_is_kept_only_while_a_later_chunk_needs_it():
    mm_schedule.init_mm_embedding_cache(0)
    encoder = _video_encoder()

    one_chunk = _video_item()
    _prefill_chunk(encoder, one_chunk, 0, VIDEO_LEN)
    assert PREFILL_ROWS_KEY not in one_chunk.model_specific_data

    # The video ends inside the first chunk of a longer prompt.
    ends_early = _video_item()
    _prefill_chunk(encoder, ends_early, 0, 21)
    assert PREFILL_ROWS_KEY not in ends_early.model_specific_data
    assert _prefill_chunk(encoder, ends_early, 21, 3) is None

    split = _video_item()
    _prefill_chunk(encoder, split, 0, 9)
    assert PREFILL_ROWS_KEY in split.model_specific_data
    assert encoder.call_count == 3


def test_pruned_video_is_encoded_once_per_request():
    mm_schedule.init_mm_embedding_cache(0)
    prompt = [1, 0, 0, 4, 5, 0, 0, 0, 9, 10, 0, 0, 12, 13]
    item = MultimodalDataItem(
        modality=Modality.VIDEO,
        hash=5,
        pad_value=0,
        feature=torch.zeros(1),
        offsets=[(1, 2), (5, 7), (10, 11)],
        model_specific_data={"pre_chunked_input_ids": prompt},
    )
    rows = torch.arange(7 * HIDDEN, dtype=torch.float32).reshape(7, HIDDEN)
    encoder = Mock(
        return_value=EVSEmbeddingResult(embedding=rows, num_tokens_per_frame=[1, 4, 2])
    )

    outs = []
    for prefix_len, extend_len in ((0, 4), (4, 5), (9, 5)):
        outs.append(
            mm_schedule._get_chunked_embedding_full(
                encoder,
                [item],
                item.offsets,
                prefix_len,
                extend_len,
                torch.tensor(prompt[prefix_len : prefix_len + extend_len]),
                _CPU,
            )
        )

    encoder.assert_called_once()
    assert [chunk.shape[0] for chunk, _ in outs] == [1, 4, 2]
    assert torch.equal(torch.cat([chunk for chunk, _ in outs]), rows)
    # Pruning moved the placeholders: [1, 0, 4, 5, 0, 0, 0, 0, 9, 10, 0, 0, 12, 13].
    assert outs[0][1].tolist() == [1, 0, 4, 5]
    assert outs[1][1].tolist() == [0, 0, 0, 0, 9]
    assert outs[2][1].tolist() == [10, 0, 0, 12, 13]
    assert PREFILL_ROWS_KEY not in item.model_specific_data


def _run_item(hash_=88, modality=Modality.VIDEO, offsets=((3, 18),)):
    return MultimodalDataItem(
        modality=modality, hash=hash_, feature=torch.zeros(1), offsets=list(offsets)
    )


@pytest.mark.parametrize("per_request", [False, True])
def test_single_run_item_is_encoded_once_per_request(monkeypatch, per_request):
    monkeypatch.setattr(mm_schedule, "_is_hip", per_request)
    mm_schedule.init_mm_embedding_cache(0)
    item = _run_item()
    encoder = _video_encoder()

    chunks = [
        _prefill_chunk(encoder, item, prefix_len, extend_len)
        for prefix_len, extend_len in VIDEO_CHUNKS
    ]

    encoder.assert_called_once()
    assert [chunk.shape[0] for chunk in chunks] == [6, 9, 1]
    assert torch.equal(torch.cat(chunks), _video_embedding(item))
    assert PREFILL_ROWS_KEY not in item.model_specific_data


def test_several_multi_run_items_are_encoded_once_per_request():
    # The fallback path keeps the rows of both videos on the first item.
    mm_schedule.init_mm_embedding_cache(0)
    first = _video_item(71)
    second = MultimodalDataItem(
        modality=Modality.VIDEO,
        hash=72,
        feature=torch.zeros(1),
        offsets=[(start + 22, end + 22) for start, end in VIDEO_RUNS],
    )
    items = [first, second]
    offsets = first.offsets + second.offsets
    encoder = _video_encoder()
    expected = torch.cat([_video_embedding(first), _video_embedding(second)])

    chunks = []
    for prefix_len in range(0, 44, 11):
        chunk, _ = mm_schedule._get_chunked_prefill_embedding(
            encoder,
            items,
            [0, 2],
            [prefix_len],
            [11],
            [offsets],
            torch.zeros(1, dtype=torch.long),
        )
        chunks.append(chunk)
        assert PREFILL_ROWS_KEY not in second.model_specific_data
        assert (PREFILL_ROWS_KEY in first.model_specific_data) == (prefix_len < 33)

    encoder.assert_called_once()
    assert torch.equal(torch.cat(chunks), expected)


def test_item_shared_by_two_requests_in_one_batch():
    # Session requests share item objects and may read the same kept rows.
    mm_schedule.init_mm_embedding_cache(0)
    item = _run_item()
    encoder = _video_encoder()
    _prefill_chunk(encoder, item, 0, 9)

    chunk, _ = mm_schedule._get_chunked_prefill_embedding(
        encoder,
        [item, item],
        [0, 1, 2],
        [9, 9],
        [VIDEO_LEN - 9, VIDEO_LEN - 9],
        [item.offsets, item.offsets],
        torch.zeros(1, dtype=torch.long),
    )

    unread = _video_embedding(item)[6:]
    assert torch.equal(chunk, torch.cat([unread, unread]))
    encoder.assert_called_once()
    assert PREFILL_ROWS_KEY not in item.model_specific_data


@pytest.mark.parametrize("make_item", [_video_item, _run_item])
def test_embedding_evicted_between_chunks_is_not_encoded_again(make_item):
    mm_schedule.init_mm_embedding_cache(1 << 30)
    item = make_item()
    encoder = _video_encoder()

    chunks = [_prefill_chunk(encoder, item, *VIDEO_CHUNKS[0])]
    mm_schedule.embedding_cache.clear()
    chunks += [
        _prefill_chunk(encoder, item, prefix_len, extend_len)
        for prefix_len, extend_len in VIDEO_CHUNKS[1:]
    ]

    encoder.assert_called_once()
    assert torch.equal(torch.cat(chunks), _video_embedding(item))
    assert PREFILL_ROWS_KEY not in item.model_specific_data


def test_request_that_starts_over_is_encoded_again():
    # The dropped rows are gone, so a prefill restarted at token 0 re-encodes.
    mm_schedule.init_mm_embedding_cache(0)
    item = _video_item()
    kept_during_encode = []

    def encode(items):
        kept_during_encode.append(PREFILL_ROWS_KEY in item.model_specific_data)
        return torch.cat([_video_embedding(i) for i in items])

    encoder = Mock(side_effect=encode)

    _prefill_chunk(encoder, item, *VIDEO_CHUNKS[0])
    chunks = [
        _prefill_chunk(encoder, item, prefix_len, extend_len)
        for prefix_len, extend_len in VIDEO_CHUNKS
    ]

    assert encoder.call_count == 2
    assert kept_during_encode == [False, False]
    assert torch.equal(torch.cat(chunks), _video_embedding(item))
    assert PREFILL_ROWS_KEY not in item.model_specific_data


def test_keeping_rows_leaves_the_cache_order_alone():
    # Keeping the first item's rows must not move it in the cache's LRU order.
    mm_schedule.init_mm_embedding_cache(1 << 30)
    items = [
        MultimodalDataItem(
            modality=Modality.AUDIO, hash=hash_, feature=torch.zeros(1), offsets=offsets
        )
        for hash_, offsets in ((11, [(2, 3), (10, 11)]), (12, [(6, 7)]))
    ]

    def encode(batch):
        return [
            torch.ones(sum(e - s + 1 for s, e in item.offsets), HIDDEN)
            for item in batch
        ]

    mm_schedule._get_chunked_prefill_embedding(
        encode,
        items,
        [0, 2],
        [0],
        [9],
        [[(2, 3), (6, 7), (10, 11)]],
        torch.zeros(1, dtype=torch.long),
    )

    assert list(mm_schedule.embedding_cache.mm_cache) == [11, 12]
    assert PREFILL_ROWS_KEY in items[0].model_specific_data


def test_rows_in_a_buffer_the_encoder_reuses_are_copied():
    # The cache holds a clone; the next encoder call overwrites the encoder's slice.
    mm_schedule.init_mm_embedding_cache(1 << 30)
    buffer = torch.empty(2 * VIDEO_TOKENS, HIDDEN)

    def encode(items):
        buffer[:VIDEO_TOKENS] = _video_embedding(items[0])
        return buffer[:VIDEO_TOKENS]

    item, other = _video_item(), _video_item(78)
    chunks = [_prefill_chunk(encode, item, *VIDEO_CHUNKS[0]).clone()]
    _prefill_chunk(encode, other, 0, VIDEO_LEN)
    chunks += [
        _prefill_chunk(encode, item, prefix_len, extend_len)
        for prefix_len, extend_len in VIDEO_CHUNKS[1:]
    ]

    assert torch.equal(torch.cat(chunks), _video_embedding(item))


def test_kept_rows_are_read_before_the_cache():
    # A colliding cache entry appears between two chunks; the kept rows still serve.
    mm_schedule.init_mm_embedding_cache(1 << 30)
    item = _video_item()
    encoder = _video_encoder()
    cache = mm_schedule.embedding_cache

    chunks = [_prefill_chunk(encoder, item, *VIDEO_CHUNKS[0])]
    (cache_hash,) = cache.mm_cache
    cache.clear()
    cache.set(cache_hash, mm_schedule.EmbeddingResult(embedding=torch.zeros(3, HIDDEN)))
    chunks += [
        _prefill_chunk(encoder, item, prefix_len, extend_len)
        for prefix_len, extend_len in VIDEO_CHUNKS[1:]
    ]

    encoder.assert_called_once()
    assert torch.equal(torch.cat(chunks), _video_embedding(item))
    assert PREFILL_ROWS_KEY not in item.model_specific_data


def test_kept_rows_are_not_served_to_a_request_with_another_layout():
    # The same clip at another position: rows one dropped are unread for the other.
    mm_schedule.init_mm_embedding_cache(0)
    item = _video_item()
    shifted = [(start + 3, end + 3) for start, end in VIDEO_RUNS]
    encoder = _video_encoder()
    expected = _video_embedding(item)

    chunks = [_prefill_chunk(encoder, item, *VIDEO_CHUNKS[0])]
    other, _ = mm_schedule._get_chunked_prefill_embedding(
        encoder,
        [item],
        [0, 1],
        [9],
        [VIDEO_LEN + 3 - 9],
        [shifted],
        torch.zeros(1, dtype=torch.long),
    )
    chunks += [
        _prefill_chunk(encoder, item, prefix_len, extend_len)
        for prefix_len, extend_len in VIDEO_CHUNKS[1:]
    ]

    assert encoder.call_count == 2
    assert torch.equal(other, expected[4:])
    assert torch.equal(torch.cat(chunks), expected)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
