"""Unit tests for per-item DataEmbeddingFunc results in the chunked mm path.

A DataEmbeddingFunc may return either one combined [tokens, hidden] tensor or
one tensor per item (see mm_schedule.DataEmbeddingFunc). These tests assert the
two forms produce bitwise-identical chunked-prefill embeddings, and that the
per-item form yields cache entries that own their storage (a torch.split view
of the combined tensor pins the whole concatenated buffer).

CPU-only: exercises mm_schedule internals directly, no engine or GPU.
"""

import logging
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from sglang.srt.managers import mm_schedule
from sglang.srt.managers.mm_utils import embed_mm_inputs
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


def _join(segments):
    return torch.cat(segments, dim=0) if segments else None


def _run_by_item_chunks(encoder):
    mm_schedule.init_mm_embedding_cache(1 << 30)
    items = _make_items()
    return [
        _join(
            mm_schedule._get_chunked_embedding_by_item(
                encoder, items, ITEM_OFFSETS, prefix_len, extend_len, _CPU
            )
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


def test_tensor_cache_entries_own_storage():
    mm_schedule.init_mm_embedding_cache(1 << 30)
    items = _make_items()
    mm_schedule._get_chunked_embedding_by_item(
        _encoder_tensor, items, ITEM_OFFSETS, 0, TOTAL_LEN, _CPU
    )
    for item in items:
        emb = mm_schedule.embedding_cache.get_single(item.hash).embedding
        assert emb.untyped_storage().nbytes() == emb.numel() * emb.element_size()


def test_by_item_mismatched_cache_entry_is_reencoded():
    mm_schedule.init_mm_embedding_cache(1 << 30)
    items = _make_items()
    first_item = items[0]
    mm_schedule.embedding_cache.set(
        first_item.hash,
        mm_schedule.EmbeddingResult(embedding=torch.zeros(1, HIDDEN)),
    )
    encoder = Mock(side_effect=_encoder_list)

    chunk = _join(
        mm_schedule._get_chunked_embedding_by_item(
            encoder, items, ITEM_OFFSETS, 0, TOTAL_LEN, _CPU
        )
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


# (modality, item offsets, chunk window) per request. Windows start and end
# inside media spans, overlap no media, or sit inside one span.
_IMG, _VID = Modality.IMAGE, Modality.VIDEO
E2E_REQUESTS = [
    ([(_IMG, [(2, 5)]), (_IMG, [(9, 14)]), (_IMG, [(20, 24)])], (4, 18)),
    ([(_IMG, [(2, 5)]), (_IMG, [(20, 24)])], (8, 8)),
    ([(_IMG, [(0, 29)])], (10, 12)),
    ([(_IMG, [(3, 6)])], (0, 10)),
]
# Items with two placeholder spans take the full (non per-image) path, which
# encodes one request at a time.
E2E_FULL_PATH_REQUESTS = [
    ([(_IMG, [(1, 3), (6, 9)])], (2, 10)),
    ([(_IMG, [(0, 2), (5, 8)])], (0, 10)),
]
E2E_VIDEO_REQUEST = ([(_VID, [(1, 4)])], (0, 8))
VOCAB = 64


def _sentinel_rows(item_index: int, num_rows: int, width: int) -> torch.Tensor:
    # Row r of item i holds 1000 * (i + 1) + r, negated in the deepstack half.
    rows = torch.arange(num_rows, dtype=torch.float32) + 1000 * (item_index + 1)
    rows = rows[:, None].expand(num_rows, width).contiguous()
    rows[:, HIDDEN:] *= -1
    return rows


class _DeepstackModel:
    deepstack_visual_indexes = [0]

    @staticmethod
    def separate_deepstack_embeds(embedding):
        return embedding[:, :HIDDEN], embedding[:, HIDDEN:]


@pytest.mark.parametrize(
    "scenario",
    ["per_image", "by_item", "full_path", "mixed_modality", "deepstack", "precomputed"],
)
def test_embed_mm_inputs_matches_whole_sequence_reference(scenario, monkeypatch):
    """Chunked assembly must place every media row where a whole-sequence merge
    would on each assembly path, and must leave cached and precomputed embeddings
    unchanged."""
    requests = list(E2E_REQUESTS)
    if scenario == "full_path":
        requests += E2E_FULL_PATH_REQUESTS
    if scenario == "mixed_modality":
        requests.append(E2E_VIDEO_REQUEST)
    if scenario == "by_item":
        monkeypatch.setattr(mm_schedule, "_is_hip", True)
    deepstack = scenario == "deepstack"
    width = 2 * HIDDEN if deepstack else HIDDEN

    mm_schedule.init_mm_embedding_cache(1 << 30)
    text_embedding = torch.nn.Embedding(VOCAB, HIDDEN)
    with torch.no_grad():
        text_embedding.weight.copy_(-torch.arange(1, VOCAB + 1.0)[:, None])

    gen = torch.Generator().manual_seed(0)
    mm_inputs_list, windows, input_ids = [], [], []
    reference, deepstack_reference = [], []
    rows_by_hash, encoded_items = {}, []
    item_index = 0
    for item_specs, (prefix_len, extend_len) in requests:
        seq_len = max(
            prefix_len + extend_len,
            max(end + 1 for _, spans in item_specs for _, end in spans),
        )
        ids = torch.randint(0, VOCAB, (seq_len,), generator=gen)
        full = text_embedding(ids).detach()
        full_deepstack = torch.zeros(seq_len, width - HIDDEN)
        items = []
        for modality, spans in item_specs:
            item = MultimodalDataItem(
                modality=modality, feature=torch.zeros(1), offsets=list(spans)
            )
            item.set_hash(5000 + item_index)
            rows = _sentinel_rows(
                item_index, sum(end - start + 1 for start, end in spans), width
            )
            rows_by_hash[item.hash] = rows
            if scenario == "precomputed":
                item.precomputed_embeddings = rows.clone()
            row = 0
            for start, end in spans:
                num_rows = end - start + 1
                ids[start : end + 1] = item.pad_value
                full[start : end + 1] = rows[row : row + num_rows, :HIDDEN]
                full_deepstack[start : end + 1] = rows[row : row + num_rows, HIDDEN:]
                row += num_rows
            items.append(item)
            item_index += 1
            if any(
                end >= prefix_len and start < prefix_len + extend_len
                for start, end in spans
            ):
                encoded_items.append(item)
        mm_inputs_list.append(SimpleNamespace(mm_items=items))
        windows.append((prefix_len, extend_len))
        input_ids.append(ids[prefix_len : prefix_len + extend_len])
        reference.append(full[prefix_len : prefix_len + extend_len])
        deepstack_reference.append(full_deepstack[prefix_len : prefix_len + extend_len])

    def encoder(items):
        return torch.cat([rows_by_hash[item.hash] for item in items])

    actual, other_info = embed_mm_inputs(
        mm_inputs_list=mm_inputs_list,
        extend_prefix_lens=[prefix for prefix, _ in windows],
        extend_seq_lens=[extend for _, extend in windows],
        input_ids=torch.cat(input_ids),
        input_embedding=text_embedding,
        multimodal_model=_DeepstackModel() if deepstack else None,
        data_embedding_func_mapping={_IMG: encoder, _VID: encoder},
        use_deepstack={_IMG: True} if deepstack else {},
    )

    torch.testing.assert_close(actual, torch.cat(reference), rtol=0, atol=0)
    if deepstack:
        torch.testing.assert_close(
            other_info["input_deepstack_embeds"],
            torch.cat(deepstack_reference),
            rtol=0,
            atol=0,
        )
    for item in encoded_items:
        rows = rows_by_hash[item.hash]
        if scenario == "precomputed":
            torch.testing.assert_close(
                item.precomputed_embeddings, rows, rtol=0, atol=0
            )
            continue
        cached = mm_schedule.embedding_cache.get_single(item.hash)
        if cached is None:
            # Multi-span items are cached under the combined request hash.
            cached = mm_schedule.embedding_cache.get([item.hash])
        assert cached is not None
        torch.testing.assert_close(cached.embedding, rows, rtol=0, atol=0)


def test_per_image_cache_hits_reach_the_scatter_as_views():
    """Cache-resident per-image embeddings must reach the scatter as views of the
    cache entries; a per-request or per-batch copy adds a transient copy of every
    media row in the prefill chunk."""
    mm_schedule.init_mm_embedding_cache(1 << 30)
    items = _make_items()
    input_ids = torch.zeros(TOTAL_LEN, dtype=torch.long)
    for item in items:
        item.set_hash(item.hash)
        start, end = item.offsets[0]
        input_ids[start : end + 1] = item.pad_value
        mm_schedule.embedding_cache.set(
            item.hash, mm_schedule.EmbeddingResult(embedding=_item_embedding(item))
        )
    cache_storages = {
        mm_schedule.embedding_cache.get_single(item.hash)
        .embedding.untyped_storage()
        .data_ptr()
        for item in items
    }

    segments, _, _ = mm_schedule.get_embedding_and_mask(
        data_embedding_func=Mock(side_effect=AssertionError("cache miss")),
        embedding_items=items,
        placeholder_tensor=torch.tensor([item.pad_value for item in items]),
        input_ids=input_ids[4:22],
        items_size=[0, len(items)],
        prefix_length=[4],
        extend_length=[18],
        items_offset_list=[ITEM_OFFSETS],
    )

    assert len(segments) == len(items)
    assert {seg.untyped_storage().data_ptr() for seg in segments} == cache_storages


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
