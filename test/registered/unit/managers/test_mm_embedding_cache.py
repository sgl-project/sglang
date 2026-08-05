from __future__ import annotations

from unittest.mock import patch

import torch

from sglang.srt.managers.mm_utils import (
    _get_chunked_embedding_full,
    _get_chunked_prefill_embedding,
    init_mm_embedding_cache,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.evs import EVSEmbeddingResult
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _item(
    *, cacheable: bool, item_hash: int = 1234, offset: tuple[int, int] = (0, 0)
) -> MultimodalDataItem:
    return MultimodalDataItem(
        modality=Modality.AUDIO,
        hash=item_hash,
        offsets=[offset],
        feature=torch.tensor([0.5]),
        embedding_cacheable=cacheable,
    )


def _embed_once(item: MultimodalDataItem, encoder):
    embedding, _ = _get_chunked_prefill_embedding(
        data_embedding_func=encoder,
        embedding_items=[item],
        items_size=[0, 1],
        prefix_length=[0],
        extend_length=[1],
        items_offset_list=[[(0, 0)]],
        input_ids=torch.tensor([0]),
    )
    return embedding


def test_embedding_cacheable_defaults_to_true() -> None:
    item = MultimodalDataItem(modality=Modality.IMAGE)
    assert item.embedding_cacheable


def test_chunked_embedding_cache_reuses_stateless_items() -> None:
    init_mm_embedding_cache(1 << 20)
    calls = 0

    def encoder(_items):
        nonlocal calls
        calls += 1
        return torch.tensor([[float(calls)]])

    first = _embed_once(_item(cacheable=True), encoder)
    second = _embed_once(_item(cacheable=True), encoder)

    assert calls == 1
    assert torch.equal(first, second)


def test_chunked_embedding_cache_never_skips_stateful_items() -> None:
    init_mm_embedding_cache(1 << 20)
    calls = 0

    def stateful_encoder(_items):
        nonlocal calls
        calls += 1
        return torch.tensor([[float(calls)]])

    first = _embed_once(_item(cacheable=False), stateful_encoder)
    second = _embed_once(_item(cacheable=False), stateful_encoder)

    assert calls == 2
    assert torch.equal(first, torch.tensor([[1.0]]))
    assert torch.equal(second, torch.tensor([[2.0]]))


def test_mixed_cacheability_reuses_only_stateless_item() -> None:
    init_mm_embedding_cache(1 << 20)
    encoded_hashes = []
    items = [
        _item(cacheable=True, item_hash=1, offset=(0, 0)),
        _item(cacheable=False, item_hash=2, offset=(1, 1)),
    ]

    def encoder(misses):
        encoded_hashes.append([item.hash for item in misses])
        return torch.tensor([[float(item.hash)] for item in misses])

    def embed():
        result, _ = _get_chunked_prefill_embedding(
            data_embedding_func=encoder,
            embedding_items=items,
            items_size=[0, 2],
            prefix_length=[0],
            extend_length=[2],
            items_offset_list=[[(0, 0), (1, 1)]],
            input_ids=torch.tensor([0, 0]),
        )
        return result

    assert torch.equal(embed(), torch.tensor([[1.0], [2.0]]))
    assert torch.equal(embed(), torch.tensor([[1.0], [2.0]]))
    assert encoded_hashes == [[1, 2], [2]]


def test_noncacheable_item_does_not_read_same_hash_cached_embedding() -> None:
    init_mm_embedding_cache(1 << 20)
    calls = 0

    def encoder(_items):
        nonlocal calls
        calls += 1
        return torch.tensor([[float(calls)]])

    assert torch.equal(
        _embed_once(_item(cacheable=True), encoder), torch.tensor([[1.0]])
    )
    assert torch.equal(
        _embed_once(_item(cacheable=False), encoder), torch.tensor([[2.0]])
    )


def test_combined_path_bypasses_cache_if_any_item_is_stateful() -> None:
    init_mm_embedding_cache(1 << 20)
    calls = 0
    item = _item(cacheable=False)
    item.offsets = [(0, 0), (1, 1)]

    def encoder(_items):
        nonlocal calls
        calls += 1
        return torch.tensor([[float(calls)], [float(calls)]])

    def embed():
        result, _ = _get_chunked_prefill_embedding(
            data_embedding_func=encoder,
            embedding_items=[item],
            items_size=[0, 1],
            prefix_length=[0],
            extend_length=[2],
            items_offset_list=[[(0, 0), (1, 1)]],
            input_ids=torch.tensor([0, 0]),
        )
        return result

    assert torch.equal(embed(), torch.tensor([[1.0], [1.0]]))
    assert torch.equal(embed(), torch.tensor([[2.0], [2.0]]))
    assert calls == 2


def test_evs_combined_result_obeys_cacheability() -> None:
    init_mm_embedding_cache(1 << 20)
    item = _item(cacheable=False)
    item.offsets = [(0, 0), (1, 1)]
    calls = 0

    def encoder(_items):
        nonlocal calls
        calls += 1
        return EVSEmbeddingResult(
            embedding=torch.tensor([[float(calls)]]),
            num_tokens_per_frame=[1],
        )

    with patch.object(
        EVSEmbeddingResult,
        "redistribute_pruned_frames_placeholders",
        return_value=(torch.tensor([0]), [(0, 0)]),
    ):
        first, _ = _get_chunked_embedding_full(
            encoder, [item], [(0, 0)], 0, 1, torch.tensor([0]), torch.device("cpu")
        )
        second, _ = _get_chunked_embedding_full(
            encoder, [item], [(0, 0)], 0, 1, torch.tensor([0]), torch.device("cpu")
        )

    assert torch.equal(first, torch.tensor([[1.0]]))
    assert torch.equal(second, torch.tensor([[2.0]]))
