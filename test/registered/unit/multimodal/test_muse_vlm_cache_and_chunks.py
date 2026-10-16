from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from sglang.srt.managers import mm_schedule
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult, MultiModalStaticCache
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


@pytest.mark.parametrize("view", [False, True])
def test_oversized_rejection_preserves_cache_and_lru(view):
    cache = MultiModalStaticCache(32)
    for key in (1, 2):
        assert cache.set(key, EmbeddingResult(embedding=torch.zeros(4)))
    cache.get_single(1)
    tensor = torch.zeros(100)
    if view:
        tensor = tensor[:9]  # Still exceeds budget after compacting its storage.
    assert not cache.set(3, EmbeddingResult(embedding=tensor))
    assert list(cache.mm_cache) == [2, 1]
    assert cache.current_size == 32
    assert cache.set(4, EmbeddingResult(embedding=torch.zeros(4)))
    assert list(cache.mm_cache) == [1, 4]
    assert cache.current_size == 32


def test_small_view_of_large_storage_is_still_admitted():
    cache = MultiModalStaticCache(32)
    tensor = torch.arange(100, dtype=torch.float32)[:8]
    assert cache.set(1, EmbeddingResult(embedding=tensor))
    cached = cache.get_single(1).embedding
    torch.testing.assert_close(cached, tensor)
    assert cached.untyped_storage().nbytes() == cache.current_size == 32


class VideoEncoder:
    def __init__(self):
        self.calls = []

    def get_video_feature(self, items):
        self.calls.append([item.hash for item in items])
        return torch.cat([item.feature for item in items])


def video_items():
    return [
        SimpleNamespace(
            hash=1, offsets=[(10, 11), (15, 16)], feature=torch.arange(4.0).view(-1, 1)
        ),
        SimpleNamespace(
            hash=2,
            offsets=[(30, 31), (35, 36)],
            feature=torch.arange(4.0, 8.0).view(-1, 1),
        ),
    ]


def run_chunk(model, items, start, length):
    return mm_schedule._get_chunked_prefill_embedding(
        model.get_video_feature,
        items,
        [0, len(items)],
        [start],
        [length],
        [[offset for item in items for offset in item.offsets]],
        torch.zeros(length, dtype=torch.long),
    )[0]


@pytest.mark.parametrize(
    "start,length,expected,calls",
    [
        (0, 10, None, []),
        (12, 3, None, []),
        (17, 13, None, []),
        (37, 10, None, []),
        (11, 5, [1.0, 2.0], [[1, 2]]),
        (16, 15, [3.0, 4.0], [[1, 2]]),
        (31, 5, [5.0, 6.0], [[1, 2]]),
    ],
)
def test_video_chunk_intersections(start, length, expected, calls):
    model = VideoEncoder()
    with patch.object(mm_schedule, "embedding_cache", MultiModalStaticCache(1024)):
        actual = run_chunk(model, video_items(), start, length)
    assert model.calls == calls
    if expected is None:
        assert actual is None or actual.numel() == 0
    else:
        torch.testing.assert_close(actual.flatten(), torch.tensor(expected))


def test_video_cache_reuses_full_item_across_its_spans():
    model = VideoEncoder()
    with (
        patch.object(mm_schedule, "embedding_cache", MultiModalStaticCache(32)),
        patch.object(mm_schedule, "_acknowledge_deferred_cuda_ipc_cache_hits") as ack,
    ):
        first = run_chunk(model, video_items(), 10, 2)
        second = run_chunk(model, video_items(), 15, 2)
    torch.testing.assert_close(first.flatten(), torch.tensor([0.0, 1.0]))
    torch.testing.assert_close(second.flatten(), torch.tensor([2.0, 3.0]))
    assert model.calls == [[1, 2]]
    ack.assert_called_once()


def test_multispan_encoders_keep_full_request_contract():
    model = VideoEncoder()
    with patch.object(mm_schedule, "embedding_cache", MultiModalStaticCache(1024)):
        actual = run_chunk(model, video_items(), 31, 5)
    assert model.calls == [[1, 2]]
    torch.testing.assert_close(actual.flatten(), torch.tensor([5.0, 6.0]))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
