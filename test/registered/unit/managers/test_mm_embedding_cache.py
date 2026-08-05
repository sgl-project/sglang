"""Focused tests for per-item multimodal embedding-cache behavior."""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest
from unittest.mock import patch

import torch

from sglang.srt.managers.mm_schedule import (
    _batch_encode_per_item_misses,
    _get_chunked_embedding_by_item,
    _PerItemRequestInfo,
)
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestMMEmbeddingCache(CustomTestCase):
    def test_noncacheable_item_is_reencoded(self):
        stable = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=1,
            offsets=[(0, 1)],
            feature=torch.ones(2, 3),
        )
        tail = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=2,
            offsets=[(2, 2)],
            feature=torch.ones(1, 3),
            use_embedding_cache=False,
        )
        encoded_item_counts = []

        def encode(items):
            encoded_item_counts.append(len(items))
            return torch.cat([item.feature for item in items])

        args = (encode, [stable, tail], [(0, 1), (2, 2)], 0, 3, torch.device("cpu"))
        cache = MultiModalStaticCache(1024 * 1024)
        with patch("sglang.srt.managers.mm_schedule.embedding_cache", cache):
            first = _get_chunked_embedding_by_item(*args)
            second = _get_chunked_embedding_by_item(*args)

        self.assertTrue(torch.equal(first, second))
        self.assertEqual(encoded_item_counts, [2, 1])
        self.assertTrue(cache.has(stable.hash))
        self.assertFalse(cache.has(tail.hash))

    def test_stale_geometry_is_reencoded(self):
        cache = MultiModalStaticCache(1024 * 1024)
        encoded_lengths = []

        def encode(items):
            encoded_lengths.append(items[0].feature.shape[0])
            return items[0].feature

        def run(token_count):
            item = MultimodalDataItem(
                modality=Modality.AUDIO,
                hash=7,
                offsets=[(0, token_count - 1)],
                feature=torch.ones(token_count, 3),
            )
            return _get_chunked_embedding_by_item(
                encode,
                [item],
                item.offsets,
                0,
                token_count,
                torch.device("cpu"),
            )

        with patch("sglang.srt.managers.mm_schedule.embedding_cache", cache):
            self.assertEqual(run(1).shape[0], 1)
            self.assertEqual(run(2).shape[0], 2)

        self.assertEqual(encoded_lengths, [1, 2])

    def test_cached_split_owns_its_storage(self):
        stable = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=1,
            offsets=[(0, 1)],
            feature=torch.ones(2, 3),
        )
        tail = MultimodalDataItem(
            modality=Modality.AUDIO,
            hash=2,
            offsets=[(2, 2)],
            feature=torch.ones(1, 3),
            use_embedding_cache=False,
        )
        encoder_outputs = []

        def encode(items):
            output = torch.cat([item.feature for item in items])
            encoder_outputs.append(output)
            return output

        request = _PerItemRequestInfo(
            req_idx=0,
            items=[stable, tail],
            items_offset=[(0, 1), (2, 2)],
            extend_prefix_len=0,
            extend_seq_len=3,
        )
        cache = MultiModalStaticCache(1024 * 1024)
        with patch("sglang.srt.managers.mm_schedule.embedding_cache", cache):
            _batch_encode_per_item_misses(encode, [request], torch.device("cpu"))

        cached = cache.get_single(stable.hash).embedding
        self.assertNotEqual(
            cached.untyped_storage().data_ptr(),
            encoder_outputs[0].untyped_storage().data_ptr(),
        )
        self.assertEqual(cached.untyped_storage().nbytes(), cached.nbytes)


if __name__ == "__main__":
    unittest.main()
