"""Regression tests for per-item prefix caching in the EPD encoder.

The prefix cache used to require a single-request batch and a whole-request
combined hash.  A repeated image in a different request or in a fused batch
therefore missed even though its item was already in the cache.
"""

import asyncio
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.disaggregation.encoder.server as encoder_server
from sglang.srt.disaggregation.encoder.server import MMEncoder
from sglang.srt.mem_cache.multimodal_cache import EmbeddingResult
from sglang.srt.multimodal.modality import Modality
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _FakeItem:
    def __init__(self, cache_key: bytes, rows):
        self.cache_key = cache_key
        self.embedding = torch.tensor(rows, dtype=torch.float32)
        self.padded = False

    def set_pad_value(self):
        self.padded = True


class _FeatureFn:
    def __init__(self):
        self.calls = 0
        self.last_keys = None

    def __call__(self, items):
        self.calls += 1
        self.last_keys = [item.cache_key for item in items]
        return torch.cat([item.embedding for item in items], dim=0)


def _ctx(items, token_counts, feature_fn, items_per_req=None):
    return SimpleNamespace(
        modality=Modality.IMAGE,
        mm_feature=None,
        num_items=len(items),
        items_per_req=items_per_req or [1] * len(items),
        preprocess_result=SimpleNamespace(
            token_counts=list(token_counts), mm_inputs={}, items=list(items)
        ),
        get_feature_fn=feature_fn,
        is_health_check=False,
        aux_data={},
    )


def _encoder():
    encoder = MMEncoder.__new__(MMEncoder)
    encoder.mm_cache = encoder_server.MultiModalStaticCache(1 << 20)
    encoder.mm_cache_lock = asyncio.Lock()
    encoder._build_model_mm_items = lambda *args: list(args[1].items)
    return encoder


class TestEncoderPrefixCacheBatch(CustomTestCase):
    def _run(self, coro):
        return asyncio.run(coro)

    def test_reordered_items_hit_the_per_item_cache(self):
        a = _FakeItem(b"a", [[1.0, 1.1], [1.2, 1.3]])
        b = _FakeItem(b"b", [[2.0, 2.1], [2.2, 2.3]])
        feature_fn = _FeatureFn()
        encoder = _encoder()

        with patch.object(
            encoder_server,
            "get_mm",
            return_value=SimpleNamespace(enable_prefix_mm_cache=True),
        ):
            first = self._run(
                encoder._compute_direct_embedding(
                    _ctx([a, b], [2, 2], feature_fn), keep_on_gpu=False
                )
            )
            self.assertEqual(feature_fn.calls, 1)

            # Same two items, different request order -> the combined hash
            # misses, but both per-item entries are present.
            second = self._run(
                encoder._compute_direct_embedding(
                    _ctx([b, a], [2, 2], feature_fn), keep_on_gpu=False
                )
            )

        self.assertEqual(feature_fn.calls, 1)
        self.assertTrue(torch.equal(first, torch.cat([a.embedding, b.embedding])))
        self.assertTrue(torch.equal(second, torch.cat([b.embedding, a.embedding])))

    def test_fused_batch_uses_prefix_cache(self):
        a = _FakeItem(b"a", [[1.0, 1.1]])
        b = _FakeItem(b"b", [[2.0, 2.1]])
        feature_fn = _FeatureFn()
        encoder = _encoder()

        with patch.object(
            encoder_server,
            "get_mm",
            return_value=SimpleNamespace(enable_prefix_mm_cache=True),
        ):
            first_ctx = _ctx([a, b], [1, 1], feature_fn, items_per_req=[1, 1])
            self._run(encoder._compute_direct_embedding(first_ctx, keep_on_gpu=False))
            self.assertEqual(feature_fn.calls, 1)
            again = self._run(
                encoder._compute_direct_embedding(
                    _ctx([a, b], [1, 1], feature_fn, items_per_req=[1, 1]),
                    keep_on_gpu=False,
                )
            )

        self.assertEqual(feature_fn.calls, 1)
        self.assertTrue(torch.equal(again, torch.cat([a.embedding, b.embedding])))

    def test_partial_item_miss_encodes_only_the_miss(self):
        a = _FakeItem(b"a", [[1.0, 1.1]])
        c = _FakeItem(b"c", [[3.0, 3.1]])
        feature_fn = _FeatureFn()
        encoder = _encoder()
        encoder.mm_cache.set(a.cache_key, EmbeddingResult(embedding=a.embedding))

        with patch.object(
            encoder_server,
            "get_mm",
            return_value=SimpleNamespace(enable_prefix_mm_cache=True),
        ):
            out = self._run(
                encoder._compute_direct_embedding(
                    _ctx([a, c], [1, 1], feature_fn), keep_on_gpu=False
                )
            )

        self.assertEqual(feature_fn.calls, 1)
        self.assertEqual(feature_fn.last_keys, [b"c"])
        self.assertTrue(torch.equal(out, torch.cat([a.embedding, c.embedding])))

    def test_wrong_sized_item_entry_does_not_poison_the_batch(self):
        a = _FakeItem(b"a", [[1.0, 1.1]])
        b = _FakeItem(b"b", [[2.0, 2.1]])
        feature_fn = _FeatureFn()
        encoder = _encoder()
        encoder.mm_cache.set(
            a.cache_key,
            EmbeddingResult(embedding=torch.zeros(3, 2)),
        )

        with patch.object(
            encoder_server,
            "get_mm",
            return_value=SimpleNamespace(enable_prefix_mm_cache=True),
        ):
            out = self._run(
                encoder._compute_direct_embedding(
                    _ctx([a, b], [1, 1], feature_fn), keep_on_gpu=False
                )
            )

        self.assertEqual(feature_fn.calls, 1)
        self.assertTrue(torch.equal(out, torch.cat([a.embedding, b.embedding])))


if __name__ == "__main__":
    unittest.main()
