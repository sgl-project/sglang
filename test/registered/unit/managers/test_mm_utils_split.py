"""Unit tests for ``get_new_expanded_mm_items`` per-image splitting.

This is the load-bearing behavioral path for multi-image requests: a bundled
``MultimodalDataItem`` (one item carrying N image offsets + a concatenated
feature) must be split back into N per-image items so RadixAttention can cache
each image independently and chunked-prefill can encode them one at a time.

The MoonViT-style models (e.g. nvidia/LocateAnything-3B) carry their per-image
grids under ``image_grid_hws`` rather than ``image_grid_thw``; the splitter must
recognize both keys, fall back cleanly when no usable grid is present, and not
mis-split a degenerate flat grid. No server / GPU / weight loading involved.
"""

from sglang.test.test_utils import maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

import unittest
from unittest.mock import patch

import numpy as np
import torch

from sglang.srt.managers.mm_schedule import _get_chunked_embedding_by_item
from sglang.srt.managers.mm_utils import get_new_expanded_mm_items
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.mem_cache.multimodal_cache import MultiModalStaticCache
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def _bundled_item(grid_key=None, grid=None, feature_len=10, num_images=2):
    """A bundled IMAGE item: `num_images` offsets, one concatenated feature."""
    model_specific_data = {}
    if grid_key is not None:
        model_specific_data[grid_key] = grid
    # Distinct per-row values so slice boundaries are checkable.
    feature = torch.arange(feature_len * 3, dtype=torch.float32).reshape(feature_len, 3)
    offsets = [(0, 5), (5, feature_len)][:num_images]
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=offsets,
        feature=feature,
        model_specific_data=model_specific_data,
    )


class TestGetNewExpandedMMItems(CustomTestCase):
    def test_image_grid_hws_splits_per_image(self):
        # grid rows [[2,3],[4,1]] -> prod = [6, 4] patches -> feature_len 10.
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=[[2, 3], [4, 1]],
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertEqual([len(o.offsets) for o in out], [1, 1])
        self.assertEqual(out[0].offsets, [(0, 5)])
        self.assertEqual(out[1].offsets, [(5, 10)])
        # Feature sliced 0:6 and 6:10 along dim-0.
        self.assertEqual(out[0].feature.shape[0], 6)
        self.assertEqual(out[1].feature.shape[0], 4)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))
        # Split items must re-hash (pad value is recomputed per image).
        self.assertTrue(all(o.hash is None for o in out))

    def test_image_grid_hws_tensor_splits_per_image(self):
        # Same as above but the grid arrives as a rank-2 tensor (HF emits these).
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=torch.tensor([[2, 3], [4, 1]], dtype=torch.long),
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))

    def test_image_grid_thw_still_splits(self):
        # The pre-existing image_grid_thw path must keep working:
        # [[1,2,3],[1,4,1]] -> [6,4].
        item = _bundled_item(
            grid_key="image_grid_thw",
            grid=[[1, 2, 3], [1, 4, 1]],
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))

    def test_missing_grid_falls_back_to_simple_split(self):
        # No grid, but feature dim-0 == num offsets -> simple per-row split.
        item = _bundled_item(grid_key=None, feature_len=2, num_images=2)
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:1]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[1:2]))

    def test_flat_1d_grid_does_not_mis_split(self):
        # A flat 1-D grid (`tensor([2, 2])`) has length == num_items so it passes
        # the length check, but prod(dim=-1) would collapse it to a scalar and
        # corrupt the slice boundaries. The rank-2 guard must reject it. With
        # feature_len != num_items, the simple-split fallback also declines, so
        # the bundled item is passed through unchanged (never mis-sliced).
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=torch.tensor([2, 2], dtype=torch.long),
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 1)
        self.assertIs(out[0], item)

    def test_numpy_grid_splits_per_image(self):
        # image_grid_hws can arrive as a numpy array from the HF image processor.
        item = _bundled_item(
            grid_key="image_grid_hws",
            grid=np.array([[2, 3], [4, 1]], dtype=np.int64),
            feature_len=10,
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 2)
        self.assertTrue(torch.equal(out[0].feature, item.feature[0:6]))
        self.assertTrue(torch.equal(out[1].feature, item.feature[6:10]))

    def test_non_bundled_item_passes_through(self):
        # A single-image item (one offset) is not bundled and is returned as-is.
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 5)],
            feature=torch.arange(18, dtype=torch.float32).reshape(6, 3),
            model_specific_data={"image_grid_hws": [[2, 3]]},
        )
        out = get_new_expanded_mm_items([item])

        self.assertEqual(len(out), 1)
        self.assertIs(out[0], item)

    def test_mutable_tail_is_reencoded_and_never_cached(self):
        """A rolling-window audio request pairs cacheable complete windows with
        a mutable tail (use_embedding_cache=False). The tail's content changes
        under the same request shape on every chunk, so it must be re-encoded
        each call and must never enter the shared embedding cache."""
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
        # Second call: the stable window hits the cache; only the tail encodes.
        self.assertEqual(encoded_item_counts, [2, 1])
        self.assertTrue(cache.has(stable.hash))
        self.assertFalse(cache.has(tail.hash))

    def test_stale_cache_geometry_is_reencoded(self):
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


if __name__ == "__main__":
    unittest.main()
