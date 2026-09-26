"""Segmented (ragged) multimodal feature storage.

The contract these tests hold to is that adopting segmented storage cannot
change what a model sees: whatever a processor used to produce with
``torch.cat``, ``SegmentedFeatures`` must reproduce exactly, and any consumer
that cannot split the item must get that same tensor back from ``dense()``.
"""

import torch

from sglang.srt.multimodal.segmented_features import (
    SegmentedFeatures,
    expand_segmented_item,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestFromParts(CustomTestCase):
    def test_matches_cat_for_independent_parts(self):
        parts = [torch.randn(3, 4), torch.randn(5, 4), torch.randn(2, 4)]
        features = SegmentedFeatures.from_parts(parts)
        self.assertEqual(features.num_parts, 3)
        self.assertEqual(features.shape, (10, 4))
        self.assertTrue(torch.equal(features.dense(), torch.cat(parts, dim=0)))
        for i, part in enumerate(parts):
            self.assertTrue(torch.equal(features.part(i), part))

    def test_no_request_wide_allocation(self):
        """The point of the class: the packed tensor is never built."""
        parts = [torch.randn(64, 8) for _ in range(16)]
        before = [p.data_ptr() for p in parts]
        features = SegmentedFeatures.from_parts(parts)
        # Every part is still the original memory, not a copy into a new buffer.
        self.assertEqual([features.part(i).data_ptr() for i in range(16)], before)

    def test_coalesces_a_shared_storage_run(self):
        """A processor that batches same-shape items emits views of one buffer."""
        buffer = torch.randn(9, 4)
        parts = [buffer[0:3], buffer[3:7], buffer[7:9]]
        features = SegmentedFeatures.from_parts(parts)
        self.assertEqual(len(features.buffers), 1)
        self.assertEqual(features.num_parts, 3)
        self.assertEqual(features.storage_bytes, buffer.untyped_storage().nbytes())
        # Even here, where one buffer already holds every row in order,
        # dense() copies: returning the buffer would alias the producer's
        # memory, which torch.cat never did.
        self.assertTrue(torch.equal(features.dense(), buffer))
        self.assertNotEqual(features.dense().data_ptr(), buffer.data_ptr())

    def test_mixed_runs_and_singletons(self):
        shared = torch.randn(6, 4)
        loose = torch.randn(2, 4)
        parts = [shared[0:2], shared[2:6], loose]
        features = SegmentedFeatures.from_parts(parts)
        self.assertEqual(len(features.buffers), 2)
        self.assertTrue(torch.equal(features.dense(), torch.cat(parts, dim=0)))

    def test_non_adjacent_slices_of_one_buffer(self):
        """Group batching produces buffer rows in group order, not item order."""
        buffer = torch.randn(6, 4)
        # Item 0 is rows 3:6, item 1 is rows 0:3 -- reordered on purpose.
        features = SegmentedFeatures(buffers=(buffer,), slices=((0, 3, 6), (0, 0, 3)))
        self.assertTrue(torch.equal(features.part(0), buffer[3:6]))
        self.assertTrue(torch.equal(features.part(1), buffer[0:3]))
        # dense() must honour item order, so here it is NOT the raw buffer.
        self.assertTrue(
            torch.equal(features.dense(), torch.cat([buffer[3:6], buffer[0:3]], dim=0))
        )
        self.assertNotEqual(features.dense().data_ptr(), buffer.data_ptr())

    def test_preserves_trailing_dims(self):
        parts = [torch.randn(2, 3, 5), torch.randn(4, 3, 5)]
        features = SegmentedFeatures.from_parts(parts)
        self.assertEqual(features.shape, (6, 3, 5))
        self.assertTrue(torch.equal(features.dense(), torch.cat(parts, dim=0)))

    def test_to_moves_each_buffer_once(self):
        buffer = torch.randn(9, 4)
        features = SegmentedFeatures.from_parts([buffer[0:3], buffer[3:9]])
        moved = features.to("cpu")
        self.assertEqual(len(moved.buffers), 1)
        self.assertEqual(moved.slices, features.slices)
        self.assertTrue(torch.equal(moved.dense(), features.dense()))

    def test_storage_bytes_counts_a_shared_storage_once(self):
        buffer = torch.randn(8, 4)
        features = SegmentedFeatures(
            buffers=(buffer[0:4], buffer[4:8]), slices=((0, 0, 4), (1, 0, 4))
        )
        self.assertEqual(features.storage_bytes, buffer.untyped_storage().nbytes())


class TestValidation(CustomTestCase):
    def test_rejects_gaps(self):
        buffer = torch.randn(6, 4)
        with self.assertRaisesRegex(ValueError, "tile each buffer"):
            SegmentedFeatures(buffers=(buffer,), slices=((0, 0, 2), (0, 3, 6)))

    def test_rejects_overlap(self):
        buffer = torch.randn(6, 4)
        with self.assertRaisesRegex(ValueError, "tile each buffer"):
            SegmentedFeatures(buffers=(buffer,), slices=((0, 0, 4), (0, 2, 6)))

    def test_rejects_unreferenced_rows(self):
        buffer = torch.randn(6, 4)
        with self.assertRaisesRegex(ValueError, "unreferenced"):
            SegmentedFeatures(buffers=(buffer,), slices=((0, 0, 4),))

    def test_rejects_mismatched_trailing_shape(self):
        with self.assertRaisesRegex(ValueError, "trailing shape"):
            SegmentedFeatures(
                buffers=(torch.randn(2, 4), torch.randn(2, 5)),
                slices=((0, 0, 2), (1, 0, 2)),
            )

    def test_rejects_mismatched_dtype(self):
        with self.assertRaisesRegex(ValueError, "trailing shape, dtype"):
            SegmentedFeatures(
                buffers=(torch.randn(2, 4), torch.randn(2, 4).half()),
                slices=((0, 0, 2), (1, 0, 2)),
            )

    def test_rejects_empty(self):
        with self.assertRaises(ValueError):
            SegmentedFeatures(buffers=(), slices=())
        with self.assertRaisesRegex(ValueError, "no parts"):
            SegmentedFeatures.from_parts([])

    def test_rejects_one_dimensional_parts(self):
        with self.assertRaisesRegex(ValueError, "leading row dimension"):
            SegmentedFeatures.from_parts([torch.randn(4)])

    def test_rejects_a_zero_row_part(self):
        """The one deliberate difference from torch.cat, which absorbs these.

        Each part stands for one placeholder, so a part with no rows is an
        item with no feature. Failing at the producer names it.
        """
        with self.assertRaisesRegex(ValueError, "non-empty"):
            SegmentedFeatures.from_parts([torch.randn(2, 4), torch.randn(0, 4)])

    def test_repr_does_not_print_buffer_contents(self):
        """These land in scheduler logs; the dataclass default dumps tensors."""
        features = SegmentedFeatures.from_parts([torch.randn(3, 4), torch.randn(2, 4)])
        text = repr(features)
        self.assertIn("parts=2", text)
        self.assertIn("shape=(5, 4)", text)
        self.assertNotIn("tensor(", text)

    def test_is_not_array_like(self):
        """HF BatchFeature must not be able to coerce this back to a tensor."""
        features = SegmentedFeatures.from_parts([torch.randn(2, 4)])
        for attr in ("__len__", "__getitem__", "__array__", "__iter__"):
            self.assertFalse(hasattr(features, attr), attr)


class _Item:
    """Stand-in for MultimodalDataItem with the fields expansion touches."""

    def __init__(self, feature, offsets, model_specific_data=None):
        self.feature = feature
        self.offsets = offsets
        self.model_specific_data = model_specific_data or {}
        self.precomputed_embeddings = None
        self.hash = 1234
        self.pad_value = 5678


def _slice_model_data(data, index, start, end, num_items, total_feature_len):
    sliced = {}
    for key, value in data.items():
        if len(value) == num_items:
            sliced[key] = value[index : index + 1]
        elif total_feature_len is not None and len(value) == total_feature_len:
            sliced[key] = value[start:end]
        else:
            sliced[key] = value
    return sliced


class TestExpansion(CustomTestCase):
    def _features(self):
        return SegmentedFeatures.from_parts(
            [torch.randn(2, 4), torch.randn(3, 4), torch.randn(1, 4)]
        )

    def test_expands_one_item_per_placeholder(self):
        features = self._features()
        item = _Item(
            features,
            offsets=[(0, 2), (2, 5), (5, 6)],
            model_specific_data={"image_grid_thw": [(1, 1, 2), (1, 1, 3), (1, 1, 1)]},
        )
        expanded = expand_segmented_item(item, _slice_model_data)
        self.assertEqual(len(expanded), 3)
        for i, new in enumerate(expanded):
            self.assertTrue(torch.equal(new.feature, features.part(i)))
            self.assertEqual(new.offsets, [item.offsets[i]])
            self.assertEqual(
                new.model_specific_data["image_grid_thw"],
                [item.model_specific_data["image_grid_thw"][i]],
            )
            # Derived from the feature, which changed.
            self.assertIsNone(new.hash)
            self.assertIsNone(new.pad_value)

    def test_a_part_that_owns_its_rows_is_handed_over_without_a_copy(self):
        features = self._features()
        item = _Item(features, offsets=[(0, 2), (2, 5), (5, 6)])
        expanded = expand_segmented_item(item, _slice_model_data)
        for i, new in enumerate(expanded):
            self.assertEqual(new.feature.data_ptr(), features.part(i).data_ptr())

    def test_a_part_cut_from_a_shared_buffer_leaves_owning_its_rows(self):
        """Pickle serialises a view's whole storage, so a view may not leave.

        Without this, each of N items would carry all N items' bytes to the
        scheduler, which is the cost segmenting exists to avoid.
        """
        buffer = torch.randn(6, 4)
        features = SegmentedFeatures.from_parts([buffer[0:2], buffer[2:4], buffer[4:6]])
        self.assertEqual(len(features.buffers), 1)

        item = _Item(features, offsets=[(0, 2), (2, 4), (4, 6)])
        expanded = expand_segmented_item(item, _slice_model_data)
        for i, new in enumerate(expanded):
            self.assertTrue(torch.equal(new.feature, features.part(i)))
            self.assertEqual(
                new.feature.untyped_storage().nbytes(),
                new.feature.numel() * new.feature.element_size(),
            )

    def test_a_part_may_own_several_placeholder_spans(self):
        """Step3 wraps every crop in boundary tokens, breaking the run up."""
        features = self._features()
        spans = [(0, 2), (4, 6), (8, 11), (13, 14)]
        item = _Item(features, offsets=spans)
        expanded = expand_segmented_item(
            item, _slice_model_data, offsets_per_part=[2, 1, 1]
        )
        self.assertEqual(
            [new.offsets for new in expanded], [spans[:2], [spans[2]], [spans[3]]]
        )

    def test_declines_a_grouping_that_does_not_account_for_every_span(self):
        features = self._features()
        item = _Item(features, offsets=[(0, 2), (4, 6), (8, 11)])
        self.assertIsNone(
            expand_segmented_item(item, _slice_model_data, offsets_per_part=[2, 1, 1])
        )
        self.assertIsNone(
            expand_segmented_item(item, _slice_model_data, offsets_per_part=[1, 1])
        )

    def test_row_aligned_metadata_is_sliced_by_rows(self):
        features = self._features()
        item = _Item(
            features,
            offsets=[(0, 2), (2, 5), (5, 6)],
            model_specific_data={"per_row": list(range(6))},
        )
        expanded = expand_segmented_item(item, _slice_model_data)
        self.assertEqual(expanded[0].model_specific_data["per_row"], [0, 1])
        self.assertEqual(expanded[1].model_specific_data["per_row"], [2, 3, 4])
        self.assertEqual(expanded[2].model_specific_data["per_row"], [5])

    def test_declines_when_offsets_do_not_match_parts(self):
        """A video whose offsets count frames, not videos. Caller uses dense()."""
        features = self._features()
        item = _Item(features, offsets=[(0, 2)])
        self.assertIsNone(expand_segmented_item(item, _slice_model_data))

    def test_declines_with_precomputed_embeddings(self):
        features = self._features()
        item = _Item(features, offsets=[(0, 2), (2, 5), (5, 6)])
        item.precomputed_embeddings = torch.randn(6, 4)
        self.assertIsNone(expand_segmented_item(item, _slice_model_data))

    def test_declines_without_offsets(self):
        item = _Item(self._features(), offsets=None)
        self.assertIsNone(expand_segmented_item(item, _slice_model_data))

    def test_slicer_gets_item_index_and_row_boundaries(self):
        """The contract _slice_segmented_model_data is built on.

        It needs the item index to pick a patch slice, and the row range to
        cut feature-aligned metadata; both must arrive per item, in order.
        """
        features = self._features()  # parts of 2, 3 and 1 rows
        item = _Item(features, offsets=[(0, 2), (2, 5), (5, 6)])
        seen = []

        def record(data, *, index, start, end, num_items, total_feature_len):
            seen.append((index, start, end, num_items, total_feature_len))
            return data

        expand_segmented_item(item, record)
        self.assertEqual(
            seen,
            [(0, 0, 2, 3, 6), (1, 2, 5, 3, 6), (2, 5, 6, 3, 6)],
        )

    def test_patch_aligned_metadata_cuts_on_its_own_boundaries(self):
        """Step3 shape: one feature row per image, but many patches per image.

        patch_pixel_values is aligned with the patch count, not the feature
        rows, so slicing it by rows would be wrong. This mirrors what
        _slice_segmented_model_data does on top of _slice_model_data.
        """
        # Three images, one feature row each; patches per image are 2, 0, 3.
        features = SegmentedFeatures.from_parts([torch.randn(1, 4)] * 3)
        num_patches = [2, 0, 3]
        patch_rows = list(range(5))
        item = _Item(
            features,
            offsets=[(0, 2), (2, 4), (4, 6)],
            model_specific_data={
                "num_patches": num_patches,
                "patch_pixel_values": patch_rows,
            },
        )

        bounds, cursor = [], 0
        for count in num_patches:
            bounds.append((cursor, cursor + count))
            cursor += count

        def slice_with_patches(data, *, index, start, end, **kw):
            sliced = _slice_model_data(data, index=index, start=start, end=end, **kw)
            lo, hi = bounds[index]
            sliced["patch_pixel_values"] = data["patch_pixel_values"][lo:hi]
            return sliced

        expanded = expand_segmented_item(item, slice_with_patches)
        self.assertEqual(
            [e.model_specific_data["patch_pixel_values"] for e in expanded],
            [[0, 1], [], [2, 3, 4]],
        )
        # num_patches is item-aligned, so it splits one entry per image.
        self.assertEqual(
            [e.model_specific_data["num_patches"] for e in expanded],
            [[2], [0], [3]],
        )


if __name__ == "__main__":
    import unittest

    unittest.main()


class TestWhyItIsNotArrayLike(CustomTestCase):
    """The reason this is a class and not a plain list of tensors.

    A processor whose output passes through HF `BatchFeature` with
    `tensor_type="pt"` -- which `process_mm_data` always requests -- cannot
    hand over a list. `convert_to_tensors` calls `torch.stack` on it, which
    either fails or, worse, silently succeeds. `SegmentedFeatures` exposes no
    `__len__`/`__getitem__`/`__array__`/`__iter__`, so it is carried through
    untouched.

    A processor that builds its own MultimodalDataItems and never reaches
    BatchFeature should keep a plain list instead; the framework splits,
    hashes and transfers one already.
    """

    def test_batch_feature_stacks_a_uniform_list_into_the_wrong_rank(self):
        from transformers.feature_extraction_utils import BatchFeature

        parts = [torch.randn(9, 3, 8, 8) for _ in range(4)]

        stacked = BatchFeature(data={"pixel_values": parts}, tensor_type="pt")[
            "pixel_values"
        ]

        # Rank 4 per image became one rank-5 tensor: the per-item structure
        # is gone, and nothing raised to say so.
        self.assertIsInstance(stacked, torch.Tensor)
        self.assertEqual(stacked.ndim, 5)

    def test_batch_feature_rejects_a_ragged_list(self):
        from transformers.feature_extraction_utils import BatchFeature

        parts = [torch.randn(9, 3, 8, 8), torch.randn(10, 3, 8, 8)]

        with self.assertRaises(ValueError):
            BatchFeature(data={"pixel_values": parts}, tensor_type="pt")

    def test_segmented_features_pass_through_batch_feature_untouched(self):
        from transformers.feature_extraction_utils import BatchFeature

        parts = [torch.randn(9, 3, 8, 8), torch.randn(10, 3, 8, 8)]
        features = SegmentedFeatures.from_parts(parts)

        carried = BatchFeature(data={"pixel_values": features}, tensor_type="pt")[
            "pixel_values"
        ]

        self.assertIs(carried, features)
        for i, part in enumerate(parts):
            self.assertTrue(torch.equal(carried.part(i), part))
