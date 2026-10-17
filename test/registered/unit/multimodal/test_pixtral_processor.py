"""Regression tests for Pixtral multimodal item processing."""

import unittest
from unittest.mock import MagicMock

import torch
from PIL import Image

from sglang.srt.managers.mm_utils import get_new_expanded_mm_items
from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputFormat,
)
from sglang.srt.multimodal.processors.pixtral import PixtralProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=11, suite="base-a-test-cpu")


class TestPixtralProcessor(CustomTestCase):
    def _make_processor(self):
        processor = object.__new__(PixtralProcessor)
        processor.use_cuda_ipc = True
        processor.image_size = 1024
        processor._effective_patch_size = 28
        processor._precompute_hashes_before_cpu_transfer = MagicMock()
        return processor

    def test_multi_image_features_are_split_before_transport(self):
        """CUDA IPC dispatch must receive per-image tensors, not a bundled proxy."""
        processor = self._make_processor()
        proxies = [object(), object()]
        processor._wrap_tensor_for_cuda_ipc = MagicMock(side_effect=proxies)

        feature = torch.arange(8).reshape(2, 4)
        image_sizes = torch.tensor([[10, 20], [30, 40]])
        bundled_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=feature,
            offsets=[(0, 0), (1, 1), (2, 2), (3, 3), (4, 4)],
            format=MultimodalInputFormat.PROCESSOR_OUTPUT,
            model_specific_data={"image_sizes": image_sizes, "extra_key": "keep"},
        )
        images = [Image.new("RGB", (28, 56)), Image.new("RGB", (28, 84))]

        items = processor._finalize_mm_items(
            [bundled_item],
            images=images,
        )

        self.assertEqual(len(items), 2)
        self.assertEqual(
            [item.offsets for item in items],
            [[(0, 0), (1, 1)], [(2, 2), (3, 3), (4, 4)]],
        )
        self.assertTrue(
            all(item.format == MultimodalInputFormat.PROCESSOR_OUTPUT for item in items)
        )
        self.assertTrue(
            all(item.model_specific_data["extra_key"] == "keep" for item in items)
        )
        self.assertTrue(all(item.pad_value is not None for item in items))
        self.assertTrue(
            torch.equal(items[0].model_specific_data["image_sizes"], image_sizes[:1])
        )
        self.assertTrue(
            torch.equal(items[1].model_specific_data["image_sizes"], image_sizes[1:])
        )
        wrapped_features = [
            call.args[0] for call in processor._wrap_tensor_for_cuda_ipc.call_args_list
        ]
        self.assertTrue(torch.equal(wrapped_features[0], feature[:1]))
        self.assertTrue(torch.equal(wrapped_features[1], feature[1:]))
        self.assertEqual([item.feature for item in items], proxies)

    def test_already_split_one_row_images_are_preserved(self):
        """Generic per-image splits must not be collapsed and re-sliced by Pixtral."""
        processor = self._make_processor()
        processor._wrap_tensor_for_cuda_ipc = MagicMock(
            side_effect=[object(), object()]
        )
        bundled_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.arange(8).reshape(2, 4),
            offsets=[(0, 0), (2, 2)],
        )
        items = get_new_expanded_mm_items([bundled_item])
        images = [Image.new("RGB", (28, 28)), Image.new("RGB", (56, 28))]

        items = processor._finalize_mm_items(items, images=images)

        self.assertEqual(len(items), 2)
        self.assertEqual([item.offsets for item in items], [[(0, 0)], [(2, 2)]])
        wrapped_features = [
            call.args[0] for call in processor._wrap_tensor_for_cuda_ipc.call_args_list
        ]
        self.assertTrue(torch.equal(wrapped_features[0], bundled_item.feature[:1]))
        self.assertTrue(torch.equal(wrapped_features[1], bundled_item.feature[1:]))

    def test_mismatched_patch_rows_fail_loudly(self):
        """Derived row counts cannot silently leave image placeholders unassigned."""
        processor = self._make_processor()
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=torch.arange(8).reshape(2, 4),
            offsets=[(0, 0), (1, 1)],
        )
        images = [Image.new("RGB", (28, 56)), Image.new("RGB", (28, 84))]

        with self.assertRaisesRegex(ValueError, "patch rows"):
            processor._finalize_mm_items([item], images=images)


if __name__ == "__main__":
    unittest.main()
