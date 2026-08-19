"""Regression tests for Pixtral multimodal item processing."""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.multimodal.processors.pixtral import PixtralProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")


class TestPixtralProcessor(CustomTestCase):
    def test_multi_image_features_are_split_before_transport(self):
        """CUDA IPC dispatch must receive per-image tensors, not a bundled proxy."""
        processor = object.__new__(PixtralProcessor)
        processor.use_cuda_ipc = True
        processor._get_image_nrows = MagicMock(return_value=[2, 3])
        proxies = [object(), object()]
        processor._wrap_tensor_for_cuda_ipc = MagicMock(side_effect=proxies)
        processor._precompute_hashes_before_cpu_transfer = MagicMock()

        feature = torch.arange(8).reshape(2, 4)
        image_sizes = torch.tensor([[10, 20], [30, 40]])
        bundled_item = MultimodalDataItem(
            modality=Modality.IMAGE,
            feature=feature,
            offsets=[0, 1, 2, 3, 4],
            model_specific_data={"image_sizes": image_sizes},
        )
        base_output = SimpleNamespace(images=[object(), object()])

        items = processor._finalize_mm_items(
            [bundled_item],
            base_output=base_output,
        )

        self.assertEqual(len(items), 2)
        self.assertEqual([item.offsets for item in items], [[0, 1], [2, 3, 4]])
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


if __name__ == "__main__":
    unittest.main()
