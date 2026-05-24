import unittest

import torch

from sglang.srt.managers.mm_utils import _get_multimodal_indices_from_offsets
from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem


class TestMultimodalOffsetPlacement(unittest.TestCase):
    def test_indices_follow_flattened_batch_starts(self):
        items = [
            MultimodalDataItem(
                modality=Modality.IMAGE,
                pad_value=-11,
                offsets=[(2, 4)],
            ),
            MultimodalDataItem(
                modality=Modality.IMAGE,
                pad_value=-22,
                offsets=[(1, 3)],
            ),
        ]
        input_ids = torch.tensor([101, -11, -11, -11, 201, -22, -22, -22])

        indices = _get_multimodal_indices_from_offsets(
            embedding_items=items,
            items_size=[0, 1, 2],
            prefix_length=[1, 0],
            extend_length=[4, 4],
            input_token_starts=[0, 4],
            input_ids=input_ids,
            embedding_length=6,
        )

        self.assertEqual(indices.tolist(), [1, 2, 3, 5, 6, 7])

    def test_rejects_mismatched_pad_values(self):
        item = MultimodalDataItem(
            modality=Modality.IMAGE,
            pad_value=-11,
            offsets=[(0, 2)],
        )

        indices = _get_multimodal_indices_from_offsets(
            embedding_items=[item],
            items_size=[0, 1],
            prefix_length=[0],
            extend_length=[3],
            input_token_starts=[0],
            input_ids=torch.tensor([-11, 0, -11]),
            embedding_length=3,
        )

        self.assertIsNone(indices)


if __name__ == "__main__":
    unittest.main()
