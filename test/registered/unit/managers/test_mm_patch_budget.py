import numpy as np
import torch

from sglang.srt.managers.schedule_batch import (
    Modality,
    MultimodalDataItem,
    MultimodalInputs,
    compute_image_patch_tokens,
)


def test_compute_image_patch_tokens_for_mixed_grids():
    items = [
        MultimodalDataItem(
            modality=Modality.IMAGE,
            model_specific_data={
                "image_grid_thw": torch.tensor([[1, 64, 64]])
            },
        ),
        MultimodalDataItem(
            modality=Modality.IMAGE,
            model_specific_data={
                "image_grid_thw": np.array([[1, 8, 12], [1, 4, 4]])
            },
        ),
        MultimodalDataItem(modality=Modality.AUDIO),
    ]

    assert compute_image_patch_tokens(items) == [4096, 96, 16]

    mm_inputs = MultimodalInputs(mm_items=items)
    assert mm_inputs.total_image_patch_tokens() == 4208
    assert mm_inputs.image_patch_tokens == [4096, 96, 16]


def test_merge_keeps_image_patch_tokens():
    left = MultimodalInputs(mm_items=[], image_patch_tokens=[4096])
    right = MultimodalInputs(mm_items=[], image_patch_tokens=[96, 16])

    left.merge(right)

    assert left.image_patch_tokens == [4096, 96, 16]
    assert left.total_image_patch_tokens() == 4208
