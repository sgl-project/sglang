"""CPU tests for Unlimited-OCR multimodal batch routing."""

import torch

from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem
from sglang.srt.models.unlimited_ocr import UnlimitedOCRForCausalLM
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _item(index: int, crop_count: int) -> MultimodalDataItem:
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        hash=index,
        feature=torch.zeros(1),
        model_specific_data={
            "images_crop": torch.zeros(1, crop_count, 3, 2, 2),
        },
    )


class _FakeUnlimitedOCR:
    def __init__(self):
        self.calls = []

    def _process_image_input_batch(self, items):
        self.calls.append(items)
        return torch.tensor([[items[0].hash]], dtype=torch.float32)


def test_unlimited_ocr_keeps_fast_path_for_matching_crop_shapes():
    model = _FakeUnlimitedOCR()
    items = [_item(1, 2), _item(2, 2)]

    output = UnlimitedOCRForCausalLM._process_image_input(model, items)

    assert len(model.calls) == 1
    assert model.calls[0] == items
    torch.testing.assert_close(output, torch.tensor([[1.0]]))


def test_unlimited_ocr_splits_mismatched_crop_shapes_in_input_order():
    model = _FakeUnlimitedOCR()
    items = [_item(10, 12), _item(20, 1), _item(30, 3)]

    output = UnlimitedOCRForCausalLM._process_image_input(model, items)

    assert model.calls == [[items[0]], [items[1]], [items[2]]]
    torch.testing.assert_close(output, torch.tensor([[10.0], [20.0], [30.0]]))
