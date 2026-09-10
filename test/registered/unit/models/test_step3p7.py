"""CPU-only coverage for Step3.7 multimodal feature batching."""

import unittest

import torch
from torch import nn

from sglang.test.test_utils import CustomTestCase, maybe_stub_sgl_kernel

maybe_stub_sgl_kernel()

from sglang.srt.managers.schedule_batch import (  # noqa: E402
    Modality,
    MultimodalDataItem,
)
from sglang.srt.models.step3p7 import (  # noqa: E402
    Step3p7ForConditionalGeneration,
)
from sglang.test.ci.ci_register import register_cpu_ci  # noqa: E402

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class _RecordingVisionModel(nn.Module):
    dtype = torch.float32

    def __init__(self):
        super().__init__()
        self.calls = []

    def forward(self, pixel_values):
        self.calls.append(pixel_values.detach().clone())
        return pixel_values.unsqueeze(1)


class _IdentityProjector(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1))

    def forward(self, image_features):
        return image_features, None


def _bare_model():
    model = Step3p7ForConditionalGeneration.__new__(Step3p7ForConditionalGeneration)
    nn.Module.__init__(model)
    model.vision_model = _RecordingVisionModel()
    model.vit_large_projector = _IdentityProjector()
    return model


def _image_item(thumbnails, num_patches, patches=None):
    model_specific_data = {"num_patches": num_patches}
    if patches is not None:
        model_specific_data["patch_pixel_values"] = patches
    return MultimodalDataItem(
        modality=Modality.IMAGE,
        offsets=[(0, 1)],
        feature=thumbnails,
        model_specific_data=model_specific_data,
    )


class TestStep3p7ImageFeatureBatching(CustomTestCase):
    def test_batches_items_and_restores_image_order(self):
        model = _bare_model()
        items = [
            _image_item(
                torch.tensor([[100.0], [200.0]]),
                [2, 0],
                torch.tensor([[1.0], [2.0]]),
            ),
            _image_item(
                torch.tensor([[300.0]]),
                torch.tensor([1]),
                torch.tensor([[3.0]]),
            ),
        ]

        output = model.get_image_feature(items)

        self.assertEqual(len(model.vision_model.calls), 2)
        self.assertTrue(
            torch.equal(
                model.vision_model.calls[0],
                torch.tensor([[100.0], [200.0], [300.0]]),
            )
        )
        self.assertTrue(
            torch.equal(
                model.vision_model.calls[1], torch.tensor([[1.0], [2.0], [3.0]])
            )
        )
        self.assertTrue(
            torch.equal(
                output,
                torch.tensor([[1.0], [2.0], [100.0], [200.0], [3.0], [300.0]]),
            )
        )

    def test_accepts_an_item_without_local_patches(self):
        model = _bare_model()
        item = _image_item(torch.tensor([[100.0]]), [0])

        output = model.get_image_feature([item])

        self.assertEqual(len(model.vision_model.calls), 1)
        self.assertTrue(torch.equal(output, torch.tensor([[100.0]])))

    def test_empty_patch_tensor_before_item_with_patches(self):
        model = _bare_model()
        items = [
            _image_item(torch.tensor([[100.0]]), [0], torch.empty(0, 1)),
            _image_item(torch.tensor([[200.0]]), [1], torch.tensor([[1.0]])),
        ]

        output = model.get_image_feature(items)

        self.assertTrue(torch.equal(output, torch.tensor([[100.0], [1.0], [200.0]])))
        self.assertEqual(len(model.vision_model.calls), 2)

    def test_batch_matches_individual_items_with_multiple_tokens(self):
        items = [
            _image_item(
                torch.arange(12, dtype=torch.float32).reshape(2, 2, 3),
                [0, 2],
                torch.arange(12, 24, dtype=torch.float32).reshape(2, 2, 3),
            ),
            _image_item(
                torch.arange(24, 30, dtype=torch.float32).reshape(1, 2, 3),
                [1],
                torch.arange(30, 36, dtype=torch.float32).reshape(1, 2, 3),
            ),
        ]

        batched = _bare_model().get_image_feature(items)
        individual = torch.cat(
            [_bare_model().get_image_feature([item]) for item in items]
        )

        self.assertEqual(batched.shape, (12, 3))
        self.assertTrue(torch.equal(batched, individual))

    def test_rejects_invalid_item_metadata(self):
        missing_num_patches = MultimodalDataItem(
            modality=Modality.IMAGE,
            offsets=[(0, 1)],
            feature=torch.tensor([[100.0]]),
        )
        cases = [
            ("empty batch", [], "at least one item"),
            ("missing num_patches", [missing_num_patches], "missing num_patches"),
            (
                "negative patch count",
                [_image_item(torch.tensor([[100.0]]), [-1])],
                "negative num_patches",
            ),
            (
                "thumbnail count mismatch",
                [_image_item(torch.tensor([[100.0], [200.0]]), [0])],
                "thumbnail and num_patches",
            ),
            (
                "patch count mismatch",
                [
                    _image_item(
                        torch.tensor([[100.0]]),
                        [2],
                        torch.tensor([[1.0]]),
                    )
                ],
                "patch_pixel_values",
            ),
        ]

        for name, items, error_pattern in cases:
            with self.subTest(name=name):
                with self.assertRaisesRegex(ValueError, error_pattern):
                    _bare_model().get_image_feature(items)


if __name__ == "__main__":
    unittest.main()
