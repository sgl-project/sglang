"""OCR encoder input assembly with different crop counts and per-image flags."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.managers.mm_utils import get_new_expanded_mm_items
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.models.unlimited_ocr import UnlimitedOCRForCausalLM
from sglang.srt.multimodal.processors.unlimited_ocr import UnlimitedOCRProcessor
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=3, suite="base-a-test-cpu")


class TestUnlimitedOCRImageInput(CustomTestCase):
    @staticmethod
    def _item(value, crops, has_local_crops):
        return SimpleNamespace(
            feature=torch.full((1, 3, 2, 2), value),
            images_crop=torch.full((1, crops, 3, 2, 2), value + 10),
            images_spatial_crop=torch.tensor([[crops, 1]]),
            has_local_crops=has_local_crops,
        )

    @staticmethod
    def _encoder(*, pixel_values, images_crop, images_spatial_crop, has_local_crops):
        # Substitute the expensive SAM/CLIP boundary with a small deterministic
        # encoder; keep production crop assembly and flag handling intact.
        if images_crop.ndim != 6 or images_spatial_crop.ndim != 3:
            raise ValueError("incorrect encoder input rank")
        results = []
        for page in range(pixel_values.shape[0]):
            global_features = pixel_values[page].mean().reshape(1, 1)
            if has_local_crops[page]:
                local_features = images_crop[page].mean(dim=(2, 3, 4)).reshape(-1, 1)
                results.append(torch.cat((local_features, global_features)))
            else:
                results.append(global_features)
        return results

    def test_different_crop_counts_preserve_page_order(self):
        """Stacking 2-crop and 3-crop pages raises before encoding either page."""
        items = [self._item(1.0, 2, True), self._item(2.0, 3, True)]
        encoder = SimpleNamespace(
            vision_model=SimpleNamespace(dtype=torch.float32),
            _collect_mm_flag=UnlimitedOCRForCausalLM._collect_mm_flag,
            _pixel_values_to_embedding=self._encoder,
        )
        result = UnlimitedOCRForCausalLM._process_image_input(encoder, items)
        torch.testing.assert_close(
            result, torch.tensor([[11.0], [11.0], [1.0], [12.0], [12.0], [12.0], [2.0]])
        )

    def test_per_page_crop_flags_stay_with_their_image(self):
        """A page without local crops must not inherit another page's flag."""
        items = [self._item(1.0, 2, False), self._item(2.0, 3, True)]
        encoder = SimpleNamespace(
            vision_model=SimpleNamespace(dtype=torch.float32),
            _collect_mm_flag=UnlimitedOCRForCausalLM._collect_mm_flag,
            _pixel_values_to_embedding=self._encoder,
        )
        result = UnlimitedOCRForCausalLM._process_image_input(encoder, items)
        torch.testing.assert_close(
            result, torch.tensor([[1.0], [12.0], [12.0], [12.0], [2.0]])
        )

    def test_bundled_pages_are_encoded_separately(self):
        """Two pages in one processor item must not reach the formatter as B=2."""
        first = self._item(1.0, 2, False)
        second = self._item(2.0, 2, True)
        bundled = SimpleNamespace(
            feature=torch.cat((first.feature, second.feature)),
            images_crop=torch.cat((first.images_crop, second.images_crop)),
            images_spatial_crop=torch.cat(
                (first.images_spatial_crop, second.images_spatial_crop)
            ),
            has_local_crops=[False, True],
        )
        encoder = SimpleNamespace(
            vision_model=SimpleNamespace(dtype=torch.float32),
            _collect_mm_flag=UnlimitedOCRForCausalLM._collect_mm_flag,
            _pixel_values_to_embedding=self._encoder,
        )
        result = UnlimitedOCRForCausalLM._process_image_input(encoder, [bundled])
        torch.testing.assert_close(result, torch.tensor([[1.0], [12.0], [12.0], [2.0]]))

    def test_separated_page_offsets_split_spatial_metadata(self):
        """HF's [1,pages,2] crop metadata must split with [pages,C,H,W]."""
        first, second = self._item(1.0, 2, False), self._item(2.0, 2, True)
        processor = UnlimitedOCRProcessor.__new__(UnlimitedOCRProcessor)
        processor.ATTR_NAME_TO_MODALITY = {
            key: Modality.IMAGE
            for key in (
                "pixel_values",
                "images_crop",
                "images_spatial_crop",
                "has_local_crops",
            )
        }
        processor.FEATURE_NAMES = ["pixel_values"]
        bundled = processor.collect_mm_items_from_processor_output(
            {
                "pixel_values": torch.cat((first.feature, second.feature)),
                "images_crop": torch.cat((first.images_crop, second.images_crop)),
                "images_spatial_crop": torch.stack(
                    (first.images_spatial_crop, second.images_spatial_crop), dim=1
                ),
                "has_local_crops": [False, True],
            }
        )
        bundled[0].offsets = [(1, 273), (275, 547)]
        items = get_new_expanded_mm_items(bundled)
        self.assertEqual(len(items), 2)
        self.assertEqual(
            [tuple(item.images_spatial_crop.shape) for item in items], [(1, 2), (1, 2)]
        )
        encoder = SimpleNamespace(
            vision_model=SimpleNamespace(dtype=torch.float32),
            _collect_mm_flag=UnlimitedOCRForCausalLM._collect_mm_flag,
            _pixel_values_to_embedding=self._encoder,
        )
        result = UnlimitedOCRForCausalLM._process_image_input(encoder, items)
        torch.testing.assert_close(result, torch.tensor([[1.0], [12.0], [12.0], [2.0]]))


if __name__ == "__main__":
    unittest.main()
