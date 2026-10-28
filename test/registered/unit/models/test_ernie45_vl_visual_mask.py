"""Ernie4.5-VL forward must route every image and video token to the vision experts.

The visual-token mask is built for any prefill batch that carries image or video
inputs, and skipped for decode and text-only batches.
"""

import unittest
from unittest.mock import MagicMock, patch

import torch

from sglang.test.ci.ci_register import register_xpu_ci
from sglang.test.test_utils import CustomTestCase

register_xpu_ci(est_time=10, suite="stage-a-test-1-gpu-xpu")

IM_PATCH_ID, VIDEO_START_ID, VIDEO_END_ID = 100, 101, 102
VIDEO_PAD = 5000


def _make_model():
    from sglang.srt.models.ernie45_vl import Ernie4_5_VLMoeForConditionalGeneration

    model = Ernie4_5_VLMoeForConditionalGeneration.__new__(
        Ernie4_5_VLMoeForConditionalGeneration
    )
    torch.nn.Module.__init__(model)
    model.is_mrope_enabled = False
    model._visual_token_ids_tensor_cache = torch.tensor(
        [IM_PATCH_ID, VIDEO_START_ID, VIDEO_END_ID]
    )
    model.visual_token_mask = None
    model.model = MagicMock()
    model.lm_head = MagicMock()
    model.logits_processor = MagicMock()
    return model


def _make_forward_batch(*, decode: bool, image: bool, video: bool):
    from sglang.srt.managers.schedule_batch import Modality, MultimodalDataItem

    item = MultimodalDataItem(modality=Modality.VIDEO, pad_value=VIDEO_PAD)
    fb = MagicMock()
    fb.forward_mode.is_decode.return_value = decode
    fb.contains_image_inputs.return_value = image
    fb.contains_video_inputs.return_value = video
    fb.mm_inputs = [MagicMock(mm_items=[item])]
    return fb


def _mask_passed_to_language_model(input_ids, forward_batch):
    with patch(
        "sglang.srt.models.ernie45_vl.general_mm_embed_routine"
    ) as embed_routine:
        _make_model().forward(input_ids, torch.zeros_like(input_ids), forward_batch)
    return embed_routine.call_args.kwargs["visual_token_mask"]


class TestErnie45VLVisualTokenMask(CustomTestCase):
    def test_video_only_prefill_masks_video_tokens(self):
        input_ids = torch.tensor(
            [1, VIDEO_START_ID, VIDEO_PAD, VIDEO_PAD, VIDEO_END_ID, 2]
        )
        mask = _mask_passed_to_language_model(
            input_ids, _make_forward_batch(decode=False, image=False, video=True)
        )
        self.assertIsNotNone(mask)
        self.assertEqual(
            mask.flatten().tolist(), [False, True, True, True, True, False]
        )

    def test_decode_and_text_only_batches_skip_the_mask(self):
        for decode, image, video in ((True, True, True), (False, False, False)):
            with self.subTest(decode=decode, image=image, video=video):
                mask = _mask_passed_to_language_model(
                    torch.tensor([1, 2, 3]),
                    _make_forward_batch(decode=decode, image=image, video=video),
                )
                self.assertIsNone(mask)


if __name__ == "__main__":
    unittest.main()
