"""The Unlimited-OCR v2 vision encoder run in slices, against the one-shot encoder on MPS."""

import unittest
from types import SimpleNamespace

import torch

from sglang.srt.models.deepseek_ocr import MlpProjector, build_clip_l, build_sam_vit_b
from sglang.srt.models.unlimited_ocr import UnlimitedOCRForCausalLM
from sglang.srt.models.unlimited_ocr_v2 import UnlimitedOCRV2ForCausalLM
from sglang.srt.models.unlimited_ocr_v2_support.crop_transfer import (
    UnlimitedOcrCropTransfer,
)
from sglang.test.ci.ci_register import register_mps_ci
from sglang.test.test_utils import CustomTestCase

register_mps_ci(est_time=60, suite="stage-a-unit-test-mps")


class _VisionOnly(UnlimitedOCRV2ForCausalLM):
    """The vision tower with random weights; no language model."""

    def __init__(self):
        torch.nn.Module.__init__(self)
        dev, dtype = "mps", torch.bfloat16
        self.sam_model = build_sam_vit_b().to(dev, dtype).eval()
        self.vision_model = build_clip_l().to(dev, dtype).eval()
        self.projector = (
            MlpProjector(projector_type="linear", input_dim=2048, n_embed=1280)
            .to(dev, dtype)
            .eval()
        )
        self.image_newline = torch.nn.Parameter(
            torch.randn(1280, device=dev, dtype=dtype), requires_grad=False
        )
        self.view_seperator = torch.nn.Parameter(
            torch.randn(1280, device=dev, dtype=dtype), requires_grad=False
        )
        self.crop_transfer = UnlimitedOcrCropTransfer(model_dtype=dtype)


@unittest.skipUnless(torch.backends.mps.is_available(), "requires MPS")
class TestUocrVisionStepsMps(CustomTestCase):
    def _drain(self, model, steps):
        n = 0
        while True:
            try:
                with model.vision_step_context():
                    next(steps)
                n += 1
            except StopIteration as done:
                return done.value, n

    def test_slices_give_the_same_tensor_as_the_one_shot_encoder(self):
        torch.manual_seed(0)
        model = _VisionOnly()
        self.addCleanup(model.crop_transfer.close)
        pv = torch.randn(1, 1, 3, 1024, 1024, device="mps", dtype=torch.bfloat16)
        # Six local tiles: past two, a SAM window block runs as two slices.
        crops = torch.randn(1, 6, 3, 640, 640, dtype=torch.bfloat16)
        ic = crops[None].to("mps")
        sc = torch.tensor([[[3, 2]]], dtype=torch.long)

        for tiles in (pv[:, 0], ic[0, 0]):
            with model.vision_step_context():
                one_shot = UnlimitedOCRForCausalLM._encode_ocr1_features(model, tiles)
            sliced, _ = self._drain(model, model._encode_ocr1_feature_steps(tiles))
            self.assertTrue(torch.equal(one_shot, sliced))

        one_shot = torch.cat(model._pixel_values_to_embedding(pv, ic, sc, [True]))
        pages, n = self._drain(
            model, model._pixel_values_to_embedding_steps(pv, ic, sc, [True])
        )
        self.assertGreater(n, 30)
        self.assertTrue(torch.equal(one_shot, torch.cat(pages)))

        item = SimpleNamespace(
            feature=pv[0],
            images_crop=crops,
            images_spatial_crop=sc,
            has_local_crops=True,
        )
        sliced, _ = self._drain(model, model.get_image_feature_steps([item]))
        self.assertTrue(torch.equal(model.get_image_feature([item]), sliced))


if __name__ == "__main__":
    unittest.main()
