"""DeepSeek-V4.1 vision routing admission: only batches that can carry image
tokens take vision_topk; text-only batches stay on the standard TopK path."""

import unittest
from types import SimpleNamespace

from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.multimodal.dsv41.vl_routing import batch_has_images
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _batch(mode, has_images):
    return SimpleNamespace(forward_mode=mode, contains_image_inputs=lambda: has_images)


class TestVisionRoutingAdmission(CustomTestCase):
    def test_unknown_batch_is_admitted(self):
        # No batch to inspect: stay conservative and keep the VL bias available.
        self.assertTrue(batch_has_images(None))

    def test_extend_with_images_is_admitted(self):
        for mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            with self.subTest(mode=mode):
                self.assertTrue(batch_has_images(_batch(mode, True)))

    def test_text_only_extend_is_rejected(self):
        for mode in (ForwardMode.EXTEND, ForwardMode.MIXED):
            with self.subTest(mode=mode):
                self.assertFalse(batch_has_images(_batch(mode, False)))

    def test_decode_modes_are_rejected_even_with_images(self):
        # Image tokens only appear in prefill-side modes; decode-side modes must
        # never take the vision path.
        for mode in (ForwardMode.DECODE, ForwardMode.IDLE):
            with self.subTest(mode=mode):
                self.assertFalse(batch_has_images(_batch(mode, True)))

    def test_rocm_alias_matches_canonical(self):
        try:
            from sglang.srt.models.deepseek_common.amd.deepseek_v2_hip_moe import (
                batch_has_images as hip_batch_has_images,
            )
        except ImportError:
            self.skipTest("ROCm module needs aiter")

        for batch in (
            None,
            _batch(ForwardMode.EXTEND, True),
            _batch(ForwardMode.EXTEND, False),
            _batch(ForwardMode.DECODE, True),
        ):
            self.assertEqual(hip_batch_has_images(batch), batch_has_images(batch))


if __name__ == "__main__":
    unittest.main()
