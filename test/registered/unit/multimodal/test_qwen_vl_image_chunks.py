import unittest

import torch
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from sglang.srt.multimodal.processors.qwen_vl_image_chunks import (
    install_cpu_transport_image_chunks,
)
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=15, stage="base-b", runner_config="1-gpu-small")


class TestQwenImageChunks(CustomTestCase):
    def test_installation_is_idempotent_and_rejects_reconfiguration(self):
        processor = Qwen2VLImageProcessor()
        install_cpu_transport_image_chunks(processor, 2)
        installed = processor._preprocess
        install_cpu_transport_image_chunks(processor, 2)
        self.assertIs(processor._preprocess, installed)
        with self.assertRaises(ValueError):
            install_cpu_transport_image_chunks(processor, 3)
        with self.assertRaises(ValueError):
            install_cpu_transport_image_chunks(Qwen2VLImageProcessor(), 0)

    def _check_images(self, device):
        reference = Qwen2VLImageProcessor()
        chunked = Qwen2VLImageProcessor()
        install_cpu_transport_image_chunks(chunked, 2)
        generator = torch.Generator().manual_seed(42)
        images = [
            torch.randint(
                0, 256, (3, height, width), dtype=torch.uint8, generator=generator
            ).to(device)
            for height, width in [
                (112, 84),
                (84, 112),
                (112, 84),
                (140, 112),
                (112, 112),
            ]
        ]
        # Covers the passthrough, exact chunk, multiple chunks, and partial tail.
        for count in [1, 2, 5]:
            with self.subTest(device=device, count=count):
                expected = reference(
                    images=images[:count], return_tensors="pt", device=device
                )
                actual = chunked(
                    images=images[:count], return_tensors="pt", device=device
                )
                self.assertEqual(set(actual), {"pixel_values", "image_grid_thw"})
                for key in expected:
                    self.assertTrue(
                        torch.equal(actual[key].cpu(), expected[key].cpu()), key
                    )
                    self.assertEqual(actual[key].dtype, expected[key].dtype)
                    self.assertEqual(actual[key].shape, expected[key].shape)
                    self.assertEqual(actual[key].stride(), expected[key].stride())
                    if device == "cuda" and count > 2:
                        self.assertEqual(actual[key].device.type, "cpu")

    def test_cpu_inputs_keep_original_behavior(self):
        self._check_images("cpu")

    @unittest.skipUnless(torch.cuda.is_available(), "requires GPU transforms")
    def test_gpu_chunks_preserve_mixed_shapes_order_and_values(self):
        self._check_images("cuda")


if __name__ == "__main__":
    unittest.main()
