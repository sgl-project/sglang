"""CPU-only regression tests for the Gemma-4 multimodal channel guard.

Two bugs combined to make a single non-RGB image (e.g. a grayscale JPEG
served by a public CDN) crash the entire Gemma-4 SGLang scheduler:

  A. The GPU-decode path in base_processor._load_single_item only
     applied RGB normalization to PIL images; torch.Tensor outputs
     (nvJPEG) kept their source channel count, so a 1-channel image
     reached the Gemma-4 patch embedder and produced patches of
     width patch**2 * 1 instead of patch**2 * 3, crashing the linear
     projection with `mat1 and mat2 shapes cannot be multiplied`.

  B. That RuntimeError propagated unchecked through
     embed_mm_inputs -> run_batch -> the scheduler event loop, where
     the unhandled exception SIGQUIT'd the scheduler subprocess and
     took every other unrelated in-flight request in the batch down
     with it.

These tests exercise both fixes without touching CUDA / the GPU.
"""

import unittest
from unittest.mock import MagicMock

import torch

from sglang.srt.managers.mm_utils import MultimodalEmbeddingError, embed_mm_inputs
from sglang.srt.managers.schedule_batch import Modality
from sglang.srt.multimodal.processors.base_processor import BaseMultimodalProcessor


class TestChannelNormalizationTensorPath(unittest.TestCase):
    """Fix A: GPU-decode (torch.Tensor) path normalizes non-3-channel images."""

    def _load(self, tensor):
        # Patch the I/O hook so we can supply a synthetic decoded tensor.
        import sglang.srt.multimodal.processors.base_processor as base_mod

        original = base_mod.load_image
        try:
            base_mod.load_image = lambda data, gpu: (tensor, None)
            return BaseMultimodalProcessor._load_single_item(
                data=b"unused",
                modality=Modality.IMAGE,
                discard_alpha_channel=True,
            )
        finally:
            base_mod.load_image = original

    def test_grayscale_tensor_expanded_to_rgb(self):
        gray = torch.zeros((1, 8, 8), dtype=torch.uint8)
        out = self._load(gray)
        self.assertEqual(out.shape, (3, 8, 8))

    def test_la_tensor_expanded_to_rgb_drops_alpha(self):
        la = torch.zeros((2, 8, 8), dtype=torch.uint8)
        out = self._load(la)
        self.assertEqual(out.shape, (3, 8, 8))

    def test_rgba_tensor_drops_alpha(self):
        rgba = torch.zeros((4, 8, 8), dtype=torch.uint8)
        out = self._load(rgba)
        self.assertEqual(out.shape, (3, 8, 8))

    def test_rgb_tensor_passthrough(self):
        rgb = torch.zeros((3, 8, 8), dtype=torch.uint8)
        out = self._load(rgb)
        # Same object — no normalization needed.
        self.assertEqual(out.shape, (3, 8, 8))


class TestMultimodalEmbeddingErrorIsolation(unittest.TestCase):
    """Fix B: embed_mm_inputs translates encoder RuntimeError to typed error."""

    def test_runtime_error_is_translated(self):
        # Simulate a mm-input list with one image item whose embedder raises.
        mm_item = MagicMock()
        mm_item.is_modality = lambda modality: modality == Modality.IMAGE
        mm_item.pad_value = 0
        mm_item.offsets = [(0, 1)]
        mm_inputs = MagicMock()
        mm_inputs.mm_items = [mm_item]

        def boom(items):
            raise RuntimeError(
                "mat1 and mat2 shapes cannot be multiplied (2520x256 and 768x1152)"
            )

        multimodal_model = MagicMock()
        # get_image_feature -> raises
        multimodal_model.get_image_feature = boom
        # No deepstack
        del multimodal_model.deepstack_visual_indexes

        input_ids = torch.zeros(8, dtype=torch.long)
        input_embedding = torch.nn.Embedding(32, 16)

        with self.assertRaises(MultimodalEmbeddingError) as ctx:
            embed_mm_inputs(
                mm_inputs_list=[mm_inputs],
                extend_prefix_lens=[0],
                extend_seq_lens=[1],
                input_ids=input_ids,
                input_embedding=input_embedding,
                multimodal_model=multimodal_model,
            )

        self.assertEqual(ctx.exception.modality, Modality.IMAGE)
        self.assertEqual(ctx.exception.num_items, 1)
        self.assertIsInstance(ctx.exception.original_error, RuntimeError)
        self.assertIn("mat1 and mat2", str(ctx.exception.original_error))


if __name__ == "__main__":
    unittest.main(verbosity=2)
