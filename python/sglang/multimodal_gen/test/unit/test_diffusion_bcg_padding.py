import os
import unittest
from unittest.mock import patch

import torch

from sglang.multimodal_gen.runtime.breakable_cuda_graph_runner import (
    _signature_kwargs,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)


class QwenImageTransformer2DModel(torch.nn.Module):
    pass


class FluxTransformer2DModel(torch.nn.Module):
    pass


class TestDiffusionBCGPadding(unittest.TestCase):
    def setUp(self):
        self.stage = DenoisingStage.__new__(DenoisingStage)
        self.qwen_model = QwenImageTransformer2DModel()
        self.flux_model = FluxTransformer2DModel()

    def _qwen_kwargs(self, seq_len: int, *, fill: float = 1.0):
        return {
            "hidden_states": torch.zeros(1, 4096, 64),
            "timestep": torch.zeros(1),
            "encoder_hidden_states": [
                torch.full((1, seq_len, 3584), fill, dtype=torch.float32)
            ],
            "encoder_hidden_states_mask": None,
            "txt_seq_lens": [seq_len],
            "freqs_cis": (
                torch.zeros(4096, 128, dtype=torch.float32),
                torch.ones(seq_len, 128, dtype=torch.float32),
            ),
            "img_shapes": [[(1, 64, 64)]],
        }

    def test_qwen_prompt_lengths_share_bucket_signature(self):
        with patch.dict(os.environ, {"SGLANG_BCG_TEXT_BUCKETS": "256,512,2048"}):
            short = self.stage._bcg_pad_prompt_kwargs(
                self._qwen_kwargs(19), current_model=self.qwen_model
            )
            longer = self.stage._bcg_pad_prompt_kwargs(
                self._qwen_kwargs(47), current_model=self.qwen_model
            )

        self.assertEqual(short["encoder_hidden_states"][0].shape, (1, 256, 3584))
        self.assertEqual(longer["encoder_hidden_states"][0].shape, (1, 256, 3584))
        self.assertEqual(short["encoder_hidden_states_mask"].shape, (1, 256))
        self.assertTrue(short["encoder_hidden_states_mask"][0, :19].all())
        self.assertFalse(short["encoder_hidden_states_mask"][0, 19:].any())
        self.assertEqual(short["freqs_cis"][1].shape, (256, 128))
        self.assertEqual(short["txt_seq_lens"], [256])
        self.assertEqual(longer["txt_seq_lens"], [256])
        self.assertEqual(_signature_kwargs(short), _signature_kwargs(longer))

    def test_qwen_prompt_content_changes_do_not_change_signature(self):
        with patch.dict(os.environ, {"SGLANG_BCG_TEXT_BUCKETS": "256,512,2048"}):
            first = self.stage._bcg_pad_prompt_kwargs(
                self._qwen_kwargs(47, fill=1.0), current_model=self.qwen_model
            )
            second = self.stage._bcg_pad_prompt_kwargs(
                self._qwen_kwargs(47, fill=2.0), current_model=self.qwen_model
            )

        self.assertFalse(
            torch.equal(
                first["encoder_hidden_states"][0],
                second["encoder_hidden_states"][0],
            )
        )
        self.assertEqual(_signature_kwargs(first), _signature_kwargs(second))

    def test_qwen_prompt_lengths_in_different_buckets_do_not_share_signature(self):
        with patch.dict(os.environ, {"SGLANG_BCG_TEXT_BUCKETS": "256,512,2048"}):
            small = self.stage._bcg_pad_prompt_kwargs(
                self._qwen_kwargs(47), current_model=self.qwen_model
            )
            medium = self.stage._bcg_pad_prompt_kwargs(
                self._qwen_kwargs(300), current_model=self.qwen_model
            )

        self.assertEqual(small["encoder_hidden_states"][0].shape[1], 256)
        self.assertEqual(medium["encoder_hidden_states"][0].shape[1], 512)
        self.assertNotEqual(_signature_kwargs(small), _signature_kwargs(medium))

    def test_non_qwen_txt_seq_lens_and_freqs_cis_do_not_take_qwen_path(self):
        kwargs = self._qwen_kwargs(47)
        with patch.dict(os.environ, {"SGLANG_BCG_TEXT_BUCKETS": "256,512,2048"}):
            out = self.stage._bcg_pad_prompt_kwargs(
                kwargs, current_model=self.flux_model
            )

        self.assertIs(out, kwargs)
        self.assertIsNone(out["encoder_hidden_states_mask"])
        self.assertEqual(out["encoder_hidden_states"][0].shape[1], 47)
        self.assertEqual(out["txt_seq_lens"], [47])

    def test_generic_prompt_padding_keeps_single_bucket_env_compatibility(self):
        kwargs = {
            "hidden_states": torch.zeros(1, 16, 64),
            "timestep": torch.zeros(1),
            "encoder_hidden_states": torch.ones(1, 17, 128),
            "encoder_attention_mask": torch.ones(1, 17, dtype=torch.bool),
        }

        with patch.dict(os.environ, {"SGLANG_BCG_TEXT_BUCKET": "64"}, clear=False):
            out = self.stage._bcg_pad_prompt_kwargs(kwargs)

        self.assertEqual(out["encoder_hidden_states"].shape, (1, 64, 128))
        self.assertEqual(out["encoder_attention_mask"].shape, (1, 64))
        self.assertTrue(out["encoder_attention_mask"][0, :17].all())
        self.assertFalse(out["encoder_attention_mask"][0, 17:].any())

    def test_generic_masked_prompt_padding_covers_text_aux_tensors(self):
        def kwargs(seq_len: int):
            return {
                "hidden_states": torch.zeros(1, 16, 64),
                "timestep": torch.zeros(1),
                "encoder_hidden_states": [torch.ones(1, seq_len, 128)],
                "encoder_hidden_states_mask": torch.ones(1, seq_len, dtype=torch.bool),
                "text_ids": torch.arange(seq_len).view(1, seq_len),
                "txt_freqs_cis": torch.zeros(seq_len, 32),
            }

        with patch.dict(os.environ, {"SGLANG_BCG_TEXT_BUCKETS": "64,128"}):
            first = self.stage._bcg_pad_prompt_kwargs(
                kwargs(17), current_model=self.flux_model
            )
            second = self.stage._bcg_pad_prompt_kwargs(
                kwargs(41), current_model=self.flux_model
            )

        self.assertEqual(first["encoder_hidden_states"][0].shape, (1, 64, 128))
        self.assertEqual(second["encoder_hidden_states"][0].shape, (1, 64, 128))
        self.assertEqual(first["encoder_hidden_states_mask"].shape, (1, 64))
        self.assertEqual(first["text_ids"].shape, (1, 64))
        self.assertEqual(first["txt_freqs_cis"].shape, (64, 32))
        self.assertEqual(_signature_kwargs(first), _signature_kwargs(second))

    def test_generic_masked_prompt_padding_supports_unbatched_text_embeddings(self):
        def kwargs(seq_len: int):
            return {
                "hidden_states": torch.zeros(1, 16, 64),
                "timestep": torch.zeros(1),
                "encoder_hidden_states": [torch.ones(seq_len, 128)],
                "encoder_attention_mask": [torch.ones(seq_len, 128, dtype=torch.long)],
                "encoder_hidden_states_mask": [
                    torch.ones(seq_len, 128, dtype=torch.long)
                ],
            }

        with patch.dict(os.environ, {"SGLANG_BCG_TEXT_BUCKETS": "64,128"}):
            first = self.stage._bcg_pad_prompt_kwargs(
                kwargs(22), current_model=self.flux_model
            )
            second = self.stage._bcg_pad_prompt_kwargs(
                kwargs(32), current_model=self.flux_model
            )

        self.assertEqual(first["encoder_hidden_states"][0].shape, (64, 128))
        self.assertEqual(first["encoder_attention_mask"][0].shape, (64, 128))
        self.assertEqual(second["encoder_hidden_states"][0].shape, (64, 128))
        self.assertEqual(_signature_kwargs(first), _signature_kwargs(second))


if __name__ == "__main__":
    unittest.main()
