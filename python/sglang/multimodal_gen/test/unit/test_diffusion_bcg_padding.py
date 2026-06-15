import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.glm_image import (
    GlmImagePipelineConfig,
)
from sglang.multimodal_gen.runtime.breakable_cuda_graph_runner import (
    _signature_kwargs,
)
from sglang.multimodal_gen.runtime.models.dits.zimage import ZImageTransformer2DModel
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import (
    DenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.glm_image import (
    GlmImageBeforeDenoisingStage,
)


class QwenImageTransformer2DModel(torch.nn.Module):
    pass


class FluxTransformer2DModel(torch.nn.Module):
    pass


class ZImageFakeTransformer2DModel(torch.nn.Module):
    def rotary_emb(self, pos_ids):
        return (
            torch.zeros(pos_ids.shape[0], 64, device=pos_ids.device),
            torch.ones(pos_ids.shape[0], 64, device=pos_ids.device),
        )


class TestDiffusionBCGPadding(unittest.TestCase):
    def setUp(self):
        self.stage = DenoisingStage.__new__(DenoisingStage)
        self.qwen_model = QwenImageTransformer2DModel()
        self.flux_model = FluxTransformer2DModel()
        self.zimage_model = ZImageFakeTransformer2DModel()

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
                "txt_seq_lens": [seq_len],
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
        self.assertEqual(first["txt_seq_lens"], [64])
        self.assertEqual(second["txt_seq_lens"], [64])
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

    def test_zimage_prompt_padding_preserves_valid_mask_and_rebuilds_cap_rope(self):
        image_freqs = (torch.zeros(4096, 64), torch.ones(4096, 64))

        def kwargs(seq_len: int):
            cap_freqs = (torch.zeros(32, 64), torch.ones(32, 64))
            return {
                "hidden_states": torch.zeros(1, 16, 1, 128, 128),
                "timestep": torch.zeros(1),
                "encoder_hidden_states": [torch.ones(seq_len, 2560)],
                "encoder_hidden_states_mask": [
                    torch.ones(1, seq_len, dtype=torch.bool)
                ],
                "freqs_cis": (cap_freqs, image_freqs),
            }

        with patch.dict(os.environ, {"SGLANG_BCG_TEXT_BUCKETS": "64,128"}):
            first = self.stage._bcg_pad_prompt_kwargs(
                kwargs(17), current_model=self.zimage_model
            )
            second = self.stage._bcg_pad_prompt_kwargs(
                kwargs(41), current_model=self.zimage_model
            )

        self.assertEqual(first["encoder_hidden_states"][0].shape, (64, 2560))
        self.assertEqual(second["encoder_hidden_states"][0].shape, (64, 2560))
        self.assertEqual(first["encoder_hidden_states_mask"][0].shape, (1, 64))
        self.assertEqual(first["encoder_hidden_states_mask"][0].sum().item(), 17)
        self.assertEqual(first["freqs_cis"][0][0].shape, (64, 64))
        self.assertEqual(second["freqs_cis"][0][0].shape, (64, 64))
        self.assertEqual(_signature_kwargs(first), _signature_kwargs(second))

    def test_zimage_caption_valid_mask_comes_from_bcg_padded_mask(self):
        mask = torch.tensor([[True, True, True, False, False]])
        valid_mask = ZImageTransformer2DModel._caption_valid_mask_from_mask(
            [mask], batch_size=1, max_seq_len=5
        )

        self.assertEqual(valid_mask.shape, (1, 5))
        self.assertTrue(torch.equal(valid_mask, mask))

    def test_zimage_mask_padding_replaces_only_invalid_caption_tokens(self):
        tensor = torch.arange(15, dtype=torch.float32).view(1, 5, 3)
        valid_mask = torch.tensor([[True, True, False, False, False]])
        pad_token = torch.tensor([[100.0, 101.0, 102.0]])

        out = ZImageTransformer2DModel._replace_padding_with_token_mask(
            tensor, valid_mask, pad_token
        )

        self.assertTrue(torch.equal(out[:, :2], tensor[:, :2]))
        self.assertTrue(torch.equal(out[:, 2:], pad_token.expand(1, 3, 3)))

    def test_glm_condition_image_uses_target_request_size_for_warmup(self):
        self.assertEqual(
            GlmImageBeforeDenoisingStage._condition_image_preprocess_size(
                height=1024, width=1024, multiple_of=16
            ),
            (1024, 1024),
        )
        self.assertEqual(
            GlmImageBeforeDenoisingStage._condition_image_preprocess_size(
                height=1025, width=769, multiple_of=16
            ),
            (1024, 768),
        )
        self.assertEqual(
            GlmImageBeforeDenoisingStage._condition_image_preprocess_size(
                height=64,
                width=64,
                multiple_of=16,
                prior_token_shape=(32, 32),
            ),
            (512, 512),
        )

    def test_glm_t2i_prompt_signature_omits_empty_kv_cache_object(self):
        cfg = GlmImagePipelineConfig.__new__(GlmImagePipelineConfig)
        cfg.get_freqs_cis = lambda *args, **kwargs: "freqs"
        batch = SimpleNamespace(
            prior_token_id=torch.ones(1, 4096, dtype=torch.long),
            prior_token_drop_cond=torch.zeros(1, 4096, dtype=torch.bool),
            prior_token_drop_uncond=torch.ones(1, 4096, dtype=torch.bool),
            crop_coords=torch.zeros(1, 2),
            target_size=torch.tensor([[1024, 1024]]),
            kv_caches=object(),
            prior_token_image_ids=None,
        )

        pos = cfg.prepare_pos_cond_kwargs(batch, None, None, None)
        neg = cfg.prepare_neg_cond_kwargs(batch, None, None, None)

        self.assertNotIn("kv_caches", pos)
        self.assertNotIn("kv_caches_mode", pos)
        self.assertNotIn("kv_caches", neg)
        self.assertNotIn("kv_caches_mode", neg)

        batch.prior_token_image_ids = [torch.ones(4096, dtype=torch.long)]
        pos = cfg.prepare_pos_cond_kwargs(batch, None, None, None)
        neg = cfg.prepare_neg_cond_kwargs(batch, None, None, None)

        self.assertIn("kv_caches", pos)
        self.assertEqual(pos["kv_caches_mode"], "read")
        self.assertIn("kv_caches", neg)
        self.assertEqual(neg["kv_caches_mode"], "skip")


if __name__ == "__main__":
    unittest.main()
