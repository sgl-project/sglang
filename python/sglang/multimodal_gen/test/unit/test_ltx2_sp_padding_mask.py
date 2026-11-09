import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising import (
    LTX2DenoisingStage,
)
from sglang.test.test_utils import CustomTestCase


class TestLTX2SPPaddingMask(CustomTestCase):
    KEY = "sp_valid_len"

    def _build(
        self, valid, *, seq_len, batch_size=2, has_padding=True, quality="lossless"
    ):
        batch = SimpleNamespace(
            quality=quality, **({self.KEY: valid} if valid is not None else {})
        )
        return LTX2DenoisingStage._build_ltx2_sp_padding_mask(
            batch,
            seq_len=seq_len,
            batch_size=batch_size,
            key=self.KEY,
            has_padding=has_padding,
            device=torch.device("cpu"),
        )

    def test_missing_valid_returns_none(self):
        # Attribute absent on the batch -> no mask.
        self.assertIsNone(self._build(None, seq_len=8))

    def test_no_padding_returns_none(self):
        # valid == seq_len: an all-True mask would be a no-op, so return None
        # to keep the fused (unmasked) attention path.
        self.assertIsNone(self._build(8, seq_len=8, has_padding=False))

    def test_exact_no_padding_preserves_reference_mask(self):
        mask = self._build(8, seq_len=8, has_padding=False, quality="exact")
        self.assertTrue(torch.equal(mask, torch.ones(2, 8, dtype=torch.bool)))

    def test_high_no_padding_returns_none(self):
        self.assertIsNone(self._build(8, seq_len=8, has_padding=False, quality="high"))

    def test_valid_greater_than_seq_len_returns_none(self):
        # Defensive: valid > seq_len still means no padding.
        self.assertIsNone(self._build(12, seq_len=8, has_padding=False))

    def test_padding_returns_real_mask(self):
        mask = self._build(5, seq_len=8, batch_size=2)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (2, 8))
        self.assertEqual(mask.dtype, torch.bool)
        expected = torch.tensor([True] * 5 + [False] * 3, dtype=torch.bool)
        for row in mask:
            self.assertTrue(torch.equal(row, expected))

    def test_zero_valid_returns_all_false_mask(self):
        mask = self._build(0, seq_len=4, batch_size=1)
        self.assertIsNotNone(mask)
        self.assertEqual(mask.shape, (1, 4))
        self.assertFalse(mask.any().item())

    def test_sharding_keeps_mask_presence_consistent_across_ranks(self):
        config = LTX2PipelineConfig()
        target = "sglang.multimodal_gen.configs.pipeline_configs.ltx_2"
        for modality in ("video", "audio"):
            for world_size in (2, 4):
                for frames in (1, 7, 8):
                    with self.subTest(
                        modality=modality, world_size=world_size, frames=frames
                    ):
                        masks = []
                        for rank in range(world_size):
                            batch = Req(
                                height=config.vae_scale_factor * config.patch_size,
                                width=config.vae_scale_factor * config.patch_size * 2,
                            )
                            tokens_per_frame = 2 if modality == "video" else 1
                            latents = torch.ones(1, frames * tokens_per_frame, 4)
                            with (
                                mock.patch(
                                    f"{target}.get_sp_world_size",
                                    return_value=world_size,
                                ),
                                mock.patch(
                                    f"{target}.get_sp_parallel_rank", return_value=rank
                                ),
                            ):
                                shard, did_shard = (
                                    config.shard_latents_for_sp(batch, latents)
                                    if modality == "video"
                                    else config.shard_audio_latents_for_sp(
                                        batch, latents
                                    )
                                )
                            self.assertTrue(did_shard)
                            mask = LTX2DenoisingStage._build_ltx2_sp_padding_mask(
                                batch,
                                seq_len=shard.shape[1],
                                batch_size=1,
                                key=f"sp_{modality}_valid_token_count",
                                has_padding=vars(batch)[f"sp_{modality}_has_padding"],
                                device=shard.device,
                            )
                            masks.append(mask)
                            if frames % world_size:
                                self.assertIsNotNone(mask)
                                self.assertTrue(
                                    torch.equal(mask, shard[:, :, 0].bool())
                                )
                            else:
                                self.assertIsNone(mask)
                        if frames % world_size:
                            gathered = torch.cat(masks, dim=1)
                            self.assertEqual(
                                gathered.sum().item(), frames * tokens_per_frame
                            )


if __name__ == "__main__":
    unittest.main()
