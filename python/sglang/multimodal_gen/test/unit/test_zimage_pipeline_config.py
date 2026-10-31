import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.encoders import BaseEncoderOutput
from sglang.multimodal_gen.configs.pipeline_configs.base import TextConditioningOutput
from sglang.multimodal_gen.configs.pipeline_configs.zimage import (
    ZImagePipelineConfig,
    zimage_postprocess_text,
)
from sglang.multimodal_gen.configs.sample.sampling_params import SamplingParams
from sglang.multimodal_gen.configs.sample.zimage import (
    ZImageSamplingParams,
    ZImageTurboSamplingParams,
)
from sglang.multimodal_gen.runtime.models.dits.zimage import (
    ZImageRMSNorm,
    ZImageTransformer2DModel,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req


class TestZImagePipelineConfig(unittest.TestCase):
    def _make_conditioning_batch(self, seq_lens, num_outputs, negative_seq_lens=None):
        config = ZImagePipelineConfig()
        config.vae_config.post_init()
        batch = Req(
            sampling_params=ZImageSamplingParams(
                prompt="A" if len(seq_lens) == 1 else ["A", "B"],
                num_outputs_per_prompt=num_outputs,
                height=16,
                width=16,
            )
        )
        for negative, lengths in ((False, seq_lens), (True, negative_seq_lens)):
            if lengths is None:
                continue
            hidden = (
                torch.arange(len(lengths) * (max(lengths) + 2) * 4, dtype=torch.float32)
                .reshape(len(lengths), max(lengths) + 2, 4)
                .to(torch.bfloat16)
            )
            # Include tokenizer padding on both sides of the real text.
            attention_mask = torch.zeros(hidden.shape[:2], dtype=torch.long)
            for idx, length in enumerate(lengths):
                attention_mask[idx, 1 : length + 1] = 1
            conditioning = zimage_postprocess_text(
                BaseEncoderOutput(hidden_states=(hidden, hidden, hidden)),
                SimpleNamespace(attention_mask=attention_mask),
            )
            if isinstance(conditioning, TextConditioningOutput):
                embeds = conditioning.prompt_embeds
                mask = conditioning.prompt_embeds_mask
                valid_lens = conditioning.prompt_seq_lens
            else:
                # Keep the legacy single-prompt output for pre-fix reproduction.
                embeds = conditioning
                mask = config.build_text_conditioning_mask(
                    {}, attention_mask, embeds, 0
                )
                valid_lens = config.seq_lens_from_text_conditioning_mask(mask)
            prefix = "negative_prompt" if negative else "prompt"
            setattr(batch, f"{prefix}_embeds", [embeds])
            setattr(batch, f"{prefix}_embeds_mask", [mask])
            setattr(batch, f"{prefix}_seq_lens", [valid_lens])
            setattr(
                batch,
                "negative_attention_mask" if negative else "prompt_attention_mask",
                [mask.to(torch.long)],
            )
        return config, batch

    def _patchify_conditioning(self, config, batch, *, negative=False):
        model = ZImageTransformer2DModel.__new__(ZImageTransformer2DModel)
        getter = (
            config.get_neg_prompt_embeds if negative else config.get_pos_prompt_embeds
        )
        prepare = (
            config.prepare_neg_cond_kwargs
            if negative
            else config.prepare_pos_cond_kwargs
        )
        with patch(
            "sglang.multimodal_gen.configs.pipeline_configs.zimage.get_sp_world_size",
            return_value=1,
        ):
            kwargs = prepare(
                batch, torch.device("cpu"), lambda ids: ids, torch.bfloat16
            )
        return model.patchify_and_embed(
            [torch.zeros(4, 1, 2, 2) for _ in range(batch.batch_size)],
            getter(batch),
            patch_size=2,
            f_patch_size=1,
            caption_valid_lens=kwargs["caption_valid_lens"],
        )

    def test_single_prompt_multi_output_caption_contract(self):
        config, batch = self._make_conditioning_batch([3], num_outputs=3)
        self.assertIs(config.expand_conditioning_to_sample_batch(batch), batch)
        images, captions, _, _, valid_lens, *_ = self._patchify_conditioning(
            config, batch
        )
        self.assertEqual(images.shape, (3, 32, 16))
        self.assertEqual(captions.shape, (3, 32, 4))
        self.assertEqual(valid_lens.tolist(), [3, 3, 3])
        self.assertEqual(batch.prompt_embeds[0].shape, (3, 3, 4))
        torch.testing.assert_close(captions[0], captions[1], rtol=0, atol=0)
        torch.testing.assert_close(captions[0], captions[2], rtol=0, atol=0)

    def test_single_output_preserves_text_and_dtype(self):
        config, batch = self._make_conditioning_batch([3], num_outputs=1)
        embeds = batch.prompt_embeds[0]
        mask = batch.prompt_embeds_mask[0]
        self.assertEqual(embeds.shape, (1, 3, 4))
        self.assertIs(config.expand_conditioning_to_sample_batch(batch), batch)
        self.assertIs(batch.prompt_embeds[0], embeds)
        self.assertIs(batch.prompt_embeds_mask[0], mask)
        captions = config.get_pos_prompt_embeds(batch)
        expected = torch.arange(4, 16).reshape(3, 4).to(torch.bfloat16)
        torch.testing.assert_close(captions[0][:3], expected, rtol=0, atol=0)
        torch.testing.assert_close(captions[0][3:], expected[-1:].repeat(29, 1))
        self.assertEqual(captions[0].dtype, torch.bfloat16)
        self.assertEqual(batch.prompt_attention_mask[0].dtype, torch.long)
        self.assertEqual(mask.dtype, torch.bool)
        self.assertEqual(self._patchify_conditioning(config, batch)[4].tolist(), [3])

    def test_multi_output_expands_positive_and_negative_conditioning(self):
        config, batch = self._make_conditioning_batch(
            [3], num_outputs=3, negative_seq_lens=[33]
        )
        config.expand_conditioning_to_sample_batch(batch)
        for negative, length in ((False, 3), (True, 33)):
            prefix = "negative_prompt" if negative else "prompt"
            embeds = getattr(batch, f"{prefix}_embeds")[0]
            mask = getattr(batch, f"{prefix}_embeds_mask")[0]
            self.assertEqual(embeds.shape, (3, length, 4))
            self.assertEqual(mask.shape, (3, length))
            self.assertTrue(mask.all())
            self.assertEqual(mask.dtype, torch.bool)
            self.assertEqual(embeds.dtype, torch.bfloat16)
            self.assertEqual(
                self._patchify_conditioning(config, batch, negative=negative)[
                    4
                ].tolist(),
                [length] * 3,
            )
        with patch(
            "sglang.multimodal_gen.configs.pipeline_configs.zimage.get_sp_world_size",
            return_value=1,
        ):
            kwargs = config.prepare_neg_cond_kwargs(
                batch, torch.device("cpu"), lambda ids: ids, torch.bfloat16
            )
        self.assertEqual(kwargs["freqs_cis"][0].shape, (64, 3))
        self.assertEqual(kwargs["freqs_cis"][1][0].tolist(), [65, 0, 0])

    def test_multi_prompt_expansion_preserves_order_masks_and_lengths(self):
        config, batch = self._make_conditioning_batch(
            [2, 5], num_outputs=2, negative_seq_lens=[4, 1]
        )
        original = {
            name: getattr(batch, name)[0].clone()
            for name in (
                "prompt_embeds",
                "negative_prompt_embeds",
                "prompt_attention_mask",
                "negative_attention_mask",
                "prompt_embeds_mask",
                "negative_prompt_embeds_mask",
            )
        }
        config.expand_conditioning_to_sample_batch(batch)
        for name, value in original.items():
            expanded = getattr(batch, name)[0]
            torch.testing.assert_close(expanded, value[[0, 0, 1, 1]], rtol=0, atol=0)
            self.assertEqual(expanded.dtype, value.dtype)
            self.assertEqual(expanded.device, value.device)
        self.assertEqual(batch.prompt_seq_lens, [[2, 2, 5, 5]])
        self.assertEqual(batch.negative_prompt_seq_lens, [[4, 4, 1, 1]])
        for negative, expected_lens in ((False, [2, 2, 5, 5]), (True, [4, 4, 1, 1])):
            patched = self._patchify_conditioning(config, batch, negative=negative)
            self.assertEqual(patched[1].shape, (4, 32, 4))
            self.assertEqual(patched[4].tolist(), expected_lens)

    def test_conditioning_expansion_is_idempotent(self):
        config, batch = self._make_conditioning_batch(
            [2, 5], num_outputs=2, negative_seq_lens=[4, 1]
        )
        config.expand_conditioning_to_sample_batch(batch)
        fields = (
            "prompt_embeds",
            "negative_prompt_embeds",
            "prompt_attention_mask",
            "negative_attention_mask",
            "prompt_embeds_mask",
            "negative_prompt_embeds_mask",
            "prompt_seq_lens",
            "negative_prompt_seq_lens",
        )
        original = {name: getattr(batch, name)[0] for name in fields}
        config.expand_conditioning_to_sample_batch(batch)
        for name, value in original.items():
            self.assertIs(getattr(batch, name)[0], value)

    def test_single_output_legacy_2d_getter_is_preserved(self):
        config, batch = self._make_conditioning_batch([3], num_outputs=1)
        with patch(
            "sglang.multimodal_gen.configs.pipeline_configs.zimage.get_sp_world_size",
            return_value=1,
        ):
            batched_kwargs = config.prepare_pos_cond_kwargs(
                batch, torch.device("cpu"), lambda ids: ids, torch.bfloat16
            )
        embeds = batch.prompt_embeds[0]
        if embeds.ndim == 3:
            embeds = embeds[0]
        batch.prompt_embeds = [embeds]
        config.expand_conditioning_to_sample_batch(batch)
        self.assertIs(batch.prompt_embeds[0], embeds)
        expected = torch.cat([embeds, embeds[-1:].repeat(29, 1)])
        torch.testing.assert_close(
            config.get_pos_prompt_embeds(batch)[0], expected, rtol=0, atol=0
        )
        with patch(
            "sglang.multimodal_gen.configs.pipeline_configs.zimage.get_sp_world_size",
            return_value=1,
        ):
            legacy_kwargs = config.prepare_pos_cond_kwargs(
                batch, torch.device("cpu"), lambda ids: ids, torch.bfloat16
            )
        for batched, legacy in zip(
            batched_kwargs["freqs_cis"], legacy_kwargs["freqs_cis"]
        ):
            torch.testing.assert_close(batched, legacy, rtol=0, atol=0)

    def test_multi_output_rejects_invalid_conditioning_dimensions(self):
        for negative in (False, True):
            field_name = "negative_prompt_embeds" if negative else "prompt_embeds"
            for shape in ((1, 4), (3, 4), (1, 1, 3, 4), (2, 3, 4)):
                with self.subTest(field=field_name, shape=shape):
                    config, batch = self._make_conditioning_batch(
                        [3], num_outputs=3, negative_seq_lens=[5]
                    )
                    setattr(batch, field_name, [torch.zeros(shape)])
                    with self.assertRaisesRegex(ValueError, "shape|batch dim"):
                        config.expand_conditioning_to_sample_batch(batch)

    def test_multi_output_rejects_misaligned_masks_and_lengths(self):
        for field_name in (
            "prompt_attention_mask",
            "negative_attention_mask",
            "prompt_embeds_mask",
            "negative_prompt_embeds_mask",
            "prompt_seq_lens",
            "negative_prompt_seq_lens",
        ):
            with self.subTest(field=field_name):
                config, batch = self._make_conditioning_batch(
                    [3], num_outputs=3, negative_seq_lens=[5]
                )
                invalid = [3, 3] if "seq_lens" in field_name else torch.ones(2, 3)
                setattr(batch, field_name, [invalid])
                with self.assertRaisesRegex(ValueError, "expected 1.*3"):
                    config.expand_conditioning_to_sample_batch(batch)

    def test_cfg_activation_and_default_negative_prompt(self):
        for cls in (ZImageSamplingParams, ZImageTurboSamplingParams):
            for scale in (0.0, 0.5, 1.0, 3.5):
                for negative in (None, "", "avoid blur"):
                    with self.subTest(cls=cls.__name__, scale=scale, negative=negative):
                        req = Req(
                            sampling_params=cls(
                                guidance_scale=scale, negative_prompt=negative
                            )
                        )
                        self.assertEqual(req.do_classifier_free_guidance, scale > 0)
                        self.assertEqual(req.negative_prompt, negative or "")
        self.assertFalse(
            Req(sampling_params=ZImageTurboSamplingParams()).do_classifier_free_guidance
        )

    def test_standard_cfg_activation_is_unchanged(self):
        self.assertTrue(
            Req(
                sampling_params=SamplingParams(guidance_scale=1.0),
                do_classifier_free_guidance=True,
            ).do_classifier_free_guidance
        )
        for scale, negative, true_scale, expected in (
            (1.0, "", None, False),
            (3.5, None, None, False),
            (3.5, "", None, True),
            (3.5, "", 1.0, False),
            (1.0, "", 3.5, True),
        ):
            req = Req(
                sampling_params=SamplingParams(
                    guidance_scale=scale,
                    negative_prompt=negative,
                    true_cfg_scale=true_scale,
                )
            )
            self.assertEqual(req.do_classifier_free_guidance, expected)

    def test_cfg_matches_diffusers_fp32_arithmetic(self):
        config = ZImagePipelineConfig()
        pos = torch.tensor([0.1, 0.25, -0.75], dtype=torch.bfloat16)
        neg = torch.tensor([-0.2, 0.5, 0.125], dtype=torch.bfloat16)
        # emulate the native DiT's sign inversion from diffusers' raw predictions
        for scale in (0.5, 1.0, 3.5):
            batch = SimpleNamespace(cfg_normalization=False, guidance_rescale=0.0)
            expected = -(pos.float() + scale * (pos.float() - neg.float()))
            for parallel in (False, True):
                actual = config.cfg_policy.combine(
                    [-pos, -neg], batch, scale, config, cfg_parallel=parallel
                )
                torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                self.assertEqual(actual.dtype, torch.float32)
        torch.testing.assert_close(
            config.cfg_policy.combine([-pos], batch, 0.0, config),
            -pos,
            rtol=0,
            atol=0,
        )
        self.assertTrue(config.cfg_policy.parallel_uses_serial_arithmetic)

    def test_rmsnorm_native_formula(self) -> None:
        norm = ZImageRMSNorm(4, eps=1e-5)
        with torch.no_grad():
            norm.weight.copy_(torch.tensor([1.0, 0.5, 1.5, 2.0]))
        x = torch.tensor(
            [[1.25, 0.5, -0.75, 3.0], [0.1, 2.3, -4.1, 0.7]],
            dtype=torch.bfloat16,
        )

        output = norm(x)
        expected = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-5)
        expected = expected * norm.weight.to(dtype=x.dtype)

        self.assertEqual(output.dtype, x.dtype)
        self.assertTrue(torch.equal(output, expected))

    def test_explicit_sigmas(self) -> None:
        """Z-Image uses the native explicit flow sigmas schedule."""
        config = ZImagePipelineConfig()

        self.assertEqual(
            config.prepare_sigmas(None, 4).tolist(),
            [1.0, 0.75, 0.5, 0.25],
        )

    def test_autocast_disabled(self) -> None:
        """Official Z-Image runs bf16 weights without an outer autocast context."""
        self.assertFalse(ZImagePipelineConfig().enable_autocast)

    @patch("sglang.multimodal_gen.configs.pipeline_configs.zimage.get_sp_world_size")
    def test_image_rope_patch_tokens(self, mock_get_sp_world_size) -> None:
        mock_get_sp_world_size.return_value = 1

        config = ZImagePipelineConfig()
        config.vae_config.post_init()
        batch = SimpleNamespace(
            prompt_embeds=[torch.ones(113, 2560)],
            prompt_seq_lens=[[113]],
            negative_prompt_embeds=None,
            height=480,
            width=640,
        )

        def rotary_emb(pos_ids):
            return pos_ids

        _, image_pos_ids = config.prepare_pos_cond_kwargs(
            batch=batch,
            device=torch.device("cpu"),
            rotary_emb=rotary_emb,
            dtype=torch.float32,
        )["freqs_cis"]

        self.assertEqual(image_pos_ids.shape, (1216, 3))
        self.assertEqual(image_pos_ids[0].tolist(), [129, 0, 0])
        self.assertEqual(image_pos_ids[1199].tolist(), [129, 29, 39])
        self.assertEqual(image_pos_ids[-1].tolist(), [0, 0, 0])

    @patch("sglang.multimodal_gen.configs.pipeline_configs.zimage.get_sp_world_size")
    def test_negative_rope_len(self, mock_get_sp_world_size) -> None:
        """Negative CFG branch should build RoPE positions from negative prompt embeds."""
        mock_get_sp_world_size.return_value = 1

        config = ZImagePipelineConfig()
        pos_seq_len = 19
        neg_seq_len = 45
        batch = SimpleNamespace(
            prompt_embeds=[torch.ones(pos_seq_len, 2560)],
            prompt_seq_lens=[[pos_seq_len]],
            negative_prompt_embeds=[torch.ones(neg_seq_len, 2560)],
            negative_prompt_seq_lens=[[neg_seq_len]],
            height=16,
            width=16,
        )

        def rotary_emb(pos_ids):
            return pos_ids

        neg_kwargs = config.prepare_neg_cond_kwargs(
            batch=batch,
            device=torch.device("cpu"),
            rotary_emb=rotary_emb,
            dtype=torch.float32,
        )

        cap_pos_ids, image_pos_ids = neg_kwargs["freqs_cis"]
        neg_cap_padded_len = 64
        self.assertEqual(cap_pos_ids.shape, (neg_cap_padded_len, 3))
        self.assertEqual(image_pos_ids[0].tolist(), [neg_cap_padded_len + 1, 0, 0])

    def test_batched_rope_offsets(self) -> None:
        model = ZImageTransformer2DModel.__new__(ZImageTransformer2DModel)

        def rotary_emb(pos_ids):
            return (
                pos_ids.to(torch.float32),
                (pos_ids + 1000).to(torch.float32),
            )

        model.rotary_emb = rotary_emb

        images = [torch.zeros(16, 1, 60, 80), torch.zeros(16, 1, 60, 80)]
        cap_feats = [torch.zeros(113, 2560), torch.zeros(177, 2560)]

        cap_freqs, image_freqs = model._build_batched_freqs_cis(
            images,
            cap_feats,
            patch_size=2,
            f_patch_size=1,
            image_target_len=1216,
            cap_target_len=192,
        )

        self.assertEqual(cap_freqs[0].shape, (2, 192, 3))
        self.assertEqual(image_freqs[0].shape, (2, 1216, 3))
        self.assertEqual(image_freqs[0][0, 0].tolist(), [129.0, 0.0, 0.0])
        self.assertEqual(image_freqs[0][1, 0].tolist(), [193.0, 0.0, 0.0])
        self.assertEqual(cap_freqs[0][0, 127].tolist(), [128.0, 0.0, 0.0])
        self.assertEqual(cap_freqs[0][0, 128].tolist(), [0.0, 0.0, 0.0])


if __name__ == "__main__":
    unittest.main()
