# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 config wiring.

These pin the handful of places where LTX-2.5 diverges from LTX-2 and where a
silent regression would produce wrong output rather than an error. Everything
here is CPU/meta-device only -- no weights, no GPU.
"""

import unittest

from sglang.multimodal_gen.configs.models.adapter.ltx_2_connector import (
    LTX2ConnectorArchConfig,
)
from sglang.multimodal_gen.configs.models.dits.ltx_2 import LTX2ArchConfig
from sglang.multimodal_gen.configs.models.dits.ltx_2_5 import LTX25ArchConfig
from sglang.multimodal_gen.configs.models.vaes.ltx_2_5_video import (
    LTX25VideoVAEArchConfig,
)
from sglang.multimodal_gen.configs.models.vaes.ltx_video import LTXVideoVAEArchConfig
from sglang.multimodal_gen.configs.models.vocoder.ltx_vocoder import LTXVocoderConfig
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import LTX2PipelineConfig
from sglang.multimodal_gen.configs.pipeline_configs.ltx_2_5 import (
    LTX25_DISTILLED_SIGMA_VALUES,
    LTX25PipelineConfig,
)


class TestLTX25DiTConfig(unittest.TestCase):
    def test_inherits_ltx23_audio_video_base(self):
        arch = LTX25ArchConfig()
        self.assertTrue(arch.apply_gated_attention)
        self.assertTrue(arch.cross_attention_adaln)
        # `use_prompt_embeddings: false` upstream -- the caption projection
        # lives in the connector, not the DiT.
        self.assertTrue(arch.caption_proj_before_connector)
        self.assertEqual(arch.rope_type.value, "split")
        self.assertTrue(arch.double_precision_rope)

    def test_feed_forward_bias_is_video_only(self):
        # LTX-2.5 checkpoints have no `ff.net.*.bias` for the video branch but
        # do for the audio one.
        arch = LTX25ArchConfig()
        self.assertFalse(arch.ff_bias)
        self.assertTrue(arch.audio_ff_bias)

    def test_param_names_mapping_extends_ltx2(self):
        # Regression: the DiT used to read this mapping off a class attribute
        # pinned to LTX2ArchConfig, so the LTX-2.5 renames never reached the
        # loader and weight loading fell back to diffusers.
        arch = LTX25ArchConfig()
        for rule in LTX2ArchConfig().param_names_mapping:
            self.assertIn(rule, arch.param_names_mapping)
        self.assertIn(r"^prompt_adaln\.(.*)$", arch.param_names_mapping)
        self.assertIn(r"^audio_prompt_adaln\.(.*)$", arch.param_names_mapping)

    def test_prompt_adaln_rename_round_trips(self):
        from sglang.multimodal_gen.runtime.loader.utils import get_param_names_mapping

        arch = LTX25ArchConfig()
        forward = get_param_names_mapping(arch.param_names_mapping)
        reverse = get_param_names_mapping(arch.reverse_param_names_mapping)
        for key in ("prompt_adaln.linear.weight", "audio_prompt_adaln.linear.bias"):
            mapped = forward(key)[0]
            self.assertTrue(mapped.startswith(key.split(".")[0] + "_single."), mapped)
            self.assertEqual(reverse(mapped)[0], key)

    def test_ltx2_defaults_unchanged(self):
        # The shared LTX-2 config must keep its original behaviour.
        arch = LTX2ArchConfig()
        self.assertTrue(arch.ff_bias)
        self.assertTrue(arch.audio_ff_bias)
        self.assertFalse(arch.use_keyframes_abs_pos_embedding)


class TestLTX25VAEConfig(unittest.TestCase):
    # Reversed like the sibling per-stage lists, `upsample_type` would give the
    # wrong upsampler strides. Upstream indexes it in decoder order.
    EXPECTED_STRIDES = {
        "spatiotemporal": (2, 2, 2),
        "temporal": (2, 1, 1),
        "spatial": (1, 2, 2),
    }

    def test_upsample_type_order_is_decoder_order(self):
        arch = LTX25VideoVAEArchConfig()
        self.assertEqual(
            list(arch.upsample_type),
            ["spatiotemporal", "spatiotemporal", "temporal", "spatial"],
        )

    def test_ltx2_defaults_to_all_spatiotemporal(self):
        # `None` must keep LTX-2 bit-identical.
        self.assertIsNone(LTXVideoVAEArchConfig().upsample_type)

    def test_decoder_builds_expected_upsampler_strides(self):
        import torch

        from sglang.multimodal_gen.runtime.models.vaes.ltx_2_vae import (
            LTX2VideoDecoder3d,
        )

        arch = LTX25VideoVAEArchConfig()
        with torch.device("meta"):
            decoder = LTX2VideoDecoder3d(
                in_channels=arch.latent_channels,
                out_channels=arch.out_channels,
                block_out_channels=arch.decoder_block_out_channels,
                spatio_temporal_scaling=arch.decoder_spatio_temporal_scaling,
                layers_per_block=arch.decoder_layers_per_block,
                patch_size=arch.patch_size,
                patch_size_t=arch.patch_size_t,
                inject_noise=arch.decoder_inject_noise,
                upsample_residual=arch.upsample_residual,
                upsample_factor=arch.upsample_factor,
                upsample_type=arch.upsample_type,
                spatial_padding_mode=arch.decoder_spatial_padding_mode,
            )

        actual = [tuple(b.upsamplers[0].stride) for b in decoder.up_blocks]
        expected = [self.EXPECTED_STRIDES[t] for t in arch.upsample_type]
        self.assertEqual(actual, expected)


class TestLTX25ConnectorConfig(unittest.TestCase):
    """The connector configures itself from the checkpoint's own config.json.

    Field names there are diffusers'; SGLang's module reads different ones. If
    the derivation breaks, LTX-2.5 silently falls back to the LTX-2.0 branch
    (one shared `text_proj_in`) and produces garbage embeddings instead of
    failing.
    """

    LTX25_CONNECTOR_CONFIG = {
        "caption_channels": 3840,
        "text_proj_in_factor": 49,
        "per_modality_projections": True,
        "video_hidden_dim": 4096,
        "audio_hidden_dim": 2048,
        "video_gated_attn": True,
        "audio_gated_attn": True,
        "video_connector_num_layers": 8,
        "audio_connector_num_layers": 8,
        "audio_connector_attention_head_dim": 64,
    }

    def test_derives_per_modality_projection_dims(self):
        arch = LTX2ConnectorArchConfig(**self.LTX25_CONNECTOR_CONFIG)
        self.assertEqual(arch.feature_extractor_in_features, 3840 * 49)
        self.assertEqual(arch.video_feature_extractor_out_features, 4096)
        self.assertEqual(arch.audio_feature_extractor_out_features, 2048)
        self.assertTrue(arch.connector_apply_gated_attention)

    def test_ltx2_keeps_shared_projection(self):
        arch = LTX2ConnectorArchConfig()
        self.assertFalse(arch.per_modality_projections)
        self.assertEqual(arch.feature_extractor_in_features, 0)
        self.assertFalse(arch.connector_apply_gated_attention)

    def test_diffusers_projection_names_are_mapped(self):
        arch = LTX2ConnectorArchConfig()
        self.assertIn(r"^video_text_proj_in\.(.*)$", arch.param_names_mapping)
        self.assertIn(r"^audio_text_proj_in\.(.*)$", arch.param_names_mapping)


class TestLTX25VocoderConfig(unittest.TestCase):
    """LTX-2.5 ships `LTX2VocoderWithBWE` with a flat diffusers config, while
    SGLang's BWE implementation expects the nested ltx-core shape."""

    LTX25_VOCODER_CONFIG = {
        "hidden_channels": 1536,
        "upsample_factors": [5, 2, 2, 2, 2, 2],
        "upsample_kernel_sizes": [11, 4, 4, 4, 4, 4],
        "resnet_kernel_sizes": [3, 7, 11],
        "act_fn": "snakebeta",
        "input_sampling_rate": 16000,
        "output_sampling_rate": 48000,
        "filter_length": 512,
        "window_length": 512,
        "hop_length": 80,
        "num_mel_channels": 64,
        "bwe_hidden_channels": 512,
        "bwe_upsample_factors": [6, 5, 2, 2, 2],
        "bwe_upsample_kernel_sizes": [12, 11, 4, 4, 4],
        "bwe_resnet_kernel_sizes": [3, 7, 11],
        "bwe_act_fn": "snakebeta",
    }

    def test_builds_nested_bwe_config(self):
        config = LTXVocoderConfig()
        config.update_model_arch(dict(self.LTX25_VOCODER_CONFIG))
        nested = config.arch_config.vocoder

        self.assertIsNotNone(nested)
        self.assertIn("bwe", nested)
        self.assertEqual(nested["vocoder"]["upsample_initial_channel"], 1536)
        self.assertEqual(nested["bwe"]["upsample_initial_channel"], 512)
        # The base stack synthesises at the BWE's input rate, not the final one.
        self.assertEqual(nested["bwe"]["input_sampling_rate"], 16000)
        self.assertEqual(nested["bwe"]["output_sampling_rate"], 48000)
        self.assertEqual(nested["bwe"]["num_mels"], 64)

    def test_ltx2_stays_on_the_non_bwe_branch(self):
        # No `bwe_upsample_factors` -> no nested config -> original code path.
        self.assertIsNone(LTXVocoderConfig().arch_config.vocoder)

    def test_vocoder_with_bwe_class_name_resolves(self):
        from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

        cls, _ = ModelRegistry.resolve_model_cls("LTX2VocoderWithBWE")
        self.assertEqual(cls.__name__, "LTX2Vocoder")


class TestLTX25PipelineConfig(unittest.TestCase):
    def test_pins_the_distilled_sigma_schedule(self):
        # The distilled DiT is driven by this schedule, not by a step count.
        config = LTX25PipelineConfig()
        self.assertEqual(config.default_sigmas, LTX25_DISTILLED_SIGMA_VALUES)
        self.assertEqual(len(LTX25_DISTILLED_SIGMA_VALUES), 8)
        self.assertEqual(LTX25_DISTILLED_SIGMA_VALUES[0], 1.0)
        self.assertTrue(
            all(
                a > b
                for a, b in zip(
                    LTX25_DISTILLED_SIGMA_VALUES, LTX25_DISTILLED_SIGMA_VALUES[1:]
                )
            )
        )

    def test_ltx2_has_no_pinned_schedule(self):
        self.assertIsNone(LTX2PipelineConfig().default_sigmas)

    def test_stays_an_ltx2_variant(self):
        # LTX-2.5 uses the LTX-2 linspace sigma path, so it must NOT be marked
        # as an LTX-2.3 native variant even though it shares 2.3's architecture.
        from sglang.multimodal_gen.configs.pipeline_configs.ltx_2 import (
            is_ltx23_native_variant,
        )

        config = LTX25PipelineConfig()
        self.assertFalse(is_ltx23_native_variant(config.vae_config.arch_config))

    def test_registry_resolves_ltx_variants_apart(self):
        from sglang.multimodal_gen.registry import get_model_info

        self.assertIs(
            get_model_info("Lightricks/LTX-2.5-Diffusers").pipeline_config_cls,
            LTX25PipelineConfig,
        )
        # The LTX-2 detector matches any "ltx-2" substring, so it must exclude
        # both point releases.
        self.assertIs(
            get_model_info("Lightricks/LTX-2").pipeline_config_cls,
            LTX2PipelineConfig,
        )


if __name__ == "__main__":
    unittest.main()
