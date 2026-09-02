# SPDX-License-Identifier: Apache-2.0
"""LTX-2.5 config wiring.

These pin the handful of places where LTX-2.5 diverges from LTX-2 and where a
silent regression would produce wrong output rather than an error. Everything
here is CPU/meta-device only -- no weights, no GPU.
"""

import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

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
        # Regression: read off a class attribute pinned to LTX2ArchConfig, the
        # LTX-2.5 renames never reached the loader.
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
    # Reversed like its sibling lists, this would give the wrong strides.
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
        # Not `get_model_info`: it also reads `model_index.json` from the Hub,
        # and offline that silently resolves to the generic diffusers config.
        from sglang.multimodal_gen.registry import _get_config_info

        self.assertIs(
            _get_config_info("Lightricks/LTX-2.5-Diffusers").pipeline_config_cls,
            LTX25PipelineConfig,
        )
        self.assertIs(
            _get_config_info("Lightricks/LTX-2").pipeline_config_cls,
            LTX2PipelineConfig,
        )
        self.assertEqual(
            _get_config_info("Lightricks/LTX-2.3").pipeline_config_cls.__name__,
            "LTX23PipelineConfig",
        )

    def test_derived_repos_keep_the_point_releases_apart(self):
        """Forks and local copies resolve by longest registered path stem.

        Resolution tries exact match, then the longest registered path that is
        a substring of the request, and only then the detectors. So a derived
        repo lands on the right config as long as it keeps the registered stem
        -- `LTX-2.5-Diffusers` is longer than `LTX-2` and wins.
        """
        from sglang.multimodal_gen.registry import _get_config_info

        self.assertIs(
            _get_config_info("myorg/LTX-2.5-Diffusers-fp8").pipeline_config_cls,
            LTX25PipelineConfig,
        )
        self.assertIs(
            _get_config_info("myorg/LTX-2-custom").pipeline_config_cls,
            LTX2PipelineConfig,
        )
        self.assertEqual(
            _get_config_info("myorg/LTX-2.3-tuned").pipeline_config_cls.__name__,
            "LTX23PipelineConfig",
        )


class TestLTX25ImageConditioningCRF(unittest.TestCase):
    """LTX-2.5 trained image conditioning at CRF 18; LTX-2 / 2.3 at 33.

    Getting this wrong does not raise -- it just feeds the model conditioning
    images from the wrong compression distribution.
    """

    def _crf_for_config(self, pipeline_config):
        """CRF for an already-resolved pipeline config.

        Driving this from a model path would route through `ServerArgs`, which
        reads `model_index.json` from the Hub; offline that falls back to the
        generic config and the assertion becomes meaningless. The resolver only
        reads `pipeline_config.text_encoder_configs`, so hand it the config
        directly.
        """
        from types import SimpleNamespace

        from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
            LTX2ImageEncodingStage,
        )

        return LTX2ImageEncodingStage._resolve_image_conditioning_crf(
            SimpleNamespace(pipeline_config=pipeline_config)
        )

    def test_ltx_2_5_uses_crf_18(self):
        self.assertEqual(self._crf_for_config(LTX25PipelineConfig()), 18)

    def test_earlier_ltx_generations_use_crf_33(self):
        self.assertEqual(self._crf_for_config(LTX2PipelineConfig()), 33)


class TestLTX25DurationHead(unittest.TestCase):
    """Frame counts must land on the VAE's causal temporal grid (8k + 1)."""

    def _head(self):
        import torch

        from sglang.multimodal_gen.configs.models.adapter.ltx_2_duration_head import (
            LTX2DurationHeadConfig,
        )
        from sglang.multimodal_gen.runtime.models.adapter.ltx_2_duration_head import (
            LTX2DurationHead,
        )

        with torch.device("meta"):
            return LTX2DurationHead(LTX2DurationHeadConfig())

    def test_predicted_frames_land_on_the_temporal_grid(self):
        from unittest import mock

        import torch

        head = self._head()
        for seconds in (1.0, 2.7, 3.28125, 7.5, 19.9):
            with mock.patch.object(
                head, "forward", return_value=torch.tensor([seconds])
            ):
                n = head.predict_num_frames(
                    frame_rate=24.0, temporal_compression_ratio=8
                )
            self.assertEqual((n - 1) % 8, 0, f"{n} frames is off-grid for {seconds}s")
            self.assertGreaterEqual(n, 1)

    def test_prediction_is_clamped_to_bounds(self):
        from unittest import mock

        import torch

        head = self._head()
        with mock.patch.object(head, "forward", return_value=torch.tensor([100.0])):
            n = head.predict_num_frames(
                frame_rate=24.0, temporal_compression_ratio=8, max_seconds=5.0
            )
        self.assertLessEqual(n / 24.0, 5.0)
        self.assertEqual((n - 1) % 8, 0)

    def test_requires_at_least_one_modality(self):
        with self.assertRaises(ValueError):
            self._head()(None, None)


class TestLTX25DiffusionDecoder(unittest.TestCase):
    """The 2.5 diffusion decoder: config shape and the geometry it implies."""

    def _config(self):
        from sglang.multimodal_gen.configs.models.decoders.ltx_2_5_diffusion_decoder import (
            LTX25DiffusionDecoderConfig,
        )

        return LTX25DiffusionDecoderConfig()

    def test_stage_channels_match_upsample_reductions(self):
        # Two views of the same thing; an inconsistent pair would only fail deep
        # inside the first block.
        arch = self._config().arch_config
        for i, reduction in enumerate(arch.decoder_upsample_channel_reductions):
            self.assertEqual(
                arch.decoder_stage_channels[i + 1],
                arch.decoder_stage_channels[i] // reduction,
            )

    def test_upsample_strides_compose_to_the_vae_ratios(self):
        arch = self._config().arch_config
        temporal = 1
        spatial = 1
        for stride_t, stride_h, _ in arch.decoder_upsample_strides:
            temporal *= stride_t
            spatial *= stride_h
        self.assertEqual(temporal, arch.temporal_compression_ratio)
        # The remaining spatial factor is the pixel patch size.
        self.assertEqual(spatial * arch.patch_size, arch.spatial_compression_ratio)

    def test_ships_as_a_single_step_x0_decoder(self):
        arch = self._config().arch_config
        self.assertEqual(arch.decoder_num_inference_steps, 1)
        self.assertEqual(arch.decoder_model_output_type, "x0")

    def test_builds_and_reports_expected_context_width(self):
        import torch

        from sglang.multimodal_gen.runtime.models.decoders.ltx_2_5_diffusion_decoder import (
            LTX2VideoDiffusionDecoderModel,
        )

        config = self._config()
        with torch.device("meta"):
            model = LTX2VideoDiffusionDecoderModel(config)
        self.assertEqual(
            model.decoder.context_channels,
            config.arch_config.decoder_stage_channels[-1],
        )
        # The window shifts inward at the border, so stages 1-4 carry replicated
        # trailing frames that stage 4 crops.
        self.assertEqual(model.decoder.trailing_pad_latent_frames, 2)

    def test_exposes_each_executable_block_group_for_layerwise_offload(self):
        import torch
        from torch import nn

        from sglang.multimodal_gen.runtime.managers.memory_managers.layerwise_offload import (
            LayerwiseOffloadableModuleMixin,
        )
        from sglang.multimodal_gen.runtime.models.decoders.ltx_2_5_diffusion_decoder import (
            LTX2VideoDiffusionDecoderModel,
        )

        with torch.device("meta"):
            model = LTX2VideoDiffusionDecoderModel(self._config())

        self.assertIsInstance(model, LayerwiseOffloadableModuleMixin)
        self.assertEqual(
            model.layer_names,
            [
                "decoder.det_stages.0",
                "decoder.det_stages.1",
                "decoder.det_stages.2",
                "decoder.det_stages.3",
                "decoder.diff_blocks",
            ],
        )
        named_modules = dict(model.named_modules())
        for layer_name in model.layer_names:
            with self.subTest(layer_name=layer_name):
                self.assertIsInstance(named_modules[layer_name], nn.ModuleList)

    def test_timestep_embedder_is_replicated_and_checkpoint_compatible(self):
        import torch
        from torch import nn

        from sglang.multimodal_gen.runtime.models.decoders.ltx_2_5_diffusion_decoder import (
            LTX2VideoDiffusionDecoderModel,
        )

        with torch.device("meta"):
            model = LTX2VideoDiffusionDecoderModel(self._config())
        timestep_embedder = model.decoder.t_embedder.timestep_embedder
        self.assertIsInstance(timestep_embedder.linear_1, nn.Linear)
        self.assertIsInstance(timestep_embedder.linear_2, nn.Linear)
        self.assertEqual(
            tuple(timestep_embedder.linear_1.weight.shape),
            (self._config().arch_config.decoder_t_emb_dim, 256),
        )
        self.assertIn(
            "decoder.t_embedder.timestep_embedder.linear_1.weight",
            model.state_dict(),
        )

    def test_class_name_resolves(self):
        from sglang.multimodal_gen.runtime.models.registry import ModelRegistry

        cls, _ = ModelRegistry.resolve_model_cls("LTX2VideoDiffusionDecoderModel")
        self.assertEqual(cls.__name__, "LTX2VideoDiffusionDecoderModel")

    def test_rotary_pair_cpu_fallback_matches_original_expression(self):
        import torch

        from sglang.multimodal_gen.runtime.models.decoders.ltx_2_5_diffusion_decoder import (
            LTX2VideoVaeRotaryPosEmbed3D,
        )

        def reference(module, hidden_states):
            outputs = []
            offset = 0
            for axis, (length, dim) in enumerate(
                zip(hidden_states.shape[1:4], module.rope_dim_split, strict=True),
                1,
            ):
                chunk = hidden_states[..., offset : offset + dim]
                pairs = chunk.reshape(*chunk.shape[:-1], dim // 2, 2)
                even = pairs[..., 0].float()
                odd = pairs[..., 1].float()
                exponents = torch.arange(0, dim, 2, dtype=torch.float64) / dim
                inv_freqs = (1.0 / module.base**exponents).to(torch.float32)
                positions = torch.arange(length, dtype=torch.float32)
                angles = positions[:, None] * inv_freqs[None, :]
                shape = [1, 1, 1, 1, 1, dim // 2]
                shape[axis] = length
                cos = angles.cos().reshape(shape)
                sin = angles.sin().reshape(shape)
                rotated = torch.stack(
                    [even * cos - odd * sin, even * sin + odd * cos], dim=-1
                )
                outputs.append(rotated.reshape(chunk.shape).to(hidden_states.dtype))
                offset += dim
            return torch.cat(outputs, dim=-1)

        torch.manual_seed(42)
        rope = LTX2VideoVaeRotaryPosEmbed3D(64)
        query = torch.randn(1, 3, 7, 7, 2, 64, dtype=torch.bfloat16)
        key = torch.randn_like(query)

        query_out, key_out = rope.forward_pair(query, key)

        self.assertTrue(torch.equal(query_out, reference(rope, query)))
        self.assertTrue(torch.equal(key_out, reference(rope, key)))

    def test_rotary_tables_are_shared_across_decoder_blocks(self):
        import torch

        from sglang.multimodal_gen.runtime.models.decoders.ltx_2_5_diffusion_decoder import (
            _ROPE_TABLE_CACHE,
            LTX2VideoVaeRotaryPosEmbed3D,
        )

        _ROPE_TABLE_CACHE.clear()
        hidden_states = torch.empty(1, 3, 7, 7, 2, 64, dtype=torch.bfloat16)
        first = LTX2VideoVaeRotaryPosEmbed3D(64)._tables(hidden_states)
        second = LTX2VideoVaeRotaryPosEmbed3D(64)._tables(hidden_states)

        self.assertIs(first, second)
        self.assertEqual(len(_ROPE_TABLE_CACHE), 1)
        _ROPE_TABLE_CACHE.clear()


class TestLTX25OptionalDecoderLoading(unittest.TestCase):
    @staticmethod
    def _server_args(load_diffusion_decoder: bool):
        return SimpleNamespace(
            load_diffusion_decoder=load_diffusion_decoder,
            model_variant=None,
            component_paths={},
        )

    @staticmethod
    def _write_model_index(model_path: str, *, include_decoder: bool = True):
        model_index = {
            "_class_name": "LTX2Pipeline",
            "duration_head": ["ltx2", "LTX2DurationHeadModel"],
        }
        if include_decoder:
            model_index["diffusion_decoder"] = [
                "ltx2",
                "LTX2VideoDiffusionDecoderModel",
            ]
        with open(f"{model_path}/model_index.json", "w") as f:
            json.dump(model_index, f)

    def test_decoder_is_not_loaded_by_default(self):
        from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import LTX2Pipeline
        from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
            LoRAPipeline,
        )

        with tempfile.TemporaryDirectory() as model_path:
            self._write_model_index(model_path)
            with mock.patch.object(LoRAPipeline, "__init__", return_value=None) as init:
                LTX2Pipeline(model_path, self._server_args(False))
        modules = init.call_args.kwargs["required_config_modules"]
        self.assertIn("duration_head", modules)
        self.assertNotIn("diffusion_decoder", modules)

    def test_decoder_load_is_explicit_and_validated(self):
        from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import LTX2Pipeline
        from sglang.multimodal_gen.runtime.pipelines_core.lora.pipeline import (
            LoRAPipeline,
        )

        with tempfile.TemporaryDirectory() as model_path:
            self._write_model_index(model_path)
            with mock.patch.object(LoRAPipeline, "__init__", return_value=None) as init:
                LTX2Pipeline(model_path, self._server_args(True))
            modules = init.call_args.kwargs["required_config_modules"]
            self.assertIn("diffusion_decoder", modules)

            self._write_model_index(model_path, include_decoder=False)
            with self.assertRaisesRegex(ValueError, "does not declare"):
                LTX2Pipeline(model_path, self._server_args(True))


class TestLTX25LatentUpsampler(unittest.TestCase):
    """LTX-2.5 turns the rational resampler off explicitly.

    Earlier LTX configs only carry `rational_spatial_scale`, so the loader
    inferred the resampler from its presence. LTX-2.5 states the choice, and
    assuming True there builds a different module than the checkpoint holds.
    """

    LTX25_UPSAMPLER_CONFIG = {
        "dims": 3,
        "in_channels": 128,
        "mid_channels": 1024,
        "num_blocks_per_stage": 4,
        "rational_spatial_scale": 2.0,
        "spatial_upsample": True,
        "temporal_upsample": False,
        "use_rational_resampler": False,
    }

    def _normalize(self, raw):
        from sglang.multimodal_gen.runtime.loader.component_loaders.upsampler_loader import (
            _normalize_config,
        )

        return _normalize_config(raw)

    def test_explicit_flag_is_honoured(self):
        config = self._normalize(dict(self.LTX25_UPSAMPLER_CONFIG))
        self.assertFalse(config["rational_resampler"])
        self.assertEqual(config["spatial_scale"], 2.0)

    def test_absent_flag_keeps_legacy_behaviour(self):
        raw = {
            k: v
            for k, v in self.LTX25_UPSAMPLER_CONFIG.items()
            if k != "use_rational_resampler"
        }
        self.assertTrue(self._normalize(raw)["rational_resampler"])

    def test_flag_changes_the_module_it_builds(self):
        # Guards the fix: the two settings are not interchangeable.
        import torch

        from sglang.multimodal_gen.runtime.models.upsampler.latent_upsampler import (
            LatentUpsampler,
        )

        kwargs = dict(
            in_channels=128,
            mid_channels=1024,
            num_blocks_per_stage=4,
            dims=3,
            spatial_upsample=True,
            temporal_upsample=False,
            spatial_scale=2.0,
        )
        with torch.device("meta"):
            without = set(
                LatentUpsampler(**kwargs, rational_resampler=False).state_dict()
            )
            with_rr = set(
                LatentUpsampler(**kwargs, rational_resampler=True).state_dict()
            )
        self.assertNotEqual(without, with_rr)


class TestLTX25DevVariant(unittest.TestCase):
    """`--model-variant dev` serves `transformer_full/`, which the index omits."""

    def _pipeline_cls(self):
        from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import (
            _BaseLTX2Pipeline,
        )

        return _BaseLTX2Pipeline

    def _args(self, variant):
        class _Args:
            model_variant = variant
            component_paths: dict = {}

        return _Args()

    def test_variant_aliases(self):
        cls = self._pipeline_cls()
        for variant in ("dev", "full", "sft", "DEV"):
            self.assertTrue(cls._is_dev_variant(self._args(variant)), variant)
        for variant in (None, "", "distilled"):
            self.assertFalse(cls._is_dev_variant(self._args(variant)), variant)

    def test_missing_weights_raises_a_clear_error(self):
        cls = self._pipeline_cls()
        with self.assertRaises(ValueError) as ctx:
            cls._maybe_route_dev_transformer("/nonexistent/model", self._args("dev"))
        self.assertIn("transformer_full", str(ctx.exception))

    def test_explicit_component_path_wins(self):
        cls = self._pipeline_cls()
        args = self._args("dev")
        args.component_paths = {"transformer": "/some/other/transformer"}
        # Must not raise, and must not overwrite the caller's choice.
        cls._maybe_route_dev_transformer("/nonexistent/model", args)
        self.assertEqual(args.component_paths["transformer"], "/some/other/transformer")


if __name__ == "__main__":
    unittest.main()
