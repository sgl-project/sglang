import argparse
import os
import sys
import tempfile
import types
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.models.dits.sana_wm import SanaWMConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import SanaWMPipelineConfig
from sglang.multimodal_gen.configs.sample.sana_wm import SanaWMSamplingParams
from sglang.multimodal_gen.registry import _get_config_info, get_pipeline_config_classes
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.models.dits.sana_wm import (
    _downscale_to_reference_rms,
    _gdn_chunk_scan_forward,
    _gdn_scan_forward,
    _RMSNorm,
    _sana_wm_chunk_index_from_chunk_size,
    _sana_wm_chunked_attention,
    _sana_wm_normalize_chunk_index,
    _single_path_delta_chunk_scan_forward,
    _single_path_delta_scan_forward,
    compute_chunk_plucker,
)
from sglang.multimodal_gen.runtime.pipelines.sana_wm_pipeline import (
    SanaWMPipeline,
    SanaWMTwoStagePipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
    SanaWMBeforeDenoisingStage,
    SanaWMDenoisingStage,
    SanaWMTextEncodingStage,
    _align_sana_wm_cfg_text_conditions,
    configure_sana_wm_ltx2_vae_for_long_video,
    parse_sana_wm_action_string,
    sana_wm_action_to_camera_to_world,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.refiner import (
    OfficialDiffusersLTX2RefinerModule,
    OfficialGemma3TextEncoderModule,
    SanaWMLTX2RefinerStage,
    SanaWMRefinerDecodingStage,
    _refiner_config_value,
    _streaming_diffusers_self_attention,
    _uses_diffusers_ltx2_refiner,
    sana_wm_skip_refiner_enabled,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args
from sglang.multimodal_gen.runtime.utils.model_overlay import (
    resolve_model_overlay_target,
)
from sglang.multimodal_gen.test.test_utils import DEFAULT_SANA_WM_MODEL_NAME_FOR_TEST

_SANA_WM_REFINER_STAGE_MODULE = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages."
    "model_specific_stages.sana_wm.refiner"
)


class _GlobalStageArgsMixin:
    def setUp(self) -> None:
        super().setUp()
        self._prev_global_server_args = server_args_module._global_server_args
        set_global_server_args(
            SimpleNamespace(
                comfyui_mode=False,
                enable_cfg_parallel=False,
                enable_torch_compile=False,
                attention_backend=None,
            )
        )

    def tearDown(self) -> None:
        set_global_server_args(self._prev_global_server_args)
        super().tearDown()


class TestSanaWMPipelineConfig(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SanaWMPipelineConfig()

    def test_task_type_is_ti2v(self) -> None:
        self.assertEqual(self.config.task_type, ModelTaskType.TI2V)

    def test_adjust_num_frames_rounds_to_temporal_stride(self) -> None:
        self.assertEqual(self.config.adjust_num_frames(50), 49)
        self.assertEqual(self.config.adjust_num_frames(49), 49)

    def test_prepare_latent_shape_returns_5d(self) -> None:
        batch = SimpleNamespace(height=704, width=1280)
        shape = self.config.prepare_latent_shape(batch, batch_size=1, num_frames=49)
        self.assertEqual(shape, (1, 128, 7, 22, 40))

    def test_prepare_latent_shape_requires_spatial_stride_alignment(self) -> None:
        batch = SimpleNamespace(height=705, width=1280)
        with self.assertRaisesRegex(ValueError, "divisible"):
            self.config.prepare_latent_shape(batch, batch_size=1, num_frames=49)

    def test_prepare_latent_shape_uses_axis_specific_spatial_strides(self) -> None:
        self.config.vae_stride = (8, 16, 32)
        batch = SimpleNamespace(height=704, width=1280)
        shape = self.config.prepare_latent_shape(batch, batch_size=1, num_frames=49)
        self.assertEqual(shape, (1, 128, 7, 44, 40))

    def test_prepare_pos_cond_kwargs_passes_camera_from_batch_extra(self) -> None:
        # SanaWMBeforeDenoisingStage packs c2w + intrinsics into the upstream
        # latent-frame (B, T_lat, 20) raymap plus a (B, 48, T_lat, H, W)
        # ``chunk_plucker`` tensor. The pipeline config just forwards those
        # tensors verbatim.
        camera_conditions = torch.zeros(1, 7, 20)
        chunk_plucker = torch.zeros(1, 48, 7, 22, 40)
        batch = SimpleNamespace(
            prompt_attention_mask=torch.ones(1, 16),
            extra={
                "camera_conditions": camera_conditions,
                "chunk_plucker": chunk_plucker,
            },
        )
        kwargs = self.config.prepare_pos_cond_kwargs(
            batch=batch,
            device=torch.device("cpu"),
            rotary_emb=None,
            dtype=torch.bfloat16,
        )
        self.assertIs(kwargs["camera_conditions"], camera_conditions)
        self.assertIs(kwargs["chunk_plucker"], chunk_plucker)

    def test_get_model_deployment_config_enables_dit_layerwise_offload(self) -> None:
        deployment = self.config.get_model_deployment_config()
        self.assertTrue(deployment.auto_dit_layerwise_offload)
        self.assertEqual(deployment.fsdp_auto_min_available_memory_gb, 60)

    def test_text_encoder_padding_matches_cfg_concat_contract(self) -> None:
        self.assertEqual(
            self.config.text_encoder_extra_args[0]["padding"], "max_length"
        )
        self.assertTrue(self.config.text_encoder_extra_args[0]["return_attention_mask"])
        self.assertTrue(self.config.chi_prompt)

    def test_inference_flow_shift_matches_official_reference(self) -> None:
        self.assertEqual(self.config.flow_shift, 9.95)
        self.assertEqual(self.config.inference_flow_shift, 9.8)

    def test_dit_arch_text_norm_defaults_match_upstream(self) -> None:
        arch = SanaWMConfig().arch_config
        self.assertTrue(arch.y_norm)
        self.assertEqual(arch.y_norm_scale_factor, 0.01)
        self.assertEqual(arch.y_norm_eps, 1e-5)
        self.assertEqual(arch.timestep_norm_scale_factor, 1.0)

    def test_prepare_neg_cond_kwargs_keeps_camera_for_cfg(self) -> None:
        camera_conditions = torch.zeros(1, 7, 20)
        chunk_plucker = torch.zeros(1, 48, 7, 22, 40)
        batch = SimpleNamespace(
            negative_attention_mask=torch.ones(1, 16),
            extra={
                "camera_conditions": camera_conditions,
                "chunk_plucker": chunk_plucker,
            },
        )
        kwargs = self.config.prepare_neg_cond_kwargs(
            batch=batch,
            device=torch.device("cpu"),
            rotary_emb=None,
            dtype=torch.bfloat16,
        )
        self.assertIn("encoder_attention_mask", kwargs)
        self.assertIs(kwargs["camera_conditions"], camera_conditions)
        self.assertIs(kwargs["chunk_plucker"], chunk_plucker)

    def test_decode_scale_and_shift_uses_ltx2_latent_stats(self) -> None:
        vae = SimpleNamespace(
            config=SimpleNamespace(scaling_factor=2.0),
            latents_mean=torch.tensor([1.0, 2.0]),
            latents_std=torch.tensor([2.0, 4.0]),
        )
        scale, shift = self.config.get_decode_scale_and_shift(
            torch.device("cpu"), torch.float32, vae
        )

        self.assertTrue(
            torch.equal(scale, torch.tensor([1.0, 0.5]).view(1, 2, 1, 1, 1))
        )
        self.assertTrue(
            torch.equal(shift, torch.tensor([1.0, 2.0]).view(1, 2, 1, 1, 1))
        )

    def test_sana_wm_ltx2_vae_tiling_defaults_match_upstream(self) -> None:
        self.assertTrue(self.config.vae_tiling)
        self.assertTrue(self.config.vae_framewise_encoding)
        self.assertTrue(self.config.vae_framewise_decoding)
        self.assertEqual(self.config.vae_tile_sample_min_num_frames, 96)
        self.assertEqual(self.config.vae_tile_sample_stride_num_frames, 64)
        self.assertEqual(self.config.vae_config.tile_sample_min_num_frames, 96)
        self.assertEqual(self.config.vae_config.tile_sample_stride_num_frames, 64)
        self.assertEqual(self.config.vae_config.blend_num_frames, 32)

    def test_configure_sana_wm_ltx2_vae_enables_framewise_decode(self) -> None:
        class FakeLTX2VAE:
            def __init__(self):
                self.use_tiling = False
                self.use_framewise_encoding = False
                self.use_framewise_decoding = False
                self.tile_sample_min_num_frames = 16
                self.tile_sample_stride_num_frames = 8
                self.enable_tiling_kwargs = None

            def enable_tiling(self, **kwargs):
                self.use_tiling = True
                self.enable_tiling_kwargs = kwargs
                self.tile_sample_min_num_frames = kwargs["tile_sample_min_num_frames"]
                self.tile_sample_stride_num_frames = kwargs[
                    "tile_sample_stride_num_frames"
                ]

        vae = FakeLTX2VAE()
        configure_sana_wm_ltx2_vae_for_long_video(vae, self.config)

        self.assertTrue(vae.use_tiling)
        self.assertTrue(vae.use_framewise_encoding)
        self.assertTrue(vae.use_framewise_decoding)
        self.assertEqual(vae.tile_sample_min_num_frames, 96)
        self.assertEqual(vae.tile_sample_stride_num_frames, 64)
        self.assertEqual(
            vae.enable_tiling_kwargs,
            {
                "tile_sample_min_num_frames": 96,
                "tile_sample_stride_num_frames": 64,
            },
        )

    def test_configure_sana_wm_ltx2_vae_honors_nested_vae_config(self) -> None:
        vae = SimpleNamespace(
            use_tiling=False,
            use_framewise_encoding=False,
            use_framewise_decoding=False,
            tile_sample_min_num_frames=16,
            tile_sample_stride_num_frames=8,
        )

        def enable_tiling(**kwargs):
            vae.use_tiling = True
            vae.enable_tiling_kwargs = kwargs

        vae.enable_tiling = enable_tiling
        self.config.vae_config.tile_sample_min_num_frames = 128
        self.config.vae_config.tile_sample_stride_num_frames = 80

        configure_sana_wm_ltx2_vae_for_long_video(vae, self.config)

        self.assertEqual(vae.tile_sample_min_num_frames, 128)
        self.assertEqual(vae.tile_sample_stride_num_frames, 80)
        self.assertEqual(
            vae.enable_tiling_kwargs,
            {
                "tile_sample_min_num_frames": 128,
                "tile_sample_stride_num_frames": 80,
            },
        )

    def test_cfg_text_conditions_pad_positive_and_negative_to_same_length(self) -> None:
        pos = torch.ones(1, 173, 4)
        neg = torch.ones(1, 1, 4) * 2
        pos_mask = torch.ones(1, 173, dtype=torch.long)
        neg_mask = torch.ones(1, 1, dtype=torch.long)

        pos, neg, pos_mask, neg_mask = _align_sana_wm_cfg_text_conditions(
            pos, neg, pos_mask, neg_mask
        )

        self.assertEqual(pos.shape, (1, 173, 4))
        self.assertEqual(neg.shape, (1, 173, 4))
        self.assertEqual(pos_mask.shape, (1, 173))
        self.assertEqual(neg_mask.shape, (1, 173))
        self.assertTrue(torch.equal(neg[:, :1], torch.ones(1, 1, 4) * 2))
        self.assertTrue(torch.equal(neg[:, 1:], torch.zeros(1, 172, 4)))
        self.assertTrue(
            torch.equal(neg_mask[:, :1], torch.ones(1, 1, dtype=torch.long))
        )
        self.assertTrue(
            torch.equal(neg_mask[:, 1:], torch.zeros(1, 172, dtype=torch.long))
        )


class TestSanaWMSamplingParams(unittest.TestCase):
    def test_defaults_match_video_ti2v_contract(self) -> None:
        params = SanaWMSamplingParams()
        self.assertEqual(params.height, 704)
        self.assertEqual(params.width, 1280)
        self.assertEqual(params.num_frames, 49)
        self.assertEqual(params.num_inference_steps, 20)
        self.assertEqual(params.guidance_scale, 4.5)
        self.assertEqual(params.negative_prompt, "")

    def test_build_request_extra_omits_camera_by_default(self) -> None:
        params = SanaWMSamplingParams()
        extra = params.build_request_extra()
        self.assertNotIn("camera_to_world", extra)
        self.assertNotIn("intrinsics", extra)

    def test_build_request_extra_includes_camera_when_set(self) -> None:
        cam = torch.eye(4).unsqueeze(0).expand(49, 4, 4)
        intr = torch.eye(3).unsqueeze(0).expand(49, 3, 3)
        params = SanaWMSamplingParams(camera_to_world=cam, intrinsics=intr)
        extra = params.build_request_extra()
        self.assertIs(extra["camera_to_world"], cam)
        self.assertIs(extra["intrinsics"], intr)

    def test_build_request_extra_includes_action_when_set(self) -> None:
        params = SanaWMSamplingParams(
            action="w-8,jw-8",
            translation_speed=0.06,
            rotation_speed_deg=2.0,
            pitch_limit_deg=70.0,
        )

        extra = params.build_request_extra()

        self.assertEqual(extra["action"], "w-8,jw-8")
        self.assertEqual(extra["translation_speed"], 0.06)
        self.assertEqual(extra["rotation_speed_deg"], 2.0)
        self.assertEqual(extra["pitch_limit_deg"], 70.0)

    def test_build_request_extra_rejects_action_with_direct_camera(self) -> None:
        cam = torch.eye(4).unsqueeze(0).expand(49, 4, 4)
        params = SanaWMSamplingParams(action="w-8", camera_to_world=cam)

        with self.assertRaisesRegex(ValueError, "either action or camera_to_world"):
            params.build_request_extra()

    def test_cli_args_accept_upstream_action_aliases(self) -> None:
        parser = argparse.ArgumentParser()
        SanaWMSamplingParams.add_cli_args(parser)

        args = parser.parse_args(
            [
                "--action",
                "w-8,jw-8",
                "--translation_speed",
                "0.06",
                "--rotation-speed-deg",
                "2.0",
            ]
        )
        cli_args = SanaWMSamplingParams.get_cli_args(args)

        self.assertEqual(cli_args["action"], "w-8,jw-8")
        self.assertEqual(cli_args["translation_speed"], 0.06)
        self.assertEqual(cli_args["rotation_speed_deg"], 2.0)


class TestSanaWMRegistry(unittest.TestCase):
    def setUp(self) -> None:
        _get_config_info.cache_clear()

    def test_model_path_resolves_sana_wm_pipeline_config(self) -> None:
        info = _get_config_info(DEFAULT_SANA_WM_MODEL_NAME_FOR_TEST)
        self.assertIsNotNone(info)
        self.assertIs(info.pipeline_config_cls, SanaWMPipelineConfig)
        self.assertIs(info.sampling_param_cls, SanaWMSamplingParams)

    def test_two_stage_pipeline_registers_sana_wm_config_classes(self) -> None:
        classes = get_pipeline_config_classes("SanaWMTwoStagePipeline")
        self.assertIsNotNone(classes)
        pipeline_config_cls, sampling_param_cls = classes
        self.assertIs(pipeline_config_cls, SanaWMPipelineConfig)
        self.assertIs(sampling_param_cls, SanaWMSamplingParams)

    def test_overlay_resolver_matches_hf_cache_snapshot_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_dir = f"{tmp_dir}/hub/models--Lightricks--LTX-2.3/snapshots/abc123"
            os.makedirs(snapshot_dir)
            target = resolve_model_overlay_target(snapshot_dir)
        self.assertIsNotNone(target)
        source_model_id, _ = target
        self.assertEqual(source_model_id, "Lightricks/LTX-2.3")


class TestSanaWMTwoStagePipeline(unittest.TestCase):
    def test_resolve_refiner_paths_defaults_to_model_refiner_dir(self) -> None:
        pipeline = object.__new__(SanaWMTwoStagePipeline)
        pipeline.model_path = "/models/sana-wm"
        server_args = SimpleNamespace(component_paths={})
        refiner_root, refiner_gemma_root = pipeline._resolve_refiner_paths(server_args)
        self.assertEqual(refiner_root, "/models/sana-wm/refiner")
        self.assertEqual(refiner_gemma_root, "/models/sana-wm/refiner/text_encoder")

    def test_resolve_refiner_paths_accepts_component_overrides(self) -> None:
        pipeline = object.__new__(SanaWMTwoStagePipeline)
        pipeline.model_path = "/models/sana-wm"
        server_args = SimpleNamespace(
            component_paths={
                "refiner": "/custom/refiner",
                "refiner_text_encoder": "/custom/refiner/text_encoder",
            }
        )
        refiner_root, refiner_gemma_root = pipeline._resolve_refiner_paths(server_args)
        self.assertEqual(refiner_root, "/custom/refiner")
        self.assertEqual(refiner_gemma_root, "/custom/refiner/text_encoder")

    def test_refiner_modules_are_loaded_from_official_subtrees(self) -> None:
        self.assertEqual(
            SanaWMTwoStagePipeline._REFINER_SUB_MODULES,
            (
                ("transformer_2", "refiner/transformer"),
                ("connectors", "refiner/connectors"),
                ("text_encoder_2", "refiner/text_encoder"),
                ("tokenizer_2", "refiner/text_encoder"),
            ),
        )


class TestSanaWMPipeline(unittest.TestCase):
    def test_validate_parallelism_rejects_tensor_parallelism(self) -> None:
        with self.assertRaisesRegex(ValueError, "tensor parallelism"):
            SanaWMPipeline._validate_parallelism_args(
                SimpleNamespace(tp_size=2, sp_degree=1)
            )

    def test_validate_parallelism_rejects_sequence_parallelism(self) -> None:
        with self.assertRaisesRegex(ValueError, "sequence parallelism"):
            SanaWMPipeline._validate_parallelism_args(
                SimpleNamespace(tp_size=1, sp_degree=2)
            )


class TestSanaWMBeforeDenoisingStage(_GlobalStageArgsMixin, unittest.TestCase):
    def test_action_string_rolls_out_camera_to_world(self) -> None:
        self.assertEqual(
            parse_sana_wm_action_string(" w-2，d-1, none-1 "),
            [["w"], ["w"], ["d"], []],
        )

        poses = sana_wm_action_to_camera_to_world(
            "w-2,d-1",
            translation_speed=0.05,
            rotation_speed_deg=1.2,
        )

        self.assertEqual(poses.shape, (4, 4, 4))
        self.assertAlmostEqual(float(poses[2, 2, 3]), 0.1, places=5)
        self.assertAlmostEqual(float(poses[3, 0, 3]), 0.05, places=5)

    def test_action_rollout_uses_upstream_strafe_yaw_coupling(self) -> None:
        # Strafe-only action yaws the camera (forward axis tilts off +Z), matching
        # the upstream SANA-WM action convention.
        strafed = sana_wm_action_to_camera_to_world(
            "d-10", translation_speed=0.04, rotation_speed_deg=1.2
        )
        self.assertGreater(abs(float(strafed[-1, 0, 2])), 1e-4)

    def test_action_string_rejects_unknown_keys_and_bad_duration(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown action keys"):
            parse_sana_wm_action_string("x-1")
        with self.assertRaisesRegex(ValueError, "invalid duration"):
            parse_sana_wm_action_string("w-0")

    def test_vae_encode_image_extracts_latent_dist_and_normalizes_ltx2(self) -> None:
        class DummyLatentDist:
            def mode(self):
                return torch.tensor([[[[[3.0]]], [[[10.0]]]]])

        class DummyVAE:
            dtype = torch.float32
            device = torch.device("cpu")
            config = SimpleNamespace(scaling_factor=2.0)
            latents_mean = torch.tensor([1.0, 2.0])
            latents_std = torch.tensor([2.0, 4.0])

            def encode(self, image):
                return SimpleNamespace(latent_dist=DummyLatentDist())

        stage = SanaWMBeforeDenoisingStage(
            vae=DummyVAE(),
            transformer=None,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )

        encoded = stage._vae_encode_image(
            torch.ones(1, 3, 1, 1),
            dtype=torch.float32,
            device=torch.device("cpu"),
        )

        expected = torch.tensor([[[[[2.0]]], [[[4.0]]]]])
        self.assertTrue(torch.equal(encoded, expected))

    def test_first_frame_vae_use_is_declared_for_component_residency(self) -> None:
        pipeline_config = SanaWMPipelineConfig()
        stage = SanaWMBeforeDenoisingStage(
            vae=object(),
            transformer=None,
            scheduler=None,
            pipeline_config=pipeline_config,
        )

        uses = stage.component_uses(
            SimpleNamespace(pipeline_config=pipeline_config),
            stage_name="sana_wm_before_denoising",
        )

        self.assertEqual([use.component_name for use in uses], ["vae"])
        self.assertEqual(uses[0].target_dtype, torch.bfloat16)

    def test_prepare_noise_latents_accepts_per_sample_generators(self) -> None:
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )

        def make_generators():
            return [
                torch.Generator(device="cpu").manual_seed(11),
                torch.Generator(device="cpu").manual_seed(23),
            ]

        latents = stage._prepare_noise_latents(
            (2, 1, 1, 2, 2),
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=make_generators(),
        )
        latents_again = stage._prepare_noise_latents(
            (2, 1, 1, 2, 2),
            dtype=torch.float32,
            device=torch.device("cpu"),
            generator=make_generators(),
        )

        self.assertEqual(latents.shape, (2, 1, 1, 2, 2))
        self.assertTrue(torch.equal(latents, latents_again))
        self.assertFalse(torch.equal(latents[0], latents[1]))

    def test_prepare_noise_latents_rejects_mismatched_generator_count(self) -> None:
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )

        with self.assertRaisesRegex(ValueError, "length must match"):
            stage._prepare_noise_latents(
                (2, 1, 1, 1, 1),
                dtype=torch.float32,
                device=torch.device("cpu"),
                generator=[
                    torch.Generator(device="cpu").manual_seed(1),
                    torch.Generator(device="cpu").manual_seed(2),
                    torch.Generator(device="cpu").manual_seed(3),
                ],
            )

    def test_generator_from_seed_accepts_per_sample_seed_list(self) -> None:
        generators = SanaWMBeforeDenoisingStage._generator_from_seed(
            [11, 23],
            batch_size=2,
            device=torch.device("cpu"),
        )

        self.assertEqual(len(generators), 2)

        with self.assertRaisesRegex(ValueError, "seed list length"):
            SanaWMBeforeDenoisingStage._generator_from_seed(
                [1, 2, 3],
                batch_size=2,
                device=torch.device("cpu"),
            )

    def test_condition_image_batch_helper_accepts_single_or_batch_list(self) -> None:
        self.assertEqual(
            SanaWMBeforeDenoisingStage._condition_images_for_batch(["img"], 2),
            ["img"],
        )
        self.assertEqual(
            SanaWMBeforeDenoisingStage._condition_images_for_batch(["a", "b"], 2),
            ["a", "b"],
        )

        with self.assertRaisesRegex(ValueError, "one image or one image per batch"):
            SanaWMBeforeDenoisingStage._condition_images_for_batch(["a", "b", "c"], 2)

    def test_first_frame_preprocess_records_official_crop_geometry(self) -> None:
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )
        latents = torch.zeros(1, 128, 7, 22, 40)
        batch = SimpleNamespace(extra={})

        with patch.object(
            stage,
            "_vae_encode_image",
            return_value=torch.ones(1, 128, 1, 22, 40),
        ):
            out = stage._splice_first_frame(
                latents,
                torch.zeros(3, 100, 100),
                dtype=torch.float32,
                device=torch.device("cpu"),
                batch=batch,
            )

        info = batch.extra["sana_wm_condition_image_preprocess"]
        self.assertEqual(info["source_size"], (100, 100))
        self.assertEqual(info["resized_size"], (1280, 1280))
        self.assertEqual(info["crop_offset"], (0, 288))
        self.assertEqual(info["target_size"], (1280, 704))
        self.assertTrue(torch.equal(out[:, :, :1], torch.ones(1, 128, 1, 22, 40)))

    def test_first_frame_splice_supports_batched_condition_images(self) -> None:
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )
        latents = torch.zeros(2, 128, 7, 22, 40)
        batch = SimpleNamespace(extra={})

        with patch.object(
            stage,
            "_vae_encode_image",
            side_effect=[
                torch.ones(1, 128, 1, 22, 40),
                torch.full((1, 128, 1, 22, 40), 2.0),
            ],
        ):
            out = stage._splice_first_frame(
                latents,
                [torch.zeros(3, 100, 100), torch.zeros(3, 200, 100)],
                dtype=torch.float32,
                device=torch.device("cpu"),
                batch=batch,
            )

        info = batch.extra["sana_wm_condition_image_preprocess"]
        self.assertEqual(len(info), 2)
        self.assertTrue(torch.equal(out[0, :, :1], torch.ones(128, 1, 22, 40)))
        self.assertTrue(torch.equal(out[1, :, :1], torch.full((128, 1, 22, 40), 2.0)))

    def test_default_static_camera_builds_latent_raymap_and_chunk_plucker(self) -> None:
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )
        batch = SimpleNamespace(extra={}, height=384, width=640)

        camera_conditions, chunk_plucker, source = stage._build_camera_conditioning(
            batch,
            batch_size=1,
            num_frames=17,
            latent_shape=(1, 128, 3, 12, 20),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(source, "default_static")
        self.assertEqual(camera_conditions.shape, (1, 3, 20))
        self.assertEqual(chunk_plucker.shape, (1, 48, 3, 12, 20))
        self.assertTrue(
            torch.equal(camera_conditions[0, 0, :16], torch.eye(4).reshape(-1))
        )
        self.assertTrue(
            torch.allclose(
                camera_conditions[0, 0, 16:],
                torch.tensor([16.0, 16.0, 10.0, 6.0]),
            )
        )

    def test_request_intrinsics_are_transformed_for_condition_image_crop(self) -> None:
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )
        batch = SimpleNamespace(
            extra={
                "diffusers_kwargs": {
                    "intrinsics": [50.0, 50.0, 50.0, 50.0],
                },
                "sana_wm_condition_image_preprocess": {
                    "source_size": (100, 100),
                    "resized_size": (1280, 1280),
                    "crop_offset": (0, 288),
                    "target_size": (1280, 704),
                },
            },
            height=704,
            width=1280,
        )

        camera_conditions, chunk_plucker, source = stage._build_camera_conditioning(
            batch,
            batch_size=1,
            num_frames=49,
            latent_shape=(1, 128, 7, 22, 40),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(source, "default_static_request_intrinsics")
        self.assertEqual(chunk_plucker.shape, (1, 48, 7, 22, 40))
        self.assertTrue(
            torch.allclose(
                camera_conditions[0, 0, 16:],
                torch.tensor([20.0, 20.0, 20.0, 11.0]),
            )
        )

    def test_explicit_camera_request_detects_diffusers_path_kwargs(self) -> None:
        batch = SimpleNamespace(
            extra={"diffusers_kwargs": {"camera_to_world_path": "/tmp/cam.npy"}}
        )

        self.assertTrue(SanaWMBeforeDenoisingStage._has_explicit_camera_request(batch))

    def test_explicit_camera_request_detects_action_kwargs(self) -> None:
        batch = SimpleNamespace(extra={"diffusers_kwargs": {"action": "w-8"}})

        self.assertTrue(SanaWMBeforeDenoisingStage._has_explicit_camera_request(batch))

    def test_action_conditioning_uses_existing_camera_path(self) -> None:
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )
        batch = SimpleNamespace(
            extra={
                "diffusers_kwargs": {
                    "action": "w-8",
                    "intrinsics": [50.0, 50.0, 32.0, 32.0],
                }
            },
            height=64,
            width=64,
        )

        camera_conditions, chunk_plucker, source = stage._build_camera_conditioning(
            batch,
            batch_size=1,
            num_frames=9,
            latent_shape=(1, 128, 2, 2, 2),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(source, "action")
        self.assertEqual(camera_conditions.shape, (1, 2, 20))
        self.assertEqual(chunk_plucker.shape, (1, 48, 2, 2, 2))
        # Z translation after 8 forward frames at the default (streaming)
        # translation speed. Reference the constant so the assertion cannot go
        # stale again if the default moves (it was 0.05 once, 0.04 now).
        from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.base import (
            _SANA_WM_DEFAULT_TRANSLATION_SPEED,
        )

        self.assertAlmostEqual(
            float(camera_conditions[0, 1, 11]),
            8 * _SANA_WM_DEFAULT_TRANSLATION_SPEED,
            places=5,
        )

    def test_camera_conditioning_accepts_unbatched_chunk_plucker(self) -> None:
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )
        batch = SimpleNamespace(
            extra={"chunk_plucker": torch.zeros(48, 3, 12, 20)},
            height=384,
            width=640,
        )

        camera_conditions, chunk_plucker, source = stage._build_camera_conditioning(
            batch,
            batch_size=1,
            num_frames=17,
            latent_shape=(1, 128, 3, 12, 20),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(source, "default_static")
        self.assertEqual(camera_conditions.shape, (1, 3, 20))
        self.assertEqual(chunk_plucker.shape, (1, 48, 3, 12, 20))

    def test_camera_conditioning_accepts_diffusers_prebuilt_raymap(self) -> None:
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )
        camera_conditions = torch.zeros(1, 17, 20)
        camera_conditions[..., :16] = torch.eye(4).reshape(1, 1, 16)
        camera_conditions[..., 16:] = torch.tensor([16.0, 16.0, 10.0, 6.0])
        batch = SimpleNamespace(
            extra={"diffusers_kwargs": {"camera_conditions": camera_conditions}},
            height=384,
            width=640,
        )

        camera_conditions, chunk_plucker, source = stage._build_camera_conditioning(
            batch,
            batch_size=1,
            num_frames=17,
            latent_shape=(1, 128, 3, 12, 20),
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(source, "prebuilt_original_frames")
        self.assertEqual(camera_conditions.shape, (1, 3, 20))
        self.assertEqual(chunk_plucker.shape, (1, 48, 3, 12, 20))

    def test_camera_conditioning_rejects_bad_prepacked_shapes(self) -> None:
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )

        with self.assertRaisesRegex(ValueError, "camera_conditions must have shape"):
            stage._build_camera_conditioning(
                SimpleNamespace(
                    extra={"camera_conditions": torch.zeros(1, 1, 3, 20)},
                    height=384,
                    width=640,
                ),
                batch_size=1,
                num_frames=17,
                latent_shape=(1, 128, 3, 12, 20),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        with self.assertRaisesRegex(ValueError, "batch dimension"):
            stage._build_camera_conditioning(
                SimpleNamespace(
                    extra={"camera_conditions": torch.zeros(3, 3, 20)},
                    height=384,
                    width=640,
                ),
                batch_size=2,
                num_frames=17,
                latent_shape=(2, 128, 3, 12, 20),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        with self.assertRaisesRegex(ValueError, "require chunk_plucker"):
            stage._build_camera_conditioning(
                SimpleNamespace(
                    extra={"camera_conditions": torch.zeros(1, 3, 20)},
                    height=384,
                    width=640,
                ),
                batch_size=1,
                num_frames=17,
                latent_shape=(1, 128, 3, 12, 20),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        with self.assertRaisesRegex(ValueError, "batch dimension"):
            stage._build_camera_conditioning(
                SimpleNamespace(
                    extra={"chunk_plucker": torch.zeros(3, 48, 3, 12, 20)},
                    height=384,
                    width=640,
                ),
                batch_size=2,
                num_frames=17,
                latent_shape=(2, 128, 3, 12, 20),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        with self.assertRaisesRegex(ValueError, "chunk_plucker shape mismatch"):
            stage._build_camera_conditioning(
                SimpleNamespace(
                    extra={"chunk_plucker": torch.zeros(48, 2, 12, 20)},
                    height=384,
                    width=640,
                ),
                batch_size=1,
                num_frames=17,
                latent_shape=(1, 128, 3, 12, 20),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def test_action_and_camera_path_are_mutually_exclusive(self) -> None:
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )
        batch = SimpleNamespace(
            extra={
                "diffusers_kwargs": {
                    "action": "w-8",
                    "camera_to_world": torch.eye(4).unsqueeze(0),
                }
            },
            height=64,
            width=64,
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            stage._build_camera_conditioning(
                batch,
                batch_size=1,
                num_frames=9,
                latent_shape=(1, 128, 2, 2, 2),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def test_action_num_frames_uses_longest_batched_action(self) -> None:
        batch = SimpleNamespace(extra={"diffusers_kwargs": {"action": ["w-8", "w-16"]}})

        self.assertEqual(
            SanaWMBeforeDenoisingStage._action_num_frames_for_request(batch),
            17,
        )

    def test_intrinsics_3x3_sequence_can_exceed_requested_frame_count(self) -> None:
        intrinsics = torch.eye(3).unsqueeze(0).repeat(321, 1, 1)

        coerced = SanaWMBeforeDenoisingStage._coerce_intrinsics_vec4(
            intrinsics,
            batch_size=1,
            num_frames=49,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(coerced.shape, (1, 49, 4))
        self.assertTrue(torch.equal(coerced[0, 0], torch.tensor([1.0, 1.0, 0.0, 0.0])))

    def test_latent_frame_camera_conditions_samples_stride_indices(self) -> None:
        original = torch.arange(1 * 17 * 20, dtype=torch.float32).reshape(1, 17, 20)

        sampled = SanaWMBeforeDenoisingStage._latent_frame_camera_conditions(
            original,
            num_frames=17,
            latent_frames=3,
            vae_temporal_stride=8,
        )

        self.assertTrue(torch.equal(sampled[:, 0], original[:, 0]))
        self.assertTrue(torch.equal(sampled[:, 1], original[:, 8]))
        self.assertTrue(torch.equal(sampled[:, 2], original[:, 16]))

    def test_chunk_plucker_accepts_ltx_frame_count(self) -> None:
        camera_conditions = torch.zeros(1, 17, 20)
        camera_conditions[..., :16] = torch.eye(4).reshape(1, 1, 16)
        camera_conditions[..., 16:] = torch.tensor([16.0, 16.0, 10.0, 6.0])

        chunk_plucker = compute_chunk_plucker(
            camera_conditions,
            HW=(3, 12, 20),
            vae_temporal_stride=8,
            patch_size=(1, 1, 1),
        )

        self.assertEqual(chunk_plucker.shape, (1, 48, 3, 12, 20))

    def test_post_ucpe_rms_stabilization_clamps_inflated_tensors(self) -> None:
        ref = torch.ones(1, 2, 4, 3)
        transformed = ref * 8.0

        stabilized = _downscale_to_reference_rms(ref, transformed)

        ref_rms = ref.square().mean(dim=2, keepdim=True).sqrt()
        stabilized_rms = stabilized.square().mean(dim=2, keepdim=True).sqrt()
        self.assertTrue(torch.all(stabilized_rms <= ref_rms + 1e-5))

    def test_rmsnorm_scale_factor_initializes_weight(self) -> None:
        norm = _RMSNorm(4, scale_factor=0.01)
        self.assertTrue(torch.allclose(norm.weight, torch.full((4,), 0.01)))

    def test_prepare_timesteps_uses_inference_flow_shift(self) -> None:
        class FakeScheduler:
            def __init__(self):
                self.shift = None
                self.timesteps = None
                self.sigmas = None

            def set_timesteps(self, num_inference_steps, device, shift=None):
                self.shift = shift
                self.timesteps = torch.arange(num_inference_steps, device=device)
                self.sigmas = torch.tensor([float(shift), 0.0], device=device)

        scheduler = FakeScheduler()
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=scheduler,
            pipeline_config=SanaWMPipelineConfig(),
        )
        batch = SimpleNamespace(num_inference_steps=3, scheduler=None)

        stage._prepare_timesteps(batch, SimpleNamespace(), torch.device("cpu"))

        self.assertEqual(scheduler.shift, 9.8)
        self.assertTrue(torch.equal(batch.timesteps, torch.arange(3)))


class TestSanaWMTextEncodingStage(unittest.TestCase):
    def test_official_prompt_window_keeps_bos_and_tail(self) -> None:
        tensor = torch.arange(5).reshape(1, 5, 1)

        selected = SanaWMTextEncodingStage._select_official_prompt_window(
            tensor, max_length=3
        )

        self.assertTrue(torch.equal(selected.squeeze(-1), torch.tensor([[0, 3, 4]])))


class TestSanaWMDenoisingStage(unittest.TestCase):
    def test_parallelism_type_follows_cfg_parallel_flag(self) -> None:
        stage = object.__new__(SanaWMDenoisingStage)

        stage.server_args = SimpleNamespace(enable_cfg_parallel=False)
        self.assertEqual(stage.parallelism_type, StageParallelismType.REPLICATED)

        stage.server_args = SimpleNamespace(enable_cfg_parallel=True)
        self.assertEqual(stage.parallelism_type, StageParallelismType.CFG_PARALLEL)

    def test_cfg_parallel_formula_matches_serial_cfg(self) -> None:
        pos = torch.tensor([2.0])
        neg = torch.tensor([-1.0])
        guidance_scale = 4.5
        serial = neg + guidance_scale * (pos - neg)

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.base.cfg_model_parallel_all_reduce",
            side_effect=lambda partial: partial + (1.0 - guidance_scale) * neg,
        ):
            combined_from_pos_rank = SanaWMDenoisingStage._combine_cfg_parallel_noise(
                pos, guidance_scale, cfg_rank=0
            )

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.base.cfg_model_parallel_all_reduce",
            side_effect=lambda partial: partial + guidance_scale * pos,
        ):
            combined_from_neg_rank = SanaWMDenoisingStage._combine_cfg_parallel_noise(
                neg, guidance_scale, cfg_rank=1
            )

        self.assertTrue(torch.allclose(combined_from_pos_rank, serial))
        self.assertTrue(torch.allclose(combined_from_neg_rank, serial))

    def test_scheduler_tensors_move_to_target_device(self) -> None:
        scheduler = SimpleNamespace(
            sigmas=torch.tensor([1.0, 0.0]),
            timesteps=torch.tensor([1000.0, 0.0]),
            untouched="value",
        )

        SanaWMDenoisingStage._move_scheduler_tensors_to_device(
            scheduler, torch.device("cpu")
        )

        self.assertEqual(scheduler.sigmas.device.type, "cpu")
        self.assertEqual(scheduler.timesteps.device.type, "cpu")
        self.assertEqual(scheduler.untouched, "value")

        class PropertyScheduler:
            def __init__(self):
                self._sigmas = torch.tensor([1.0, 0.0])
                self._timesteps = torch.tensor([1000.0, 0.0])
                self.untouched = "value"

            @property
            def sigmas(self):
                return self._sigmas

            @property
            def timesteps(self):
                return self._timesteps

        property_scheduler = PropertyScheduler()
        SanaWMDenoisingStage._move_scheduler_tensors_to_device(
            property_scheduler, torch.device("cpu")
        )

        self.assertEqual(property_scheduler.sigmas.device.type, "cpu")
        self.assertEqual(property_scheduler.timesteps.device.type, "cpu")
        self.assertEqual(property_scheduler.untouched, "value")


class TestSanaWMNativeDiTChunking(unittest.TestCase):
    def test_softmax_chunking_is_disabled_by_default_for_upstream_parity(self) -> None:
        arch = SanaWMConfig().arch_config

        self.assertFalse(arch.use_chunked_softmax_attention)

    def test_main_gdn_backend_defaults_to_auto(self) -> None:
        arch = SanaWMConfig().arch_config

        self.assertEqual(arch.gdn_backend, "auto")

    def test_first_chunk_plus_one_chunk_indices_match_upstream(self) -> None:
        self.assertEqual(
            _sana_wm_chunk_index_from_chunk_size(
                21, 4, strategy="first_chunk_plus_one"
            ),
            [0, 5, 9, 13, 17],
        )

    def test_normalize_chunk_index_adds_start_and_final_boundary(self) -> None:
        self.assertEqual(_sana_wm_normalize_chunk_index([3], 5), [0, 3, 5])

    def test_chunked_attention_matches_prefix_attention_per_chunk(self) -> None:
        torch.manual_seed(0)
        q = torch.randn(1, 5, 1, 4)
        k = torch.randn(1, 5, 1, 4)
        v = torch.randn(1, 5, 1, 4)
        scale = 0.5

        out = _sana_wm_chunked_attention(
            q,
            k,
            v,
            HW=(5, 1, 1),
            chunk_size=2,
            chunk_split_strategy="first_chunk_plus_one",
            chunk_index=None,
            softmax_scale=scale,
        )

        first = torch.nn.functional.scaled_dot_product_attention(
            q[:, :3].transpose(1, 2),
            k[:, :3].transpose(1, 2),
            v[:, :3].transpose(1, 2),
            dropout_p=0.0,
            scale=scale,
        ).transpose(1, 2)
        second = torch.nn.functional.scaled_dot_product_attention(
            q[:, 3:].transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            scale=scale,
        ).transpose(1, 2)
        expected = torch.cat([first, second], dim=1)

        self.assertTrue(torch.allclose(out, expected))

    def test_chunked_attention_returns_none_when_bidirectional(self) -> None:
        q = torch.randn(1, 3, 1, 4)

        out = _sana_wm_chunked_attention(
            q,
            q,
            q,
            HW=(3, 1, 1),
            chunk_size=10,
            chunk_split_strategy="first_chunk_plus_one",
            chunk_index=None,
            softmax_scale=0.5,
        )

        self.assertIsNone(out)

    def test_gdn_chunk_scan_matches_recurrent_scan(self) -> None:
        torch.manual_seed(0)
        B, H, D, T, S = 1, 2, 4, 5, 3
        N = T * S
        q = torch.randn(B, H, D, N, dtype=torch.float64)
        k = torch.randn(B, H, D, N, dtype=torch.float64)
        v = torch.randn(B, H, D, N, dtype=torch.float64)
        q_rot = torch.randn(B, H, D, N, dtype=torch.float64)
        k_rot = torch.randn(B, H, D, N, dtype=torch.float64)
        beta = torch.sigmoid(torch.randn(B, H, T, S, dtype=torch.float64))
        decay = torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64))

        recurrent = _gdn_scan_forward(
            q,
            k,
            v,
            q_rot,
            k_rot,
            beta,
            decay,
            return_components=True,
        )
        chunked = _gdn_chunk_scan_forward(
            q,
            k,
            v,
            q_rot,
            k_rot,
            beta,
            decay,
            chunk_size=2,
            return_components=True,
        )

        self.assertTrue(torch.allclose(chunked[0], recurrent[0], atol=1e-9))
        self.assertTrue(torch.allclose(chunked[1], recurrent[1], atol=1e-9))

    def test_camera_chunk_scan_matches_recurrent_scan(self) -> None:
        torch.manual_seed(0)
        B, H, D, T, S = 1, 2, 4, 5, 3
        N = T * S
        q_rot = torch.randn(B, H, D, N, dtype=torch.float64)
        k_rot = torch.randn(B, H, D, N, dtype=torch.float64)
        v = torch.randn(B, H, D, N, dtype=torch.float64)
        beta = torch.sigmoid(torch.randn(B, H, T, S, dtype=torch.float64))
        decay = torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64))

        recurrent = _single_path_delta_scan_forward(q_rot, k_rot, v, beta, decay)
        chunked = _single_path_delta_chunk_scan_forward(
            q_rot,
            k_rot,
            v,
            beta,
            decay,
            chunk_size=2,
        )

        self.assertTrue(torch.allclose(chunked, recurrent, atol=1e-9))


class TestSanaWMRefinerStage(_GlobalStageArgsMixin, unittest.TestCase):
    def test_diffusers_refiner_detection_uses_official_class_name(self) -> None:
        class LTX2VideoTransformer3DModel(torch.nn.Module):
            pass

        self.assertTrue(_uses_diffusers_ltx2_refiner(LTX2VideoTransformer3DModel()))
        self.assertTrue(
            _uses_diffusers_ltx2_refiner(
                OfficialDiffusersLTX2RefinerModule(LTX2VideoTransformer3DModel())
            )
        )

    def test_refiner_config_value_prefers_diffusers_config(self) -> None:
        module = SimpleNamespace(
            patch_size=999,
            config=SimpleNamespace(patch_size=1),
        )

        self.assertEqual(_refiner_config_value(module, "patch_size"), 1)

    def test_official_refiner_wrappers_expose_layerwise_blocks(self) -> None:
        self.assertEqual(
            OfficialDiffusersLTX2RefinerModule.layer_names,
            ["module.transformer_blocks"],
        )
        self.assertIn(
            "module.model.language_model.layers",
            OfficialGemma3TextEncoderModule.layer_names,
        )

    def test_refiner_component_uses_follow_execution_order(self) -> None:
        stage = SanaWMLTX2RefinerStage(
            transformer=torch.nn.Identity(),
            connectors=torch.nn.Identity(),
            text_encoder=torch.nn.Identity(),
            tokenizer=SimpleNamespace(pad_token="<pad>", eos_token="<eos>"),
            dtype=torch.bfloat16,
        )

        names = [
            use.component_name
            for use in stage.component_uses(
                SimpleNamespace(), stage_name="sana_wm_refiner"
            )
        ]

        self.assertEqual(names, ["text_encoder_2", "connectors", "transformer_2"])

    def test_refiner_parallelism_runs_on_main_rank_when_cfg_parallel(self) -> None:
        stage = SanaWMLTX2RefinerStage(
            transformer=torch.nn.Identity(),
            connectors=torch.nn.Identity(),
            text_encoder=torch.nn.Identity(),
            tokenizer=SimpleNamespace(pad_token="<pad>", eos_token="<eos>"),
            dtype=torch.bfloat16,
        )

        stage.server_args = SimpleNamespace(enable_cfg_parallel=False)
        self.assertEqual(stage.parallelism_type, StageParallelismType.REPLICATED)

        stage.server_args = SimpleNamespace(enable_cfg_parallel=True)
        self.assertEqual(stage.parallelism_type, StageParallelismType.MAIN_RANK_ONLY)

    def test_refiner_component_uses_skip_non_main_cfg_rank(self) -> None:
        stage = SanaWMLTX2RefinerStage(
            transformer=torch.nn.Identity(),
            connectors=torch.nn.Identity(),
            text_encoder=torch.nn.Identity(),
            tokenizer=SimpleNamespace(pad_token="<pad>", eos_token="<eos>"),
            dtype=torch.bfloat16,
        )

        with (
            patch(
                f"{_SANA_WM_REFINER_STAGE_MODULE}.torch.distributed.is_available",
                return_value=True,
            ),
            patch(
                f"{_SANA_WM_REFINER_STAGE_MODULE}.torch.distributed.is_initialized",
                return_value=True,
            ),
            patch(
                f"{_SANA_WM_REFINER_STAGE_MODULE}.get_classifier_free_guidance_rank",
                return_value=1,
            ),
        ):
            uses = stage.component_uses(
                SimpleNamespace(enable_cfg_parallel=True),
                stage_name="sana_wm_refiner",
            )

        self.assertEqual(uses, [])

    def test_streaming_diffusers_attention_accepts_ungated_ltx2_attention(self) -> None:
        """Diffusers 0.37 LTX2Attention omits `to_gate_logits` for ungated configs."""

        class IdentityLinear(torch.nn.Module):
            def forward(self, x):
                return x

        class FakeProcessor:
            _attention_backend = None
            _parallel_config = None

        class UngatedAttention(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.to_q = IdentityLinear()
                self.to_k = IdentityLinear()
                self.to_v = IdentityLinear()
                self.norm_q = IdentityLinear()
                self.norm_k = IdentityLinear()
                self.to_out = torch.nn.ModuleList(
                    [IdentityLinear(), torch.nn.Identity()]
                )
                self.heads = 2
                self.rope_type = "split"
                self.processor = FakeProcessor()

        def dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=None,
            dropout_p=0.0,
            is_causal=False,
            backend=None,
            parallel_config=None,
        ):
            return query

        attention_dispatch = types.ModuleType("diffusers.models.attention_dispatch")
        attention_dispatch.dispatch_attention_fn = dispatch_attention_fn
        transformer_ltx2 = types.ModuleType(
            "diffusers.models.transformers.transformer_ltx2"
        )
        transformer_ltx2.apply_interleaved_rotary_emb = lambda x, _rope: x
        transformer_ltx2.apply_split_rotary_emb = lambda x, _rope: x

        with patch.dict(
            sys.modules,
            {
                "diffusers": types.ModuleType("diffusers"),
                "diffusers.models": types.ModuleType("diffusers.models"),
                "diffusers.models.attention_dispatch": attention_dispatch,
                "diffusers.models.transformers": types.ModuleType(
                    "diffusers.models.transformers"
                ),
                "diffusers.models.transformers.transformer_ltx2": transformer_ltx2,
            },
        ):
            hidden_states = torch.randn(1, 3, 4)
            rotary = (torch.empty(0), torch.empty(0))
            out = _streaming_diffusers_self_attention(
                attn=UngatedAttention(),
                hidden_states=hidden_states,
                query_rotary_emb=rotary,
                n_context_tokens=1,
            )

        self.assertEqual(out.shape, hidden_states.shape)

    def test_skip_refiner_flag_accepts_request_extra(self) -> None:
        batch = SimpleNamespace(extra={"diffusers_kwargs": {"skip_refiner": True}})
        self.assertTrue(sana_wm_skip_refiner_enabled(batch))

    def test_prompt_resolution_broadcasts_single_prompt(self) -> None:
        batch = Req(prompt="drive forward")
        prompts = SanaWMLTX2RefinerStage._prompts_for_batch(batch, batch_size=2)
        self.assertEqual(prompts, ["drive forward", "drive forward"])

    def test_prompt_resolution_accepts_batch_prompt_list(self) -> None:
        batch = Req(prompt=["left", "right"])
        prompts = SanaWMLTX2RefinerStage._prompts_for_batch(batch, batch_size=2)
        self.assertEqual(prompts, ["left", "right"])

    def test_refiner_prompt_encoding_uses_hf_gemma_backbone(self) -> None:
        class DummyTokenizer:
            padding_side = "right"
            pad_token = None
            eos_token = "<eos>"

            def __call__(self, *args, **kwargs):
                return SimpleNamespace(
                    input_ids=torch.tensor([[1, 2, 0, 0]]),
                    attention_mask=torch.tensor([[1, 1, 0, 0]]),
                )

        class DummyBackbone:
            def __init__(self):
                self.called = False

            def __call__(self, **kwargs):
                self.called = True
                hidden0 = torch.arange(8, dtype=torch.float32).reshape(1, 4, 2)
                hidden1 = hidden0 + 10
                return SimpleNamespace(hidden_states=(hidden0, hidden1))

        class DummyTextEncoder:
            def __init__(self):
                self.called = False
                self.model = DummyBackbone()

            def __call__(self, **kwargs):
                self.called = True
                raise AssertionError(
                    "Gemma3ForConditionalGeneration.model was not used"
                )

        class DummyConnectors:
            def __call__(self, prompt_embeds, attention_mask):
                return prompt_embeds, None, attention_mask

        stage = SanaWMLTX2RefinerStage(
            transformer=torch.nn.Identity(),
            connectors=DummyConnectors(),
            text_encoder=DummyTextEncoder(),
            tokenizer=DummyTokenizer(),
            dtype=torch.float32,
            text_max_sequence_length=4,
        )

        prompt_embeds, attention_mask = stage._encode_prompt(
            "drive forward", torch.device("cpu")
        )

        self.assertTrue(stage.text_encoder.model.called)
        self.assertFalse(stage.text_encoder.called)
        self.assertEqual(prompt_embeds.shape, (1, 4, 4))
        self.assertTrue(torch.equal(attention_mask, torch.tensor([[1, 1, 0, 0]])))

    def test_refiner_decoding_drops_clean_sink_frame_after_decode(self) -> None:
        stage = SanaWMRefinerDecodingStage(vae=None)
        decoded = torch.arange(1 * 3 * 4 * 2 * 2).reshape(1, 3, 4, 2, 2)

        with patch.object(DecodingStage, "decode", return_value=decoded):
            frames = stage.decode(
                torch.empty(1, 128, 4, 2, 2),
                SimpleNamespace(pipeline_config=SanaWMPipelineConfig()),
                vae_dtype=torch.bfloat16,
            )

        self.assertTrue(torch.equal(frames, decoded[:, :, 1:]))

    def test_refiner_decoding_keeps_sink_frame_when_refiner_skipped(self) -> None:
        stage = SanaWMRefinerDecodingStage(vae=None)
        decoded = torch.arange(1 * 3 * 4 * 2 * 2).reshape(1, 3, 4, 2, 2)
        stage._drop_refiner_sink = False

        with patch.object(DecodingStage, "decode", return_value=decoded):
            frames = stage.decode(
                torch.empty(1, 128, 4, 2, 2),
                SimpleNamespace(pipeline_config=SanaWMPipelineConfig()),
                vae_dtype=torch.bfloat16,
            )

        self.assertTrue(torch.equal(frames, decoded))


if __name__ == "__main__":
    unittest.main()
