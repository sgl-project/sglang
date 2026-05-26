import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import SanaWMPipelineConfig
from sglang.multimodal_gen.configs.sample.sana_wm import SanaWMSamplingParams
from sglang.multimodal_gen.configs.models.dits.sana_wm import SanaWMConfig
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.registry import _get_config_info, get_pipeline_config_classes
from sglang.multimodal_gen.runtime.pipelines.sana_wm_pipeline import (
    SanaWMTwoStagePipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm_refiner import (
    SanaWMLTX2RefinerStage,
    SanaWMRefinerDecodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
    SanaWMBeforeDenoisingStage,
    SanaWMDenoisingStage,
    SanaWMTextEncodingStage,
    _align_sana_wm_cfg_text_conditions,
    configure_sana_wm_ltx2_vae_for_long_video,
)
from sglang.multimodal_gen.runtime.models.dits.sana_wm import (
    _RMSNorm,
    _downscale_to_reference_rms,
    compute_chunk_plucker,
)
from sglang.multimodal_gen.runtime.utils.model_overlay import (
    resolve_model_overlay_target,
)
from sglang.multimodal_gen.test.test_utils import DEFAULT_SANA_WM_MODEL_NAME_FOR_TEST


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

    def test_text_encoder_padding_matches_cfg_concat_contract(self) -> None:
        self.assertEqual(self.config.text_encoder_extra_args[0]["padding"], "max_length")
        self.assertTrue(self.config.text_encoder_extra_args[0]["return_attention_mask"])
        self.assertTrue(self.config.chi_prompt)

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
                self.tile_sample_min_num_frames = kwargs[
                    "tile_sample_min_num_frames"
                ]
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
        self.assertTrue(torch.equal(neg_mask[:, :1], torch.ones(1, 1, dtype=torch.long)))
        self.assertTrue(torch.equal(neg_mask[:, 1:], torch.zeros(1, 172, dtype=torch.long)))


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
            snapshot_dir = (
                f"{tmp_dir}/hub/models--Lightricks--LTX-2.3/snapshots/abc123"
            )
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

    def test_refiner_text_encoder_config_is_scoped_to_manual_load(self) -> None:
        config = SanaWMPipelineConfig()
        server_args = SimpleNamespace(pipeline_config=config)
        original_configs = config.text_encoder_configs
        original_precisions = config.text_encoder_precisions

        saved = SanaWMTwoStagePipeline._ensure_refiner_text_encoder_config(server_args)
        try:
            self.assertEqual(len(config.text_encoder_configs), 2)
            self.assertEqual(len(config.text_encoder_precisions), 2)
            self.assertEqual(config.text_encoder_precisions[1], "bf16")
            refiner_config = config.text_encoder_configs[1]
            refiner_config.update_model_arch(
                {
                    "architectures": ["Gemma3ForConditionalGeneration"],
                    "text_config": {"vocab_size": 10},
                    "vision_config": {"hidden_size": 20},
                }
            )
            self.assertEqual(refiner_config.text_config.vocab_size, 10)
            self.assertEqual(refiner_config.vision_config.hidden_size, 20)
        finally:
            config.text_encoder_configs, config.text_encoder_precisions = saved

        self.assertIs(config.text_encoder_configs, original_configs)
        self.assertIs(config.text_encoder_precisions, original_precisions)


class TestSanaWMBeforeDenoisingStage(unittest.TestCase):
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
        self.assertTrue(torch.equal(camera_conditions[0, 0, :16], torch.eye(4).reshape(-1)))
        self.assertTrue(
            torch.allclose(
                camera_conditions[0, 0, 16:],
                torch.tensor([16.0, 16.0, 10.0, 6.0]),
            )
        )

    def test_explicit_camera_request_detects_diffusers_path_kwargs(self) -> None:
        batch = SimpleNamespace(
            extra={"diffusers_kwargs": {"camera_to_world_path": "/tmp/cam.npy"}}
        )

        self.assertTrue(SanaWMBeforeDenoisingStage._has_explicit_camera_request(batch))

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


class TestSanaWMTextEncodingStage(unittest.TestCase):
    def test_official_prompt_window_keeps_bos_and_tail(self) -> None:
        tensor = torch.arange(5).reshape(1, 5, 1)

        selected = SanaWMTextEncodingStage._select_official_prompt_window(
            tensor, max_length=3
        )

        self.assertTrue(torch.equal(selected.squeeze(-1), torch.tensor([[0, 3, 4]])))


class TestSanaWMRefinerStage(unittest.TestCase):
    def test_prompt_resolution_broadcasts_single_prompt(self) -> None:
        batch = Req(prompt="drive forward")
        prompts = SanaWMLTX2RefinerStage._prompts_for_batch(batch, batch_size=2)
        self.assertEqual(prompts, ["drive forward", "drive forward"])

    def test_prompt_resolution_accepts_batch_prompt_list(self) -> None:
        batch = Req(prompt=["left", "right"])
        prompts = SanaWMLTX2RefinerStage._prompts_for_batch(batch, batch_size=2)
        self.assertEqual(prompts, ["left", "right"])

    def test_refiner_decoding_drops_clean_sink_frame_after_decode(self) -> None:
        stage = SanaWMRefinerDecodingStage(vae=None)
        decoded = torch.arange(1 * 3 * 4 * 2 * 2).reshape(1, 3, 4, 2, 2)

        with patch.object(DecodingStage, "decode", return_value=decoded):
            frames = stage.decode(
                torch.empty(1, 128, 4, 2, 2),
                SimpleNamespace(),
                vae_dtype=torch.bfloat16,
            )

        self.assertTrue(torch.equal(frames, decoded[:, :, 1:]))


if __name__ == "__main__":
    unittest.main()
