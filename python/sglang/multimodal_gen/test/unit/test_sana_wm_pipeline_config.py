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
from sglang.multimodal_gen.configs.models.dits.sana_wm_refiner import (
    SanaWMRefinerConfig,
)
from sglang.multimodal_gen.configs.pipeline_configs.base import ModelTaskType
from sglang.multimodal_gen.configs.pipeline_configs.sana_wm import SanaWMPipelineConfig
from sglang.multimodal_gen.configs.sample.sana_wm import SanaWMSamplingParams
from sglang.multimodal_gen.registry import (
    _get_config_info,
    get_non_diffusers_pipeline_name,
    get_pipeline_config_classes,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.models.dits.sana_wm import (
    _downscale_to_reference_rms,
    _gdn_chunk_scan_forward,
    _gdn_scan_forward,
    _UpstreamMlp,
    BidirectionalGDNUCPESinglePathLiteLA,
    GLUMBConvTemp,
    MultiHeadCrossAttention,
    PatchEmbedMS3D,
    _RMSNorm,
    _SanaWMPaddedLocalAttention,
    SanaWMBlock,
    SanaWMTransformer3DModel,
    T2IFinalLayer,
    TimestepEmbedder,
    _make_sana_wm_local_attention,
    _sana_wm_chunk_index_from_chunk_size,
    _sana_wm_chunked_attention,
    _sana_wm_normalize_chunk_index,
    _sana_wm_padded_attention_head_size,
    _sana_wm_sequence_shard_enabled,
    _sana_wm_sp_rank,
    _sana_wm_sp_world_size,
    _single_path_delta_chunk_scan_forward,
    _single_path_delta_scan_forward,
    _tensor_cache_key,
    compute_chunk_plucker,
)
from sglang.multimodal_gen.runtime.pipelines.sana_wm_pipeline import (
    SanaWMPipeline,
    SanaWMTwoStagePipeline,
    _configured_sana_wm_refiner_backend,
    _configured_sana_wm_two_stage_residency,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_resident_strategies import (
    VanillaD2HStrategy,
)
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
    SanaWMBeforeDenoisingStage,
    SanaWMDenoisingStage,
    SanaWMTextEncodingStage,
    _align_sana_wm_cfg_text_conditions,
    _sana_wm_effective_guidance_scale,
    _sana_wm_should_do_cfg,
    configure_sana_wm_ltx2_vae_for_long_video,
)
from sglang.multimodal_gen.runtime.utils.sana_wm_camera import (
    coerce_sana_wm_intrinsics_vec4,
    latent_frame_sana_wm_camera_conditions,
    parse_sana_wm_action_string,
    sana_wm_action_to_camera_to_world,
)
from sglang.multimodal_gen.runtime.utils.sana_wm_runtime_cache import (
    SANA_WM_REQUEST_RUNTIME_CACHE_KEY,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm_refiner import (
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
    "model_specific_stages.sana_wm_refiner"
)
_SANA_WM_STAGE_MODULE = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages."
    "model_specific_stages.sana_wm"
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

    def test_pipeline_config_public_export(self) -> None:
        from sglang.multimodal_gen.configs.pipeline_configs import (
            SanaWMPipelineConfig as ExportedSanaWMPipelineConfig,
        )

        self.assertIs(ExportedSanaWMPipelineConfig, SanaWMPipelineConfig)

    def test_sana_wm_runtime_controls_default_to_auto(self) -> None:
        self.assertEqual(self.config.sana_wm_refiner_backend, "auto")
        self.assertEqual(self.config.sana_wm_two_stage_residency, "auto")
        self.assertFalse(self.config.sana_wm_skip_refiner)

    def test_sana_wm_runtime_controls_normalize_from_config(self) -> None:
        config = SanaWMPipelineConfig(
            sana_wm_refiner_backend="NATIVE",
            sana_wm_two_stage_residency="SEQUENTIAL",
            sana_wm_skip_refiner=True,
        )

        self.assertEqual(config.sana_wm_refiner_backend, "native")
        self.assertEqual(config.sana_wm_two_stage_residency, "sequential")
        self.assertTrue(config.sana_wm_skip_refiner)

    def test_sana_wm_runtime_controls_reject_invalid_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "sana_wm_refiner_backend"):
            SanaWMPipelineConfig(sana_wm_refiner_backend="bad-backend")
        with self.assertRaisesRegex(ValueError, "sana_wm_two_stage_residency"):
            SanaWMPipelineConfig(sana_wm_two_stage_residency="bad-residency")

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

    def test_sp_latent_sharding_is_replicated_for_stage1_safety(self) -> None:
        latents = torch.zeros(1, 128, 7, 22, 40)
        batch = SimpleNamespace(enable_sequence_shard=True)

        sharded, did_shard = self.config.shard_latents_for_sp(batch, latents)
        gathered = self.config.gather_latents_for_sp(sharded, batch=batch)

        self.assertIs(sharded, latents)
        self.assertIs(gathered, latents)
        self.assertFalse(did_shard)

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

    def test_get_model_deployment_config_prefers_native_tp_residency(self) -> None:
        deployment = self.config.get_model_deployment_config()
        self.assertEqual(deployment.auto_full_device_tp_size_candidates, (2, 4))
        self.assertFalse(deployment.auto_dit_layerwise_offload)
        self.assertEqual(
            deployment.auto_disable_default_layerwise_offload_min_available_memory_gb,
            70,
        )
        self.assertEqual(
            deployment.auto_disable_component_offload_min_available_memory_gb, 70
        )
        self.assertEqual(
            deployment.auto_disable_component_offload_components,
            ("dit", "text_encoder", "image_encoder", "vae"),
        )
        self.assertIsNone(deployment.fsdp_auto_min_available_memory_gb)

    def test_text_encoder_padding_matches_cfg_concat_contract(self) -> None:
        self.assertEqual(
            self.config.text_encoder_extra_args[0]["padding"], "max_length"
        )
        self.assertTrue(self.config.text_encoder_extra_args[0]["return_attention_mask"])
        self.assertTrue(self.config.chi_prompt)

    def test_inference_flow_shift_matches_official_reference(self) -> None:
        self.assertEqual(self.config.flow_shift, 9.95)
        self.assertEqual(self.config.inference_flow_shift, 9.8)

    def test_dit_attention_flags_sync_to_arch_config(self) -> None:
        dit_config = SanaWMConfig(
            use_chunked_softmax_attention=True,
            pad_attention_head_dim_to_flash=True,
            use_triton_kernels=True,
        )

        self.assertTrue(dit_config.arch_config.use_chunked_softmax_attention)
        self.assertTrue(dit_config.arch_config.pad_attention_head_dim_to_flash)
        self.assertTrue(dit_config.arch_config.use_triton_kernels)

    def test_dit_triton_kernels_default_enabled(self) -> None:
        self.assertTrue(SanaWMConfig().arch_config.use_triton_kernels)
        self.assertFalse(
            SanaWMConfig(use_triton_kernels=False).arch_config.use_triton_kernels
        )

    def test_pipeline_config_dict_can_enable_dit_attention_flags(self) -> None:
        self.config.update_pipeline_config(
            {
                "dit_config": {
                    "use_chunked_softmax_attention": True,
                    "pad_attention_head_dim_to_flash": True,
                    "use_triton_kernels": True,
                }
            }
        )
        arch = self.config.dit_config.arch_config

        self.assertTrue(arch.use_chunked_softmax_attention)
        self.assertTrue(arch.pad_attention_head_dim_to_flash)
        self.assertTrue(arch.use_triton_kernels)

    def test_dit_arch_text_norm_defaults_match_upstream(self) -> None:
        arch = SanaWMConfig().arch_config
        self.assertTrue(arch.y_norm)
        self.assertEqual(arch.y_norm_scale_factor, 0.01)
        self.assertEqual(arch.y_norm_eps, 1e-5)
        self.assertEqual(arch.timestep_norm_scale_factor, 1.0)

    def test_refiner_dit_config_is_available_for_native_backend(self) -> None:
        self.assertIsInstance(self.config.refiner_dit_config, SanaWMRefinerConfig)
        arch = self.config.refiner_dit_config.arch_config
        self.assertEqual(arch.num_attention_heads, 32)
        self.assertEqual(arch.attention_head_dim, 64)

    def test_dit_tp_config_requires_heads_divisible_by_tp_size(self) -> None:
        arch = SanaWMConfig().arch_config
        SanaWMTransformer3DModel._validate_tp_config(arch, 2)
        with self.assertRaisesRegex(ValueError, "num_attention_heads"):
            SanaWMTransformer3DModel._validate_tp_config(arch, 3)

    def test_dit_stage1_tp_uses_parallel_linear_for_unsharded_edges(self) -> None:
        module_path = "sglang.multimodal_gen.runtime.models.dits.sana_wm"

        class FakeColumnParallelLinear(torch.nn.Linear):
            def __init__(
                self,
                input_size,
                output_size,
                bias=True,
                gather_output=False,
                **kwargs,
            ) -> None:
                super().__init__(input_size, output_size // 2, bias=bias)
                self.gather_output = gather_output

            def forward(self, x):
                return super().forward(x), None

        class FakeMergedColumnParallelLinear(FakeColumnParallelLinear):
            def __init__(
                self,
                input_size,
                output_sizes,
                bias=True,
                gather_output=False,
                **kwargs,
            ) -> None:
                output_sizes = tuple(output_sizes)
                super().__init__(
                    input_size,
                    sum(output_sizes),
                    bias=bias,
                    gather_output=gather_output,
                )
                self.output_sizes = output_sizes

        class FakeRowParallelLinear(torch.nn.Linear):
            def __init__(
                self,
                input_size,
                output_size,
                bias=True,
                input_is_parallel=True,
                **kwargs,
            ) -> None:
                super().__init__(input_size // 2, output_size, bias=bias)
                self.input_is_parallel = input_is_parallel

            def forward(self, x):
                return super().forward(x), None

        class FakeLocalAttention(torch.nn.Module):
            def __init__(
                self,
                num_heads,
                head_size,
                num_kv_heads=None,
                softmax_scale=None,
                causal=False,
                **extra_impl_args,
            ) -> None:
                super().__init__()
                self.softmax_scale = softmax_scale

        with (
            patch(f"{module_path}._sana_wm_tp_world_size", return_value=2),
            patch(f"{module_path}._sana_wm_tp_rank", return_value=0),
            patch(f"{module_path}.ColumnParallelLinear", FakeColumnParallelLinear),
            patch(
                f"{module_path}.MergedColumnParallelLinear",
                FakeMergedColumnParallelLinear,
            ),
            patch(f"{module_path}.RowParallelLinear", FakeRowParallelLinear),
            patch(f"{module_path}.LocalAttention", FakeLocalAttention),
        ):
            patch_embed = PatchEmbedMS3D((1, 1, 1), 3, 16)
            glumb = GLUMBConvTemp(16, 24, t_kernel_size=3)
            timestep = TimestepEmbedder(16, frequency_embedding_size=8)
            caption_mlp = _UpstreamMlp(12, 16, 16)
            final_layer = T2IFinalLayer(16, (1, 1, 1), 8)
            block = SanaWMBlock(
                hidden_size=16,
                num_heads=4,
                head_dim=4,
                mlp_ratio=1.5,
                t_kernel_size=3,
                qk_norm=True,
                cross_norm=True,
                conv_kernel_size=0,
                k_conv_only=True,
                softmax_main=False,
                use_chunk_plucker_post_attn=True,
                use_triton_kernels=False,
            )
            attn = BidirectionalGDNUCPESinglePathLiteLA(
                in_dim=16,
                heads=4,
                head_dim=4,
                conv_kernel_size=0,
                softmax_main=False,
                use_triton_kernels=False,
            )

        self.assertEqual(patch_embed.proj.tp_size, 2)
        self.assertTrue(patch_embed.proj.gather_output)
        self.assertEqual(tuple(patch_embed.proj.weight.shape), (8, 3, 1, 1, 1))

        self.assertEqual(glumb.inverted_conv.conv.tp_size, 2)
        self.assertTrue(glumb.inverted_conv.conv.paired_output)
        self.assertFalse(glumb.inverted_conv.conv.gather_output)
        self.assertEqual(
            tuple(glumb.inverted_conv.conv.weight.shape), (24, 16, 1, 1)
        )
        self.assertEqual(glumb.depth_conv.conv.tp_size, 2)
        self.assertEqual(tuple(glumb.depth_conv.conv.weight.shape), (24, 1, 3, 3))
        self.assertEqual(glumb.point_conv.conv.tp_size, 2)
        self.assertEqual(tuple(glumb.point_conv.conv.weight.shape), (16, 12, 1, 1))
        self.assertEqual(glumb.t_conv.tp_size, 2)
        self.assertTrue(glumb.t_conv.gather_output)
        self.assertEqual(tuple(glumb.t_conv.weight.shape), (8, 16, 3, 1))

        self.assertIsInstance(block.plucker_proj, FakeColumnParallelLinear)
        self.assertTrue(block.plucker_proj.gather_output)

        self.assertIsInstance(timestep.mlp[0], FakeColumnParallelLinear)
        self.assertFalse(timestep.mlp[0].gather_output)
        self.assertIsInstance(timestep.mlp[2], FakeRowParallelLinear)
        self.assertTrue(timestep.mlp[2].input_is_parallel)

        self.assertIsInstance(caption_mlp.fc1, FakeColumnParallelLinear)
        self.assertFalse(caption_mlp.fc1.gather_output)
        self.assertIsInstance(caption_mlp.fc2, FakeRowParallelLinear)
        self.assertTrue(caption_mlp.fc2.input_is_parallel)

        self.assertIsInstance(final_layer.linear, FakeColumnParallelLinear)
        self.assertTrue(final_layer.linear.gather_output)

        self.assertIsInstance(attn.beta_proj, FakeColumnParallelLinear)
        self.assertFalse(attn.beta_proj.gather_output)
        self.assertIsInstance(attn.gate_proj, FakeColumnParallelLinear)
        self.assertFalse(attn.gate_proj.gather_output)
        self.assertIsInstance(attn.output_gate, FakeColumnParallelLinear)
        self.assertFalse(attn.output_gate.gather_output)
        self.assertIsInstance(attn.out_proj_cam, FakeColumnParallelLinear)
        self.assertFalse(attn.out_proj_cam.gather_output)
        self.assertIsInstance(attn.proj, FakeRowParallelLinear)
        self.assertTrue(attn.proj.input_is_parallel)

    def test_dit_sp_helpers_default_to_noop_without_model_parallel(self) -> None:
        self.assertEqual(_sana_wm_sp_world_size(), 1)
        self.assertEqual(_sana_wm_sp_rank(), 0)
        self.assertFalse(_sana_wm_sequence_shard_enabled(1))
        with self.assertRaisesRegex(NotImplementedError, "sequence parallelism"):
            _sana_wm_sequence_shard_enabled(2)

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
    @staticmethod
    def _server_args():
        return SimpleNamespace(pipeline_config=SanaWMPipelineConfig(), output_path=None)

    def test_defaults_match_video_ti2v_contract(self) -> None:
        params = SanaWMSamplingParams()
        self.assertEqual(params.height, 704)
        self.assertEqual(params.width, 1280)
        self.assertEqual(params.num_frames, 49)
        self.assertEqual(params.num_inference_steps, 20)
        self.assertEqual(params.guidance_scale, 4.5)
        self.assertEqual(params.negative_prompt, "")

    def test_adjust_omits_camera_condition_inputs_by_default(self) -> None:
        params = SanaWMSamplingParams()
        params._adjust(self._server_args())
        self.assertNotIn("camera_to_world", params.condition_inputs)
        self.assertNotIn("intrinsics", params.condition_inputs)

    def test_adjust_includes_camera_condition_inputs_when_set(self) -> None:
        cam = torch.eye(4).unsqueeze(0).expand(49, 4, 4)
        intr = torch.eye(3).unsqueeze(0).expand(49, 3, 3)
        params = SanaWMSamplingParams(camera_to_world=cam, intrinsics=intr)
        params._adjust(self._server_args())
        self.assertIs(params.condition_inputs["camera_to_world"], cam)
        self.assertIs(params.condition_inputs["intrinsics"], intr)

    def test_adjust_includes_action_condition_inputs_when_set(self) -> None:
        params = SanaWMSamplingParams(
            action="w-8,jw-8",
            translation_speed=0.06,
            rotation_speed_deg=2.0,
            pitch_limit_deg=70.0,
        )

        params._adjust(self._server_args())

        self.assertEqual(params.condition_inputs["action"], "w-8,jw-8")
        self.assertEqual(params.condition_inputs["translation_speed"], 0.06)
        self.assertEqual(params.condition_inputs["rotation_speed_deg"], 2.0)
        self.assertEqual(params.condition_inputs["pitch_limit_deg"], 70.0)

    def test_adjust_rejects_action_with_direct_camera(self) -> None:
        cam = torch.eye(4).unsqueeze(0).expand(49, 4, 4)
        params = SanaWMSamplingParams(action="w-8", camera_to_world=cam)

        with self.assertRaisesRegex(ValueError, "either action or camera_to_world"):
            params._adjust(self._server_args())

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

    def test_raw_sana_wm_path_uses_native_two_stage_pipeline(self) -> None:
        self.assertEqual(
            get_non_diffusers_pipeline_name("Efficient-Large-Model/SANA-WM_bidirectional"),
            "SanaWMTwoStagePipeline",
        )
        self.assertEqual(
            get_non_diffusers_pipeline_name("/models/sana_wm_bidirectional"),
            "SanaWMTwoStagePipeline",
        )

    def test_overlay_resolver_matches_hf_cache_snapshot_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_dir = f"{tmp_dir}/hub/models--Lightricks--LTX-2.3/snapshots/abc123"
            os.makedirs(snapshot_dir)
            target = resolve_model_overlay_target(snapshot_dir)
        self.assertIsNotNone(target)
        source_model_id, _ = target
        self.assertEqual(source_model_id, "Lightricks/LTX-2.3")


class TestSanaWMTwoStagePipeline(unittest.TestCase):
    @staticmethod
    def _make_two_stage_pipeline() -> SanaWMTwoStagePipeline:
        pipeline = object.__new__(SanaWMTwoStagePipeline)
        pipeline.modules = {
            "text_encoder": torch.nn.Linear(1, 1),
            "transformer": torch.nn.Linear(1, 1),
            "text_encoder_2": torch.nn.Linear(1, 1),
            "connectors": torch.nn.Linear(1, 1),
            "transformer_2": torch.nn.Linear(1, 1),
            "tokenizer_2": object(),
        }
        pipeline.component_residency_strategies = {}
        return pipeline

    @staticmethod
    def _make_two_stage_server_args(**overrides) -> SimpleNamespace:
        values = {
            "performance_mode": "auto",
            "tp_size": 1,
            "enable_cfg_parallel": False,
            "use_fsdp_inference": False,
            "dit_cpu_offload": False,
            "text_encoder_cpu_offload": False,
            "vae_cpu_offload": False,
            "dit_layerwise_offload": False,
            "layerwise_offload_components": None,
        }
        values.update(overrides)
        return SimpleNamespace(**values)

    def _assert_two_stage_sequential_residency(
        self, pipeline: SanaWMTwoStagePipeline
    ) -> None:
        for component_name in (
            "text_encoder",
            "transformer",
            "text_encoder_2",
            "connectors",
            "transformer_2",
        ):
            self.assertIsInstance(
                pipeline.component_residency_strategies[component_name],
                VanillaD2HStrategy,
            )
        self.assertNotIn("tokenizer_2", pipeline.component_residency_strategies)

    def test_two_stage_pipeline_auto_residency_enables_for_manual_tp_overlap(
        self,
    ) -> None:
        pipeline = self._make_two_stage_pipeline()
        custom_strategy = object()
        pipeline.component_residency_strategies = {"transformer": custom_strategy}
        server_args = self._make_two_stage_server_args(
            performance_mode="manual",
            tp_size=2,
            enable_cfg_parallel=False,
        )

        with patch.dict(
            os.environ,
            {"SGLANG_SANA_WM_TWO_STAGE_RESIDENCY": "auto"},
        ):
            pipeline._configure_two_stage_component_residency(server_args)

        self.assertIs(
            pipeline.component_residency_strategies["transformer"],
            custom_strategy,
        )
        for component_name in (
            "text_encoder",
            "text_encoder_2",
            "connectors",
            "transformer_2",
        ):
            self.assertIsInstance(
                pipeline.component_residency_strategies[component_name],
                VanillaD2HStrategy,
            )
        self.assertNotIn("tokenizer_2", pipeline.component_residency_strategies)

    def test_two_stage_pipeline_auto_residency_skips_safe_auto_path(self) -> None:
        pipeline = self._make_two_stage_pipeline()
        server_args = self._make_two_stage_server_args(performance_mode="auto")

        with patch.dict(
            os.environ,
            {"SGLANG_SANA_WM_TWO_STAGE_RESIDENCY": "auto"},
        ):
            pipeline._configure_two_stage_component_residency(server_args)

        self.assertEqual(pipeline.component_residency_strategies, {})

    def test_two_stage_pipeline_auto_residency_skips_fsdp_policy(self) -> None:
        pipeline = self._make_two_stage_pipeline()
        server_args = self._make_two_stage_server_args(
            performance_mode="manual",
            tp_size=2,
            use_fsdp_inference=True,
        )

        with patch.dict(
            os.environ,
            {"SGLANG_SANA_WM_TWO_STAGE_RESIDENCY": "auto"},
        ):
            pipeline._configure_two_stage_component_residency(server_args)

        self.assertEqual(pipeline.component_residency_strategies, {})

    def test_two_stage_pipeline_auto_residency_skips_dit_layerwise_policy(
        self,
    ) -> None:
        pipeline = self._make_two_stage_pipeline()
        server_args = self._make_two_stage_server_args(
            performance_mode="manual",
            tp_size=2,
            layerwise_offload_components=["dit"],
        )

        with patch.dict(
            os.environ,
            {"SGLANG_SANA_WM_TWO_STAGE_RESIDENCY": "auto"},
        ):
            pipeline._configure_two_stage_component_residency(server_args)

        self.assertEqual(pipeline.component_residency_strategies, {})

    def test_two_stage_pipeline_auto_residency_allows_text_layerwise_policy(
        self,
    ) -> None:
        pipeline = self._make_two_stage_pipeline()
        server_args = self._make_two_stage_server_args(
            performance_mode="manual",
            tp_size=2,
            layerwise_offload_components=["text_encoder", "vae"],
        )

        with patch.dict(
            os.environ,
            {"SGLANG_SANA_WM_TWO_STAGE_RESIDENCY": "auto"},
        ):
            pipeline._configure_two_stage_component_residency(server_args)

        self._assert_two_stage_sequential_residency(pipeline)

    def test_two_stage_pipeline_auto_residency_ignores_vae_only_offload(self) -> None:
        pipeline = self._make_two_stage_pipeline()
        server_args = self._make_two_stage_server_args(
            performance_mode="manual",
            tp_size=2,
            vae_cpu_offload=True,
        )

        with patch.dict(
            os.environ,
            {"SGLANG_SANA_WM_TWO_STAGE_RESIDENCY": "auto"},
        ):
            pipeline._configure_two_stage_component_residency(server_args)

        self._assert_two_stage_sequential_residency(pipeline)

    def test_two_stage_pipeline_residency_env_can_force_resident_path(self) -> None:
        pipeline = self._make_two_stage_pipeline()
        server_args = self._make_two_stage_server_args(
            performance_mode="manual",
            tp_size=2,
        )

        with patch.dict(
            os.environ,
            {"SGLANG_SANA_WM_TWO_STAGE_RESIDENCY": "resident"},
        ):
            pipeline._configure_two_stage_component_residency(server_args)

        self.assertEqual(pipeline.component_residency_strategies, {})

    def test_two_stage_pipeline_residency_env_can_force_sequential_path(self) -> None:
        pipeline = self._make_two_stage_pipeline()
        server_args = self._make_two_stage_server_args(performance_mode="auto")

        with patch.dict(
            os.environ,
            {"SGLANG_SANA_WM_TWO_STAGE_RESIDENCY": "sequential"},
        ):
            pipeline._configure_two_stage_component_residency(server_args)

        self._assert_two_stage_sequential_residency(pipeline)

    def test_two_stage_pipeline_residency_config_can_force_sequential_path(
        self,
    ) -> None:
        pipeline = self._make_two_stage_pipeline()
        server_args = self._make_two_stage_server_args(
            performance_mode="auto",
            pipeline_config=SanaWMPipelineConfig(
                sana_wm_two_stage_residency="sequential"
            ),
        )

        with patch.dict(os.environ, {}, clear=True):
            pipeline._configure_two_stage_component_residency(server_args)

        self._assert_two_stage_sequential_residency(pipeline)

    def test_two_stage_pipeline_residency_config_overrides_env(self) -> None:
        server_args = self._make_two_stage_server_args(
            pipeline_config=SanaWMPipelineConfig(
                sana_wm_two_stage_residency="resident"
            )
        )

        with patch.dict(
            os.environ,
            {"SGLANG_SANA_WM_TWO_STAGE_RESIDENCY": "sequential"},
            clear=True,
        ):
            self.assertEqual(
                _configured_sana_wm_two_stage_residency(server_args),
                "resident",
            )

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

    def test_refiner_backend_auto_uses_native_for_tp(self) -> None:
        with patch.dict(
            os.environ,
            {"SGLANG_SANA_WM_REFINER_BACKEND": "auto"},
        ):
            self.assertEqual(
                _configured_sana_wm_refiner_backend(
                    self._make_two_stage_server_args(tp_size=2)
                ),
                "native",
            )
            self.assertEqual(
                _configured_sana_wm_refiner_backend(
                    self._make_two_stage_server_args(
                        tp_size=2,
                        enable_cfg_parallel=True,
                    )
                ),
                "native",
            )
            self.assertEqual(
                _configured_sana_wm_refiner_backend(
                    self._make_two_stage_server_args(tp_size=1)
                ),
                "official",
            )

    def test_refiner_backend_env_can_force_official(self) -> None:
        with patch.dict(
            os.environ,
            {"SGLANG_SANA_WM_REFINER_BACKEND": "official"},
        ):
            self.assertEqual(
                _configured_sana_wm_refiner_backend(
                    self._make_two_stage_server_args(
                        tp_size=2,
                        pipeline_config=SanaWMPipelineConfig(),
                    )
                ),
                "official",
            )

    def test_refiner_backend_config_can_force_native(self) -> None:
        server_args = self._make_two_stage_server_args(
            tp_size=1,
            pipeline_config=SanaWMPipelineConfig(sana_wm_refiner_backend="native"),
        )

        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(_configured_sana_wm_refiner_backend(server_args), "native")

    def test_refiner_backend_config_overrides_env(self) -> None:
        server_args = self._make_two_stage_server_args(
            tp_size=2,
            pipeline_config=SanaWMPipelineConfig(sana_wm_refiner_backend="native"),
        )

        with patch.dict(
            os.environ,
            {"SGLANG_SANA_WM_REFINER_BACKEND": "official"},
            clear=True,
        ):
            self.assertEqual(
                _configured_sana_wm_refiner_backend(server_args),
                "native",
            )

    def test_initialize_pipeline_validates_native_refiner_tp_before_loading(
        self,
    ) -> None:
        pipeline = object.__new__(SanaWMTwoStagePipeline)
        server_args = self._make_two_stage_server_args(
            tp_size=5,
            pipeline_config=SanaWMPipelineConfig(sana_wm_refiner_backend="native"),
        )

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(SanaWMTwoStagePipeline, "_load_refiner_modules") as loader,
            self.assertRaisesRegex(ValueError, "Valid common tp_size values"),
        ):
            pipeline.initialize_pipeline(server_args)

        loader.assert_not_called()

    def test_initialize_pipeline_allows_cfg_parallel_native_refiner(
        self,
    ) -> None:
        pipeline = object.__new__(SanaWMTwoStagePipeline)
        pipeline.modules = {}
        pipeline.memory_usages = {}
        pipeline.component_residency_strategies = {}
        server_args = self._make_two_stage_server_args(
            tp_size=2,
            enable_cfg_parallel=True,
            cfg_parallel_degree=2,
            pipeline_config=SanaWMPipelineConfig(sana_wm_refiner_backend="native"),
        )

        with (
            patch.dict(os.environ, {}, clear=True),
            patch.object(SanaWMTwoStagePipeline, "_load_refiner_modules") as loader,
            patch.object(
                SanaWMTwoStagePipeline,
                "_configure_two_stage_component_residency",
            ) as residency_configurer,
        ):
            pipeline.initialize_pipeline(server_args)

        loader.assert_called_once_with(server_args)
        residency_configurer.assert_called_once_with(server_args)

    def test_refiner_modules_use_native_loader_for_transformer_2(self) -> None:
        pipeline = object.__new__(SanaWMTwoStagePipeline)
        pipeline.model_path = "/models/sana-wm"
        pipeline.modules = {}
        pipeline.memory_usages = {}
        server_args = self._make_two_stage_server_args(tp_size=2, component_paths={})
        native_transformer = torch.nn.Linear(1, 1)

        with (
            patch.dict(
                os.environ,
                {"SGLANG_SANA_WM_REFINER_BACKEND": "auto"},
            ),
            patch.object(
                SanaWMTwoStagePipeline,
                "_load_native_refiner_transformer",
                return_value=(native_transformer, 1.0),
            ) as native_loader,
            patch.object(
                SanaWMTwoStagePipeline,
                "_load_official_refiner_component",
                side_effect=lambda module_name, _path, _args: (
                    f"official:{module_name}",
                    0.5,
                ),
            ) as official_loader,
        ):
            pipeline._load_refiner_modules(server_args)

        native_loader.assert_called_once_with(
            "/models/sana-wm/refiner/transformer",
            server_args,
        )
        self.assertEqual(official_loader.call_count, 3)
        self.assertIs(pipeline.modules["transformer_2"], native_transformer)
        self.assertEqual(pipeline.modules["connectors"], "official:connectors")
        self.assertEqual(pipeline.memory_usages["transformer_2"], 1.0)


class TestSanaWMPipeline(unittest.TestCase):
    def test_validate_parallelism_allows_tensor_parallelism(self) -> None:
        SanaWMPipeline._validate_parallelism_args(
            SimpleNamespace(
                tp_size=2,
                sp_degree=1,
                pipeline_config=SanaWMPipelineConfig(),
            )
        )

    def test_validate_parallelism_rejects_bad_tensor_parallelism(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_attention_heads"):
            SanaWMPipeline._validate_parallelism_args(
                SimpleNamespace(
                    tp_size=3,
                    sp_degree=1,
                    pipeline_config=SanaWMPipelineConfig(),
                )
            )

    def test_validate_parallelism_rejects_cfg_parallel_degree_above_two(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive and negative prompt"):
            SanaWMPipeline._validate_parallelism_args(
                SimpleNamespace(
                    tp_size=1,
                    sp_degree=1,
                    enable_cfg_parallel=True,
                    cfg_parallel_degree=4,
                    pipeline_config=SanaWMPipelineConfig(),
                )
            )

    def test_two_stage_validate_parallelism_allows_native_refiner_common_tp(
        self,
    ) -> None:
        with patch.dict(os.environ, {}, clear=True):
            SanaWMTwoStagePipeline._validate_parallelism_args(
                SimpleNamespace(
                    tp_size=4,
                    sp_degree=1,
                    enable_cfg_parallel=False,
                    pipeline_config=SanaWMPipelineConfig(
                        sana_wm_refiner_backend="native"
                    ),
                )
            )

    def test_two_stage_validate_parallelism_allows_cfg_parallel_native_refiner(
        self,
    ) -> None:
        with patch.dict(os.environ, {}, clear=True):
            SanaWMTwoStagePipeline._validate_parallelism_args(
                SimpleNamespace(
                    tp_size=2,
                    sp_degree=1,
                    enable_cfg_parallel=True,
                    cfg_parallel_degree=2,
                    pipeline_config=SanaWMPipelineConfig(
                        sana_wm_refiner_backend="native"
                    ),
                )
            )

    def test_two_stage_validate_parallelism_allows_cfg_parallel_official_refiner(
        self,
    ) -> None:
        with patch.dict(os.environ, {}, clear=True):
            SanaWMTwoStagePipeline._validate_parallelism_args(
                SimpleNamespace(
                    tp_size=5,
                    sp_degree=1,
                    enable_cfg_parallel=True,
                    cfg_parallel_degree=2,
                    pipeline_config=SanaWMPipelineConfig(
                        sana_wm_refiner_backend="official"
                    ),
                )
            )

    def test_two_stage_validate_parallelism_rejects_native_refiner_bad_tp(
        self,
    ) -> None:
        for tp_size in (5, 10):
            with self.subTest(tp_size=tp_size), patch.dict(
                os.environ, {}, clear=True
            ):
                with self.assertRaisesRegex(
                    ValueError,
                    r"stage-1 num_attention_heads \(20\).*refiner "
                    r"num_attention_heads \(32\).*Valid common tp_size values: "
                    r"\[1, 2, 4\]",
                ):
                    SanaWMTwoStagePipeline._validate_parallelism_args(
                        SimpleNamespace(
                            tp_size=tp_size,
                            sp_degree=1,
                            enable_cfg_parallel=False,
                            pipeline_config=SanaWMPipelineConfig(),
                        )
                    )

    def test_two_stage_validate_parallelism_allows_official_refiner_tp(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            SanaWMTwoStagePipeline._validate_parallelism_args(
                SimpleNamespace(
                    tp_size=5,
                    sp_degree=1,
                    enable_cfg_parallel=False,
                    pipeline_config=SanaWMPipelineConfig(
                        sana_wm_refiner_backend="official"
                    ),
                )
            )

    def test_two_stage_validate_parallelism_allows_skipped_refiner_tp(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            SanaWMTwoStagePipeline._validate_parallelism_args(
                SimpleNamespace(
                    tp_size=5,
                    sp_degree=1,
                    enable_cfg_parallel=False,
                    pipeline_config=SanaWMPipelineConfig(
                        sana_wm_refiner_backend="native",
                        sana_wm_skip_refiner=True,
                    ),
                )
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

    def test_action_string_rejects_unknown_keys_and_bad_duration(self) -> None:
        with self.assertRaisesRegex(ValueError, "unknown keys"):
            parse_sana_wm_action_string("x-1")
        with self.assertRaisesRegex(ValueError, "non-positive"):
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

    def test_before_denoising_declares_transformer_static_conditioning_use(
        self,
    ) -> None:
        pipeline_config = SanaWMPipelineConfig()
        stage = SanaWMBeforeDenoisingStage(
            vae=object(),
            transformer=object(),
            scheduler=None,
            pipeline_config=pipeline_config,
        )

        uses = stage.component_uses(
            SimpleNamespace(pipeline_config=pipeline_config),
            stage_name="sana_wm_before_denoising",
        )

        self.assertEqual([use.component_name for use in uses], ["vae", "transformer"])
        transformer_use = uses[1]
        self.assertEqual(transformer_use.phase, "transformer")
        self.assertTrue(transformer_use.memory_intensive)
        self.assertTrue(transformer_use.preferred_ready_after_request)

    def test_static_conditioning_precompute_expands_cfg_batch(self) -> None:
        class FakeTransformer:
            def __init__(self):
                self.camera_shape = None
                self.chunk_shape = None

            def prepare_sana_wm_static_conditioning(
                self,
                *,
                camera_conditions,
                chunk_plucker,
                latent_shape,
            ):
                self.camera_shape = tuple(camera_conditions.shape)
                self.chunk_shape = tuple(chunk_plucker.shape)
                self.latent_shape = tuple(latent_shape)
                return {
                    "precomputed_prope_fns": ("prepared",),
                    "precomputed_plucker_emb": torch.ones(4, 60, 8),
                }

        transformer = FakeTransformer()
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=transformer,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )

        out = stage._prepare_transformer_static_conditioning(
            camera_conditions=torch.zeros(2, 3, 20),
            chunk_plucker=torch.zeros(2, 48, 3, 4, 5),
            latent_shape=(2, 128, 3, 4, 5),
            do_cfg=True,
            cfg_parallel=False,
        )

        self.assertEqual(transformer.camera_shape, (4, 3, 20))
        self.assertEqual(transformer.chunk_shape, (4, 48, 3, 4, 5))
        self.assertEqual(transformer.latent_shape, (2, 128, 3, 4, 5))
        self.assertIn("precomputed_prope_fns", out)
        self.assertIn("precomputed_plucker_emb", out)

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
        self.assertTrue(
            torch.equal(out[1, :, :1], torch.full((128, 1, 22, 40), 2.0))
        )

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
            condition_inputs={"intrinsics": [50.0, 50.0, 50.0, 50.0]},
            extra={
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

    def test_explicit_camera_request_detects_camera_condition_inputs(self) -> None:
        batch = SimpleNamespace(condition_inputs={"camera_to_world": "/tmp/cam.npy"})

        self.assertTrue(SanaWMBeforeDenoisingStage._has_explicit_camera_request(batch))

    def test_explicit_camera_request_detects_action_condition_inputs(self) -> None:
        batch = SimpleNamespace(condition_inputs={"action": "w-8"})

        self.assertTrue(SanaWMBeforeDenoisingStage._has_explicit_camera_request(batch))

    def test_action_conditioning_uses_condition_inputs(self) -> None:
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )
        batch = SimpleNamespace(
            condition_inputs={
                "action": "w-8",
                "intrinsics": [50.0, 50.0, 32.0, 32.0],
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
        self.assertAlmostEqual(float(camera_conditions[0, 1, 11]), 0.4, places=5)

    def test_camera_conditioning_accepts_unbatched_chunk_plucker(self) -> None:
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )
        batch = SimpleNamespace(
            condition_inputs={"chunk_plucker": torch.zeros(48, 3, 12, 20)},
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

    def test_camera_conditioning_accepts_prebuilt_raymap_condition_inputs(self) -> None:
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
            condition_inputs={"camera_conditions": camera_conditions},
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
                    condition_inputs={
                        "camera_conditions": torch.zeros(1, 1, 3, 20)
                    },
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
                    condition_inputs={"camera_conditions": torch.zeros(3, 3, 20)},
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
                    condition_inputs={"camera_conditions": torch.zeros(1, 3, 20)},
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
                    condition_inputs={
                        "chunk_plucker": torch.zeros(3, 48, 3, 12, 20)
                    },
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
                    condition_inputs={"chunk_plucker": torch.zeros(48, 2, 12, 20)},
                    height=384,
                    width=640,
                ),
                batch_size=1,
                num_frames=17,
                latent_shape=(1, 128, 3, 12, 20),
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

    def test_action_and_camera_to_world_are_mutually_exclusive(self) -> None:
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )
        batch = SimpleNamespace(
            condition_inputs={
                "action": "w-8",
                "camera_to_world": torch.eye(4).unsqueeze(0),
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
        batch = SimpleNamespace(condition_inputs={"action": ["w-8", "w-16"]})

        self.assertEqual(
            SanaWMBeforeDenoisingStage._action_num_frames_for_request(batch),
            17,
        )

    def test_intrinsics_3x3_sequence_can_exceed_requested_frame_count(self) -> None:
        intrinsics = torch.eye(3).unsqueeze(0).repeat(321, 1, 1)

        coerced = coerce_sana_wm_intrinsics_vec4(
            intrinsics,
            batch_size=1,
            num_frames=49,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )

        self.assertEqual(coerced.shape, (1, 49, 4))
        self.assertTrue(
            torch.equal(coerced[0, 0], torch.tensor([1.0, 1.0, 0.0, 0.0]))
        )

    def test_latent_frame_camera_conditions_samples_stride_indices(self) -> None:
        original = torch.arange(1 * 17 * 20, dtype=torch.float32).reshape(1, 17, 20)

        sampled = latent_frame_sana_wm_camera_conditions(
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

    def test_tensor_cache_key_accepts_inference_tensors(self) -> None:
        with torch.inference_mode():
            tensor = torch.ones(2, 3)
            key = _tensor_cache_key(tensor)

        self.assertEqual(key[0], (2, 3))
        self.assertEqual(key[-1], 0)

    def test_camera_qkv_projection_fuses_params_for_inference(self) -> None:
        attn = BidirectionalGDNUCPESinglePathLiteLA(
            in_dim=8,
            heads=2,
            head_dim=4,
        )
        attn.eval()
        x = torch.randn(2, 3, 8)

        with torch.no_grad():
            q_ref = attn.q_proj_cam(x)
            k_ref = attn.k_proj_cam(x)
            v_ref = attn.v_proj_cam(x)

            self.assertTrue(attn.fuse_cam_qkv_projection())
            fused_weight = attn._fused_cam_qkv_weight
            q, k, v = attn._cam_qkv(x)

            self.assertIs(attn._fused_cam_qkv_weight, fused_weight)
            self.assertFalse(attn._fused_cam_qkv_weight.requires_grad)
            self.assertFalse(attn._fused_cam_qkv_bias.requires_grad)
            torch.testing.assert_close(q, q_ref)
            torch.testing.assert_close(k, k_ref)
            torch.testing.assert_close(v, v_ref)

    def test_cross_attention_kv_request_cache_reuses_static_condition(self) -> None:
        attn = MultiHeadCrossAttention(d_model=8, num_heads=2, qk_norm=True)
        attn.request_cache_name = "cross_attn_kv_0"
        attn.eval()
        x = torch.randn(1, 3, 8)
        cond = torch.randn(1, 4, 8)
        batch = Req(prompt="drive forward")

        with patch.dict(
            os.environ,
            {
                "SGLANG_SANA_WM_REQUEST_RUNTIME_CACHE": "1",
                "SGLANG_SANA_WM_CROSS_ATTN_KV_CACHE_MAX_BYTES": "-1",
            },
        ), torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=batch
        ), patch.object(
            attn.kv_linear,
            "forward",
            wraps=attn.kv_linear.forward,
        ) as kv_forward:
            out = attn(x, cond)
            cached_out = attn(x, cond)

            self.assertEqual(kv_forward.call_count, 1)
            request_cache = batch.extra[SANA_WM_REQUEST_RUNTIME_CACHE_KEY]
            self.assertIn("cross_attn_kv_0", request_cache)
            self.assertIn("entry", request_cache["cross_attn_kv_0"])
            torch.testing.assert_close(cached_out, out)

            cond.add_(0.25)
            attn(x, cond)

            self.assertEqual(kv_forward.call_count, 2)

    def test_cross_attention_kv_request_cache_respects_max_bytes(self) -> None:
        attn = MultiHeadCrossAttention(d_model=8, num_heads=2, qk_norm=True)
        attn.request_cache_name = "cross_attn_kv_0"
        attn.eval()
        x = torch.randn(1, 3, 8)
        cond = torch.randn(1, 4, 8)
        batch = Req(prompt="drive forward")

        with patch.dict(
            os.environ,
            {
                "SGLANG_SANA_WM_REQUEST_RUNTIME_CACHE": "1",
                "SGLANG_SANA_WM_CROSS_ATTN_KV_CACHE_MAX_BYTES": "1",
            },
        ), torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=batch
        ), patch.object(
            attn.kv_linear,
            "forward",
            wraps=attn.kv_linear.forward,
        ) as kv_forward:
            out = attn(x, cond)
            uncached_out = attn(x, cond)

            self.assertEqual(kv_forward.call_count, 2)
            request_cache = batch.extra[SANA_WM_REQUEST_RUNTIME_CACHE_KEY]
            self.assertEqual(request_cache["cross_attn_kv_0"], {})
            torch.testing.assert_close(uncached_out, out)

    def test_y_projection_request_cache_reuses_static_encoder_states(self) -> None:
        model = SanaWMTransformer3DModel.__new__(SanaWMTransformer3DModel)
        torch.nn.Module.__init__(model)
        model.y_embedder = torch.nn.Linear(4, 8)
        model.y_norm = True
        model.attention_y_norm = _RMSNorm(8)
        model.blocks = torch.nn.ModuleList()
        encoder_hidden_states = torch.randn(1, 5, 4)
        batch = Req(prompt="drive forward")

        with patch.dict(
            os.environ,
            {"SGLANG_SANA_WM_REQUEST_RUNTIME_CACHE": "1"},
        ), torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=batch
        ), patch.object(
            model.y_embedder,
            "forward",
            wraps=model.y_embedder.forward,
        ) as y_forward:
            y = model._get_projected_y(encoder_hidden_states, batch_size=2)
            cached_y = model._get_projected_y(encoder_hidden_states, batch_size=2)

            self.assertEqual(y_forward.call_count, 1)
            self.assertEqual(tuple(cached_y.shape), (2, 5, 8))
            request_cache = batch.extra[SANA_WM_REQUEST_RUNTIME_CACHE_KEY]
            self.assertIn("y_projection", request_cache)
            self.assertIn("entry", request_cache["y_projection"])
            torch.testing.assert_close(cached_y, y)

    def test_request_runtime_cache_can_be_disabled(self) -> None:
        model = SanaWMTransformer3DModel.__new__(SanaWMTransformer3DModel)
        torch.nn.Module.__init__(model)
        model.y_embedder = torch.nn.Linear(4, 8)
        model.y_norm = False
        model.blocks = torch.nn.ModuleList()
        encoder_hidden_states = torch.randn(1, 5, 4)
        batch = Req(prompt="drive forward")

        with patch.dict(
            os.environ,
            {"SGLANG_SANA_WM_REQUEST_RUNTIME_CACHE": "0"},
        ), torch.no_grad(), set_forward_context(
            current_timestep=0, attn_metadata=None, forward_batch=batch
        ), patch.object(
            model.y_embedder,
            "forward",
            wraps=model.y_embedder.forward,
        ) as y_forward:
            model._get_projected_y(encoder_hidden_states, batch_size=1)
            model._get_projected_y(encoder_hidden_states, batch_size=1)

            self.assertEqual(y_forward.call_count, 2)
            self.assertNotIn(SANA_WM_REQUEST_RUNTIME_CACHE_KEY, batch.extra)

    def test_cfg_uses_true_cfg_scale_when_present(self) -> None:
        batch = Req(
            prompt="drive forward",
            negative_prompt="",
            guidance_scale=1.0,
            true_cfg_scale=4.5,
        )

        self.assertTrue(batch.do_classifier_free_guidance)
        self.assertTrue(_sana_wm_should_do_cfg(batch))
        self.assertEqual(_sana_wm_effective_guidance_scale(batch), 4.5)

    def test_cfg_accepts_preencoded_negative_prompt_embeds(self) -> None:
        batch = Req(
            prompt="drive forward",
            negative_prompt=None,
            negative_prompt_embeds=[torch.zeros(1, 4, 2304)],
            guidance_scale=4.5,
        )

        self.assertTrue(batch.do_classifier_free_guidance)
        self.assertTrue(_sana_wm_should_do_cfg(batch))

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


class TestSanaWMTextEncodingStage(_GlobalStageArgsMixin, unittest.TestCase):
    def test_official_prompt_window_keeps_bos_and_tail(self) -> None:
        tensor = torch.arange(5).reshape(1, 5, 1)

        selected = SanaWMTextEncodingStage._select_official_prompt_window(
            tensor, max_length=3
        )

        self.assertTrue(torch.equal(selected.squeeze(-1), torch.tensor([[0, 3, 4]])))

    def test_forward_accepts_batched_negative_prompts(self) -> None:
        tokenizer = SimpleNamespace(encode=lambda text: [0, 1])
        stage = SanaWMTextEncodingStage(text_encoders=[object()], tokenizers=[tokenizer])
        batch = Req(
            prompt=["turn left", "turn right"],
            negative_prompt=["blur", "distortion"],
            guidance_scale=4.5,
        )
        server_args = SimpleNamespace(pipeline_config=SanaWMPipelineConfig())
        pos_mask = torch.ones(2, 4, dtype=torch.long)
        neg_mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.long)
        pos_outputs = (
            [torch.ones(2, 4, 2304)],
            [pos_mask],
            [],
            [pos_mask.bool()],
            [[4, 4]],
        )
        neg_outputs = (
            [torch.zeros(2, 4, 2304)],
            [neg_mask],
            [],
            [neg_mask.bool()],
            [[2, 1]],
        )

        with patch.object(stage, "encode_text", side_effect=[pos_outputs, neg_outputs]):
            out = stage.forward(batch, server_args)

        self.assertEqual(out.negative_prompt_embeds[0].shape[0], 2)
        self.assertTrue(torch.equal(out.negative_attention_mask[0], neg_mask))
        self.assertEqual(out.negative_prompt_seq_lens[0], [2, 1])

    def test_forward_reuses_preencoded_negative_prompt_embeds(self) -> None:
        tokenizer = SimpleNamespace(encode=lambda text: [0, 1])
        stage = SanaWMTextEncodingStage(text_encoders=[object()], tokenizers=[tokenizer])
        neg_embeds = torch.zeros(1, 4, 2304)
        batch = Req(
            prompt="drive forward",
            negative_prompt=None,
            negative_prompt_embeds=[neg_embeds],
            guidance_scale=4.5,
        )
        server_args = SimpleNamespace(pipeline_config=SanaWMPipelineConfig())
        pos_mask = torch.ones(1, 4, dtype=torch.long)
        pos_outputs = (
            [torch.ones(1, 4, 2304)],
            [pos_mask],
            [],
            [pos_mask.bool()],
            [[4]],
        )

        with patch.object(stage, "encode_text", return_value=pos_outputs) as encode_text:
            out = stage.forward(batch, server_args)

        self.assertEqual(encode_text.call_count, 1)
        self.assertIs(out.negative_prompt_embeds[0], neg_embeds)

    def test_forward_receives_text_outputs_from_tp_rank0(self) -> None:
        tokenizer = SimpleNamespace(encode=lambda text: [0, 1])
        stage = SanaWMTextEncodingStage(text_encoders=[object()], tokenizers=[tokenizer])
        batch = Req(prompt="drive forward", guidance_scale=1.0)
        server_args = SimpleNamespace(pipeline_config=SanaWMPipelineConfig())
        prompt_embeds = torch.ones(1, 4, 2304)
        prompt_mask = torch.ones(1, 4, dtype=torch.long)
        payload = {
            "embeds_count": 1,
            "embeds": {"0": prompt_embeds},
            "masks_count": 1,
            "masks": {"0": prompt_mask},
            "pooled_count": 0,
            "pooled": {},
            "embeds_masks_count": 1,
            "embeds_masks": {"0": prompt_mask.bool()},
            "seq_lens": [[4]],
        }

        with (
            patch(
                f"{_SANA_WM_STAGE_MODULE}._sana_wm_stage_tp_world_size",
                return_value=2,
            ),
            patch(
                f"{_SANA_WM_STAGE_MODULE}._sana_wm_is_tp_rank0",
                return_value=False,
            ),
            patch(
                f"{_SANA_WM_STAGE_MODULE}._sana_wm_broadcast_tensor_dict_from_tp_rank0",
                return_value=payload,
            ),
            patch.object(
                stage,
                "encode_text",
                side_effect=AssertionError("nonzero TP rank encoded text"),
            ),
        ):
            out = stage.forward(batch, server_args)

        self.assertTrue(torch.equal(out.prompt_embeds[0], prompt_embeds))
        self.assertTrue(torch.equal(out.prompt_attention_mask[0], prompt_mask))
        self.assertEqual(out.prompt_seq_lens[0], [4])


class TestSanaWMDenoisingStage(unittest.TestCase):
    def test_reuses_shared_denoising_forward(self) -> None:
        self.assertIs(SanaWMDenoisingStage.forward, DenoisingStage.forward)

    def test_stage_attention_backend_head_size_uses_sana_wm_padding(self) -> None:
        stage = object.__new__(SanaWMDenoisingStage)
        stage.server_args = SimpleNamespace(pipeline_config=SanaWMPipelineConfig())

        self.assertEqual(stage._infer_transformer_attention_head_size(), 128)

        pipeline_config = SanaWMPipelineConfig()
        pipeline_config.update_pipeline_config(
            {"dit_config": {"pad_attention_head_dim_to_flash": False}}
        )
        stage.server_args = SimpleNamespace(pipeline_config=pipeline_config)

        self.assertEqual(stage._infer_transformer_attention_head_size(), 112)

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
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.cfg_model_parallel_all_reduce",
            side_effect=lambda partial: partial + (1.0 - guidance_scale) * neg,
        ):
            combined_from_pos_rank = SanaWMDenoisingStage._combine_cfg_parallel_noise(
                pos, guidance_scale, cfg_rank=0
            )

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.cfg_model_parallel_all_reduce",
            side_effect=lambda partial: partial + guidance_scale * pos,
        ):
            combined_from_neg_rank = SanaWMDenoisingStage._combine_cfg_parallel_noise(
                neg, guidance_scale, cfg_rank=1
            )

        self.assertTrue(torch.allclose(combined_from_pos_rank, serial))
        self.assertTrue(torch.allclose(combined_from_neg_rank, serial))

    def test_cfg_parallel_idle_rank_contributes_zero(self) -> None:
        local_noise = torch.tensor([7.0])

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm.cfg_model_parallel_all_reduce",
            side_effect=lambda partial: partial,
        ) as all_reduce:
            combined = SanaWMDenoisingStage._combine_cfg_parallel_noise(
                local_noise, guidance_scale=4.5, cfg_rank=2
            )

        expected = torch.zeros_like(local_noise)
        self.assertTrue(torch.equal(combined, expected))
        self.assertTrue(torch.equal(all_reduce.call_args.args[0], expected))


class TestSanaWMOptionalAttentionPadding(unittest.TestCase):
    def test_attention_head_padding_default_can_be_disabled(self) -> None:
        self.assertTrue(SanaWMConfig().arch_config.pad_attention_head_dim_to_flash)
        self.assertFalse(
            SanaWMConfig(
                pad_attention_head_dim_to_flash=False
            ).arch_config.pad_attention_head_dim_to_flash
        )
        self.assertEqual(_sana_wm_padded_attention_head_size(112), 128)

        created_attn = []

        class RecordingLocalAttention(torch.nn.Module):
            def __init__(
                self,
                num_heads,
                head_size,
                num_kv_heads=None,
                softmax_scale=None,
                causal=False,
                **extra_impl_args,
            ) -> None:
                super().__init__()
                self.num_heads = num_heads
                self.head_size = head_size
                self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
                self.softmax_scale = softmax_scale
                self.backend = "recording"
                self.dtype = torch.float32
                created_attn.append(self)

            def forward(self, q, k, v, attn_mask=None):
                return torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=self.softmax_scale,
                ).transpose(1, 2)

        with patch(
            "sglang.multimodal_gen.runtime.models.dits.sana_wm.LocalAttention",
            RecordingLocalAttention,
        ):
            attn = _make_sana_wm_local_attention(num_heads=2, head_size=112)

        self.assertIsInstance(attn, RecordingLocalAttention)
        self.assertEqual(created_attn[0].head_size, 112)

    def test_opt_in_padding_preserves_unpadded_attention_result(self) -> None:
        torch.manual_seed(0)
        created_attn = []

        class RecordingLocalAttention(torch.nn.Module):
            def __init__(
                self,
                num_heads,
                head_size,
                num_kv_heads=None,
                softmax_scale=None,
                causal=False,
                **extra_impl_args,
            ) -> None:
                super().__init__()
                self.num_heads = num_heads
                self.head_size = head_size
                self.num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
                self.softmax_scale = softmax_scale
                self.backend = "recording"
                self.dtype = torch.float32
                self.last_q_shape = None
                created_attn.append(self)

            def forward(self, q, k, v, attn_mask=None):
                self.last_q_shape = tuple(q.shape)
                return torch.nn.functional.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    attn_mask=attn_mask,
                    dropout_p=0.0,
                    is_causal=False,
                    scale=self.softmax_scale,
                ).transpose(1, 2)

        with patch(
            "sglang.multimodal_gen.runtime.models.dits.sana_wm.LocalAttention",
            RecordingLocalAttention,
        ):
            attn = _make_sana_wm_local_attention(
                num_heads=2,
                head_size=112,
                pad_head_dim_to_flash=True,
            )
            q = torch.randn(1, 5, 2, 112)
            k = torch.randn(1, 5, 2, 112)
            v = torch.randn(1, 5, 2, 112)
            out = attn(q, k, v)

        self.assertIsInstance(attn, _SanaWMPaddedLocalAttention)
        self.assertEqual(attn.padded_head_size, 128)
        self.assertEqual(created_attn[0].head_size, 128)
        self.assertEqual(created_attn[0].last_q_shape, (1, 5, 2, 128))
        self.assertEqual(tuple(out.shape), tuple(q.shape))

        expected = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            dropout_p=0.0,
            is_causal=False,
            scale=112**-0.5,
        ).transpose(1, 2)
        torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-5)


class TestSanaWMGLUMBConvTemp(unittest.TestCase):
    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_channels_last_spatial_path_matches_nchw_path(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda")
        dtype = torch.float16
        batch_size, channels = 1, 8
        hw = (2, 4, 4)
        tokens = hw[0] * hw[1] * hw[2]
        module = GLUMBConvTemp(channels, hidden_features=16).to(
            device=device, dtype=dtype
        )
        module.eval()
        x = torch.randn(batch_size, tokens, channels, device=device, dtype=dtype)

        def nchw_forward() -> torch.Tensor:
            t, h, w = hw
            x_sp = (
                x.reshape(batch_size * t, h, w, channels)
                .permute(0, 3, 1, 2)
                .contiguous()
            )
            x_sp = module._apply_spatial_autochunked(x_sp)
            x_t = (
                x_sp.view(batch_size, t, channels, h * w)
                .permute(0, 2, 1, 3)
                .contiguous()
            )
            x_out = x_t + module.t_conv(x_t)
            return x_out.permute(0, 2, 3, 1).reshape(batch_size, tokens, channels)

        with torch.no_grad():
            expected = nchw_forward()
            actual = module(x, hw)

        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


class TestSanaWMNativeDiTChunking(unittest.TestCase):
    def test_softmax_chunking_is_disabled_by_default_for_upstream_parity(self) -> None:
        arch = SanaWMConfig().arch_config

        self.assertFalse(arch.use_chunked_softmax_attention)

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

        uses = stage.component_uses(SimpleNamespace(), stage_name="sana_wm_refiner")
        names = [use.component_name for use in uses]

        self.assertEqual(names, ["text_encoder_2", "connectors", "transformer_2"])
        transformer_use = uses[-1]
        self.assertTrue(transformer_use.memory_intensive)
        self.assertFalse(transformer_use.allow_prefetch)

    def test_refiner_parallelism_avoids_duplicate_tp_execution(self) -> None:
        stage = SanaWMLTX2RefinerStage(
            transformer=torch.nn.Identity(),
            connectors=torch.nn.Identity(),
            text_encoder=torch.nn.Identity(),
            tokenizer=SimpleNamespace(pad_token="<pad>", eos_token="<eos>"),
            dtype=torch.bfloat16,
        )

        stage.server_args = SimpleNamespace(enable_cfg_parallel=False, tp_size=1)
        self.assertEqual(stage.parallelism_type, StageParallelismType.REPLICATED)

        stage.server_args = SimpleNamespace(enable_cfg_parallel=False, tp_size=2)
        self.assertEqual(
            stage.parallelism_type,
            StageParallelismType.MAIN_RANK_ONLY_AND_SEND_TO_OTHERS,
        )

        stage.server_args = SimpleNamespace(enable_cfg_parallel=True, tp_size=2)
        self.assertEqual(stage.parallelism_type, StageParallelismType.MAIN_RANK_ONLY)

    def test_native_refiner_parallelism_runs_on_all_tp_ranks(self) -> None:
        class SanaWMLTX2VideoRefiner(torch.nn.Identity):
            pass

        stage = SanaWMLTX2RefinerStage(
            transformer=SanaWMLTX2VideoRefiner(),
            connectors=torch.nn.Identity(),
            text_encoder=torch.nn.Identity(),
            tokenizer=SimpleNamespace(pad_token="<pad>", eos_token="<eos>"),
            dtype=torch.bfloat16,
        )

        stage.server_args = SimpleNamespace(enable_cfg_parallel=False, tp_size=2)
        self.assertEqual(stage.parallelism_type, StageParallelismType.REPLICATED)

    def test_native_refiner_component_uses_keep_transformer_on_nonzero_tp_rank(
        self,
    ) -> None:
        class SanaWMLTX2VideoRefiner(torch.nn.Identity):
            pass

        stage = SanaWMLTX2RefinerStage(
            transformer=SanaWMLTX2VideoRefiner(),
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
                f"{_SANA_WM_REFINER_STAGE_MODULE}.torch.distributed.get_rank",
                return_value=1,
            ),
            patch(
                f"{_SANA_WM_REFINER_STAGE_MODULE}._sana_wm_is_tp_rank0",
                return_value=False,
            ),
        ):
            uses = stage.component_uses(
                SimpleNamespace(enable_cfg_parallel=False, tp_size=2),
                stage_name="sana_wm_refiner",
            )

        self.assertEqual(
            [use.component_name for use in uses],
            ["transformer_2"],
        )

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

    def test_refiner_component_uses_skip_non_main_tp_rank(self) -> None:
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
                f"{_SANA_WM_REFINER_STAGE_MODULE}.torch.distributed.get_rank",
                return_value=1,
            ),
        ):
            uses = stage.component_uses(
                SimpleNamespace(enable_cfg_parallel=False, tp_size=2),
                stage_name="sana_wm_refiner",
            )

        self.assertEqual(uses, [])

    def test_refiner_component_uses_keep_main_tp_rank(self) -> None:
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
                f"{_SANA_WM_REFINER_STAGE_MODULE}.torch.distributed.get_rank",
                return_value=0,
            ),
        ):
            uses = stage.component_uses(
                SimpleNamespace(enable_cfg_parallel=False, tp_size=2),
                stage_name="sana_wm_refiner",
            )

        self.assertEqual(
            [use.component_name for use in uses],
            ["text_encoder_2", "connectors", "transformer_2"],
        )

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
        batch = SimpleNamespace(extra={"skip_refiner": True})
        self.assertTrue(sana_wm_skip_refiner_enabled(batch))

    def test_skip_refiner_flag_accepts_pipeline_config(self) -> None:
        self.assertTrue(
            sana_wm_skip_refiner_enabled(
                pipeline_config=SanaWMPipelineConfig(sana_wm_skip_refiner=True)
            )
        )

    def test_skip_refiner_flag_parses_string_values(self) -> None:
        self.assertTrue(
            sana_wm_skip_refiner_enabled(
                SimpleNamespace(extra={"skip_refiner": "true"})
            )
        )
        self.assertFalse(
            sana_wm_skip_refiner_enabled(
                SimpleNamespace(extra={"skip_refiner": "false"})
            )
        )

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

    def test_native_refiner_prompt_encoding_receives_tp_broadcast(self) -> None:
        class SanaWMLTX2VideoRefiner(torch.nn.Identity):
            pass

        class RaisingTokenizer:
            pad_token = "<pad>"
            eos_token = "<eos>"

            def __call__(self, *args, **kwargs):
                raise AssertionError("nonzero TP rank tokenized refiner prompt")

        class RaisingModule:
            def __call__(self, *args, **kwargs):
                raise AssertionError("nonzero TP rank ran refiner prompt module")

        prompt_embeds = torch.ones(1, 4, 8)
        attention_mask = torch.tensor([[1, 1, 0, 0]])
        stage = SanaWMLTX2RefinerStage(
            transformer=SanaWMLTX2VideoRefiner(),
            connectors=RaisingModule(),
            text_encoder=RaisingModule(),
            tokenizer=RaisingTokenizer(),
            dtype=torch.float32,
            text_max_sequence_length=4,
        )
        stage.server_args = SimpleNamespace(enable_cfg_parallel=False, tp_size=2)

        with (
            patch(
                f"{_SANA_WM_REFINER_STAGE_MODULE}._sana_wm_stage_tp_world_size",
                return_value=2,
            ),
            patch(
                f"{_SANA_WM_REFINER_STAGE_MODULE}._sana_wm_is_tp_rank0",
                return_value=False,
            ),
            patch(
                f"{_SANA_WM_REFINER_STAGE_MODULE}."
                "_sana_wm_broadcast_tensor_dict_from_tp_rank0",
                return_value={
                    "prompt_embeds": prompt_embeds,
                    "attention_mask": attention_mask,
                },
            ) as broadcast,
        ):
            out_embeds, out_mask = stage._encode_prompt(
                "drive forward", torch.device("cpu")
            )

        self.assertIsNone(broadcast.call_args.args[0])
        self.assertTrue(torch.equal(out_embeds, prompt_embeds))
        self.assertTrue(torch.equal(out_mask, attention_mask))

    def test_refiner_forward_refines_batch_in_single_call(self) -> None:
        stage = SanaWMLTX2RefinerStage(
            transformer=torch.nn.Identity(),
            connectors=torch.nn.Identity(),
            text_encoder=torch.nn.Identity(),
            tokenizer=SimpleNamespace(pad_token="<pad>", eos_token="<eos>"),
            dtype=torch.float32,
        )
        original_latents = torch.zeros(2, 128, 3, 1, 1)
        refined_latents = torch.ones_like(original_latents)
        batch = Req(
            prompt=["left", "right"],
            latents=original_latents,
            seeds=[11, 23],
            extra={},
        )

        with (
            patch.object(
                stage, "_refine_batch", return_value=refined_latents
            ) as refine_batch,
            patch.object(stage, "_refine_one", side_effect=AssertionError),
        ):
            out = stage.forward(batch, SimpleNamespace())

        refine_batch.assert_called_once()
        call_args = refine_batch.call_args
        self.assertIs(call_args.args[0], original_latents)
        self.assertEqual(call_args.args[1], ["left", "right"])
        self.assertEqual(call_args.kwargs["seeds"], [11, 23])
        self.assertTrue(torch.equal(out.latents, refined_latents))
        self.assertTrue(out.extra["sana_wm_refiner_applied"])

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
