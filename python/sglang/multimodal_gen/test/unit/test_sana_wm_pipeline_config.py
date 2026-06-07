import argparse
import math
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
    SanaWMRefinerArchConfig,
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
    _apply_ray_projmat,
    _downscale_to_reference_rms,
    _gdn_chunk_scan_forward,
    _gdn_scan_bidirectional,
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
    _build_ucpe_apply_fns,
    _make_sana_wm_local_attention,
    _record_sana_wm_triton_import_error,
    _sana_wm_add_repeated_batch,
    _sana_wm_chunk_index_from_chunk_size,
    _sana_wm_triton_import_error,
    _invert_SE3,
    _sana_wm_normalize_chunk_index,
    _sana_wm_padded_attention_head_size,
    _sana_wm_sequence_shard_enabled,
    _sana_wm_sp_rank,
    _sana_wm_sp_world_size,
    _sana_wm_triton_context,
    _module_inference_cache_key,
    _single_path_delta_scan_bidirectional,
    _single_path_delta_chunk_scan_forward,
    _single_path_delta_scan_forward,
    _tensor_cache_key,
    _sana_wm_materialize_repeated_raymats,
    compute_chunk_plucker,
)
from sglang.multimodal_gen.runtime.models.dits.sana_wm_refiner_transformer import (
    SanaWMLTX2VideoRefiner,
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
from sglang.multimodal_gen.runtime.disaggregation.roles import RoleType
from sglang.multimodal_gen.runtime.managers.forward_context import set_forward_context
from sglang.multimodal_gen.runtime.pipelines_core.schedule_batch import Req
from sglang.multimodal_gen.runtime.pipelines_core.stages.base import (
    StageParallelismType,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.denoising import DenoisingStage
from sglang.multimodal_gen.runtime.pipelines_core.stages.input_validation import (
    InputValidationStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.sana_wm import (
    SanaWMBeforeDenoisingStage,
    SanaWMDecodingStage,
    SanaWMDenoisingStage,
    SanaWMTextEncodingStage,
    _align_sana_wm_cfg_text_conditions,
    _sana_wm_effective_guidance_scale,
    _sana_wm_should_do_cfg,
    configure_sana_wm_ltx2_vae_for_long_video,
)
from sglang.multimodal_gen.runtime.utils.sana_wm_camera import (
    SANA_WM_DEFAULT_HORIZONTAL_FOV_ENV,
    coerce_sana_wm_intrinsics_vec4,
    default_sana_wm_intrinsics_vec4,
    latent_frame_sana_wm_camera_conditions,
    parse_sana_wm_action_string,
    sana_wm_action_to_camera_to_world,
    scale_sana_wm_intrinsics_to_latent,
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
_SANA_WM_DIT_MODULE = "sglang.multimodal_gen.runtime.models.dits.sana_wm"
_SANA_WM_STAGE_MODULE = (
    "sglang.multimodal_gen.runtime.pipelines_core.stages."
    "model_specific_stages.sana_wm"
)


class _GlobalStageArgsMixin:
    def setUp(self) -> None:
        super().setUp()
        from sglang.multimodal_gen.runtime.models.dits.sana_wm import (
            _sana_wm_cross_attn_kv_cache_max_bytes,
        )
        _sana_wm_cross_attn_kv_cache_max_bytes.cache_clear()
        self._prev_global_server_args = server_args_module._global_server_args
        set_global_server_args(
            SimpleNamespace(
                comfyui_mode=False,
                enable_cfg_parallel=False,
                enable_torch_compile=False,
                attention_backend=None,
                pipeline_config=SanaWMPipelineConfig(),
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

    def test_adjust_num_frames_rounds_down_to_temporal_stride(self) -> None:
        self.assertEqual(self.config.adjust_num_frames(50), 49)
        self.assertEqual(self.config.adjust_num_frames(53), 49)
        self.assertEqual(self.config.adjust_num_frames(55), 49)
        self.assertEqual(self.config.adjust_num_frames(57), 57)
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
            pad_attention_head_dim_to_flash=True,
            use_triton_kernels=True,
            allow_triton_fallback=True,
        )

        self.assertTrue(dit_config.arch_config.pad_attention_head_dim_to_flash)
        self.assertTrue(dit_config.arch_config.use_triton_kernels)
        self.assertTrue(dit_config.arch_config.allow_triton_fallback)

    def test_dit_triton_kernels_default_enabled(self) -> None:
        self.assertTrue(SanaWMConfig().arch_config.use_triton_kernels)
        self.assertFalse(SanaWMConfig().arch_config.allow_triton_fallback)
        self.assertFalse(
            SanaWMConfig(use_triton_kernels=False).arch_config.use_triton_kernels
        )
        self.assertTrue(
            SanaWMConfig(allow_triton_fallback=True).arch_config.allow_triton_fallback
        )

    def test_pipeline_config_dict_can_enable_dit_attention_flags(self) -> None:
        self.config.update_pipeline_config(
            {
                "dit_config": {
                    "pad_attention_head_dim_to_flash": True,
                    "use_triton_kernels": True,
                    "allow_triton_fallback": True,
                }
            }
        )
        arch = self.config.dit_config.arch_config

        self.assertTrue(arch.pad_attention_head_dim_to_flash)
        self.assertTrue(arch.use_triton_kernels)
        self.assertTrue(arch.allow_triton_fallback)

    def test_sana_wm_triton_fallback_defaults_to_fail_fast(self) -> None:
        attn = BidirectionalGDNUCPESinglePathLiteLA(
            in_dim=16,
            heads=4,
            head_dim=4,
            conv_kernel_size=0,
            softmax_main=False,
            use_triton_kernels=True,
        )

        with self.assertRaisesRegex(RuntimeError, "refuses to silently fall back"):
            attn._handle_triton_fallback(
                kernel_name="Triton test kernel",
                reason="is unavailable",
                warning_attr="_triton_main_gdn_warning_logged",
            )

    def test_sana_wm_triton_strict_message_includes_context(self) -> None:
        attn = BidirectionalGDNUCPESinglePathLiteLA(
            in_dim=16,
            heads=4,
            head_dim=4,
            conv_kernel_size=0,
            softmax_main=False,
            use_triton_kernels=True,
        )

        with self.assertRaisesRegex(RuntimeError, "shape=\\(1, 2, 3, 4, 4\\)") as cm:
            attn._handle_triton_fallback(
                kernel_name="Triton test kernel",
                reason="does not support the current tensors",
                warning_attr="_triton_main_gdn_warning_logged",
                details=_sana_wm_triton_context(
                    path="unit",
                    qkv=torch.zeros(1, 2, 3, 4, 4),
                    head_dim=4,
                ),
            )
        self.assertIn("SANA-WM Triton strict mode", str(cm.exception))
        self.assertIn("path='unit'", str(cm.exception))

    def test_sana_wm_triton_fallback_can_be_explicitly_allowed(self) -> None:
        attn = BidirectionalGDNUCPESinglePathLiteLA(
            in_dim=16,
            heads=4,
            head_dim=4,
            conv_kernel_size=0,
            softmax_main=False,
            use_triton_kernels=True,
            allow_triton_fallback=True,
        )

        attn._handle_triton_fallback(
            kernel_name="Triton test kernel",
            reason="failed",
            warning_attr="_triton_main_gdn_warning_logged",
            error=RuntimeError("boom"),
            disable_fast_path=True,
        )

        self.assertFalse(attn.use_triton_kernels)
        self.assertTrue(attn._triton_main_gdn_warning_logged)

    def test_sana_wm_triton_import_errors_are_keyed_by_kernel_family(self) -> None:
        qkv_error = RuntimeError("qkv missing")
        scan_error = RuntimeError("scan missing")

        with patch(
            "sglang.multimodal_gen.runtime.models.dits.sana_wm."
            "_SANA_WM_TRITON_KERNELS_IMPORT_ERRORS",
            {},
        ):
            _record_sana_wm_triton_import_error("qkv_preprocess", qkv_error)
            _record_sana_wm_triton_import_error("cam_scan", scan_error)

            self.assertIs(_sana_wm_triton_import_error("qkv_preprocess"), qkv_error)
            self.assertIs(_sana_wm_triton_import_error("cam_scan"), scan_error)

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

    def test_repeated_raymat_materialization_reuses_request_cache(self) -> None:
        raymats = torch.eye(4).view(1, 1, 4, 4).repeat(1, 3, 1, 1)
        raymats_t = raymats.transpose(-1, -2).contiguous()
        raymats_inv = raymats.clone()
        cache = {}

        with torch.no_grad():
            first = _sana_wm_materialize_repeated_raymats(
                raymats,
                raymats_t,
                raymats_inv,
                2,
                name="test",
                cache=cache,
            )
            second = _sana_wm_materialize_repeated_raymats(
                raymats,
                raymats_t,
                raymats_inv,
                2,
                name="test",
                cache=cache,
            )

        self.assertEqual(first[0].shape, (2, 3, 4, 4))
        self.assertIs(first[0], second[0])
        self.assertIs(first[1], second[1])
        self.assertIs(first[2], second[2])
        self.assertEqual(len(cache), 1)

    def test_auto_refiner_backend_uses_official_for_cfg_parallel_tp(self) -> None:
        server_args = SimpleNamespace(
            tp_size=2,
            enable_cfg_parallel=True,
            pipeline_config=SanaWMPipelineConfig(),
        )

        self.assertEqual(_configured_sana_wm_refiner_backend(server_args), "official")

    def test_invalid_refiner_backend_server_arg_fails_fast(self) -> None:
        server_args = SimpleNamespace(
            tp_size=1,
            enable_cfg_parallel=False,
            pipeline_config=SanaWMPipelineConfig(),
            sana_wm_refiner_backend="bad-backend",
        )

        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            ValueError, "server_args.sana_wm_refiner_backend"
        ):
            _configured_sana_wm_refiner_backend(server_args)

    def test_invalid_refiner_backend_env_fails_fast(self) -> None:
        server_args = SimpleNamespace(
            tp_size=1,
            enable_cfg_parallel=False,
            pipeline_config=SanaWMPipelineConfig(),
        )

        with patch.dict(
            os.environ,
            {"SGLANG_SANA_WM_REFINER_BACKEND": "bad-backend"},
            clear=True,
        ), self.assertRaisesRegex(ValueError, "SGLANG_SANA_WM_REFINER_BACKEND"):
            _configured_sana_wm_refiner_backend(server_args)

    def test_native_refiner_rejects_cfg_parallel_tp(self) -> None:
        server_args = SimpleNamespace(
            tp_size=2,
            enable_cfg_parallel=True,
            cfg_parallel_degree=2,
            sp_degree=1,
            pipeline_config=SanaWMPipelineConfig(
                sana_wm_refiner_backend="native",
            ),
        )

        with self.assertRaisesRegex(ValueError, "native refiner.*enable_cfg_parallel"):
            SanaWMTwoStagePipeline._validate_parallelism_args(server_args)

    def test_dit_tp_config_requires_heads_divisible_by_tp_size(self) -> None:
        arch = SanaWMConfig().arch_config
        SanaWMTransformer3DModel._validate_tp_config(arch, 2)
        with self.assertRaisesRegex(ValueError, "num_attention_heads"):
            SanaWMTransformer3DModel._validate_tp_config(arch, 3)

    def test_dit_stage1_tp_keeps_low_compute_convs_replicated(self) -> None:
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

        self.assertIsInstance(patch_embed.proj, torch.nn.Conv3d)
        self.assertFalse(hasattr(patch_embed.proj, "tp_size"))
        self.assertEqual(tuple(patch_embed.proj.weight.shape), (16, 3, 1, 1, 1))

        self.assertIsInstance(glumb.inverted_conv.conv, torch.nn.Conv2d)
        self.assertFalse(hasattr(glumb.inverted_conv.conv, "tp_size"))
        self.assertEqual(
            tuple(glumb.inverted_conv.conv.weight.shape), (48, 16, 1, 1)
        )
        self.assertIsInstance(glumb.depth_conv.conv, torch.nn.Conv2d)
        self.assertEqual(tuple(glumb.depth_conv.conv.weight.shape), (48, 1, 3, 3))
        self.assertIsInstance(glumb.point_conv.conv, torch.nn.Conv2d)
        self.assertEqual(tuple(glumb.point_conv.conv.weight.shape), (16, 24, 1, 1))
        self.assertIsInstance(glumb.t_conv, torch.nn.Conv2d)
        self.assertFalse(hasattr(glumb.t_conv, "tp_size"))
        self.assertEqual(tuple(glumb.t_conv.weight.shape), (16, 16, 3, 1))

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


class TestSanaWMTritonCudaSmoke(unittest.TestCase):
    @staticmethod
    def _require_triton() -> None:
        try:
            import triton  # noqa: F401
        except Exception as exc:
            raise unittest.SkipTest("Triton is required for Sana-WM CUDA smoke") from exc

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_main_gdn_default_strict_cuda_smoke_matches_torch_path(self) -> None:
        self._require_triton()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        device = torch.device("cuda")
        hidden_size = 64
        heads = 2
        head_dim = 32
        hw = (3, 1, 5)
        tokens = math.prod(hw)

        fast = BidirectionalGDNUCPESinglePathLiteLA(
            in_dim=hidden_size,
            heads=heads,
            head_dim=head_dim,
            conv_kernel_size=0,
            softmax_main=False,
            use_triton_kernels=True,
            allow_triton_fallback=False,
        ).to(device=device)
        reference = BidirectionalGDNUCPESinglePathLiteLA(
            in_dim=hidden_size,
            heads=heads,
            head_dim=head_dim,
            conv_kernel_size=0,
            softmax_main=False,
            use_triton_kernels=False,
            allow_triton_fallback=False,
        ).to(device=device)
        reference.load_state_dict(fast.state_dict())
        fast.eval()
        reference.eval()
        x = torch.randn(1, tokens, hidden_size, device=device) * 0.02

        with torch.no_grad():
            actual = fast(x, HW=hw, rotary_emb=None, prope_fns=None)
            expected = reference(x, HW=hw, rotary_emb=None, prope_fns=None)
        torch.cuda.synchronize()

        self.assertEqual(tuple(actual.shape), (1, tokens, hidden_size))
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertFalse(fast._triton_main_gdn_warning_logged)
        self.assertTrue(fast.use_triton_kernels)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_main_gdn_d112_long_t_strict_cuda_smoke_matches_torch_path(self) -> None:
        self._require_triton()
        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        device = torch.device("cuda")
        hidden_size = 224
        heads = 2
        head_dim = 112
        hw = (22, 1, 4)
        tokens = math.prod(hw)

        fast = BidirectionalGDNUCPESinglePathLiteLA(
            in_dim=hidden_size,
            heads=heads,
            head_dim=head_dim,
            conv_kernel_size=0,
            softmax_main=False,
            use_triton_kernels=True,
            allow_triton_fallback=False,
        ).to(device=device)
        reference = BidirectionalGDNUCPESinglePathLiteLA(
            in_dim=hidden_size,
            heads=heads,
            head_dim=head_dim,
            conv_kernel_size=0,
            softmax_main=False,
            use_triton_kernels=False,
            allow_triton_fallback=False,
        ).to(device=device)
        reference.load_state_dict(fast.state_dict())
        fast.eval()
        reference.eval()
        x = torch.randn(1, tokens, hidden_size, device=device) * 0.02

        with torch.no_grad():
            actual = fast(x, HW=hw, rotary_emb=None, prope_fns=None)
            expected = reference(x, HW=hw, rotary_emb=None, prope_fns=None)
        torch.cuda.synchronize()

        self.assertEqual(tuple(actual.shape), (1, tokens, hidden_size))
        self.assertTrue(torch.isfinite(actual).all().item())
        self.assertFalse(fast._triton_main_gdn_warning_logged)
        self.assertTrue(fast.use_triton_kernels)
        torch.testing.assert_close(actual, expected, atol=6e-2, rtol=6e-2)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_camera_scan_d112_long_t_cuda_smoke_matches_torch_chunk(self) -> None:
        self._require_triton()
        from sglang.jit_kernel.diffusion.sana_wm.camera_scan import (
            sana_wm_cam_scan_bidi_chunkwise,
        )

        torch.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        device = torch.device("cuda")
        B, H, D, T, S = 1, 2, 112, 22, 4
        N = T * S
        q = (torch.randn(B, H, D, N, device=device) * 0.02).contiguous()
        k = (torch.randn(B, H, D, N, device=device) * 0.02).contiguous()
        v = (torch.randn(B, H, D, N, device=device) * 0.02).contiguous()
        beta = torch.sigmoid(torch.randn(B, H, T, S, device=device)).contiguous()
        decay = torch.sigmoid(torch.randn(B, H, T, device=device)).contiguous()

        with torch.no_grad():
            actual = sana_wm_cam_scan_bidi_chunkwise(q, k, v, beta, decay)
            expected_chunk = _single_path_delta_scan_bidirectional(
                q,
                k,
                v,
                beta,
                decay,
                HW=(T, 1, S),
                chunk_size=21,
            )
            expected_full = _single_path_delta_scan_bidirectional(
                q,
                k,
                v,
                beta,
                decay,
                HW=(T, 1, S),
                chunk_size=None,
            )
        torch.cuda.synchronize()

        self.assertEqual(tuple(actual.shape), (B, H, D, N))
        self.assertTrue(torch.isfinite(actual).all().item())
        torch.testing.assert_close(expected_chunk, expected_full, atol=2e-4, rtol=2e-4)
        torch.testing.assert_close(actual, expected_chunk, atol=6e-2, rtol=6e-2)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_main_gdn_strict_import_missing_reports_context(self) -> None:
        torch.manual_seed(0)
        device = torch.device("cuda")
        B, H, D, T, S = (1, 2, 32, 3, 5)
        N = T * S
        attn = BidirectionalGDNUCPESinglePathLiteLA(
            in_dim=H * D,
            heads=H,
            head_dim=D,
            conv_kernel_size=0,
            softmax_main=False,
            use_triton_kernels=True,
            allow_triton_fallback=False,
        ).to(device=device)
        attn.eval()
        qkv = torch.randn(B, N, 3, H, D, device=device).contiguous()
        beta = torch.rand(B, H, T, S, device=device)
        decay = torch.rand(B, H, T, device=device)

        with (
            patch(
                f"{_SANA_WM_DIT_MODULE}._get_sana_wm_triton_main_gdn",
                return_value=None,
            ),
            patch(
                f"{_SANA_WM_DIT_MODULE}._sana_wm_triton_import_error",
                return_value=ImportError("missing main_gdn"),
            ),
            torch.no_grad(),
        ):
            with self.assertRaisesRegex(
                RuntimeError, "Triton main GDN kernel is unavailable"
            ) as cm:
                attn._try_triton_main_gdn(
                    qkv,
                    beta,
                    decay,
                    HW=(T, 1, S),
                    rotary_emb=None,
                    k_scale=(D**-0.5) * (S**-0.5),
                )

        message = str(cm.exception)
        self.assertIn("SANA-WM Triton strict mode", message)
        self.assertIn("qkv: shape=(1, 15, 3, 2, 32)", message)
        self.assertIn("beta: shape=(1, 2, 3, 5)", message)
        self.assertIn("decay: shape=(1, 2, 3)", message)


class TestSanaWMTritonIdentityCache(unittest.TestCase):
    @staticmethod
    def _load_kernel_module():
        try:
            from sglang.jit_kernel.diffusion.sana_wm import fused_gdn_chunkwise
        except ModuleNotFoundError as exc:
            if exc.name == "triton":
                raise unittest.SkipTest("triton is not installed") from exc
            raise
        return fused_gdn_chunkwise

    def test_cam_identity_cache_evicts_lru_entry(self) -> None:
        module = self._load_kernel_module()
        env = {
            module._CAM_IDENTITY_CACHE_MAX_ENTRIES_ENV: "2",
            module._CAM_IDENTITY_CACHE_MAX_BYTES_ENV: "-1",
        }
        device = torch.device("cpu")

        with patch.dict(os.environ, env):
            module._clear_cam_identity_cache()
            try:
                first = module._cam_identity_tables(B=1, N=2, H=1, D=2, device=device)
                module._cam_identity_tables(B=1, N=3, H=1, D=2, device=device)
                self.assertIs(
                    module._cam_identity_tables(B=1, N=2, H=1, D=2, device=device),
                    first,
                )
                module._cam_identity_tables(B=1, N=4, H=1, D=2, device=device)

                keys = list(module._CAM_IDENTITY_CACHE.keys())
                self.assertEqual(len(keys), 2)
                self.assertIn(("cpu", None, 1, 2, 2, 2), keys)
                self.assertNotIn(("cpu", None, 1, 3, 2, 2), keys)
                self.assertIn(("cpu", None, 1, 4, 2, 2), keys)
            finally:
                module._clear_cam_identity_cache()

    def test_cam_identity_cache_can_be_disabled(self) -> None:
        module = self._load_kernel_module()
        env = {
            module._CAM_IDENTITY_CACHE_MAX_ENTRIES_ENV: "8",
            module._CAM_IDENTITY_CACHE_MAX_BYTES_ENV: "0",
        }
        device = torch.device("cpu")

        with patch.dict(os.environ, env):
            module._clear_cam_identity_cache()
            try:
                first = module._cam_identity_tables(B=1, N=2, H=1, D=2, device=device)
                second = module._cam_identity_tables(B=1, N=2, H=1, D=2, device=device)

                self.assertEqual(len(module._CAM_IDENTITY_CACHE), 0)
                self.assertIsNot(first, second)
                self.assertIsNot(first[0], second[0])
            finally:
                module._clear_cam_identity_cache()

    def test_phase_b_dtile_tuning_cache_is_device_specific(self) -> None:
        module = self._load_kernel_module()
        module._PHASE_B_DTILE_ARCH_CACHE.clear()

        with patch.dict(os.environ, {}, clear=True), patch.object(
            module.torch.cuda, "is_available", return_value=True
        ), patch.object(
            module.torch.cuda, "current_device", side_effect=[0, 1]
        ), patch.object(
            module.torch.cuda,
            "get_device_capability",
            side_effect=lambda dev: (8, 0) if dev == 0 else (12, 1),
        ):
            self.assertEqual(module._pick_phase_b_d_splits(112, 0), (4, 8, 2, False))
            self.assertEqual(
                module._pick_phase_b_d_splits(112, 0), (1, None, None, None)
            )

        module._PHASE_B_DTILE_ARCH_CACHE.clear()


class TestSanaWMSamplingParams(unittest.TestCase):
    @staticmethod
    def _server_args():
        return SimpleNamespace(
            pipeline_config=SanaWMPipelineConfig(),
            output_path=None,
            num_gpus=1,
            comfyui_mode=False,
        )

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

    def test_adjust_includes_prepacked_camera_condition_inputs_when_set(self) -> None:
        camera_conditions = torch.zeros(49, 20)
        chunk_plucker = torch.zeros(48, 6, 22, 40)
        params = SanaWMSamplingParams(
            camera_conditions=camera_conditions,
            chunk_plucker=chunk_plucker,
        )

        params._adjust(self._server_args())

        self.assertIs(params.condition_inputs["camera_conditions"], camera_conditions)
        self.assertIs(params.condition_inputs["chunk_plucker"], chunk_plucker)

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

    def test_adjust_normalizes_action_condition_inputs_motion_params(self) -> None:
        params = SanaWMSamplingParams(
            condition_inputs={
                "action": "w-8",
                "translation_speed": "0.06",
            }
        )

        params._adjust(self._server_args())

        self.assertEqual(params.condition_inputs["action"], "w-8")
        self.assertEqual(params.condition_inputs["translation_speed"], 0.06)
        self.assertEqual(params.condition_inputs["rotation_speed_deg"], 1.2)
        self.assertEqual(params.condition_inputs["pitch_limit_deg"], 85.0)
        self.assertEqual(params.translation_speed, 0.06)

    def test_build_request_extra_includes_refiner_controls_when_set(self) -> None:
        params = SanaWMSamplingParams(
            skip_refiner=True,
            refiner_prompt="refine this world state",
        )

        extra = params.build_request_extra()

        self.assertTrue(extra["skip_refiner"])
        self.assertEqual(extra["refiner_prompt"], "refine this world state")

    def test_adjust_rejects_action_with_direct_camera(self) -> None:
        cam = torch.eye(4).unsqueeze(0).expand(49, 4, 4)
        params = SanaWMSamplingParams(action="w-8", camera_to_world=cam)

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            params._adjust(self._server_args())

    def test_adjust_rejects_action_with_prepacked_camera_inputs(self) -> None:
        params = SanaWMSamplingParams(
            action="w-8",
            camera_conditions=torch.zeros(49, 20),
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            params._adjust(self._server_args())

        params = SanaWMSamplingParams(
            action="w-8",
            chunk_plucker=torch.zeros(48, 6, 22, 40),
        )

        with self.assertRaisesRegex(ValueError, "mutually exclusive"):
            params._adjust(self._server_args())

    def test_adjust_rejects_prepacked_camera_conditions_with_intrinsics(self) -> None:
        params = SanaWMSamplingParams(
            camera_conditions=torch.zeros(49, 20),
            intrinsics=torch.eye(3),
        )

        with self.assertRaisesRegex(ValueError, "already contains"):
            params._adjust(self._server_args())

    def test_adjust_rejects_invalid_action_motion_params(self) -> None:
        params = SanaWMSamplingParams(action="w-8", translation_speed=0)

        with self.assertRaisesRegex(ValueError, "translation_speed"):
            params._adjust(self._server_args())

    def test_adjust_rejects_unknown_condition_input_keys(self) -> None:
        params = SanaWMSamplingParams(
            condition_inputs={
                "camrea_to_world": torch.eye(4).unsqueeze(0),
            }
        )

        with self.assertRaisesRegex(ValueError, "unknown keys"):
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

    def test_video_api_sampling_params_exposes_sana_wm_controls(self) -> None:
        from sglang.multimodal_gen.runtime.entrypoints.openai.protocol import (
            VideoGenerationsRequest,
        )
        from sglang.multimodal_gen.runtime.entrypoints.openai.video_api import (
            _build_video_sampling_params,
        )

        prev_global = server_args_module._global_server_args
        server_args = SimpleNamespace(
            backend="sglang",
            comfyui_mode=False,
            model_id=None,
            model_path=DEFAULT_SANA_WM_MODEL_NAME_FOR_TEST,
            num_gpus=1,
            output_path=None,
            pipeline_class_name=None,
            pipeline_config=SanaWMPipelineConfig(),
        )
        set_global_server_args(server_args)
        try:
            self.assertNotIn("action", VideoGenerationsRequest.model_fields)
            self.assertNotIn("skip_refiner", VideoGenerationsRequest.model_fields)
            request = VideoGenerationsRequest(
                prompt="drive forward",
                input_reference="/tmp/sana-wm-first-frame.png",
                action="w-8,jw-8",
                condition_inputs={"intrinsics": [50.0, 50.0, 32.0, 32.0]},
                skip_refiner=True,
                refiner_prompt="cinematic refiner prompt",
                true_cfg_scale=4.5,
            )

            params = _build_video_sampling_params("api-rid", request)
        finally:
            server_args_module._global_server_args = prev_global

        self.assertIsInstance(params, SanaWMSamplingParams)
        self.assertEqual(params.condition_inputs["action"], "w-8,jw-8")
        self.assertEqual(
            params.condition_inputs["intrinsics"], [50.0, 50.0, 32.0, 32.0]
        )
        self.assertEqual(params.true_cfg_scale, 4.5)
        extra = params.build_request_extra()
        self.assertTrue(extra["skip_refiner"])
        self.assertEqual(extra["refiner_prompt"], "cinematic refiner prompt")


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

    def test_realtime_registry_does_not_register_sana_wm(self) -> None:
        from sglang.multimodal_gen.runtime.entrypoints.openai.realtime.registry import (
            get_realtime_model_adapter,
        )

        with self.assertRaisesRegex(ValueError, "Realtime video is not supported"):
            get_realtime_model_adapter(
                SimpleNamespace(pipeline_config=SanaWMPipelineConfig())
            )

    def test_overlay_resolver_matches_hf_cache_snapshot_paths(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            snapshot_dir = f"{tmp_dir}/hub/models--Lightricks--LTX-2.3/snapshots/abc123"
            os.makedirs(snapshot_dir)
            target = resolve_model_overlay_target(snapshot_dir)
        self.assertIsNotNone(target)
        source_model_id, _ = target
        self.assertEqual(source_model_id, "Lightricks/LTX-2.3")


class TestSanaWMTwoStagePipeline(_GlobalStageArgsMixin, unittest.TestCase):
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

    def test_invalid_two_stage_residency_server_arg_fails_fast(self) -> None:
        server_args = self._make_two_stage_server_args(
            sana_wm_two_stage_residency="bad-residency"
        )

        with patch.dict(os.environ, {}, clear=True), self.assertRaisesRegex(
            ValueError, "server_args.sana_wm_two_stage_residency"
        ):
            _configured_sana_wm_two_stage_residency(server_args)

    def test_invalid_two_stage_residency_env_fails_fast(self) -> None:
        server_args = self._make_two_stage_server_args()

        with patch.dict(
            os.environ,
            {"SGLANG_SANA_WM_TWO_STAGE_RESIDENCY": "bad-residency"},
            clear=True,
        ), self.assertRaisesRegex(ValueError, "SGLANG_SANA_WM_TWO_STAGE_RESIDENCY"):
            _configured_sana_wm_two_stage_residency(server_args)

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

    def test_refiner_backend_auto_prefers_native_except_cfg_parallel_tp(self) -> None:
        with patch.dict(
            os.environ,
            {"SGLANG_SANA_WM_REFINER_BACKEND": "auto"},
        ):
            self.assertEqual(
                _configured_sana_wm_refiner_backend(
                    self._make_two_stage_server_args(tp_size=1)
                ),
                "native",
            )
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

    def test_initialize_pipeline_rejects_cfg_parallel_native_refiner(
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
            self.assertRaisesRegex(ValueError, "does not support enable_cfg_parallel"),
        ):
            pipeline.initialize_pipeline(server_args)

        loader.assert_not_called()
        residency_configurer.assert_not_called()

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


class TestSanaWMPipeline(_GlobalStageArgsMixin, unittest.TestCase):
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

    def test_validate_parallelism_rejects_cfg_parallel_degree_below_two(self) -> None:
        with self.assertRaisesRegex(ValueError, "exactly two branches"):
            SanaWMPipeline._validate_parallelism_args(
                SimpleNamespace(
                    tp_size=1,
                    sp_degree=1,
                    enable_cfg_parallel=True,
                    cfg_parallel_degree=1,
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

    def test_two_stage_validate_parallelism_rejects_cfg_parallel_native_refiner(
        self,
    ) -> None:
        with (
            patch.dict(os.environ, {}, clear=True),
            self.assertRaisesRegex(ValueError, "does not support enable_cfg_parallel"),
        ):
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

    def test_create_pipeline_stages_uses_standard_sana_wm_stages(self) -> None:
        pipeline = object.__new__(SanaWMPipeline)
        pipeline.modules = {
            "text_encoder": object(),
            "tokenizer": object(),
            "vae": object(),
            "transformer": torch.nn.Module(),
            "scheduler": object(),
        }
        pipeline._stages = []
        pipeline._stage_name_mapping = {}
        pipeline._disagg_role = RoleType.MONOLITHIC

        pipeline.create_pipeline_stages(
            SimpleNamespace(
                tp_size=1,
                sp_degree=1,
                enable_cfg_parallel=False,
                pipeline_config=SanaWMPipelineConfig(),
            )
        )

        self.assertIsInstance(pipeline._stages[0], InputValidationStage)
        self.assertIsInstance(pipeline._stages[1], SanaWMTextEncodingStage)
        self.assertEqual(
            pipeline._stage_name_mapping["prompt_encoding_stage"],
            pipeline._stages[1],
        )


class TestSanaWMBeforeDenoisingStage(_GlobalStageArgsMixin, unittest.TestCase):
    @staticmethod
    def _make_verify_batch(**overrides) -> Req:
        kwargs = {
            "prompt": "drive forward",
            "negative_prompt": "",
            "guidance_scale": 4.5,
            "height": 704,
            "width": 1280,
            "num_frames": 49,
            "num_inference_steps": 4,
            "condition_image": torch.zeros(3, 8, 8),
            "prompt_embeds": [torch.zeros(1, 4, 8)],
            "negative_prompt_embeds": [torch.zeros(1, 4, 8)],
            "condition_inputs": {},
        }
        kwargs.update(overrides)
        return Req(**kwargs)

    def test_verify_input_accepts_prepared_sana_wm_request(self) -> None:
        pipeline_config = SanaWMPipelineConfig()
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=pipeline_config,
        )

        result = stage.verify_input(
            self._make_verify_batch(),
            SimpleNamespace(pipeline_config=pipeline_config),
        )

        self.assertTrue(result.is_valid(), result.get_failure_summary())

    def test_verify_input_requires_condition_image(self) -> None:
        pipeline_config = SanaWMPipelineConfig()
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=pipeline_config,
        )

        result = stage.verify_input(
            self._make_verify_batch(condition_image=None),
            SimpleNamespace(pipeline_config=pipeline_config),
        )

        self.assertIn("condition_image", result.get_failed_fields())

    def test_verify_input_rejects_stride_misaligned_size(self) -> None:
        pipeline_config = SanaWMPipelineConfig()
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=pipeline_config,
        )

        result = stage.verify_input(
            self._make_verify_batch(height=705),
            SimpleNamespace(pipeline_config=pipeline_config),
        )

        self.assertIn("height", result.get_failed_fields())

    def test_verify_input_rejects_action_camera_conflict(self) -> None:
        pipeline_config = SanaWMPipelineConfig()
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=pipeline_config,
        )

        result = stage.verify_input(
            self._make_verify_batch(
                condition_inputs={
                    "action": "w-8",
                    "camera_to_world": torch.eye(4).unsqueeze(0),
                }
            ),
            SimpleNamespace(pipeline_config=pipeline_config),
        )

        self.assertIn("condition_inputs", result.get_failed_fields())

    def test_verify_input_rejects_invalid_action_motion_params(self) -> None:
        pipeline_config = SanaWMPipelineConfig()
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=pipeline_config,
        )

        result = stage.verify_input(
            self._make_verify_batch(
                condition_inputs={
                    "action": "w-8",
                    "translation_speed": -0.1,
                }
            ),
            SimpleNamespace(pipeline_config=pipeline_config),
        )

        self.assertIn("condition_inputs", result.get_failed_fields())

    def test_verify_input_rejects_unknown_condition_input_keys(self) -> None:
        pipeline_config = SanaWMPipelineConfig()
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=pipeline_config,
        )

        result = stage.verify_input(
            self._make_verify_batch(
                condition_inputs={
                    "camrea_to_world": torch.eye(4).unsqueeze(0),
                }
            ),
            SimpleNamespace(pipeline_config=pipeline_config),
        )

        self.assertIn("condition_inputs", result.get_failed_fields())

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

    def test_ucpe_apply_fns_expose_reusable_projection_matrices(self) -> None:
        raymats = torch.eye(4).reshape(1, 1, 4, 4).repeat(1, 3, 1, 1)

        prope_fns = _build_ucpe_apply_fns(
            head_dim=112,
            raymats=raymats,
            rotary_emb=None,
        )

        self.assertEqual(len(prope_fns), 7)
        P, P_T, P_inv = prope_fns[3:6]
        self.assertTrue(P.is_contiguous())
        self.assertTrue(P_T.is_contiguous())
        self.assertTrue(P_inv.is_contiguous())
        self.assertTrue(torch.equal(P, raymats))
        self.assertTrue(torch.equal(P_T, raymats.transpose(-1, -2).contiguous()))
        self.assertTrue(torch.equal(P_inv, _invert_SE3(raymats)))

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

    def test_static_conditioning_precompute_keeps_single_cfg_batch(self) -> None:
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
                    "precomputed_plucker_emb": torch.ones(2, 60, 8),
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

        self.assertEqual(transformer.camera_shape, (2, 3, 20))
        self.assertEqual(transformer.chunk_shape, (2, 48, 3, 4, 5))
        self.assertEqual(transformer.latent_shape, (2, 128, 3, 4, 5))
        self.assertIn("precomputed_prope_fns", out)
        self.assertIn("precomputed_plucker_emb", out)

    def test_ucpe_ray_projection_repeats_static_cfg_batch(self) -> None:
        feats = torch.randn(4, 1, 2, 4)
        matrix = (
            torch.eye(4)
            .view(1, 1, 4, 4)
            .expand(2, 2, 4, 4)
            .contiguous()
        )

        out = _apply_ray_projmat(feats, matrix)

        torch.testing.assert_close(out, feats)

    def test_repeated_batch_add_repeats_static_cfg_batch_without_precat(self) -> None:
        x = torch.zeros(4, 3, 4)
        addend = torch.randn(2, 3, 4)

        with torch.no_grad():
            out = _sana_wm_add_repeated_batch(
                x,
                addend,
                name="test addend",
            )

        torch.testing.assert_close(out[:2], addend)
        torch.testing.assert_close(out[2:], addend)

    def test_plucker_post_attn_repeats_static_cfg_batch_without_precat(self) -> None:
        block = SanaWMBlock(
            hidden_size=4,
            num_heads=1,
            head_dim=4,
            mlp_ratio=1.0,
            t_kernel_size=3,
            qk_norm=False,
            cross_norm=False,
            conv_kernel_size=0,
            k_conv_only=True,
            softmax_main=True,
            use_chunk_plucker_post_attn=True,
            use_triton_kernels=False,
        )
        assert block.plucker_proj is not None
        with torch.no_grad():
            block.plucker_proj.weight.copy_(torch.eye(4))
            block.plucker_proj.bias.zero_()

        x = torch.zeros(4, 3, 4)
        plucker_emb = torch.randn(2, 3, 4)

        with torch.no_grad():
            out = block._add_plucker_post_attn(x, plucker_emb)

        torch.testing.assert_close(out[:2], plucker_emb)
        torch.testing.assert_close(out[2:], plucker_emb)

    def test_static_conditioning_precompute_without_hook_falls_back(self) -> None:
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=object(),
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )

        out = stage._prepare_transformer_static_conditioning(
            camera_conditions=torch.zeros(1, 3, 20),
            chunk_plucker=torch.zeros(1, 48, 3, 4, 5),
            latent_shape=(1, 128, 3, 4, 5),
            do_cfg=False,
            cfg_parallel=False,
        )

        self.assertEqual(out, {})

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

    def test_condition_image_tensor_preprocess_matches_pil_lanczos_path(self) -> None:
        import PIL.Image

        src_h, src_w = 23, 37
        pixels = torch.arange(src_h * src_w * 3, dtype=torch.uint8).reshape(
            src_h, src_w, 3
        )
        pil_image = PIL.Image.fromarray(pixels.numpy())
        tensor_image = pixels.permute(2, 0, 1)

        pil_out, pil_info = SanaWMBeforeDenoisingStage._preprocess_condition_image(
            pil_image,
            target_h=17,
            target_w=31,
        )
        tensor_out, tensor_info = SanaWMBeforeDenoisingStage._preprocess_condition_image(
            tensor_image,
            target_h=17,
            target_w=31,
        )

        self.assertEqual(tensor_info, pil_info)
        torch.testing.assert_close(tensor_out, pil_out, rtol=0.0, atol=0.0)

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

        encoded_first_frames = torch.cat(
            [
                torch.ones(1, 128, 1, 22, 40),
                torch.full((1, 128, 1, 22, 40), 2.0),
            ],
            dim=0,
        )
        with patch.object(
            stage,
            "_vae_encode_image",
            return_value=encoded_first_frames,
        ) as encode_mock:
            out = stage._splice_first_frame(
                latents,
                [torch.zeros(3, 100, 100), torch.zeros(3, 200, 100)],
                dtype=torch.float32,
                device=torch.device("cpu"),
                batch=batch,
            )

        encode_mock.assert_called_once()
        encoded_pixels = encode_mock.call_args[0][0]
        self.assertEqual(tuple(encoded_pixels.shape), (2, 3, 704, 1280))
        info = batch.extra["sana_wm_condition_image_preprocess"]
        self.assertEqual(len(info), 2)
        self.assertTrue(torch.equal(out[0, :, :1], torch.ones(128, 1, 22, 40)))
        self.assertTrue(
            torch.equal(out[1, :, :1], torch.full((128, 1, 22, 40), 2.0))
        )

    def test_default_intrinsics_use_horizontal_fov_strategy(self) -> None:
        with patch.dict(os.environ, {SANA_WM_DEFAULT_HORIZONTAL_FOV_ENV: "70.0"}):
            intrinsics = default_sana_wm_intrinsics_vec4(
                batch_size=2,
                num_frames=3,
                pixel_h=384,
                pixel_w=640,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        expected_focal = 640.0 / (2.0 * math.tan(math.radians(70.0) / 2.0))
        self.assertEqual(tuple(intrinsics.shape), (2, 3, 4))
        torch.testing.assert_close(
            intrinsics[0, 0],
            torch.tensor([expected_focal, expected_focal, 320.0, 192.0]),
        )
        torch.testing.assert_close(intrinsics[0], intrinsics[1])

    def test_default_intrinsics_env_override_changes_horizontal_fov(self) -> None:
        with patch.dict(os.environ, {SANA_WM_DEFAULT_HORIZONTAL_FOV_ENV: "60.0"}):
            intrinsics = default_sana_wm_intrinsics_vec4(
                batch_size=1,
                num_frames=1,
                pixel_h=384,
                pixel_w=640,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )

        expected_focal = 640.0 / (2.0 * math.tan(math.radians(60.0) / 2.0))
        torch.testing.assert_close(
            intrinsics[0, 0],
            torch.tensor([expected_focal, expected_focal, 320.0, 192.0]),
        )

    def test_default_intrinsics_rejects_out_of_range_fov(self) -> None:
        with patch.dict(os.environ, {SANA_WM_DEFAULT_HORIZONTAL_FOV_ENV: "10.0"}):
            with self.assertRaisesRegex(ValueError, "horizontal FOV"):
                default_sana_wm_intrinsics_vec4(
                    batch_size=1,
                    num_frames=1,
                    pixel_h=384,
                    pixel_w=640,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )

    def test_default_static_camera_builds_latent_raymap_and_chunk_plucker(self) -> None:
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )
        batch = SimpleNamespace(extra={}, height=384, width=640)

        with patch.dict(os.environ, {SANA_WM_DEFAULT_HORIZONTAL_FOV_ENV: "70.0"}):
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
        expected_pixel_intrinsics = default_sana_wm_intrinsics_vec4(
            batch_size=1,
            num_frames=17,
            pixel_h=384,
            pixel_w=640,
            device=torch.device("cpu"),
            dtype=torch.float32,
            horizontal_fov_deg=70.0,
        )
        expected_latent_intrinsics = scale_sana_wm_intrinsics_to_latent(
            expected_pixel_intrinsics,
            pixel_h=384,
            pixel_w=640,
            latent_h=12,
            latent_w=20,
        )
        torch.testing.assert_close(
            camera_conditions[0, 0, 16:],
            expected_latent_intrinsics[0, 0],
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

    def test_action_length_does_not_cap_requested_num_frames(self) -> None:
        stage = SanaWMBeforeDenoisingStage(
            vae=None,
            transformer=None,
            scheduler=None,
            pipeline_config=SanaWMPipelineConfig(),
        )
        batch = SimpleNamespace(num_frames=49, condition_inputs={"action": "w-8"})

        self.assertEqual(stage._adjust_num_frames_for_request(batch), 49)

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

    def test_gdn_camera_fallback_does_not_apply_post_ucpe_renorm(self) -> None:
        attn = BidirectionalGDNUCPESinglePathLiteLA(
            in_dim=8,
            heads=2,
            head_dim=4,
            conv_kernel_size=0,
            softmax_main=False,
            cam_update_rule="torch_recurrent",
            use_triton_kernels=False,
        )
        attn.eval()
        x = torch.randn(1, 4, 8)
        beta = torch.full((1, 2, 2), 0.1)
        decay = torch.full((1, 2, 2), 0.9)

        def fail_downscale(*args, **kwargs):
            raise AssertionError("GDN camera fallback should match BothTriton prep")

        with patch(
            f"{_SANA_WM_DIT_MODULE}._downscale_to_reference_rms",
            side_effect=fail_downscale,
        ):
            with torch.no_grad():
                out = attn._cam_branch(
                    x,
                    HW=(2, 1, 2),
                    apply_q=lambda tensor: tensor,
                    apply_kv=lambda tensor: tensor,
                    apply_o=lambda tensor: tensor,
                    beta=beta,
                    decay=decay,
                    rotary_emb=None,
                    raymats_flat=None,
                )

        self.assertEqual(tuple(out.shape), (1, 4, 8))
        self.assertFalse(torch.isnan(out).any())

    def test_rmsnorm_scale_factor_initializes_weight(self) -> None:
        norm = _RMSNorm(4, scale_factor=0.01)
        self.assertTrue(torch.allclose(norm.weight, torch.full((4,), 0.01)))

    def test_tensor_cache_key_accepts_inference_tensors(self) -> None:
        with torch.inference_mode():
            tensor = torch.ones(2, 3)
            key = _tensor_cache_key(tensor)

        self.assertEqual(key[0], (2, 3))
        self.assertEqual(key[-1], tensor.data_ptr())

    def test_module_inference_cache_key_does_not_scan_module_state(self) -> None:
        class NoScanModule(torch.nn.Module):
            def named_parameters(self, *args, **kwargs):
                raise AssertionError("module cache key should not scan parameters")

            def named_buffers(self, *args, **kwargs):
                raise AssertionError("module cache key should not scan buffers")

        module = NoScanModule()

        key = _module_inference_cache_key(module)
        cached_key = _module_inference_cache_key(module)

        self.assertIs(cached_key, key)
        self.assertEqual(key[2], id(module))

    def test_zero_output_camera_projection_skips_camera_branch_for_inference(
        self,
    ) -> None:
        attn = BidirectionalGDNUCPESinglePathLiteLA(
            in_dim=8,
            heads=2,
            head_dim=4,
            conv_kernel_size=0,
            softmax_main=False,
            use_triton_kernels=False,
        )
        attn.eval()
        x = torch.randn(1, 4, 8)

        def fail_camera_branch(*args, **kwargs):
            raise AssertionError("zero out_proj_cam should skip camera branch")

        prope_fns = (fail_camera_branch, fail_camera_branch, fail_camera_branch)

        with torch.no_grad():
            actual = attn(x, HW=(2, 1, 2), rotary_emb=None, prope_fns=prope_fns)
            expected = attn(x, HW=(2, 1, 2), rotary_emb=None, prope_fns=None)

        torch.testing.assert_close(actual, expected)

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

            changed_cond = cond + 0.25
            attn(x, changed_cond)

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

    def test_cfg_treats_empty_negative_prompt_as_unconditional_condition(self) -> None:
        batch = SimpleNamespace(
            do_classifier_free_guidance=False,
            negative_prompt="",
            negative_prompt_embeds=None,
            guidance_scale=4.5,
            true_cfg_scale=None,
        )

        self.assertTrue(_sana_wm_should_do_cfg(batch))

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

    def test_prepare_timesteps_uses_request_flow_shift_override(self) -> None:
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
        batch = SimpleNamespace(
            num_inference_steps=3,
            scheduler=None,
            flow_shift=7.25,
        )

        stage._prepare_timesteps(batch, SimpleNamespace(), torch.device("cpu"))

        self.assertEqual(scheduler.shift, 7.25)


class TestSanaWMTextEncodingStage(_GlobalStageArgsMixin, unittest.TestCase):
    def test_verify_input_allows_preencoded_negative_tensor(self) -> None:
        stage = SanaWMTextEncodingStage(text_encoders=[object()], tokenizers=[object()])
        batch = Req(
            prompt="drive forward",
            negative_prompt=None,
            negative_prompt_embeds=torch.zeros(1, 4, 8),
            guidance_scale=4.5,
        )

        result = stage.verify_input(
            batch,
            SimpleNamespace(pipeline_config=SanaWMPipelineConfig()),
        )

        self.assertTrue(result.is_valid(), result.get_failure_summary())

    def test_verify_input_requires_single_sana_wm_text_encoder(self) -> None:
        stage = SanaWMTextEncodingStage(text_encoders=[], tokenizers=[])
        batch = Req(prompt="drive forward", guidance_scale=1.0)

        result = stage.verify_input(
            batch,
            SimpleNamespace(pipeline_config=SanaWMPipelineConfig()),
        )

        self.assertIn("text_encoders", result.get_failed_fields())
        self.assertIn("tokenizers", result.get_failed_fields())

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
                f"{_SANA_WM_STAGE_MODULE}.sana_wm_stage_tp_world_size",
                return_value=2,
            ),
            patch(
                f"{_SANA_WM_STAGE_MODULE}.sana_wm_is_tp_rank0",
                return_value=False,
            ),
            patch(
                f"{_SANA_WM_STAGE_MODULE}.sana_wm_broadcast_tensor_dict_from_tp_rank0",
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
    @staticmethod
    def _make_verify_batch(**overrides) -> Req:
        kwargs = {
            "prompt": "drive forward",
            "guidance_scale": 1.0,
            "latents": torch.zeros(1, 128, 3, 2, 2),
            "timesteps": torch.arange(2),
            "num_inference_steps": 2,
            "prompt_embeds": [torch.zeros(1, 4, 8)],
            "generator": torch.Generator(device="cpu").manual_seed(0),
            "extra": {
                "camera_conditions": torch.zeros(1, 3, 20),
                "chunk_plucker": torch.zeros(1, 48, 3, 2, 2),
            },
        }
        kwargs.update(overrides)
        return Req(**kwargs)

    def test_verify_input_accepts_prepared_denoising_request(self) -> None:
        stage = object.__new__(SanaWMDenoisingStage)

        result = stage.verify_input(
            self._make_verify_batch(),
            SimpleNamespace(pipeline_config=SanaWMPipelineConfig()),
        )

        self.assertTrue(result.is_valid(), result.get_failure_summary())

    def test_verify_input_requires_5d_latents(self) -> None:
        stage = object.__new__(SanaWMDenoisingStage)

        result = stage.verify_input(
            self._make_verify_batch(latents=torch.zeros(1, 128, 2, 2)),
            SimpleNamespace(pipeline_config=SanaWMPipelineConfig()),
        )

        self.assertIn("latents", result.get_failed_fields())

    def test_verify_input_rejects_bad_camera_condition_shapes(self) -> None:
        stage = object.__new__(SanaWMDenoisingStage)

        result = stage.verify_input(
            self._make_verify_batch(
                extra={
                    "camera_conditions": torch.zeros(1, 3, 19),
                    "chunk_plucker": torch.zeros(1, 48, 3, 2, 2),
                }
            ),
            SimpleNamespace(pipeline_config=SanaWMPipelineConfig()),
        )

        self.assertIn("camera_conditions", result.get_failed_fields())

    def test_denoising_forward_wraps_base_cleanup(self) -> None:
        stage = object.__new__(SanaWMDenoisingStage)
        batch = Req(prompt="drive forward")
        server_args = SimpleNamespace(enable_cfg_parallel=False)

        with (
            patch.object(
                DenoisingStage,
                "forward",
                return_value=batch,
            ) as base_forward,
            patch(
                f"{_SANA_WM_STAGE_MODULE}._clear_sana_wm_precomputed_static_conditioning"
            ) as clear_static,
            patch(
                f"{_SANA_WM_STAGE_MODULE}.clear_sana_wm_request_runtime_cache"
            ) as clear_runtime,
        ):
            result = stage.forward(batch, server_args)

        self.assertIs(result, batch)
        base_forward.assert_called_once_with(batch, server_args)
        clear_static.assert_not_called()
        clear_runtime.assert_not_called()

        with (
            patch.object(
                DenoisingStage,
                "forward",
                side_effect=RuntimeError("boom"),
            ),
            patch(
                f"{_SANA_WM_STAGE_MODULE}._clear_sana_wm_precomputed_static_conditioning"
            ) as clear_static,
            patch(
                f"{_SANA_WM_STAGE_MODULE}.clear_sana_wm_request_runtime_cache"
            ) as clear_runtime,
            self.assertRaisesRegex(RuntimeError, "boom"),
        ):
            stage.forward(batch, server_args)

        clear_static.assert_called_once_with(batch)
        clear_runtime.assert_called_once_with(batch)

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

    def test_cfg_parallel_rejects_invalid_rank(self) -> None:
        local_noise = torch.tensor([7.0])

        with self.assertRaisesRegex(ValueError, "cfg_rank 0 or 1"):
            SanaWMDenoisingStage._combine_cfg_parallel_noise(
                local_noise, guidance_scale=4.5, cfg_rank=2
            )

    def test_serial_cfg_latent_model_input_reuses_buffer(self) -> None:
        latents = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        buffer = torch.empty(4, 3, 2)

        out = SanaWMDenoisingStage._write_serial_cfg_latent_model_input(
            buffer,
            latents,
        )

        self.assertIs(out, buffer)
        torch.testing.assert_close(out[:2], latents)
        torch.testing.assert_close(out[2:], latents)

        updated = latents + 100
        out_again = SanaWMDenoisingStage._write_serial_cfg_latent_model_input(
            buffer,
            updated,
        )

        self.assertIs(out_again, buffer)
        torch.testing.assert_close(out_again[:2], updated)
        torch.testing.assert_close(out_again[2:], updated)

    def test_serial_cfg_noise_combine_reuses_text_branch_storage(self) -> None:
        noise_uncond = torch.tensor([[[1.0, 2.0]]])
        noise_text = torch.tensor([[[3.0, 5.0]]])
        noise_pred = torch.cat([noise_uncond, noise_text], dim=0)

        combined = SanaWMDenoisingStage._combine_serial_cfg_noise_in_place(
            noise_pred,
            guidance_scale=2.0,
        )

        expected = noise_uncond + 2.0 * (noise_text - noise_uncond)
        self.assertEqual(combined.data_ptr(), noise_pred[1:].data_ptr())
        torch.testing.assert_close(combined, expected)

    def test_serial_cfg_step_timestep_avoids_spatial_cfg_duplication(self) -> None:
        condition_mask = torch.zeros(2, 1, 3, 4, 5)
        condition_mask[:, :, :1] = 1
        frame_limit = (1.0 - condition_mask[:, :, :, 0, 0].float()) * 1000.0
        token_limit = (1.0 - condition_mask.flatten(2).squeeze(1).float()) * 1000.0

        model_timestep, per_token_timesteps = (
            SanaWMDenoisingStage._prepare_step_timesteps(
                torch.tensor(800),
                frame_limit,
                token_limit,
                do_cfg=True,
                cfg_parallel=False,
            )
        )

        self.assertEqual(tuple(model_timestep.shape), (4, 1, 3))
        self.assertEqual(tuple(per_token_timesteps.shape), (2, 60))
        self.assertTrue(torch.equal(model_timestep[:2], model_timestep[2:]))
        self.assertTrue(torch.equal(model_timestep[:, :, 0], torch.zeros(4, 1)))
        self.assertTrue(
            torch.equal(
                model_timestep[:, :, 1:],
                torch.full((4, 1, 2), 800.0),
            )
        )
        self.assertTrue(torch.equal(per_token_timesteps[:, :20], torch.zeros(2, 20)))
        self.assertTrue(
            torch.equal(per_token_timesteps[:, 20:], torch.full((2, 40), 800.0))
        )


class TestSanaWMDecodingStage(_GlobalStageArgsMixin, unittest.TestCase):
    def test_verify_input_requires_5d_latents(self) -> None:
        stage = SanaWMDecodingStage(vae=None)
        batch = Req(
            prompt="drive forward",
            latents=torch.zeros(1, 128, 3, 2, 2),
        )

        result = stage.verify_input(
            batch,
            SimpleNamespace(pipeline_config=SanaWMPipelineConfig()),
        )

        self.assertTrue(result.is_valid(), result.get_failure_summary())

        batch.latents = torch.zeros(1, 128, 2, 2)
        result = stage.verify_input(
            batch,
            SimpleNamespace(pipeline_config=SanaWMPipelineConfig()),
        )

        self.assertIn("latents", result.get_failed_fields())


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
    def test_first_chunk_plus_one_chunk_indices_match_upstream(self) -> None:
        self.assertEqual(
            _sana_wm_chunk_index_from_chunk_size(
                21, 4, strategy="first_chunk_plus_one"
            ),
            [0, 5, 9, 13, 17],
        )

    def test_normalize_chunk_index_adds_start_and_final_boundary(self) -> None:
        self.assertEqual(_sana_wm_normalize_chunk_index([3], 5), [0, 3, 5])

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

    def test_bidirectional_gdn_chunk_size_matches_full_recurrence_for_long_t(
        self,
    ) -> None:
        torch.manual_seed(0)
        B, H, D, T, S = 1, 2, 4, 22, 2
        N = T * S
        q = torch.randn(B, H, D, N, dtype=torch.float64)
        k = torch.randn(B, H, D, N, dtype=torch.float64)
        v = torch.randn(B, H, D, N, dtype=torch.float64)
        q_rot = torch.randn(B, H, D, N, dtype=torch.float64)
        k_rot = torch.randn(B, H, D, N, dtype=torch.float64)
        beta = torch.sigmoid(torch.randn(B, H, T, S, dtype=torch.float64))
        decay = torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64))

        chunked = _gdn_scan_bidirectional(
            q,
            k,
            v,
            q_rot,
            k_rot,
            beta,
            decay,
            HW=(T, 1, S),
            chunk_size=21,
        )
        full = _gdn_scan_bidirectional(
            q,
            k,
            v,
            q_rot,
            k_rot,
            beta,
            decay,
            HW=(T, 1, S),
            chunk_size=None,
        )

        torch.testing.assert_close(chunked, full, atol=1e-9, rtol=1e-9)

    def test_bidirectional_camera_chunk_size_matches_full_recurrence_for_long_t(
        self,
    ) -> None:
        torch.manual_seed(0)
        B, H, D, T, S = 1, 2, 4, 22, 2
        N = T * S
        q_rot = torch.randn(B, H, D, N, dtype=torch.float64)
        k_rot = torch.randn(B, H, D, N, dtype=torch.float64)
        v = torch.randn(B, H, D, N, dtype=torch.float64)
        beta = torch.sigmoid(torch.randn(B, H, T, S, dtype=torch.float64))
        decay = torch.sigmoid(torch.randn(B, H, T, dtype=torch.float64))

        chunked = _single_path_delta_scan_bidirectional(
            q_rot,
            k_rot,
            v,
            beta,
            decay,
            HW=(T, 1, S),
            chunk_size=21,
        )
        full = _single_path_delta_scan_bidirectional(
            q_rot,
            k_rot,
            v,
            beta,
            decay,
            HW=(T, 1, S),
            chunk_size=None,
        )

        torch.testing.assert_close(chunked, full, atol=1e-9, rtol=1e-9)


class TestSanaWMRefinerStage(_GlobalStageArgsMixin, unittest.TestCase):
    @staticmethod
    def _make_refiner_stage() -> SanaWMLTX2RefinerStage:
        return SanaWMLTX2RefinerStage(
            transformer=torch.nn.Identity(),
            connectors=torch.nn.Identity(),
            text_encoder=torch.nn.Identity(),
            tokenizer=SimpleNamespace(pad_token="<pad>", eos_token="<eos>"),
            dtype=torch.float32,
        )

    def test_verify_input_accepts_prepared_refiner_request(self) -> None:
        stage = self._make_refiner_stage()
        batch = Req(
            prompt=["left", "right"],
            latents=torch.zeros(2, 128, 3, 2, 2),
            seeds=[11, 23],
            extra={},
        )

        result = stage.verify_input(
            batch,
            SimpleNamespace(pipeline_config=SanaWMPipelineConfig()),
        )

        self.assertTrue(result.is_valid(), result.get_failure_summary())

    def test_verify_input_rejects_bad_refiner_prompt_batch(self) -> None:
        stage = self._make_refiner_stage()
        batch = Req(
            prompt=["left", "right", "forward"],
            latents=torch.zeros(2, 128, 3, 2, 2),
            extra={},
        )

        result = stage.verify_input(
            batch,
            SimpleNamespace(pipeline_config=SanaWMPipelineConfig()),
        )

        self.assertIn("refiner_prompt", result.get_failed_fields())

    def test_verify_input_rejects_bad_refiner_seed_batch(self) -> None:
        stage = self._make_refiner_stage()
        batch = Req(
            prompt=["left", "right"],
            latents=torch.zeros(2, 128, 3, 2, 2),
            seeds=[1, 2, 3],
            extra={},
        )

        result = stage.verify_input(
            batch,
            SimpleNamespace(pipeline_config=SanaWMPipelineConfig()),
        )

        self.assertIn("seeds", result.get_failed_fields())

    def test_verify_input_requires_refiner_5d_latents(self) -> None:
        stage = self._make_refiner_stage()
        batch = Req(
            prompt="drive forward",
            latents=torch.zeros(1, 128, 2, 2),
            extra={},
        )

        result = stage.verify_input(
            batch,
            SimpleNamespace(pipeline_config=SanaWMPipelineConfig()),
        )

        self.assertIn("latents", result.get_failed_fields())

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

    def test_native_refiner_preserves_packed_and_5d_return_contract(self) -> None:
        arch = SanaWMRefinerArchConfig(
            in_channels=1,
            out_channels=1,
            patch_size=1,
            patch_size_t=1,
            num_layers=0,
            num_attention_heads=1,
            attention_head_dim=4,
            cross_attention_dim=4,
            caption_channels=4,
            base_num_frames=2,
            base_height=2,
            base_width=2,
        )
        model = SanaWMLTX2VideoRefiner(
            SanaWMRefinerConfig(arch_config=arch),
            hf_config={},
        ).eval()
        prompt_embeds = torch.randn(1, 3, 4)
        timestep = torch.zeros(1, 8)

        with torch.no_grad():
            packed = torch.randn(1, 8, 1)
            packed_out = model(
                hidden_states=packed,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                num_frames=2,
                height=2,
                width=2,
                fps=16.0,
                n_context_tokens=1,
            )

            latents = torch.randn(1, 1, 2, 2, 2)
            latents_out = model(
                hidden_states=latents,
                encoder_hidden_states=prompt_embeds,
                timestep=timestep,
                fps=16.0,
                n_context_tokens=1,
            )

        self.assertEqual(tuple(packed_out.shape), tuple(packed.shape))
        self.assertEqual(tuple(latents_out.shape), tuple(latents.shape))

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
                f"{_SANA_WM_REFINER_STAGE_MODULE}.sana_wm_is_tp_rank0",
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
                f"{_SANA_WM_REFINER_STAGE_MODULE}.sana_wm_stage_tp_world_size",
                return_value=2,
            ),
            patch(
                f"{_SANA_WM_REFINER_STAGE_MODULE}.sana_wm_is_tp_rank0",
                return_value=False,
            ),
            patch(
                f"{_SANA_WM_REFINER_STAGE_MODULE}."
                "sana_wm_broadcast_tensor_dict_from_tp_rank0",
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
