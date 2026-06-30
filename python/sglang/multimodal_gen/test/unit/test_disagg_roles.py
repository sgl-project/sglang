# SPDX-License-Identifier: Apache-2.0
"""Unit tests for disaggregation role-based module filtering."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.multimodal_gen.configs.pipeline_configs.hunyuan3d import (
    Hunyuan3D2PipelineConfig,
)
from sglang.multimodal_gen.runtime import server_args as server_args_module
from sglang.multimodal_gen.runtime.disaggregation.roles import (
    RoleType,
    filter_modules_for_role,
    get_module_role,
)
from sglang.multimodal_gen.runtime.pipelines.flux_2 import Flux2Pipeline
from sglang.multimodal_gen.runtime.pipelines.glm_image import GlmImagePipeline
from sglang.multimodal_gen.runtime.pipelines.hunyuan3d_pipeline import (
    Hunyuan3D2Pipeline,
)
from sglang.multimodal_gen.runtime.pipelines.ltx_2_pipeline import LTX2Pipeline
from sglang.multimodal_gen.runtime.pipelines.mova_pipeline import (
    MOVAPipeline,
    MOVAPipelineAlias,
)
from sglang.multimodal_gen.runtime.pipelines.qwen_image import (
    QwenImageEditPipeline,
    QwenImageLayeredPipeline,
)
from sglang.multimodal_gen.runtime.pipelines.wan_i2v_dmd_pipeline import (
    WanImageToVideoDmdPipeline,
)
from sglang.multimodal_gen.runtime.pipelines.wan_i2v_pipeline import (
    WanImageToVideoPipeline,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.image_encoding import (
    ImageVAEEncodingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.helios_denoising import (
    HeliosChunkedDenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan3d.shape import (
    Hunyuan3DShapeBeforeDenoisingStage,
    Hunyuan3DShapeExportStage,
    Hunyuan3DShapeSaveStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.ltx_2.denoising_av import (
    LTX2RefinementStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.mova import (
    MOVADecodingStage,
    MOVADenoisingStage,
)
from sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image_layered import (
    QwenImageLayeredBeforeDenoisingStage,
    _resolve_layered_image_path,
    _resolve_text_encoder_dtype,
)
from sglang.multimodal_gen.runtime.server_args import set_global_server_args


class _GlobalStageArgsMixin:
    def _install_stage_server_args(self, **kwargs):
        server_args = SimpleNamespace(
            comfyui_mode=False,
            enable_torch_compile=False,
            enable_cfg_parallel=False,
            attention_backend=None,
            **kwargs,
        )
        set_global_server_args(server_args)
        return server_args

    def setUp(self):
        super().setUp()
        self._prev_global_server_args = server_args_module._global_server_args
        self._install_stage_server_args()

    def tearDown(self):
        set_global_server_args(self._prev_global_server_args)
        super().tearDown()


class TestRoleType(unittest.TestCase):
    def test_from_string(self):
        self.assertEqual(RoleType.from_string("monolithic"), RoleType.MONOLITHIC)
        self.assertEqual(RoleType.from_string("encoder"), RoleType.ENCODER)
        self.assertEqual(RoleType.from_string("denoiser"), RoleType.DENOISER)
        self.assertEqual(RoleType.from_string("decoder"), RoleType.DECODER)
        self.assertEqual(RoleType.from_string("ENCODER"), RoleType.ENCODER)

    def test_from_string_backward_compat(self):
        self.assertEqual(RoleType.from_string("denoising"), RoleType.DENOISER)

    def test_from_string_invalid(self):
        with self.assertRaises(ValueError):
            RoleType.from_string("invalid")

    def test_choices(self):
        choices = RoleType.choices()
        self.assertIn("monolithic", choices)
        self.assertIn("encoder", choices)
        self.assertIn("denoiser", choices)
        self.assertIn("denoising", choices)
        self.assertIn("decoder", choices)


class TestGetModuleRole(unittest.TestCase):
    def test_encoder_modules(self):
        self.assertEqual(get_module_role("text_encoder"), RoleType.ENCODER)
        self.assertEqual(get_module_role("text_encoder_2"), RoleType.ENCODER)
        self.assertEqual(get_module_role("tokenizer"), RoleType.ENCODER)
        self.assertEqual(get_module_role("tokenizer_2"), RoleType.ENCODER)
        self.assertEqual(get_module_role("image_encoder"), RoleType.ENCODER)
        self.assertEqual(get_module_role("image_processor"), RoleType.ENCODER)
        self.assertEqual(get_module_role("connectors"), RoleType.ENCODER)
        self.assertEqual(get_module_role("vision_language_encoder"), RoleType.ENCODER)
        self.assertEqual(get_module_role("hy3dshape_conditioner"), RoleType.ENCODER)
        self.assertEqual(get_module_role("hy3dshape_image_processor"), RoleType.ENCODER)

    def test_denoiser_modules(self):
        self.assertEqual(get_module_role("transformer"), RoleType.DENOISER)
        self.assertEqual(get_module_role("transformer_2"), RoleType.DENOISER)
        self.assertEqual(get_module_role("video_dit"), RoleType.DENOISER)
        self.assertEqual(get_module_role("video_dit_2"), RoleType.DENOISER)
        self.assertEqual(get_module_role("audio_dit"), RoleType.DENOISER)
        self.assertEqual(get_module_role("dual_tower_bridge"), RoleType.DENOISER)
        self.assertEqual(get_module_role("hy3dshape_model"), RoleType.DENOISER)

    def test_decoder_modules(self):
        self.assertEqual(get_module_role("vae"), RoleType.DECODER)
        self.assertEqual(get_module_role("audio_vae"), RoleType.DECODER)
        self.assertEqual(get_module_role("video_vae"), RoleType.DECODER)
        self.assertEqual(get_module_role("vocoder"), RoleType.DECODER)
        self.assertEqual(get_module_role("hy3dshape_vae"), RoleType.DECODER)

    def test_shared_modules(self):
        self.assertIsNone(get_module_role("scheduler"))
        self.assertIsNone(get_module_role("hy3dshape_scheduler"))


class TestFilterModulesForRole(unittest.TestCase):
    WAN_MODULES = ["text_encoder", "tokenizer", "vae", "transformer", "scheduler"]

    def test_monolithic_keeps_all(self):
        result = filter_modules_for_role(self.WAN_MODULES, RoleType.MONOLITHIC)
        self.assertEqual(result, self.WAN_MODULES)

    def test_encoder_does_not_keep_decoder_modules_by_default(self):
        result = filter_modules_for_role(self.WAN_MODULES, RoleType.ENCODER)
        self.assertEqual(result, ["text_encoder", "tokenizer", "scheduler"])

    def test_encoder_can_keep_explicit_cross_role_modules(self):
        result = filter_modules_for_role(
            self.WAN_MODULES,
            RoleType.ENCODER,
            extra_allowed_modules={"vae"},
        )
        self.assertEqual(result, ["text_encoder", "tokenizer", "vae", "scheduler"])

    def test_denoiser_skips_encoders_and_vae(self):
        result = filter_modules_for_role(self.WAN_MODULES, RoleType.DENOISER)
        self.assertEqual(result, ["transformer", "scheduler"])

    def test_decoder_keeps_vae_and_scheduler(self):
        result = filter_modules_for_role(self.WAN_MODULES, RoleType.DECODER)
        self.assertEqual(result, ["vae", "scheduler"])


class TestFilterModulesLTX2(unittest.TestCase):
    LTX2_MODULES = [
        "transformer",
        "text_encoder",
        "tokenizer",
        "scheduler",
        "vae",
        "audio_vae",
        "vocoder",
        "connectors",
    ]

    def test_decoder_includes_audio(self):
        result = filter_modules_for_role(self.LTX2_MODULES, RoleType.DECODER)
        self.assertEqual(result, ["scheduler", "vae", "audio_vae", "vocoder"])

    def test_encoder_does_not_keep_decoder_modules_by_default(self):
        result = filter_modules_for_role(self.LTX2_MODULES, RoleType.ENCODER)
        self.assertEqual(
            result, ["text_encoder", "tokenizer", "scheduler", "connectors"]
        )

    def test_denoiser_can_keep_ti2v_decoder_components(self):
        result = filter_modules_for_role(
            self.LTX2_MODULES,
            RoleType.DENOISER,
            extra_allowed_modules={"vae", "audio_vae"},
        )
        self.assertEqual(result, ["transformer", "scheduler", "vae", "audio_vae"])


# Consolidated from test_pipeline_stage_role_filter.py.
class _FakePipeline(ComposedPipelineBase):
    pipeline_name = "FakePipeline"
    _required_config_modules = []

    def initialize_pipeline(self, server_args):
        pass

    def create_pipeline_stages(self, server_args) -> None:
        pass


def _make_pipeline(role: RoleType) -> _FakePipeline:
    pipeline = object.__new__(_FakePipeline)
    pipeline.modules = {}
    pipeline._stages = []
    pipeline._stage_name_mapping = {}
    pipeline._disagg_role = role
    return pipeline


class _FakeStage:
    def __init__(self, role_affinity: RoleType):
        self.role_affinity = role_affinity
        self.registered_stage_name = None
        self.profile_stage_name = None

    def set_registered_stage_name(self, stage_name: str) -> None:
        self.registered_stage_name = stage_name

    def set_profile_stage_name(self, stage_name: str) -> None:
        self.profile_stage_name = stage_name


class TestPipelineStageRoleFilter(unittest.TestCase):
    def test_stage_factory_skips_without_constructing_for_other_role(self):
        pipeline = _make_pipeline(RoleType.ENCODER)

        def should_not_construct():
            raise AssertionError("stage factory should have been skipped")

        pipeline.add_stage_factory(
            RoleType.DENOISER,
            should_not_construct,
            "denoising_stage",
        )

        self.assertEqual(pipeline.stages, [])

    def test_stage_factory_constructs_for_matching_role(self):
        pipeline = _make_pipeline(RoleType.DENOISER)
        stage = _FakeStage(RoleType.DENOISER)
        events = []

        def create_stage():
            events.append("called")
            return stage

        pipeline.add_stage_factory(
            RoleType.DENOISER,
            create_stage,
            "denoising_stage",
        )

        self.assertEqual(events, ["called"])
        self.assertIs(pipeline.get_stage("denoising_stage"), stage)
        self.assertEqual(stage.registered_stage_name, "denoising_stage")

    def test_encoder_role_does_not_construct_standard_denoising_stage(self):
        pipeline = _make_pipeline(RoleType.ENCODER)

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base.DenoisingStage",
            side_effect=AssertionError("DenoisingStage should not be constructed"),
        ):
            pipeline.add_standard_denoising_stage()

        self.assertEqual(pipeline.stages, [])

    def test_encoder_role_does_not_construct_standard_decoding_stage(self):
        pipeline = _make_pipeline(RoleType.ENCODER)

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base.DecodingStage",
            side_effect=AssertionError("DecodingStage should not be constructed"),
        ):
            pipeline.add_standard_decoding_stage()

        self.assertEqual(pipeline.stages, [])


# Consolidated from test_disagg_pipeline_alignment.py.
class TestPipelineSpecificExtraModules(unittest.TestCase):
    def _get_extra_modules(
        self, pipeline_cls, role: RoleType, task_name: str
    ) -> set[str]:
        pipeline = object.__new__(pipeline_cls)
        return pipeline._get_extra_allowed_modules_for_role(role, task_name)

    def test_flux_encoder_keeps_vae(self):
        extras = self._get_extra_modules(Flux2Pipeline, RoleType.ENCODER, "ti2i")
        filtered = filter_modules_for_role(
            Flux2Pipeline._required_config_modules,
            RoleType.ENCODER,
            extra_allowed_modules=extras,
        )
        self.assertEqual(extras, {"vae"})
        self.assertEqual(
            set(filtered), {"text_encoder", "tokenizer", "vae", "scheduler"}
        )

    def test_qwen_image_edit_encoder_keeps_vae(self):
        extras = self._get_extra_modules(
            QwenImageEditPipeline, RoleType.ENCODER, "ti2i"
        )
        filtered = filter_modules_for_role(
            QwenImageEditPipeline._required_config_modules,
            RoleType.ENCODER,
            extra_allowed_modules=extras,
        )
        self.assertEqual(extras, {"vae"})
        self.assertEqual(
            set(filtered),
            {"processor", "scheduler", "text_encoder", "tokenizer", "vae"},
        )

    def test_qwen_image_layered_encoder_keeps_required_cross_role_modules(self):
        extras = self._get_extra_modules(
            QwenImageLayeredPipeline, RoleType.ENCODER, "ti2i"
        )
        filtered = filter_modules_for_role(
            QwenImageLayeredPipeline._required_config_modules,
            RoleType.ENCODER,
            extra_allowed_modules=extras,
        )
        self.assertEqual(extras, {"vae", "transformer"})
        self.assertNotIn(
            "text_encoder", QwenImageLayeredPipeline._required_config_modules
        )
        self.assertEqual(
            set(filtered),
            {
                "vae",
                "tokenizer",
                "processor",
                "transformer",
                "scheduler",
            },
        )

    def test_glm_image_encoder_keeps_vae_and_transformer(self):
        extras = self._get_extra_modules(GlmImagePipeline, RoleType.ENCODER, "ti2i")
        filtered = filter_modules_for_role(
            GlmImagePipeline._required_config_modules,
            RoleType.ENCODER,
            extra_allowed_modules=extras,
        )
        self.assertEqual(extras, {"vae", "transformer"})
        self.assertEqual(
            set(filtered),
            {
                "text_encoder",
                "tokenizer",
                "vae",
                "vision_language_encoder",
                "processor",
                "transformer",
                "scheduler",
            },
        )

    def test_wan_ti2v_denoiser_keeps_vae(self):
        for pipeline_cls in (WanImageToVideoPipeline, WanImageToVideoDmdPipeline):
            extras = self._get_extra_modules(pipeline_cls, RoleType.DENOISER, "ti2v")
            filtered = filter_modules_for_role(
                pipeline_cls._required_config_modules,
                RoleType.DENOISER,
                extra_allowed_modules=extras,
            )
            self.assertEqual(extras, {"vae"})
            self.assertEqual(set(filtered), {"vae", "transformer", "scheduler"})

    def test_ltx2_encoder_does_not_keep_decoder_modules(self):
        extras = self._get_extra_modules(LTX2Pipeline, RoleType.ENCODER, "ti2v")
        filtered = filter_modules_for_role(
            LTX2Pipeline._required_config_modules,
            RoleType.ENCODER,
            extra_allowed_modules=extras,
        )
        self.assertEqual(extras, set())
        self.assertEqual(
            set(filtered),
            {"text_encoder", "tokenizer", "scheduler", "connectors"},
        )

    def test_ltx2_ti2v_denoiser_keeps_vae_and_audio_vae(self):
        extras = self._get_extra_modules(LTX2Pipeline, RoleType.DENOISER, "ti2v")
        filtered = filter_modules_for_role(
            LTX2Pipeline._required_config_modules,
            RoleType.DENOISER,
            extra_allowed_modules=extras,
        )
        self.assertEqual(extras, {"vae", "audio_vae"})
        self.assertEqual(
            set(filtered), {"transformer", "scheduler", "vae", "audio_vae"}
        )

    def test_mova_encoder_keeps_video_and_audio_vaes(self):
        extras = self._get_extra_modules(MOVAPipeline, RoleType.ENCODER, "i2v")
        filtered = filter_modules_for_role(
            MOVAPipeline._required_config_modules,
            RoleType.ENCODER,
            extra_allowed_modules=extras,
        )
        self.assertEqual(extras, {"video_vae", "audio_vae"})
        self.assertEqual(
            set(filtered),
            {"video_vae", "audio_vae", "text_encoder", "tokenizer", "scheduler"},
        )

    def test_mova_alias_uses_same_encoder_extras(self):
        extras = self._get_extra_modules(MOVAPipelineAlias, RoleType.ENCODER, "i2v")
        self.assertEqual(extras, {"video_vae", "audio_vae"})


class TestQwenImageLayeredDtype(_GlobalStageArgsMixin, unittest.TestCase):
    def test_layered_image_path_accepts_string_and_list(self):
        self.assertEqual(
            _resolve_layered_image_path("/tmp/input.png"),
            "/tmp/input.png",
        )
        self.assertEqual(
            _resolve_layered_image_path(["/tmp/input.png"]),
            "/tmp/input.png",
        )

    def test_layered_image_path_rejects_empty_list(self):
        with self.assertRaisesRegex(ValueError, "non-empty image_path"):
            _resolve_layered_image_path([])

    def test_text_encoder_dtype_uses_parameter_dtype_without_dtype_attr(self):
        text_encoder = torch.nn.Linear(1, 1, bias=False).to(dtype=torch.bfloat16)
        self.assertEqual(
            _resolve_text_encoder_dtype(text_encoder),
            torch.bfloat16,
        )

    def test_component_uses_keep_standard_text_encoder_and_configured_dtypes(self):
        class _DummyVAE:
            temperal_downsample = []
            z_dim = 16

            def to(self, *args, **kwargs):
                return self

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.qwen_image_layered.get_local_torch_device",
            return_value=torch.device("cpu"),
        ):
            stage = QwenImageLayeredBeforeDenoisingStage(
                vae=_DummyVAE(),
                text_encoder=torch.nn.Linear(1, 1),
                tokenizer=object(),
                processor=object(),
                transformer=object(),
                scheduler=object(),
                model_path="/unused",
                vae_dtype=torch.float32,
                text_encoder_dtype=torch.float16,
            )

        uses = stage.component_uses(SimpleNamespace(), "qwen_layered")
        self.assertEqual(
            [(use.component_name, use.target_dtype) for use in uses],
            [("text_encoder", torch.float16), ("vae", torch.float32)],
        )


class TestImageVAEEncodingStageComponentName(_GlobalStageArgsMixin, unittest.TestCase):
    def test_component_name_can_follow_non_default_vae_key(self):
        stage = ImageVAEEncodingStage(vae=object(), component_name="video_vae")
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(vae_precision="bf16")
        )

        uses = stage.component_uses(server_args, "image_vae_encoding")
        self.assertEqual(len(uses), 1)
        self.assertEqual(uses[0].component_name, "video_vae")
        self.assertEqual(uses[0].target_dtype, torch.bfloat16)


class TestStageAffinityAndValidation(_GlobalStageArgsMixin, unittest.TestCase):
    def _make_hunyuan_pipeline(
        self, role: RoleType, *, paint_enable: bool
    ) -> Hunyuan3D2Pipeline:
        pipeline = object.__new__(Hunyuan3D2Pipeline)
        pipeline.server_args = self._install_stage_server_args(
            pipeline_config=Hunyuan3D2PipelineConfig(paint_enable=paint_enable)
        )
        pipeline._disagg_role = role
        pipeline.modules = {
            "hy3dshape_image_processor": object(),
            "hy3dshape_conditioner": object(),
            "hy3dshape_scheduler": object(),
            "hy3dshape_model": torch.nn.Linear(1, 1),
            "hy3dshape_vae": object(),
        }
        pipeline._stages = []
        pipeline._stage_name_mapping = {}
        return pipeline

    def test_helios_denoising_stage_is_denoiser_affine(self):
        stage = object.__new__(HeliosChunkedDenoisingStage)
        self.assertEqual(stage.role_affinity, RoleType.DENOISER)

    def test_mova_denoising_stage_is_denoiser_affine(self):
        stage = object.__new__(MOVADenoisingStage)
        self.assertEqual(stage.role_affinity, RoleType.DENOISER)

    def test_mova_decoding_stage_is_decoder_affine(self):
        stage = object.__new__(MOVADecodingStage)
        self.assertEqual(stage.role_affinity, RoleType.DECODER)

    def test_mova_skips_torch_compile_on_rocm(self):
        class _CompileTrackingModule(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.compile_called = False

            def compile(self, *args, **kwargs):
                self.compile_called = True

        stage = object.__new__(MOVADenoisingStage)
        module = _CompileTrackingModule()
        server_args = SimpleNamespace(enable_torch_compile=True)

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.mova.current_platform.is_hip",
            return_value=True,
        ):
            stage._maybe_enable_torch_compile(module, server_args)

        self.assertFalse(module.compile_called)

    def test_hunyuan3d_shape_only_disagg_accepts_non_monolithic_roles(self):
        pipeline = self._make_hunyuan_pipeline(RoleType.ENCODER, paint_enable=False)
        pipeline.validate_disagg_role(RoleType.ENCODER)
        pipeline.validate_disagg_role(RoleType.MONOLITHIC)

    def test_hunyuan3d_disagg_rejects_paint_pipeline(self):
        pipeline = self._make_hunyuan_pipeline(RoleType.ENCODER, paint_enable=True)
        with self.assertRaisesRegex(ValueError, "shape-only disaggregation"):
            pipeline.validate_disagg_role(RoleType.ENCODER)

    def test_hunyuan3d_shape_export_and_save_are_decoder_affine(self):
        export_stage = Hunyuan3DShapeExportStage(
            vae=object(),
            config=Hunyuan3D2PipelineConfig(paint_enable=False),
        )
        save_stage = Hunyuan3DShapeSaveStage(
            config=Hunyuan3D2PipelineConfig(paint_enable=False),
        )

        self.assertEqual(export_stage.role_affinity, RoleType.DECODER)
        self.assertEqual(save_stage.role_affinity, RoleType.DECODER)

    def test_hunyuan3d_stage_filtering_matches_shape_only_roles(self):
        expected = {
            RoleType.ENCODER: ["shape_before_denoising"],
            RoleType.DENOISER: ["shape_denoising"],
            RoleType.DECODER: ["shape_export", "shape_save"],
        }

        for role, stage_names in expected.items():
            pipeline = self._make_hunyuan_pipeline(role, paint_enable=False)
            pipeline.create_pipeline_stages(pipeline.server_args)
            self.assertEqual(list(pipeline._stage_name_mapping.keys()), stage_names)

    def test_hunyuan3d_shape_stage_no_longer_stores_model_dtype(self):
        pipeline = self._make_hunyuan_pipeline(RoleType.ENCODER, paint_enable=False)
        pipeline.create_pipeline_stages(pipeline.server_args)
        stage = pipeline._stage_name_mapping["shape_before_denoising"]
        self.assertIsInstance(stage, Hunyuan3DShapeBeforeDenoisingStage)
        self.assertFalse(hasattr(stage, "model_dtype"))

    def test_ltx2_refinement_stage_keeps_class_name_stage_key(self):
        stage = object.__new__(LTX2RefinementStage)
        self.assertEqual(
            ComposedPipelineBase._infer_stage_name(stage), "LTX2RefinementStage"
        )


class TestHunyuan3DShapeStageRuntimeDtype(_GlobalStageArgsMixin, unittest.TestCase):
    def test_conditioner_parameter_dtype_wins_over_sample_dtype(self):
        conditioner = torch.nn.Linear(4, 4, bias=False).to(dtype=torch.float32)
        stage = Hunyuan3DShapeBeforeDenoisingStage(
            image_processor=object(),
            conditioner=conditioner,
            scheduler=SimpleNamespace(init_noise_sigma=1.0),
            config=Hunyuan3D2PipelineConfig(),
            latent_shape=(1, 2, 2),
            guidance_embed=False,
        )

        self.assertEqual(
            stage._resolve_runtime_dtype(torch.zeros(1, dtype=torch.float16)),
            torch.float32,
        )

    def test_runtime_dtype_falls_back_to_sample_tensor_without_module_dtype(self):
        stage = Hunyuan3DShapeBeforeDenoisingStage(
            image_processor=object(),
            conditioner=object(),
            scheduler=SimpleNamespace(init_noise_sigma=1.0),
            config=Hunyuan3D2PipelineConfig(),
            latent_shape=(1, 2, 2),
            guidance_embed=False,
        )

        self.assertEqual(
            stage._resolve_runtime_dtype(torch.zeros(1, dtype=torch.bfloat16)),
            torch.bfloat16,
        )

    def test_runtime_dtype_warns_and_falls_back_after_non_iterable_parameters(self):
        conditioner = SimpleNamespace(
            parameters=lambda: (_ for _ in ()).throw(TypeError("not iterable")),
            buffers=lambda: iter(()),
        )
        stage = Hunyuan3DShapeBeforeDenoisingStage(
            image_processor=object(),
            conditioner=conditioner,
            scheduler=SimpleNamespace(init_noise_sigma=1.0),
            config=Hunyuan3D2PipelineConfig(),
            latent_shape=(1, 2, 2),
            guidance_embed=False,
        )

        with patch(
            "sglang.multimodal_gen.runtime.pipelines_core.stages.model_specific_stages.hunyuan3d.shape.logger.warning"
        ) as mock_warning:
            dtype = stage._resolve_runtime_dtype(torch.zeros(1, dtype=torch.float16))

        self.assertEqual(dtype, torch.float16)
        mock_warning.assert_called_once()
        self.assertEqual(mock_warning.call_args.args[1], "parameters")


if __name__ == "__main__":
    unittest.main()
