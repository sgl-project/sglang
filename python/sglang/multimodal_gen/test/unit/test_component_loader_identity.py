# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch
import torch.nn as nn

from sglang.multimodal_gen.runtime.disaggregation.roles import (
    RoleType,
    filter_modules_for_role,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.adapter_loader import (
    AdapterLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentLoader,
    PipelineComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    TransformerLoader,
    _server_args_for_transformer_component,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import VAELoader
from sglang.multimodal_gen.runtime.loader.component_loaders.vl_encoder_loader import (
    VisionLanguageEncoderLoader,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
)
from sglang.multimodal_gen.runtime.pipelines_core.component_loading import (
    load_transformer_if_needed,
    register_loaded_transformer,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)
from sglang.multimodal_gen.runtime.platforms import current_platform
from sglang.multimodal_gen.runtime.server_args import ServerArgs


class _AliasPipeline(ComposedPipelineBase):
    def initialize_pipeline(self, _server_args):
        pass

    def create_pipeline_stages(self, _server_args):
        pass


class TestComponentLoaderIdentity(unittest.TestCase):
    def test_lazy_transformer_load_register_and_reuse(self):
        for native_fallback in (False, True):
            with self.subTest(native_fallback=native_fallback):
                server_args = ServerArgs(
                    model_path="x",
                    component_residency={"transformer": COMPONENT_OFFLOAD},
                )
                server_args.model_loaded = {"transformer": False}
                server_args.model_paths = {"transformer": "/model/transformer"}
                stage = SimpleNamespace(transformer=None)
                pipeline = object.__new__(_AliasPipeline)
                pipeline.modules = {}
                transformer = nn.Linear(2, 2)

                with (
                    patch.object(
                        current_platform, "get_available_gpu_memory", return_value=16.0
                    ),
                    patch.object(
                        TransformerLoader,
                        "load_customized",
                        autospec=True,
                        return_value=transformer,
                        side_effect=(
                            ValueError("Unsupported model architecture")
                            if native_fallback
                            else None
                        ),
                    ) as customized,
                    patch.object(
                        TransformerLoader,
                        "load_native",
                        autospec=True,
                        return_value=transformer,
                    ) as native,
                ):
                    self.assertTrue(load_transformer_if_needed(stage, server_args))
                    self.assertIs(stage.transformer, transformer)
                    self.assertFalse(server_args.model_loaded["transformer"])
                    register_loaded_transformer(stage, server_args, pipeline)
                    self.assertTrue(server_args.model_loaded["transformer"])
                    self.assertIs(pipeline.get_module("transformer"), transformer)
                    self.assertFalse(load_transformer_if_needed(stage, server_args))
                    customized.assert_called_once()
                    if native_fallback:
                        self.assertEqual(
                            native.call_args.args[1:],
                            (
                                "/model/transformer",
                                server_args,
                                "diffusers",
                                "transformer",
                            ),
                        )
                    else:
                        native.assert_not_called()
                self.assertFalse(transformer.training)
                self.assertEqual(stage.transformer(torch.ones(1, 2)).shape, (1, 2))

    def test_lazy_transformer_load_failure_preserves_state(self):
        server_args = ServerArgs(model_path="x")
        server_args.model_loaded = {"transformer": False}
        server_args.model_paths = {"transformer": "/missing/transformer"}
        stage = SimpleNamespace(transformer=None)
        with (
            patch.object(
                current_platform, "get_available_gpu_memory", return_value=16.0
            ),
            patch.object(
                TransformerLoader, "load_customized", side_effect=FileNotFoundError
            ),
            patch.object(TransformerLoader, "load_native") as native,
        ):
            with self.assertRaises(FileNotFoundError):
                load_transformer_if_needed(stage, server_args)
        self.assertIsNone(stage.transformer)
        self.assertFalse(server_args.model_loaded["transformer"])
        native.assert_not_called()

    def test_structural_identity_selects_roles_and_preserves_exact_policy_keys(self):
        aliases = {
            "conditioning": "text_encoder_2",
            "refiner": "transformer_3",
            "decoder": "video_vae",
        }
        self.assertEqual(
            filter_modules_for_role(
                aliases, RoleType.ENCODER, structural_component_names=aliases
            ),
            ["conditioning"],
        )
        self.assertEqual(
            filter_modules_for_role(
                aliases, RoleType.DENOISER, structural_component_names=aliases
            ),
            ["refiner"],
        )
        self.assertEqual(
            filter_modules_for_role(
                aliases, RoleType.DECODER, structural_component_names=aliases
            ),
            ["decoder"],
        )

        loader = ComponentLoader.for_component_type(
            "duration_head_2", "ltx2", "LTX2DurationHeadModel"
        )
        self.assertIsInstance(loader, AdapterLoader)
        self.assertEqual(
            loader.structural_component_name("auxiliary_head"), "duration_head_2"
        )
        self.assertEqual(
            loader.structural_component_type("auxiliary_head"), "duration_head"
        )

    def test_declared_alias_loads_by_exact_key_and_structural_source(self):
        pipeline = object.__new__(_AliasPipeline)
        pipeline.model_path = "/model"
        pipeline.memory_usages = {}
        pipeline._disagg_role = RoleType.MONOLITHIC
        pipeline._required_config_modules = ["auxiliary_head"]
        pipeline._extra_config_module_map = {"auxiliary_head": "duration_head_2"}
        pipeline._load_config = lambda: {
            "_class_name": "TestPipeline",
            "_diffusers_version": "0",
            "duration_head_2": ["ltx2", "LTX2DurationHeadModel"],
            "scheduler": ["diffusers", "Scheduler"],
        }
        server_args = SimpleNamespace(
            component_paths={},
            component_direct_gpu_weight_loading=set(),
            resolve_component_attention_backend=lambda *_names: (None, None),
            comfyui_mode=False,
        )

        with patch.object(
            PipelineComponentLoader,
            "load_component",
            return_value=(nn.Linear(1, 1), 0.25),
        ) as load_component:
            loaded = pipeline.load_modules(server_args)

        self.assertIn("auxiliary_head", loaded)
        self.assertEqual(pipeline.memory_usages, {"auxiliary_head": 0.25})
        load_component.assert_called_once_with(
            component_name="auxiliary_head",
            component_type="duration_head_2",
            loader_cls=None,
            component_model_path="/model/duration_head_2",
            transformers_or_diffusers="ltx2",
            server_args=server_args,
            component_architecture="LTX2DurationHeadModel",
            component_attn_backend=None,
            component_attn_name="auxiliary_head",
        )

    def test_skipped_alias_keeps_exact_override_and_structural_config(self):
        pipeline = object.__new__(_AliasPipeline)
        pipeline.model_path = "/model"
        pipeline._disagg_role = RoleType.ENCODER
        pipeline._required_config_modules = []
        pipeline._unfiltered_required_config_modules = ("vae_2",)
        pipeline._extra_config_module_map = {"vae_2": "video_vae_2"}
        vae_config = Mock()
        server_args = SimpleNamespace(
            component_paths={"vae_2": "/exact/decoder"},
            pipeline_config=SimpleNamespace(vae_config=vae_config),
        )

        with (
            patch(
                "sglang.multimodal_gen.runtime.pipelines_core."
                "composed_pipeline_base.prepare_diffusers_component_path_for_loading",
                return_value="/resolved/decoder",
            ) as prepare_path,
            patch(
                "sglang.multimodal_gen.runtime.utils.hf_diffusers_utils."
                "get_diffusers_component_config",
                return_value={"sample_size": 32},
            ) as get_config,
        ):
            pipeline._init_skipped_component_configs(
                {
                    "vae_2": ["diffusers", "VideoVAE"],
                    "video_vae_2": ["diffusers", "VideoVAE"],
                },
                server_args,
            )

        prepare_path.assert_called_once_with("/exact/decoder")
        get_config.assert_called_once_with(component_path="/resolved/decoder")
        vae_config.update_model_arch.assert_called_once_with({"sample_size": 32})

    def test_structural_aliases_select_loader_config_and_weight_behavior(self):
        selected = ["selected.safetensors"]
        select_weight_files = Mock(return_value=selected)
        vae_loader = VAELoader()
        vae_loader.component_type = "video_vae"
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(select_vae_weight_files=select_weight_files)
        )
        self.assertIs(
            vae_loader.select_weight_files(
                ["candidate.safetensors"],
                "/decoder",
                server_args,
                "decoder",
                "fp32",
            ),
            selected,
        )
        select_weight_files.assert_called_once_with(
            safetensors_list=["candidate.safetensors"],
            component_model_path="/decoder",
            component_name="video_vae",
            vae_precision="fp32",
        )

    def test_transformer_exact_override_wins_without_leaking_global_flags(self):
        server_args = SimpleNamespace(
            component_weights_paths={},
            component_quantizations={},
            component_quantization_ignored_layers={},
            transformer_weights_path="global.safetensors",
            nunchaku_config="global-nunchaku",
        )
        secondary = _server_args_for_transformer_component(
            server_args, "refiner", "transformer_3"
        )
        self.assertIsNone(secondary.transformer_weights_path)
        self.assertIsNone(secondary.nunchaku_config)

        server_args.component_weights_paths["refiner"] = "refiner.safetensors"
        exact = _server_args_for_transformer_component(
            server_args, "refiner", "transformer_3"
        )
        self.assertEqual(exact.transformer_weights_path, "refiner.safetensors")
        self.assertIs(
            _server_args_for_transformer_component(
                server_args, "denoiser", "transformer"
            ),
            server_args,
        )

    def test_native_fallback_prioritizes_exact_component_precision(self):
        loader = ComponentLoader()
        loader.component_type = "video_vae_2"
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(),
            revision="test-revision",
            trust_remote_code=False,
        )
        native_model = nn.Linear(1, 1)

        with (
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "component_loader.resolve_component_precision",
                side_effect=[torch.float32],
            ) as resolve_precision,
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "component_loader.prepare_diffusers_component_path_for_loading",
                return_value="/resolved/decoder",
            ),
            patch("diffusers.AutoModel.from_pretrained", return_value=native_model),
        ):
            loaded = loader.load_native("/decoder", server_args, "diffusers", "decoder")

        self.assertIs(loaded, native_model)
        self.assertEqual(
            [call.args[1] for call in resolve_precision.call_args_list], ["decoder"]
        )

    def test_vision_language_loader_uses_exact_residency_key(self):
        requested = []
        server_args = SimpleNamespace(
            srt_encoder_url=None,
            trust_remote_code=False,
            revision=None,
            should_start_component_on_cpu=lambda name: requested.append(name) or True,
        )
        loader = VisionLanguageEncoderLoader()
        loader.component_type = "vision_language_encoder"

        with (
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "vl_encoder_loader.get_hf_config",
                return_value=object(),
            ),
            patch(
                "transformers.GlmImageForConditionalGeneration.from_pretrained",
                return_value=nn.Linear(1, 1),
            ),
        ):
            loader.load_customized("unused", server_args, "prompt_conditioner")

        self.assertEqual(requested, ["prompt_conditioner"])


if __name__ == "__main__":
    unittest.main()
