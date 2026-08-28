# SPDX-License-Identifier: Apache-2.0

import unittest
from types import SimpleNamespace
from unittest.mock import patch

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
from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    TextEncoderLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.transformer_loader import (
    _server_args_for_transformer_component,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import VAELoader
from sglang.multimodal_gen.runtime.loader.component_loaders.vl_encoder_loader import (
    VisionLanguageEncoderLoader,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)


class TestComponentLoaderIdentity(unittest.TestCase):
    def test_disaggregation_classifies_structural_aliases(self):
        aliases = {
            "conditioning": "text_encoder_2",
            "refiner": "transformer_3",
            "decoder": "video_vae",
        }
        components = list(aliases)

        self.assertEqual(
            filter_modules_for_role(
                components,
                RoleType.ENCODER,
                structural_component_names=aliases,
            ),
            ["conditioning"],
        )
        self.assertEqual(
            filter_modules_for_role(
                components,
                RoleType.DENOISER,
                structural_component_names=aliases,
            ),
            ["refiner"],
        )
        self.assertEqual(
            filter_modules_for_role(
                components,
                RoleType.DECODER,
                structural_component_names=aliases,
            ),
            ["decoder"],
        )

    def test_declared_alias_loads_under_its_exact_key(self):
        class _Pipeline(ComposedPipelineBase):
            def initialize_pipeline(self, _server_args):
                pass

            def create_pipeline_stages(self, _server_args):
                pass

        pipeline = object.__new__(_Pipeline)
        pipeline.model_path = "/model"
        pipeline.memory_usages = {}
        pipeline._disagg_role = RoleType.MONOLITHIC
        pipeline._required_config_modules = ["auxiliary_head"]
        pipeline._extra_config_module_map = {"auxiliary_head": "duration_head_2"}
        pipeline._load_config = lambda: {
            "_class_name": "TestPipeline",
            "_diffusers_version": "0",
            "duration_head_2": ["ltx2", "LTX2DurationHeadModel"],
        }
        server_args = SimpleNamespace(
            component_paths={},
            resolve_component_attention_backend=lambda *_names: (None, None),
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
            component_model_path="/model/duration_head_2",
            transformers_or_diffusers="ltx2",
            server_args=server_args,
            component_architecture="LTX2DurationHeadModel",
            component_attn_backend=None,
            component_attn_name="auxiliary_head",
        )

    def test_factory_preserves_the_structural_slot(self):
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

    def test_pipeline_passes_the_exact_policy_key_to_the_loader(self):
        observed = {}

        class _RecordingLoader:
            allow_global_attention_backend_fallback = True

            def load(self, _path, _server_args, component_name, library):
                observed["component_name"] = component_name
                observed["library"] = library
                return object(), 0.0

        with patch.object(
            ComponentLoader, "for_component_type", return_value=_RecordingLoader()
        ) as factory:
            PipelineComponentLoader.load_component(
                component_name="auxiliary_head",
                component_type="duration_head_2",
                component_model_path="unused",
                transformers_or_diffusers="ltx2",
                server_args=object(),
            )

        factory.assert_called_once_with("duration_head_2", "ltx2", None)
        self.assertEqual(
            observed, {"component_name": "auxiliary_head", "library": "ltx2"}
        )

    def test_structural_aliases_select_config_roles_without_masking_exact_keys(self):
        text_loader = TextEncoderLoader()
        text_loader.component_type = "text_encoder_3"
        self.assertEqual(
            text_loader._extract_encoder_index(
                text_loader.structural_component_name("conditioning_encoder")
            ),
            2,
        )

        vae_loader = VAELoader()
        vae_loader.component_type = "video_vae"
        self.assertEqual(vae_loader.structural_component_type("decoder"), "video_vae")

    def test_structural_native_only_contract_applies_to_an_exact_alias(self):
        loader = ComponentLoader()
        loader.component_type = "video_vae"
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(native_only_components=("video_vae",))
        )

        self.assertTrue(
            loader.should_raise_customized_load_error(server_args, "decoder")
        )

    def test_native_fallback_uses_exact_then_structural_precision(self):
        loader = ComponentLoader()
        loader.component_type = "video_vae"
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
                side_effect=[None, torch.bfloat16],
            ) as resolve_precision,
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "component_loader.prepare_diffusers_component_path_for_loading",
                return_value="/resolved/decoder",
            ),
            patch(
                "diffusers.AutoModel.from_pretrained",
                return_value=native_model,
            ) as native_load,
        ):
            loaded = loader.load_native("/decoder", server_args, "diffusers", "decoder")

        self.assertIs(loaded, native_model)
        self.assertEqual(
            [call.args[1] for call in resolve_precision.call_args_list],
            ["decoder", "video_vae"],
        )
        native_load.assert_called_once_with(
            "/resolved/decoder",
            revision="test-revision",
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
        )

    def test_transformer_alias_masks_global_weights_but_exact_override_wins(self):
        server_args = SimpleNamespace(
            component_weights_paths={},
            component_quantizations={},
            component_quantization_ignored_layers={},
            transformer_weights_path="global.safetensors",
            nunchaku_config="global-nunchaku",
        )

        masked = _server_args_for_transformer_component(
            server_args, "refiner", "transformer_3"
        )
        self.assertIsNone(masked.transformer_weights_path)
        self.assertIsNone(masked.nunchaku_config)

        server_args.component_weights_paths["refiner"] = "refiner.safetensors"
        exact = _server_args_for_transformer_component(
            server_args, "refiner", "transformer_3"
        )
        self.assertEqual(exact.transformer_weights_path, "refiner.safetensors")

    def test_vision_language_encoder_uses_exact_residency_key(self):
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
