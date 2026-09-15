# SPDX-License-Identifier: Apache-2.0

import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from torch import nn

from sglang.multimodal_gen.runtime.loader.component_loaders.adapter_loader import (
    AdapterLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.bridge_loader import (
    BridgeLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
    ComponentLoader,
    NativeComponentLoaderRequired,
    PlainStateDictComponentLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.diffusion_decoder_loader import (
    DiffusionDecoderLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.sound_tokenizer_loader import (
    SoundTokenizerLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.upsampler_loader import (
    UpsamplerLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import VAELoader
from sglang.multimodal_gen.runtime.loader.component_loaders.vocoder_loader import (
    VocoderLoader,
)
from sglang.multimodal_gen.runtime.pipelines_core.composed_pipeline_base import (
    ComposedPipelineBase,
)


class _TestLoader(PlainStateDictComponentLoader):
    pass


class _WeightOverrideLoader(PlainStateDictComponentLoader):
    def __init__(self, error: Exception):
        super().__init__()
        self.error = error
        self.native_called = False

    def load_customized(self, *args, **kwargs):
        raise self.error

    def load_native(self, *args, **kwargs):
        self.native_called = True
        return object()


class _ModuleLoader(ComponentLoader):
    def load_customized(self, *args, **kwargs):
        return nn.Linear(1, 1)


def _loader_server_args(component_weights_paths, *, fsdp_requested=False):
    disabled_components = set()
    return SimpleNamespace(
        component_precisions={},
        component_quantizations={},
        component_weights_paths=component_weights_paths,
        pipeline_config=SimpleNamespace(native_only_components=()),
        resolve_component_attention_backend=lambda _name: (None, None),
        requested_component_attention_backend=lambda _name: None,
        should_direct_gpu_weight_load_component=lambda _name: False,
        should_start_component_on_cpu=lambda _name: True,
        should_use_fsdp_for_component=lambda _name: fsdp_requested,
        disable_fsdp_for_component=lambda name: disabled_components.add(name),
        disabled_components=disabled_components,
    )


class TestComponentQuantizationAdmission(unittest.TestCase):
    def test_plain_loader_admits_its_exact_precision(self):
        server_args = SimpleNamespace(component_precisions={"vocoder": "fp16"})

        self.assertEqual(
            _TestLoader().component_load_precision(server_args, "vocoder"), "fp16"
        )

    def test_direct_gpu_selection_requires_a_declared_component(self):
        server_args = SimpleNamespace(
            component_direct_gpu_weight_loading={"missing_vae": True}
        )

        with self.assertRaisesRegex(ValueError, "missing_vae"):
            ComposedPipelineBase._validate_direct_gpu_component_selection(
                {"vae": ["diffusers", "AutoencoderKL"]}, server_args
            )

    def test_direct_gpu_selector_is_rejected_by_unqualified_loader(self):
        server_args = SimpleNamespace(
            component_precisions={},
            component_quantizations={},
            component_weights_paths={},
            should_direct_gpu_weight_load_component=lambda component: (
                component == "vocoder"
            ),
        )

        with self.assertRaisesRegex(
            ComponentCheckpointUnsupportedError, "does not support direct GPU"
        ):
            ComponentLoader().load(
                "/model/vocoder", server_args, "vocoder", "diffusers"
            )

    def test_direct_gpu_selector_is_rejected_by_unqualified_component(self):
        server_args = SimpleNamespace(
            component_precisions={},
            component_quantizations={},
            component_weights_paths={},
            should_direct_gpu_weight_load_component=lambda component: (
                component == "audio_vae"
            ),
        )

        with self.assertRaisesRegex(
            ComponentCheckpointUnsupportedError,
            "Direct GPU loading is not implemented",
        ):
            VAELoader().load("/model/audio_vae", server_args, "audio_vae", "diffusers")

    def test_plain_loader_resolves_weights_separately_from_config(self):
        server_args = SimpleNamespace(
            component_weights_paths={"vocoder": "owner/repo/vocoder.safetensors"}
        )
        with (
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "component_loader.resolve_weight",
                return_value="resolved",
            ),
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "component_loader.materialize_weight",
                return_value="/cache/vocoder.safetensors",
            ),
        ):
            self.assertEqual(
                _TestLoader().resolve_component_weights_path(
                    "/base/vocoder", server_args, "vocoder"
                ),
                "/cache/vocoder.safetensors",
            )

    def test_unsupported_loader_rejects_weight_override_before_loading(self):
        server_args = _loader_server_args({"pe": "/weights/model.safetensors"})

        with self.assertRaisesRegex(
            ComponentCheckpointUnsupportedError,
            r"--component-paths\.pe.*config and weights",
        ):
            ComponentLoader().load("/base/pe", server_args, "pe", "transformers")

    def test_weight_override_failure_never_falls_back_to_base_component(self):
        loader = _WeightOverrideLoader(ValueError("incompatible checkpoint"))
        server_args = _loader_server_args(
            {"text_encoder": "/weights/model.safetensors"}
        )

        with (
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "component_loader.current_platform.get_available_gpu_memory",
                return_value=10.0,
            ),
            self.assertRaisesRegex(RuntimeError, "fallback would ignore it"),
        ):
            loader.load(
                "/base/text_encoder", server_args, "text_encoder", "transformers"
            )

        self.assertFalse(loader.native_called)

    def test_library_fallback_requires_a_complete_component_override(self):
        loader = _WeightOverrideLoader(
            NativeComponentLoaderRequired("delegate to Transformers")
        )
        server_args = _loader_server_args(
            {"text_encoder": "/weights/model.safetensors"}
        )

        with (
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "component_loader.current_platform.get_available_gpu_memory",
                return_value=10.0,
            ),
            self.assertRaisesRegex(
                ComponentCheckpointUnsupportedError,
                r"--component-paths\.text_encoder.*config and weights",
            ),
        ):
            loader.load(
                "/base/text_encoder", server_args, "text_encoder", "transformers"
            )

        self.assertFalse(loader.native_called)

    def test_materialized_module_decides_fsdp_support(self):
        server_args = _loader_server_args({}, fsdp_requested=True)
        with patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "component_loader.current_platform.get_available_gpu_memory",
            side_effect=(10.0, 9.0),
        ):
            _ModuleLoader().load("/base/module", server_args, "module", "diffusers")

        self.assertEqual(server_args.disabled_components, {"module"})

    def test_plain_checkpoint_config_is_accepted(self):
        config = {"_class_name": "TestModel"}

        with patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "component_loader.get_diffusers_component_config",
            return_value=config,
        ):
            loaded = _TestLoader().load_component_config("/model/component", "test")

        self.assertIs(loaded, config)

    def test_all_quantization_metadata_layouts_fail_closed(self):
        configs = {
            "quantization_config": {
                "quantization_config": {"quant_method": "bitsandbytes"}
            },
            "text_config.quantization_config": {
                "text_config": {"quantization_config": {"quant_method": "fp8"}}
            },
            "compression_config": {
                "compression_config": {"quant_method": "compressed-tensors"}
            },
        }

        for source, config in configs.items():
            with (
                self.subTest(source=source),
                self.assertRaisesRegex(
                    ComponentCheckpointUnsupportedError,
                    rf"{re.escape(source)}.*quant_method=.*cannot restore",
                ),
            ):
                _TestLoader.ensure_plain_state_dict_checkpoint(config, "test_component")

        with self.assertRaisesRegex(
            ComponentCheckpointUnsupportedError,
            "Cannot parse checkpoint quantization metadata",
        ):
            _TestLoader.ensure_plain_state_dict_checkpoint(
                {"quantization_config": "invalid"}, "test_component"
            )

    def test_native_raw_state_loaders_share_the_admission_boundary(self):
        loader_classes = (
            AdapterLoader,
            BridgeLoader,
            DiffusionDecoderLoader,
            SoundTokenizerLoader,
            UpsamplerLoader,
            VocoderLoader,
        )

        for loader_class in loader_classes:
            with self.subTest(loader=loader_class.__name__):
                self.assertTrue(issubclass(loader_class, PlainStateDictComponentLoader))

    def test_adapter_rejects_quantization_before_model_construction(self):
        config = {
            "_class_name": "LTX2ConnectorModel",
            "quantization_config": {"quant_method": "bitsandbytes"},
        }

        with (
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "component_loader.get_diffusers_component_config",
                return_value=config,
            ),
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "component_loader.ModelRegistry.resolve_model_cls"
            ) as resolve_model,
            self.assertRaises(ComponentCheckpointUnsupportedError),
        ):
            AdapterLoader().load_customized("/model/connectors", None, "connectors")

        resolve_model.assert_not_called()

    def test_upsampler_rejects_quantization_before_loading_weights(self):
        config = {
            "_class_name": "LatentUpsampler",
            "quantization_config": {"quant_method": "bitsandbytes"},
        }
        server_args = SimpleNamespace(component_weights_paths={})

        with (
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "upsampler_loader._find_safetensors_file",
                return_value="/model/spatial_upsampler/model.safetensors",
            ),
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "upsampler_loader._load_explicit_config",
                return_value=config,
            ),
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "upsampler_loader.safetensors_load_file"
            ) as load_weights,
            self.assertRaises(ComponentCheckpointUnsupportedError),
        ):
            UpsamplerLoader().load_customized(
                "/model/spatial_upsampler", server_args, "spatial_upsampler"
            )

        load_weights.assert_not_called()

    def test_upsampler_uses_exact_component_weight_override(self):
        server_args = SimpleNamespace(
            component_weights_paths={"spatial_upsampler": "owner/repo/upsampler"}
        )
        with (
            patch.object(
                UpsamplerLoader,
                "resolve_component_weights_path",
                return_value="/cache/upsampler.safetensors",
            ) as resolve_weights,
            patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "upsampler_loader._find_safetensors_file",
                side_effect=RuntimeError("stop after routing"),
            ) as find_weights,
            self.assertRaisesRegex(RuntimeError, "stop after routing"),
        ):
            UpsamplerLoader().load_customized(
                "/base/spatial_upsampler", server_args, "spatial_upsampler"
            )

        resolve_weights.assert_called_once_with(
            "/base/spatial_upsampler", server_args, "spatial_upsampler"
        )
        find_weights.assert_called_once_with("/cache/upsampler.safetensors")


if __name__ == "__main__":
    unittest.main()
