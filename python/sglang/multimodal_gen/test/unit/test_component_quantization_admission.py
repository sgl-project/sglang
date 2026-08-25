# SPDX-License-Identifier: Apache-2.0

import re
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from sglang.multimodal_gen.runtime.loader.component_loaders.adapter_loader import (
    AdapterLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.bridge_loader import (
    BridgeLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
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
from sglang.multimodal_gen.runtime.loader.component_loaders.vocoder_loader import (
    VocoderLoader,
)


class _TestLoader(PlainStateDictComponentLoader):
    pass


class TestComponentQuantizationAdmission(unittest.TestCase):
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
                "adapter_loader.ModelRegistry.resolve_model_cls"
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
                "/model/spatial_upsampler", None, "spatial_upsampler"
            )

        load_weights.assert_not_called()


if __name__ == "__main__":
    unittest.main()
