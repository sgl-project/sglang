import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch import nn

from sglang.multimodal_gen.configs.models.encoders.clip import CLIPVisionConfig
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
    NativeComponentLoaderRequired,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.image_encoder_loader import (
    ImageEncoderLoader,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    _configure_encoder_quantization,
)
from sglang.multimodal_gen.runtime.managers.memory_managers.component_residency import (
    COMPONENT_OFFLOAD,
)
from sglang.multimodal_gen.runtime.models.encoders.clip import CLIPVisionModel
from sglang.srt.layers.quantization.fp8 import Fp8Config as SRTFp8Config


class TestImageEncoderQuantizationAdmission(unittest.TestCase):
    def setUp(self):
        self.loader = ImageEncoderLoader()
        load_native_patcher = mock.patch.object(
            self.loader, "load_native", return_value=object()
        )
        self.load_native = load_native_patcher.start()
        self.addCleanup(load_native_patcher.stop)
        self.server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(
                image_encoder_config=CLIPVisionConfig(),
                image_encoder_precision="bf16",
                native_only_components=(),
            ),
            component_weights_paths={},
            component_quantizations={},
            encoder_parallel="replicate",
            resolve_component_attention_backend=lambda _name: (None, None),
            should_direct_gpu_weight_load_component=lambda _name: False,
            should_use_fsdp_for_component=lambda _name: False,
        )

    def _component_config(self, architecture, *, quantized):
        config = {"architectures": [architecture]}
        if quantized:
            config["quantization_config"] = {
                "quant_method": "fp8",
                "activation_scheme": "dynamic",
            }
        return config

    def _config_patch(self, config):
        return mock.patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "image_encoder_loader.get_diffusers_component_config",
            return_value=config,
        )

    def _load(self):
        return self.loader.load(
            "/model/image_encoder", self.server_args, "image_encoder", "transformers"
        )

    def test_clip_serialized_fp8_uses_srt_quantization(self):
        encoder_config = CLIPVisionConfig()
        _configure_encoder_quantization(
            encoder_config,
            CLIPVisionModel,
            self._component_config("CLIPVisionModelWithProjection", quantized=True),
            "/model/image_encoder",
            "/model/image_encoder",
            "image_encoder",
        )

        self.assertIsInstance(encoder_config.quant_config, SRTFp8Config)
        self.assertEqual(
            encoder_config.quant_config.packed_modules_mapping,
            {"qkv_proj": ["q_proj", "k_proj", "v_proj"]},
        )

    def test_unknown_transformers_quantized_architecture_falls_back(self):
        config = self._component_config("UnknownVisionModel", quantized=True)
        with self._config_patch(config):
            self._load()
        self.load_native.assert_called_once()

    def test_native_only_quantized_architecture_does_not_fall_back(self):
        self.server_args.pipeline_config.native_only_components = ("image_encoder",)
        config = self._component_config("UnknownVisionModel", quantized=True)
        with self._config_patch(config), self.assertRaises(
            NativeComponentLoaderRequired
        ):
            self._load()
        self.load_native.assert_not_called()

    def test_unknown_unsupported_quantized_architecture_does_not_fall_back(self):
        config = {
            "architectures": ["UnknownVisionModel"],
            "quantization_config": {"quant_method": "not-a-format"},
        }
        with self._config_patch(config), self.assertRaises(
            ComponentCheckpointUnsupportedError
        ):
            self._load()
        self.load_native.assert_not_called()

    def test_unknown_unquantized_architecture_keeps_native_fallback(self):
        config = self._component_config("UnknownVisionModel", quantized=False)
        with self._config_patch(config):
            self._load()
        self.load_native.assert_called_once()


class TestImageEncoderNativeLoading(unittest.TestCase):
    def test_bnb4_uses_shared_transformers_path_and_image_precision(self):
        component_config = SimpleNamespace(
            is_encoder_decoder=False,
            architectures=["CLIPVisionModelWithProjection"],
            quantization_config={
                "load_in_4bit": True,
                "quant_method": "bitsandbytes",
            },
        )
        loaded_encoder = object()
        model_class = SimpleNamespace(
            from_pretrained=mock.Mock(return_value=loaded_encoder)
        )
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(image_encoder_precision="bf16"),
            explicit_residency_mode=mock.Mock(return_value=None),
            require_component_resident=mock.Mock(),
            should_use_fsdp_for_component=mock.Mock(return_value=False),
            revision=None,
            trust_remote_code=False,
        )
        loader = ImageEncoderLoader()

        with mock.patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "component_loader.get_hf_config",
            return_value=component_config,
        ), mock.patch.object(
            loader,
            "resolve_native_transformers_model_class",
            return_value=model_class,
        ), mock.patch.object(
            loader,
            "target_device",
            return_value=torch.device("cuda:0"),
        ):
            component = loader.load_native(
                "/model/image_encoder",
                server_args,
                "transformers",
                "image_encoder",
            )

        self.assertIs(component, loaded_encoder)
        server_args.require_component_resident.assert_called_once_with(
            "image_encoder",
            feature_name="Transformers quantized component",
        )
        model_class.from_pretrained.assert_called_once_with(
            "/model/image_encoder",
            config=component_config,
            trust_remote_code=False,
            revision=None,
            torch_dtype=torch.bfloat16,
            device_map={"": torch.device("cuda:0")},
        )

    def test_explicit_offload_is_rejected_before_transformers_load(self):
        component_config = SimpleNamespace(
            is_encoder_decoder=False,
            architectures=["ThirdPartyVisionModel"],
            quantization_config={"quant_method": "fp8"},
        )
        model_class = SimpleNamespace(from_pretrained=mock.Mock())
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(image_encoder_precision="bf16"),
            explicit_residency_mode=mock.Mock(return_value=COMPONENT_OFFLOAD),
            require_component_resident=mock.Mock(),
            should_use_fsdp_for_component=mock.Mock(return_value=False),
            revision=None,
            trust_remote_code=False,
        )
        loader = ImageEncoderLoader()

        with mock.patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "component_loader.get_hf_config",
            return_value=component_config,
        ), mock.patch.object(
            loader,
            "resolve_native_transformers_model_class",
            return_value=model_class,
        ), self.assertRaisesRegex(
            ComponentCheckpointUnsupportedError, "requires resident placement"
        ):
            loader.load_native(
                "/model/image_encoder",
                server_args,
                "transformers",
                "image_encoder",
            )

        model_class.from_pretrained.assert_not_called()
        server_args.require_component_resident.assert_not_called()

    def test_full_loader_does_not_move_quantized_component_again(self):
        class RejectMoveModule(nn.Module):
            def to(self, *args, **kwargs):
                raise AssertionError("quantized component must not be moved again")

        component_config = SimpleNamespace(
            is_encoder_decoder=False,
            architectures=["ThirdPartyVisionModel"],
            quantization_config={"quant_method": "fp8"},
        )
        loaded_encoder = RejectMoveModule()
        model_class = SimpleNamespace(
            from_pretrained=mock.Mock(return_value=loaded_encoder)
        )
        server_args = SimpleNamespace(
            component_quantizations={},
            pipeline_config=SimpleNamespace(
                image_encoder_precision="bf16",
                native_only_components=(),
            ),
            resolve_component_attention_backend=lambda _name: (None, None),
            explicit_residency_mode=lambda _name: None,
            require_component_resident=mock.Mock(),
            should_use_fsdp_for_component=lambda _name: False,
            should_start_component_on_cpu=lambda _name: False,
            should_direct_gpu_weight_load_component=lambda _name: False,
            revision=None,
            trust_remote_code=False,
        )
        loader = ImageEncoderLoader()

        with mock.patch.object(
            loader,
            "load_customized",
            side_effect=NativeComponentLoaderRequired("use Transformers"),
        ), mock.patch.object(
            loader,
            "resolve_native_transformers_model_class",
            return_value=model_class,
        ), mock.patch.object(
            loader,
            "target_device",
            return_value=torch.device("cuda:0"),
        ), mock.patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "component_loader.get_hf_config",
            return_value=component_config,
        ), mock.patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "component_loader.current_platform.get_available_gpu_memory",
            return_value=10.0,
        ), mock.patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "component_loader.get_memory_usage_of_component",
            return_value=0.0,
        ), mock.patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "component_loader.format_component_residency",
            return_value="resident",
        ):
            component, _ = loader.load(
                "/model/image_encoder",
                server_args,
                "image_encoder",
                "transformers",
            )

        self.assertIs(component, loaded_encoder)
