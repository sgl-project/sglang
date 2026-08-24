import json
import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import transformers
from safetensors.torch import save_file
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import LinearBase
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_int8_config import (
    KitchenInt8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_w4a8_config import (
    KitchenW4A8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
    NativeComponentLoaderRequired,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    TextEncoderLoader,
    _configure_encoder_quantization,
    _process_quantized_encoder_weights,
    _require_quantized_encoder_layers,
    _resolve_and_configure_encoder_quantization,
)
from sglang.multimodal_gen.runtime.models.encoders.base import TextEncoder
from sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl import (
    MiniMaxH3ConditioningProjection,
    MiniMaxH3Qwen3VLEncoder,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import Qwen3VLTextModel


class TestTextEncoderClassResolution(unittest.TestCase):
    """load_native must not load encoder-decoder text encoders via AutoModel.

    AutoModel maps T5/UMT5 model types to the full seq2seq class
    (T5Model/UMT5Model), whose forward needs decoder inputs and raises when the
    module is used purely as a text encoder.
    """

    def _resolve(self, is_encoder_decoder, architectures):
        config = SimpleNamespace(
            is_encoder_decoder=is_encoder_decoder, architectures=architectures
        )
        return TextEncoderLoader().resolve_native_transformers_model_class(config)

    def test_umt5_encoder_decoder_uses_encoder_only_class(self):
        self.assertIs(
            self._resolve(True, ["UMT5EncoderModel"]), transformers.UMT5EncoderModel
        )
        self.assertIs(self._resolve(True, ["UMT5Model"]), transformers.UMT5EncoderModel)
        self.assertIs(
            self._resolve(True, ["UMT5ForConditionalGeneration"]),
            transformers.UMT5EncoderModel,
        )

    def test_t5_encoder_decoder_uses_encoder_only_class(self):
        self.assertIs(
            self._resolve(True, ["T5EncoderModel"]), transformers.T5EncoderModel
        )
        self.assertIs(self._resolve(True, ["T5Model"]), transformers.T5EncoderModel)
        self.assertIs(
            self._resolve(True, ["T5ForConditionalGeneration"]),
            transformers.T5EncoderModel,
        )

    def test_mt5_encoder_decoder_uses_encoder_only_class(self):
        self.assertIs(
            self._resolve(True, ["MT5EncoderModel"]), transformers.MT5EncoderModel
        )
        self.assertIs(self._resolve(True, ["MT5Model"]), transformers.MT5EncoderModel)
        self.assertIs(
            self._resolve(True, ["MT5ForConditionalGeneration"]),
            transformers.MT5EncoderModel,
        )

    def test_non_encoder_decoder_keeps_automodel(self):
        # e.g. CLIP/Mistral/Qwen text encoders are not encoder-decoder.
        self.assertIs(self._resolve(False, ["CLIPTextModel"]), transformers.AutoModel)

    def test_qwen_text_model_constructs_checkpoint_owned_embedding(self):
        config = SimpleNamespace(
            pad_token_id=0,
            vocab_size=64,
            hidden_size=256,
            num_hidden_layers=0,
            rms_norm_eps=1e-6,
        )
        quant_config = mock.Mock()
        quant_config.quantizes_embedding.return_value = True
        replacement = nn.Embedding(64, 256)
        with mock.patch(
            "sglang.multimodal_gen.runtime.models.encoders.qwen3vl."
            "VocabParallelEmbedding",
            return_value=replacement,
        ) as embedding_cls:
            model = Qwen3VLTextModel(
                config,
                quant_config=quant_config,
                use_tensor_parallel=True,
                prefix="model.language_model",
            )

        self.assertIs(model.embed_tokens, replacement)
        quant_config.quantizes_embedding.assert_called_once_with(
            "model.language_model.embed_tokens"
        )
        embedding_cls.assert_called_once_with(
            64,
            256,
            params_dtype=torch.get_default_dtype(),
            quant_config=quant_config,
            prefix="model.language_model.embed_tokens",
        )

    def test_unknown_architecture_falls_back_to_automodel(self):
        self.assertIs(self._resolve(True, ["NotARealClass"]), transformers.AutoModel)

    def test_bitsandbytes_native_load_requires_resident_encoder(self):
        loaded_encoder = nn.Linear(1, 1)
        transformers_model_class = SimpleNamespace(
            from_pretrained=mock.Mock(return_value=loaded_encoder)
        )
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(text_encoder_precisions=["bf16"]),
            require_component_resident=mock.Mock(),
            revision=None,
            trust_remote_code=False,
        )
        component_config = {
            "quantization_config": {
                "load_in_4bit": True,
                "quant_method": "bitsandbytes",
            }
        }

        with mock.patch.object(
            TextEncoderLoader,
            "resolve_native_transformers_model_class",
            return_value=transformers_model_class,
        ), mock.patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "component_loader.get_hf_config",
            return_value=component_config,
        ):
            encoder = TextEncoderLoader().load_native(
                "/model/text_encoder",
                server_args,
                "transformers",
                "text_encoder",
            )

        self.assertIs(encoder, loaded_encoder)
        server_args.require_component_resident.assert_called_once_with(
            "text_encoder",
            feature_name="Transformers bitsandbytes component",
        )
        transformers_model_class.from_pretrained.assert_called_once_with(
            "/model/text_encoder",
            config=component_config,
            trust_remote_code=False,
            revision=None,
            torch_dtype=torch.bfloat16,
        )


class TestMiniMaxH3CheckpointFilter(unittest.TestCase):
    def test_only_known_unconsumed_weights_are_filtered(self):
        encoder = MiniMaxH3Qwen3VLEncoder.__new__(MiniMaxH3Qwen3VLEncoder)
        encoder.selected_lm_layer = 50
        should_load = encoder.should_materialize_checkpoint_weight
        expected = {
            "model.language_model.layers.49.self_attn.q_proj.weight": True,
            "model.language_model.layers.50.self_attn.q_proj.weight": False,
            "model.language_model.layers.63.mlp.down_proj.weight": False,
            "model.language_model.norm.weight": False,
            "lm_head.weight": False,
            "model.language_model.rotary_emb.inv_freq": False,
            "model.visual.blocks.0.attn.qkv.weight": True,
            "model.layers.49.self_attn.q_proj.weight": True,
            "model.layers.50.self_attn.q_proj.weight": False,
            "visual.blocks.0.attn.qkv.weight": True,
            "language_model.layers.63.mlp.down_proj.weight": True,
            "module.model.language_model.layers.63.mlp.down_proj.weight": True,
        }
        self.assertEqual(
            {name: should_load(name) for name in expected},
            expected,
        )
        encoder.selected_lm_layer = 24
        self.assertTrue(
            should_load("model.language_model.layers.23.mlp.down_proj.weight")
        )
        self.assertFalse(
            should_load("model.language_model.layers.24.mlp.down_proj.weight")
        )

    def test_vision_qkv_checkpoint_name_maps_to_native_projection(self):
        encoder = MiniMaxH3Qwen3VLEncoder.__new__(MiniMaxH3Qwen3VLEncoder)
        torch.nn.Module.__init__(encoder)
        encoder.selected_lm_layer = 50
        encoder.model = torch.nn.Module()
        encoder.model.visual = torch.nn.Module()
        block = torch.nn.Module()
        block.attn = torch.nn.Module()
        block.attn.qkv_proj = torch.nn.Linear(2, 2)
        encoder.model.visual.blocks = torch.nn.ModuleList([block])

        loaded = encoder.load_weights(
            [("model.visual.blocks.0.attn.qkv.bias", torch.tensor([1.0, 2.0]))]
        )

        self.assertEqual(loaded, {"model.visual.blocks.0.attn.qkv_proj.bias"})
        torch.testing.assert_close(
            encoder.model.visual.blocks[0].attn.qkv_proj.bias,
            torch.tensor([1.0, 2.0]),
        )

    def test_comfy_language_checkpoint_name_maps_to_native_namespace(self):
        encoder = MiniMaxH3Qwen3VLEncoder.__new__(MiniMaxH3Qwen3VLEncoder)
        torch.nn.Module.__init__(encoder)
        encoder.selected_lm_layer = 50
        encoder.model = torch.nn.Module()
        encoder.model.language_model = torch.nn.Module()
        layer = torch.nn.Module()
        layer.self_attn = torch.nn.Module()
        layer.self_attn.q_proj = torch.nn.Linear(2, 2, bias=False)
        encoder.model.language_model.layers = torch.nn.ModuleList([layer])
        source = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        loaded = encoder.load_weights(
            [("model.layers.0.self_attn.q_proj.weight", source)]
        )

        target_name = "model.language_model.layers.0.self_attn.q_proj.weight"
        self.assertEqual(loaded, {target_name})
        torch.testing.assert_close(layer.self_attn.q_proj.weight, source)


class TestMiniMaxH3ConditioningProjection(unittest.TestCase):
    @staticmethod
    def _save_projection(path, tensors):
        save_file(tensors, path, metadata={"tap": "2"})

    def test_linear_projection_and_attention_sink(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as checkpoint:
            tensors = {
                "W": torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
                "mean_in": torch.tensor([1.0, -1.0]),
                "std_in": torch.tensor([2.0, 4.0]),
                "mean_out": torch.tensor([0.5, -0.5]),
                "std_out": torch.tensor([2.0, 3.0]),
                "sink_out": torch.tensor([9.0, 8.0]),
            }
            self._save_projection(checkpoint.name, tensors)
            projection = MiniMaxH3ConditioningProjection(checkpoint.name)
            hidden = torch.tensor([[5.0, 3.0], [3.0, -1.0]])
            expected = (hidden - tensors["mean_in"]) / tensors["std_in"]
            expected = expected @ tensors["W"]
            expected = expected * tensors["std_out"] + tensors["mean_out"]
            expected[0] = tensors["sink_out"]

            torch.testing.assert_close(projection(hidden), expected)
            self.assertEqual(projection.tap, 2)

    def test_mlp_only_projection(self):
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as checkpoint:
            tensors = {
                "mean_in": torch.zeros(2),
                "std_in": torch.ones(2),
                "mean_out": torch.tensor([0.5]),
                "std_out": torch.tensor([2.0]),
                "mlp.0.weight": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
                "mlp.0.bias": torch.tensor([0.25, -0.25]),
                "mlp.2.weight": torch.tensor([[2.0, -1.0]]),
                "mlp.2.bias": torch.tensor([0.75]),
            }
            self._save_projection(checkpoint.name, tensors)
            projection = MiniMaxH3ConditioningProjection(checkpoint.name)
            hidden = torch.tensor([[1.0, 2.0]])
            residual = torch.nn.functional.linear(
                hidden, tensors["mlp.0.weight"], tensors["mlp.0.bias"]
            )
            residual = torch.nn.functional.gelu(residual)
            residual = torch.nn.functional.linear(
                residual, tensors["mlp.2.weight"], tensors["mlp.2.bias"]
            )
            expected = residual * tensors["std_out"] + tensors["mean_out"]

            torch.testing.assert_close(projection(hidden), expected)

    def test_small_encoder_requires_matching_projection(self):
        config = SimpleNamespace(
            arch_config=SimpleNamespace(
                hidden_size=2560,
                checkpoint_num_hidden_layers=36,
            )
        )
        with self.assertRaisesRegex(ValueError, "conditioning_projection"):
            MiniMaxH3Qwen3VLEncoder.configure_component_paths(config, {})


class TestTextEncoderQuantization(unittest.TestCase):
    def setUp(self):
        serialized = Fp8Config(
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
            weight_block_size=[128, 128],
        )
        self.quant_config_patcher = mock.patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "text_encoder_loader.get_quant_config",
            return_value=serialized,
        )
        self.get_quant_config = self.quant_config_patcher.start()
        self.addCleanup(self.quant_config_patcher.stop)
        self.serialized = serialized

    def test_serialized_checkpoint_configures_native_encoder(self):
        model_config = SimpleNamespace(quant_config=None)
        _configure_encoder_quantization(
            model_config,
            TextEncoder,
            {},
            "/model/text_encoder",
            "/model/text_encoder",
            "text_encoder",
        )
        self.assertIs(model_config.quant_config, self.serialized)

    def test_weight_file_metadata_configures_native_encoder(self):
        model_config = SimpleNamespace(quant_config=None)
        self.get_quant_config.return_value = None
        with mock.patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "text_encoder_loader.get_quant_config_from_safetensors_metadata",
            return_value=self.serialized,
        ) as get_file_quant_config:
            _configure_encoder_quantization(
                model_config,
                TextEncoder,
                {},
                "/model/text_encoder",
                "/weights/encoder.safetensors",
                "text_encoder",
            )

        self.assertIs(model_config.quant_config, self.serialized)
        get_file_quant_config.assert_called_once_with("/weights/encoder.safetensors")

    def test_comfy_int8_weight_file_configures_native_encoder(self):
        self.get_quant_config.return_value = None
        marker = json.dumps(
            {
                "format": "int8_tensorwise",
                "convrot": True,
                "convrot_groupsize": 256,
            }
        ).encode()
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as checkpoint:
            save_file(
                {
                    "model.layers.0.self_attn.q_proj.weight": torch.ones(
                        (2, 256), dtype=torch.int8
                    ),
                    "model.layers.0.self_attn.q_proj.weight_scale": torch.ones((2, 1)),
                    "model.layers.0.self_attn.q_proj.comfy_quant": torch.tensor(
                        list(marker), dtype=torch.uint8
                    ),
                },
                checkpoint.name,
            )
            model_config = SimpleNamespace(quant_config=None)
            with mock.patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "text_encoder_loader.get_quant_config_from_safetensors_metadata",
                return_value=None,
            ):
                _configure_encoder_quantization(
                    model_config,
                    MiniMaxH3Qwen3VLEncoder,
                    {},
                    "/model/text_encoder",
                    checkpoint.name,
                    "text_encoder",
                )

        self.assertIsInstance(model_config.quant_config, KitchenInt8Config)
        self.assertEqual(
            set(model_config.quant_config.layer_markers),
            {"model.language_model.layers.0.self_attn.q_proj"},
        )

    def test_mixed_w4a8_weight_file_maps_embedding_and_linear_markers(self):
        self.get_quant_config.return_value = None
        layers = {
            "model.embed_tokens": {"format": "int8_tensorwise"},
            "model.layers.0.mlp.down_proj": {
                "format": "asym_w4a8_int8",
                "convrot": True,
                "group_size": 16,
                "convrot_groupsize": 256,
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as checkpoint:
            save_file(
                {
                    "model.embed_tokens.weight": torch.ones((4, 256), dtype=torch.int8),
                    "model.embed_tokens.weight_scale": torch.tensor(0.25),
                    "model.layers.0.mlp.down_proj.weight": torch.ones(
                        (2, 128), dtype=torch.int8
                    ),
                    "model.layers.0.mlp.down_proj.weight_s_rel": torch.ones(
                        (2, 16), dtype=torch.float8_e4m3fn
                    ),
                    "model.layers.0.mlp.down_proj.weight_s_channel": torch.ones(2),
                    "model.layers.0.mlp.down_proj.weight_codebook": torch.ones(16),
                },
                checkpoint.name,
                metadata={"_quantization_metadata": json.dumps({"layers": layers})},
            )
            model_config = SimpleNamespace(quant_config=None)
            with mock.patch(
                "sglang.multimodal_gen.runtime.loader.component_loaders."
                "text_encoder_loader.get_quant_config_from_safetensors_metadata",
                return_value=None,
            ):
                _configure_encoder_quantization(
                    model_config,
                    MiniMaxH3Qwen3VLEncoder,
                    {},
                    "/model/text_encoder",
                    checkpoint.name,
                    "text_encoder",
                )

        self.assertIsInstance(model_config.quant_config, KitchenW4A8Config)
        self.assertTrue(
            model_config.quant_config.quantizes_embedding(
                "model.language_model.embed_tokens"
            )
        )
        self.assertIn(
            "model.language_model.layers.0.mlp.down_proj",
            model_config.quant_config.layer_markers,
        )

    def test_encoder_must_use_native_loader(self):
        model_config = SimpleNamespace(quant_config=None)
        with self.assertRaisesRegex(
            ComponentCheckpointUnsupportedError, "requires an in-tree native encoder"
        ):
            _configure_encoder_quantization(
                model_config,
                nn.Module,
                {},
                "/model/text_encoder",
                "/model/text_encoder",
                "text_encoder",
            )

    def test_standard_bitsandbytes_delegates_to_transformers(self):
        component_config = {
            "quantization_config": {
                "load_in_4bit": True,
                "quant_method": "bitsandbytes",
            }
        }

        for architecture in (
            "T5EncoderModel",
            "CLIPTextModel",
            "ThirdPartyTextEncoder",
        ):
            with self.subTest(architecture=architecture), self.assertRaisesRegex(
                NativeComponentLoaderRequired,
                "delegates serialized bitsandbytes checkpoint loading to Transformers",
            ):
                _resolve_and_configure_encoder_quantization(
                    SimpleNamespace(architectures=[architecture], quant_config=None),
                    component_config,
                    "/model/text_encoder",
                    "/model/text_encoder",
                    "text_encoder",
                )
        self.get_quant_config.assert_not_called()

    def test_rejects_nonstandard_bitsandbytes_metadata_location(self):
        with self.assertRaisesRegex(
            ComponentCheckpointUnsupportedError,
            "requires a top-level quantization_config",
        ):
            _configure_encoder_quantization(
                SimpleNamespace(quant_config=None),
                TextEncoder,
                {
                    "compression_config": {
                        "load_in_4bit": True,
                        "quant_method": "bitsandbytes",
                    }
                },
                "/model/text_encoder",
                "/model/text_encoder",
                "text_encoder",
            )

    def test_rejects_bitsandbytes_8bit(self):
        with self.assertRaisesRegex(
            ComponentCheckpointUnsupportedError,
            "supports only serialized BitsAndBytes 4-bit checkpoints",
        ):
            _resolve_and_configure_encoder_quantization(
                SimpleNamespace(
                    architectures=["ThirdPartyTextEncoder"], quant_config=None
                ),
                {
                    "quantization_config": {
                        "load_in_4bit": False,
                        "load_in_8bit": True,
                        "quant_method": "bitsandbytes",
                    }
                },
                "/model/text_encoder",
                "/model/text_encoder",
                "text_encoder",
            )

    def test_model_managed_quantization_bypasses_generic_lifecycle(self):
        model_config = SimpleNamespace(quant_config=None)
        with mock.patch.object(
            TextEncoder,
            "manages_checkpoint_quantization",
            True,
        ):
            _configure_encoder_quantization(
                model_config,
                TextEncoder,
                {
                    "quantization_config": {
                        "load_in_4bit": True,
                        "quant_method": "bitsandbytes",
                    }
                },
                "/model/text_encoder",
                "/model/text_encoder",
                "text_encoder",
            )

        self.assertIsNone(model_config.quant_config)
        self.get_quant_config.assert_not_called()


class _RecordingQuantMethod:
    def __init__(self, *, error: Exception | None = None):
        self.error = error
        self.devices = []

    def process_weights_after_loading(self, layer):
        self.devices.append(layer.weight.device)
        if self.error is not None:
            raise self.error


class _QuantizedLinear(LinearBase):
    def __init__(self, quant_method):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.empty(2, 2), requires_grad=False)
        self.quant_method = quant_method


class _QuantizedEncoder(nn.Module):
    def __init__(self, quant_method):
        super().__init__()
        self.quantized = _QuantizedLinear(quant_method)
        self.unquantized = nn.Linear(2, 2, bias=False)


class TestQuantizedTextEncoderPostprocess(unittest.TestCase):
    def test_rejects_native_encoder_without_quantized_layers(self):
        with self.assertRaisesRegex(
            ComponentCheckpointUnsupportedError,
            "does not construct quantized linear layers",
        ):
            _require_quantized_encoder_layers(nn.Linear(2, 2), "text_encoder")

    def test_rejects_unconsumed_comfy_marker(self):
        config = KitchenInt8Config(
            layer_markers={
                "visual.proj": {
                    "format": "int8_tensorwise",
                    "convrot": True,
                    "convrot_groupsize": 256,
                }
            }
        )
        with self.assertRaisesRegex(
            ComponentCheckpointUnsupportedError, "did not consume"
        ):
            _require_quantized_encoder_layers(
                _QuantizedEncoder(_RecordingQuantMethod()),
                "text_encoder",
                quant_config=config,
            )

    def test_processes_quantized_layers_without_moving_the_model(self):
        quant_method = _RecordingQuantMethod()
        model = _QuantizedEncoder(quant_method)

        processed = _process_quantized_encoder_weights(
            model,
            torch.device("cpu"),
            "text_encoder",
        )

        self.assertEqual(processed, 1)
        self.assertEqual(quant_method.devices, [torch.device("cpu")])
        self.assertEqual(model.unquantized.weight.device, torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_stages_only_the_quantized_layer_and_restores_it(self):
        quant_method = _RecordingQuantMethod()
        model = _QuantizedEncoder(quant_method)

        processed = _process_quantized_encoder_weights(
            model,
            torch.device("cuda", torch.cuda.current_device()),
            "text_encoder",
        )

        self.assertEqual(processed, 1)
        self.assertEqual(quant_method.devices[0].type, "cuda")
        self.assertEqual(model.quantized.weight.device, torch.device("cpu"))
        self.assertEqual(model.unquantized.weight.device, torch.device("cpu"))

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_restores_staged_layer_when_postprocess_fails(self):
        model = _QuantizedEncoder(_RecordingQuantMethod(error=RuntimeError("boom")))

        with self.assertRaisesRegex(RuntimeError, "boom"):
            _process_quantized_encoder_weights(
                model,
                torch.device("cuda", torch.cuda.current_device()),
                "text_encoder",
            )

        self.assertEqual(model.quantized.weight.device, torch.device("cpu"))
        self.assertEqual(model.unquantized.weight.device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
