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
from sglang.multimodal_gen.runtime.layers.quantization.comfy_nvfp4 import (
    ComfyFullPrecisionNvfp4LinearMethod,
    ComfyNvfp4Config,
    ComfyRowwiseInt8EmbeddingMethod,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_int8_config import (
    KitchenInt8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_w4a4_config import (
    KitchenW4A4Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.configs.kitchen_w4a8_config import (
    KitchenW4A8Config,
)
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from sglang.multimodal_gen.runtime.layers.quantization.gguf import GGUFConfig
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
    NativeComponentLoaderRequired,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    TextEncoderLoader,
    _configure_encoder_quantization,
    _get_encoder_quant_config,
    _process_quantized_encoder_weights,
    _require_quantized_encoder_layers,
    _resolve_and_configure_encoder_quantization,
)
from sglang.multimodal_gen.runtime.loader.gguf_weights import GGUFTensorMeta
from sglang.multimodal_gen.runtime.models.encoders.base import (
    EncoderTensorParallelMixin,
    TextEncoder,
)
from sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl import (
    MiniMaxH3ConditioningProjection,
    MiniMaxH3Qwen3VLEncoder,
)
from sglang.multimodal_gen.runtime.models.encoders.qwen3vl import Qwen3VLTextModel
from sglang.srt.layers.linear import LinearBase as SrtLinearBase


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

    def test_tensor_parallel_encoder_keeps_checkpoint_weights_by_default(self):
        self.assertTrue(
            EncoderTensorParallelMixin.should_materialize_checkpoint_weight(
                "model.layers.0.self_attn.q_proj.weight"
            )
        )

    def test_bitsandbytes_native_load_requires_resident_encoder(self):
        loaded_encoder = nn.Linear(1, 1)
        transformers_model_class = SimpleNamespace(
            from_pretrained=mock.Mock(return_value=loaded_encoder)
        )
        server_args = SimpleNamespace(
            pipeline_config=SimpleNamespace(text_encoder_precisions=["bf16"]),
            explicit_residency_mode=mock.Mock(return_value=None),
            require_component_resident=mock.Mock(),
            should_use_fsdp_for_component=mock.Mock(return_value=False),
            revision=None,
            trust_remote_code=False,
        )
        component_config = {
            "quantization_config": {
                "load_in_4bit": True,
                "quant_method": "bitsandbytes",
            }
        }

        loader = TextEncoderLoader()
        with mock.patch.object(
            TextEncoderLoader,
            "resolve_native_transformers_model_class",
            return_value=transformers_model_class,
        ), mock.patch.object(
            loader,
            "target_device",
            return_value=torch.device("cuda:0"),
        ), mock.patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "component_loader.get_hf_config",
            return_value=component_config,
        ):
            encoder = loader.load_native(
                "/model/text_encoder",
                server_args,
                "transformers",
                "text_encoder",
            )

        self.assertIs(encoder, loaded_encoder)
        server_args.require_component_resident.assert_called_once_with(
            "text_encoder",
            feature_name="Transformers quantized component",
        )
        transformers_model_class.from_pretrained.assert_called_once_with(
            "/model/text_encoder",
            config=component_config,
            trust_remote_code=False,
            revision=None,
            torch_dtype=torch.bfloat16,
            device_map={"": torch.device("cuda:0")},
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
            "language_model.layers.63.mlp.down_proj.weight": False,
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

    def test_explicit_online_quantization_configures_native_encoder(self):
        model_config = SimpleNamespace(quant_config=None)
        self.get_quant_config.return_value = None

        _configure_encoder_quantization(
            model_config,
            TextEncoder,
            {},
            "/model/text_encoder",
            "/model/text_encoder",
            "text_encoder",
            explicit_quantization="kitchen_int8",
            ignored_layers=["lm_head"],
        )

        self.assertIsInstance(model_config.quant_config, KitchenInt8Config)
        self.assertFalse(model_config.quant_config.is_checkpoint_int8_serialized)
        self.assertEqual(model_config.quant_config.ignored_layers, ["lm_head"])

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
                    "visual.blocks.0.attn.qkv.weight": torch.ones(
                        (2, 256), dtype=torch.int8
                    ),
                    "visual.blocks.0.attn.qkv.weight_scale": torch.ones((2, 1)),
                    "visual.blocks.0.attn.qkv.comfy_quant": torch.tensor(
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
            {"model.visual.blocks.0.attn.qkv_proj"},
        )

    def test_comfy_w4a4_weight_file_configures_native_encoder(self):
        self.get_quant_config.return_value = None
        marker = json.dumps(
            {"format": "convrot_w4a4", "convrot_groupsize": 256}
        ).encode()
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as checkpoint:
            save_file(
                {
                    "model.layers.0.self_attn.q_proj.weight": torch.ones(
                        (2, 128), dtype=torch.int8
                    ),
                    "model.layers.0.self_attn.q_proj.weight_scale": torch.ones(2),
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

        self.assertIsInstance(model_config.quant_config, KitchenW4A4Config)
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

    def test_nvfp4_awq_weight_file_maps_embedding_and_linear_markers(self):
        self.get_quant_config.return_value = None
        layers = {
            "model.embed_tokens": {"format": "int8_tensorwise"},
            "model.layers.0.self_attn.o_proj": {
                "format": "nvfp4",
                "full_precision_matrix_mult": True,
            },
        }
        with tempfile.NamedTemporaryFile(suffix=".safetensors") as checkpoint:
            save_file(
                {
                    "model.embed_tokens.weight": torch.ones((4, 64), dtype=torch.int8),
                    "model.embed_tokens.weight_scale": torch.ones(4, 1),
                    "model.layers.0.self_attn.o_proj.weight": torch.full(
                        (128, 32), 0x21, dtype=torch.uint8
                    ),
                    "model.layers.0.self_attn.o_proj.weight_scale": torch.ones(
                        (128, 4), dtype=torch.float8_e4m3fn
                    ),
                    "model.layers.0.self_attn.o_proj.weight_scale_2": torch.tensor(0.5),
                    "model.layers.0.self_attn.o_proj.pre_quant_scale": torch.ones(
                        64, dtype=torch.bfloat16
                    ),
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

        self.assertIsInstance(model_config.quant_config, ComfyNvfp4Config)
        self.assertTrue(
            model_config.quant_config.quantizes_embedding(
                "model.language_model.embed_tokens"
            )
        )
        marker = model_config.quant_config.layer_markers[
            "model.language_model.layers.0.self_attn.o_proj"
        ]
        self.assertTrue(marker["_has_pre_quant_scale"])

    def test_nvfp4_awq_portable_linear_and_rowwise_embedding(self):
        config = ComfyNvfp4Config(
            {
                "proj": {
                    "format": "nvfp4",
                    "full_precision_matrix_mult": True,
                    "_has_pre_quant_scale": True,
                }
            }
        )
        method = ComfyFullPrecisionNvfp4LinearMethod(config, has_pre_quant_scale=True)
        layer = nn.Module()
        layer.weight = nn.Parameter(
            torch.full((128, 32), 0x21, dtype=torch.uint8), requires_grad=False
        )
        layer.weight_scale = nn.Parameter(
            torch.ones((128, 4), dtype=torch.float8_e4m3fn), requires_grad=False
        )
        layer.weight_scale_2 = nn.Parameter(torch.tensor(0.5), requires_grad=False)
        layer.pre_quant_scale = nn.Parameter(
            torch.full((64,), 2.0), requires_grad=False
        )
        inputs = torch.zeros(1, 64)
        inputs[:, 0::2] = 1

        output = method.apply(layer, inputs)

        torch.testing.assert_close(output, torch.full((1, 128), 32.0))

        embedding_method = ComfyRowwiseInt8EmbeddingMethod()
        embedding = nn.Module()
        embedding_method.create_weights(
            embedding,
            input_size_per_partition=2,
            output_partition_sizes=[3],
            input_size=2,
            output_size=3,
            params_dtype=torch.bfloat16,
        )
        embedding.weight.data.copy_(torch.tensor([[1, 2], [3, 4], [5, 6]]))
        embedding.weight_scale.data.copy_(torch.tensor([[0.5], [1.0], [2.0]]))
        rows = embedding_method.embedding(embedding, torch.tensor([2, 0]))
        torch.testing.assert_close(
            rows,
            torch.tensor([[10.0, 12.0], [0.5, 1.0]], dtype=torch.bfloat16),
        )

    def test_gguf_maps_h3_names_and_drops_unused_language_layers(self):
        self.get_quant_config.return_value = None

        def meta(name: str) -> GGUFTensorMeta:
            return GGUFTensorMeta(
                ggml_type=12,
                logical_shape=(2, 256),
                stored_shape=(2, 144),
                stored_dtype=torch.uint8,
                param_name=f"{name.removesuffix('.weight')}.qweight",
            )

        checkpoint_meta = {
            name: meta(name)
            for name in (
                "model.layers.49.self_attn.q_proj.weight",
                "model.layers.50.self_attn.q_proj.weight",
                "visual.blocks.0.attn.qkv.weight",
            )
        }
        with mock.patch(
            "sglang.multimodal_gen.runtime.loader.component_loaders."
            "text_encoder_loader.read_gguf_tensor_meta",
            return_value=checkpoint_meta,
        ):
            config = _get_encoder_quant_config(
                {},
                "/model/text_encoder",
                "/weights/encoder.gguf",
                MiniMaxH3Qwen3VLEncoder,
            )

        self.assertIsInstance(config, GGUFConfig)
        encoder = MiniMaxH3Qwen3VLEncoder.__new__(MiniMaxH3Qwen3VLEncoder)
        encoder.selected_lm_layer = 50
        config.retain_tensor_meta(encoder.should_materialize_checkpoint_weight)
        self.assertEqual(
            config.quantized_prefixes,
            {"model.language_model.layers.49.self_attn.q_proj"},
        )
        vision_meta = config.tensor_meta["model.visual.blocks.0.attn.qkv_proj.weight"]
        self.assertTrue(vision_meta.dequantize_on_load)
        self.assertEqual(
            vision_meta.param_name,
            "model.visual.blocks.0.attn.qkv_proj.weight",
        )
        self.assertNotIn(
            "model.language_model.layers.50.self_attn.q_proj.weight",
            config.tensor_meta,
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
                "delegates serialized quant_method='bitsandbytes' checkpoint",
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

    def test_bitsandbytes_8bit_delegates_to_transformers(self):
        with self.assertRaisesRegex(
            NativeComponentLoaderRequired,
            "delegates serialized quant_method='bitsandbytes' checkpoint",
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

    def test_unknown_fp8_architecture_delegates_to_transformers(self):
        with self.assertRaisesRegex(
            NativeComponentLoaderRequired,
            "delegates serialized quant_method='fp8' checkpoint",
        ):
            _resolve_and_configure_encoder_quantization(
                SimpleNamespace(
                    architectures=["ThirdPartyTextEncoder"], quant_config=None
                ),
                {
                    "quantization_config": {
                        "quant_method": "fp8",
                        "activation_scheme": "dynamic",
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


class _SRTQuantizedLinear(SrtLinearBase):
    def __init__(self, quant_method):
        nn.Module.__init__(self)
        self.weight = nn.Parameter(torch.empty(2, 2), requires_grad=False)
        self.quant_method = quant_method


class TestQuantizedTextEncoderPostprocess(unittest.TestCase):
    def test_processes_srt_quantized_linear(self):
        quant_method = _RecordingQuantMethod()
        model = _SRTQuantizedLinear(quant_method)

        processed = _process_quantized_encoder_weights(
            model,
            torch.device("cpu"),
            "image_encoder",
        )

        self.assertEqual(processed, 1)
        self.assertEqual(quant_method.devices, [torch.device("cpu")])

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
