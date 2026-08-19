import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import transformers
from torch import nn

from sglang.multimodal_gen.runtime.layers.linear import LinearBase
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import Fp8Config
from sglang.multimodal_gen.runtime.loader.component_loaders.component_loader import (
    ComponentCheckpointUnsupportedError,
)
from sglang.multimodal_gen.runtime.loader.component_loaders.text_encoder_loader import (
    TextEncoderLoader,
    _configure_encoder_quantization,
    _process_quantized_encoder_weights,
)
from sglang.multimodal_gen.runtime.models.encoders.base import (
    CheckpointQuantizationCapability,
    TextEncoder,
)
from sglang.multimodal_gen.runtime.models.encoders.minimax_h3_qwen3vl import (
    MiniMaxH3Qwen3VLEncoder,
)


class TestTextEncoderClassResolution(unittest.TestCase):
    """load_native must not load encoder-decoder text encoders via AutoModel.

    AutoModel maps T5/UMT5 model types to the full seq2seq class
    (T5Model/UMT5Model), whose forward needs decoder inputs and raises when the
    module is used purely as a text encoder.
    """

    server_args = SimpleNamespace(trust_remote_code=False, revision=None)

    def _resolve(self, is_encoder_decoder, architectures):
        config = SimpleNamespace(
            is_encoder_decoder=is_encoder_decoder, architectures=architectures
        )
        with mock.patch.object(
            transformers.AutoConfig, "from_pretrained", return_value=config
        ):
            return TextEncoderLoader._resolve_transformers_text_encoder_class(
                "dummy/path", self.server_args
            )

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

    def test_unknown_architecture_falls_back_to_automodel(self):
        self.assertIs(self._resolve(True, ["NotARealClass"]), transformers.AutoModel)

    def test_config_load_failure_falls_back_to_automodel(self):
        with mock.patch.object(
            transformers.AutoConfig,
            "from_pretrained",
            side_effect=OSError("no config"),
        ):
            cls = TextEncoderLoader._resolve_transformers_text_encoder_class(
                "dummy/path", self.server_args
            )
        self.assertIs(cls, transformers.AutoModel)


class TestMiniMaxH3CheckpointFilter(unittest.TestCase):
    def test_only_known_unconsumed_weights_are_filtered(self):
        should_load = MiniMaxH3Qwen3VLEncoder.should_materialize_checkpoint_weight
        expected = {
            "model.language_model.layers.49.self_attn.q_proj.weight": True,
            "model.language_model.layers.50.self_attn.q_proj.weight": False,
            "model.language_model.layers.63.mlp.down_proj.weight": False,
            "model.language_model.norm.weight": False,
            "lm_head.weight": False,
            "model.language_model.rotary_emb.inv_freq": False,
            "model.visual.blocks.0.attn.qkv.weight": True,
            "language_model.layers.63.mlp.down_proj.weight": True,
            "module.model.language_model.layers.63.mlp.down_proj.weight": True,
        }
        self.assertEqual(
            {name: should_load(name) for name in expected},
            expected,
        )

    def test_vision_qkv_checkpoint_name_maps_to_native_projection(self):
        encoder = MiniMaxH3Qwen3VLEncoder.__new__(MiniMaxH3Qwen3VLEncoder)
        torch.nn.Module.__init__(encoder)
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

    def test_serialized_fp8_checkpoint_configures_h3_encoder(self):
        model_config = SimpleNamespace(quant_config=None)
        _configure_encoder_quantization(
            model_config,
            MiniMaxH3Qwen3VLEncoder,
            {},
            "/model/text_encoder",
            "text_encoder",
        )
        self.assertIs(model_config.quant_config, self.serialized)

    def test_encoder_class_must_opt_in(self):
        model_config = SimpleNamespace(quant_config=None)
        with self.assertRaisesRegex(
            ComponentCheckpointUnsupportedError, "does not support"
        ):
            _configure_encoder_quantization(
                model_config,
                TextEncoder,
                {},
                "/model/text_encoder",
                "text_encoder",
            )

    def test_srt_backend_is_not_admitted_without_an_adapter(self):
        model_config = SimpleNamespace(quant_config=None)
        capability = CheckpointQuantizationCapability(
            backend="srt",
            methods=frozenset({"fp8"}),
        )
        with mock.patch.object(
            MiniMaxH3Qwen3VLEncoder,
            "checkpoint_quantization_capability",
            capability,
        ), self.assertRaisesRegex(
            ComponentCheckpointUnsupportedError,
            "'srt'.*only the 'diffusion' backend",
        ):
            _configure_encoder_quantization(
                model_config,
                MiniMaxH3Qwen3VLEncoder,
                {},
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
                {},
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
