"""CPU tests: Whisper's encoder and decoder layers build ``fc1``/``fc2`` with the
quantization config, as they build the attention projections.

A checkpoint quantized on every Linear layer (for example W4A16 with
``weight_packed``/``weight_scale``/``weight_shape``) stores ``fc1``/``fc2``
quantized. Built without the config, they have only ``weight`` and ``bias``, so
the stored tensors have no parameter, are skipped, and the layers keep their
initial weights.

The tests replace the parallel layers with recorders, so they only check what
each layer is built with and need no distributed setup.
"""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

import unittest
from unittest import mock

import torch
from transformers import WhisperConfig

from sglang.srt.models import whisper
from sglang.test.test_utils import CustomTestCase


class _Recorder(torch.nn.Module):
    """Stands in for a parallel Linear or attention layer and keeps its config."""

    def __init__(self, *args, quant_config=None, **kwargs):
        super().__init__()
        self.quant_config = quant_config


def _config():
    return WhisperConfig(
        d_model=8,
        encoder_layers=1,
        decoder_layers=1,
        encoder_attention_heads=2,
        decoder_attention_heads=2,
        encoder_ffn_dim=16,
        decoder_ffn_dim=16,
    )


class TestWhisperFfnQuantConfig(CustomTestCase):
    def _build(self, layer_cls, quant_config):
        with (
            mock.patch.object(whisper, "ColumnParallelLinear", _Recorder),
            mock.patch.object(whisper, "RowParallelLinear", _Recorder),
            mock.patch.object(whisper, "WhisperAttention", _Recorder),
            mock.patch.object(
                whisper, "get_act_fn", lambda *args, **kwargs: torch.nn.GELU()
            ),
        ):
            return layer_cls(_config(), layer_id=0, quant_config=quant_config)

    def test_ffn_gets_the_quant_config(self):
        quant_config = object()
        for layer_cls in (whisper.WhisperEncoderLayer, whisper.WhisperDecoderLayer):
            with self.subTest(layer=layer_cls.__name__):
                layer = self._build(layer_cls, quant_config)
                self.assertIs(layer.fc1.quant_config, quant_config)
                self.assertIs(layer.fc2.quant_config, quant_config)
                # The attention projections of the same layer get it too.
                self.assertIs(layer.self_attn.quant_config, quant_config)

    def test_unquantized_model_is_unchanged(self):
        for layer_cls in (whisper.WhisperEncoderLayer, whisper.WhisperDecoderLayer):
            with self.subTest(layer=layer_cls.__name__):
                layer = self._build(layer_cls, None)
                self.assertIsNone(layer.fc1.quant_config)
                self.assertIsNone(layer.fc2.quant_config)


if __name__ == "__main__":
    unittest.main()
