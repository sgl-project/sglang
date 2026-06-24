"""Regression tests for Ministral3 Llama base-class initialization."""

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=4, suite="base-a-test-cpu")

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import sglang.srt.models.ministral3 as ministral3_module
from sglang.srt.models.ministral3 import Ministral3Attention, Ministral3DecoderLayer
from sglang.test.test_utils import CustomTestCase


def _make_config():
    return SimpleNamespace(
        hidden_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        rope_parameters={"rope_theta": 1000000.0, "llama_4_scaling_beta": 8.0},
        original_max_position_embeddings=16384,
        attention_bias=False,
        bias=False,
    )


class TestMinistral3Init(CustomTestCase):
    def test_attention_preserves_llama_init_arguments(self):
        captured = {}
        quant_config = object()

        def fake_llama_attention_init(self, *args, **kwargs):
            torch.nn.Module.__init__(self)
            captured["args"] = args
            captured["kwargs"] = kwargs

        with patch.object(
            ministral3_module.LlamaAttention,
            "__init__",
            fake_llama_attention_init,
        ):
            Ministral3Attention(
                config=_make_config(),
                hidden_size=128,
                num_heads=4,
                num_kv_heads=2,
                layer_id=3,
                start_layer=2,
                rope_theta=1000000.0,
                rope_scaling={"rope_type": "llama3"},
                rope_is_neox_style=True,
                max_position_embeddings=16384,
                quant_config=quant_config,
                prefix="model.layers.3.self_attn",
                bias=True,
            )

        self.assertEqual(captured["args"], ())
        self.assertEqual(captured["kwargs"]["layer_id"], 3)
        self.assertEqual(captured["kwargs"]["start_layer"], 2)
        self.assertEqual(captured["kwargs"]["rope_theta"], 1000000.0)
        self.assertEqual(captured["kwargs"]["rope_scaling"], {"rope_type": "llama3"})
        self.assertEqual(captured["kwargs"]["max_position_embeddings"], 16384)
        self.assertIs(captured["kwargs"]["quant_config"], quant_config)
        self.assertEqual(captured["kwargs"]["prefix"], "model.layers.3.self_attn")
        self.assertTrue(captured["kwargs"]["bias"])

    def test_decoder_layer_preserves_llama_init_arguments(self):
        captured = {}
        quant_config = object()

        def fake_llama_decoder_layer_init(self, *args, **kwargs):
            torch.nn.Module.__init__(self)
            self.hidden_size = kwargs["config"].hidden_size
            captured["args"] = args
            captured["kwargs"] = kwargs

        with (
            patch.object(
                ministral3_module.LlamaDecoderLayer,
                "__init__",
                fake_llama_decoder_layer_init,
            ),
            patch.object(
                ministral3_module,
                "Ministral3Attention",
                return_value=torch.nn.Identity(),
            ) as mock_attention,
        ):
            Ministral3DecoderLayer(
                config=_make_config(),
                layer_id=5,
                start_layer=4,
                quant_config=quant_config,
                prefix="model.layers.5",
            )

        self.assertEqual(captured["args"], ())
        self.assertEqual(captured["kwargs"]["layer_id"], 5)
        self.assertEqual(captured["kwargs"]["start_layer"], 4)
        self.assertIs(captured["kwargs"]["quant_config"], quant_config)
        self.assertEqual(captured["kwargs"]["prefix"], "model.layers.5")

        attention_kwargs = mock_attention.call_args.kwargs
        self.assertEqual(attention_kwargs["layer_id"], 5)
        self.assertEqual(attention_kwargs["start_layer"], 4)
        self.assertIs(attention_kwargs["quant_config"], quant_config)
        self.assertEqual(attention_kwargs["prefix"], "model.layers.5.self_attn")


if __name__ == "__main__":
    unittest.main()
