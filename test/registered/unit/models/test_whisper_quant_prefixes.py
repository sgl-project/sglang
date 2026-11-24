"""Whisper passes checkpoint-aligned prefixes to its quantized linears.

Regression: fc1/fc2 received quant_config but no prefix, so a compressed-tensors
`ignore` entry naming them could not match. An MLP the checkpoint keeps in floating
point was then built with int8 parameters, and load_weights failed on its float weight
("Unsupported copy between dtypes: target.dtype=torch.int8").
"""

import unittest
from unittest import mock

import torch
from transformers import WhisperConfig

from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsConfig,
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.models.whisper import WhisperForConditionalGeneration
from sglang.srt.runtime_context import SpawnRanks, get_parallel, publish, reset_context
from sglang.srt.server_args import ServerArgs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")

D_MODEL, FFN = 16, 32


def _w8a8_config(ignore):
    return CompressedTensorsConfig.from_config(
        {
            "quant_method": "compressed-tensors",
            "format": "int-quantized",
            "ignore": ignore,
            # the model loader injects the model class's mapping the same way
            "packed_modules_mapping": WhisperForConditionalGeneration.packed_modules_mapping,
            "config_groups": {
                "group_0": {
                    "targets": ["Linear"],
                    "weights": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "channel",
                        "symmetric": True,
                        "dynamic": False,
                    },
                    "input_activations": {
                        "num_bits": 8,
                        "type": "int",
                        "strategy": "token",
                        "symmetric": True,
                        "dynamic": True,
                    },
                }
            },
        }
    )


class TestWhisperQuantPrefixes(CustomTestCase):
    def setUp(self):
        reset_context()
        self.addCleanup(reset_context)
        publish(
            ServerArgs(model_path="dummy", device="cpu", tp_size=1),
            role="test",
            ranks=SpawnRanks(world_rank=0),
        )
        get_parallel().override_permanently(tp_group=None, attn_tp_group=None)
        hf_config = WhisperConfig(
            vocab_size=64,
            num_mel_bins=8,
            d_model=D_MODEL,
            encoder_layers=1,
            decoder_layers=1,
            encoder_attention_heads=2,
            decoder_attention_heads=2,
            encoder_ffn_dim=FFN,
            decoder_ffn_dim=FFN,
            max_source_positions=8,
            max_target_positions=8,
            pad_token_id=1,  # special-token ids must fit the tiny vocabulary
            bos_token_id=2,
            eos_token_id=3,
            decoder_start_token_id=2,
        )
        quant_config = _w8a8_config(
            ignore=[
                "model.encoder.layers.0.fc1",  # exact checkpoint name, unfused MLP
                r"re:.*decoder\.layers\.0\.encoder_attn\.[kv]_proj$",  # fused K/V shards
            ]
        )
        # The device capability query is a hardware boundary; this test is about names.
        with mock.patch.object(
            CompressedTensorsConfig, "_check_scheme_supported", return_value=True
        ):
            self.model = WhisperForConditionalGeneration(hf_config, quant_config)

    def test_ignore_entries_match_checkpoint_names(self):
        enc, dec = self.model.encoder.layers[0], self.model.decoder.layers[0]
        self.assertIsInstance(enc.fc1.quant_method, UnquantizedLinearMethod)
        self.assertIsInstance(
            dec.encoder_attn.kv_proj.quant_method, UnquantizedLinearMethod
        )
        for quantized in (
            enc.fc2,
            enc.self_attn.qkv_proj,
            dec.encoder_attn.q_proj,
            dec.fc1,
            dec.fc2,
        ):
            self.assertIsInstance(quantized.quant_method, CompressedTensorsLinearMethod)

    def test_load_weights_keeps_ignored_mlp_float_and_loads_scales(self):
        torch.manual_seed(0)
        fc1_w = torch.randn(FFN, D_MODEL)
        fc2_w = torch.randint(-127, 128, (D_MODEL, FFN), dtype=torch.int8)
        fc2_scale = torch.rand(D_MODEL, 1)
        self.model.load_weights(
            {
                "model.decoder.embed_tokens.weight": torch.randn(64, D_MODEL),
                "model.encoder.layers.0.fc1.weight": fc1_w,
                "model.encoder.layers.0.fc2.weight": fc2_w,
                "model.encoder.layers.0.fc2.weight_scale": fc2_scale,
            }.items()
        )
        enc = self.model.encoder.layers[0]
        self.assertTrue(enc.fc1.weight.is_floating_point())
        torch.testing.assert_close(enc.fc1.weight.data.float(), fc1_w, rtol=0, atol=0)
        torch.testing.assert_close(enc.fc2.weight.data, fc2_w, rtol=0, atol=0)
        torch.testing.assert_close(enc.fc2.weight_scale.data, fc2_scale, rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
