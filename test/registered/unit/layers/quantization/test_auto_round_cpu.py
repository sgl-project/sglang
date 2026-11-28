"""Unit tests for AutoRound mixed-precision CPU loading."""

import unittest
from unittest.mock import patch

import torch

from sglang.srt.hardware_backend.cpu.quantization.gptq_kernels import (
    _gptq_w8_to_channelwise_w8a8,
    _unpack_gptq_uint8,
)
from sglang.srt.layers.quantization.auto_round import AutoRoundConfig
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=8, suite="base-a-test-cpu")


def _pack_gptq_uint8(values: torch.Tensor, pack_dim: int) -> torch.Tensor:
    """Pack uint8 ``values`` four-per-int32 along ``pack_dim``."""

    values = values.to(torch.int32)
    if pack_dim == 0:
        values = values.reshape(values.shape[0] // 4, 4, values.shape[1])
        packed = torch.zeros(values.shape[0], values.shape[2], dtype=torch.int32)
        chunks = [values[:, byte_index, :] for byte_index in range(4)]
    else:
        values = values.reshape(values.shape[0], values.shape[1] // 4, 4)
        packed = torch.zeros(values.shape[0], values.shape[1], dtype=torch.int32)
        chunks = [values[:, :, byte_index] for byte_index in range(4)]
    for byte_index, chunk in enumerate(chunks):
        packed |= chunk.to(torch.int32) << (8 * byte_index)
    return packed


def _make_mixed_config(**overrides):
    config = {
        "bits": 4,
        "group_size": 128,
        "sym": True,
        "packing_format": "auto_round:auto_gptq",
        "block_name_to_quantize": "model.layers",
        "packed_modules_mapping": {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
        },
        "extra_config": {
            "model.layers.0.self_attn.qkv_proj.base_layer": {"bits": 8},
            "model.layers.0.self_attn.o_proj.base_layer": {"bits": 8},
            "model.layers.0.mlp.gate_up_proj.base_layer": {"bits": 8},
            "model.layers.1.mlp.gate_up_proj.base_layer": {"bits": 8},
            "model.layers.16.mlp.gate_up_proj.base_layer": {"bits": 4},
            "model.layers.0.mlp.down_proj.base_layer": {"bits": 4},
            "model.layers.0.self_attn.qkv_proj.lora_A.vision": {"bits": 16},
        },
    }
    config.update(overrides)
    return AutoRoundConfig.from_config(config)


class TestAutoRoundMixedConfig(CustomTestCase):
    def test_mixed_checkpoint_resolves_per_layer_bits(self):
        quant_config = _make_mixed_config()
        layer = torch.nn.Module()

        expected = {
            "model.layers.0.self_attn.qkv_proj": 8,
            "model.layers.0.self_attn.o_proj": 8,
            "model.layers.0.self_attn.qkv_proj.lora_A.vision": 16,
            "model.layers.0.mlp.gate_up_proj": 8,
            "model.layers.0.mlp.down_proj": 4,
            "model.layers.16.mlp.gate_up_proj": 4,
        }
        for name, bits in expected.items():
            with self.subTest(name=name):
                self.assertEqual(quant_config.get_layer_config(layer, name)[0], bits)

    def test_checkpoint_and_runtime_prefixes_resolve_equally(self):
        quant_config = _make_mixed_config()
        layer = torch.nn.Module()
        names = (
            "model.layers.0.self_attn.qkv_proj",
            "model.layers.0.self_attn.qkv_proj.base_layer",
            "language_model.model.layers.0.self_attn.qkv_proj",
            "model.language_model.layers.0.self_attn.qkv_proj",
            "layers.0.self_attn.qkv_proj",
        )
        self.assertEqual(
            {quant_config.get_layer_config(layer, name)[0] for name in names}, {8}
        )

    def test_base_layer_regex_matches_runtime_name(self):
        for pattern in (
            r".*layers\.(2)\.mlp\.gate_up_proj\.base_layer$",
            r".*layers\.(2)\.mlp\.gate_up_proj.base_layer$",
        ):
            with self.subTest(pattern=pattern):
                quant_config = _make_mixed_config(extra_config={pattern: {"bits": 8}})
                layer = torch.nn.Module()
                self.assertEqual(
                    quant_config.get_layer_config(
                        layer, "language_model.model.layers.2.mlp.gate_up_proj"
                    )[0],
                    8,
                )
                self.assertEqual(
                    quant_config.get_layer_config(
                        layer, "language_model.model.layers.3.mlp.gate_up_proj"
                    )[0],
                    4,
                )

    def test_non_decoder_layers_stay_unquantized(self):
        quant_config = _make_mixed_config()
        layer = torch.nn.Module()
        for name in (
            "model.embed_tokens",
            "model.vision_embed_tokens",
            "model.embed_tokens_extend.audio_embed",
        ):
            with self.subTest(name=name):
                self.assertEqual(quant_config.get_layer_config(layer, name)[0], 16)

    def test_fused_shards_expand_through_packed_modules_mapping(self):
        quant_config = _make_mixed_config(
            extra_config={
                "model.layers.0.self_attn.q_proj.base_layer": {"bits": 8},
                "model.layers.0.self_attn.k_proj.base_layer": {"bits": 8},
                "model.layers.0.self_attn.v_proj.base_layer": {"bits": 8},
            }
        )
        layer = torch.nn.Module()
        self.assertEqual(
            quant_config.get_layer_config(layer, "model.layers.0.self_attn.qkv_proj")[
                0
            ],
            8,
        )

    def test_inconsistent_fused_shards_are_rejected(self):
        quant_config = _make_mixed_config(
            extra_config={
                "model.layers.0.self_attn.q_proj.base_layer": {"bits": 8},
                "model.layers.0.self_attn.k_proj.base_layer": {"bits": 8},
                "model.layers.0.self_attn.v_proj.base_layer": {"bits": 4},
            }
        )
        layer = torch.nn.Module()
        with self.assertRaisesRegex(ValueError, "consistent quant config"):
            quant_config.get_layer_config(layer, "model.layers.0.self_attn.qkv_proj")

    def test_direct_fused_rule_wins_over_shard_expansion(self):
        for extra_config in (
            {
                "model.layers.0.self_attn.qkv_proj.base_layer": {"bits": 8},
                "model.layers.0.self_attn.q_proj.base_layer": {"bits": 4},
                "model.layers.0.self_attn.k_proj.base_layer": {"bits": 4},
                "model.layers.0.self_attn.v_proj.base_layer": {"bits": 4},
            },
            {
                r".*self_attn\.qkv_proj\.base_layer$": {"bits": 8},
                "model.layers.0.self_attn.q_proj.base_layer": {"bits": 4},
                "model.layers.0.self_attn.k_proj.base_layer": {"bits": 4},
                "model.layers.0.self_attn.v_proj.base_layer": {"bits": 4},
            },
        ):
            with self.subTest(extra_config=extra_config):
                quant_config = _make_mixed_config(extra_config=extra_config)
                layer = torch.nn.Module()
                self.assertEqual(
                    quant_config.get_layer_config(
                        layer, "model.layers.0.self_attn.qkv_proj"
                    )[0],
                    8,
                )

    def test_conflicting_normalized_exact_entries_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Conflicting AutoRound extra_config"):
            _make_mixed_config(
                extra_config={
                    "model.layers.0.mlp.down_proj.base_layer": {"bits": 4},
                    "language_model.model.layers.0.mlp.down_proj": {"bits": 8},
                }
            )

    def test_amx_activation_hint_is_logged_once(self):
        quant_config = _make_mixed_config(act_bits=16, act_dynamic=False)
        logger_name = "sglang.srt.layers.quantization.auto_round"
        with patch(
            "sglang.srt.layers.quantization.auto_round._is_cpu_amx_available", True
        ):
            with self.assertLogs(logger_name, level="INFO") as captured:
                quant_config.check_cpu_support(4)
                quant_config.check_cpu_support(8)

        hint_records = [
            record
            for record in captured.records
            if "dynamic per-token INT8 activations" in record.getMessage()
        ]
        self.assertEqual(len(hint_records), 1)
        message = hint_records[0].getMessage()
        self.assertIn("act_bits=16", message)
        self.assertIn("act_dynamic=False", message)
        self.assertIn("act_bits=8, act_dynamic=True", message)

    def test_unsupported_cpu_precisions_are_rejected(self):
        quant_config = _make_mixed_config()
        with patch(
            "sglang.srt.layers.quantization.auto_round._is_cpu_amx_available", True
        ):
            with self.assertRaisesRegex(ValueError, "4-bit experts only"):
                quant_config.check_cpu_support(8, is_moe=True)
            with self.assertRaisesRegex(ValueError, "GPTQ packing format"):
                _make_mixed_config(
                    packing_format="auto_round:auto_awq",
                ).check_cpu_support(8)


class TestAutoRoundW8Conversion(CustomTestCase):
    def test_unpack_preserves_checkpoint_order(self):
        qweight = torch.tensor(
            [[1 | (2 << 8) | (3 << 16) | (4 << 24)]], dtype=torch.int32
        )
        unpacked = _unpack_gptq_uint8(qweight, output_channels=1, input_channels=4)
        self.assertEqual(unpacked.flatten().tolist(), [1, 2, 3, 4])

    def test_symmetric_conversion_requantizes_per_channel(self):
        torch.manual_seed(0)
        output_channels, input_channels, group_size = 64, 256, 128
        groups = input_channels // group_size
        qvalue = torch.randint(
            0, 256, (input_channels, output_channels), dtype=torch.int16
        )
        scales = torch.rand(groups, output_channels, dtype=torch.float16) * 0.01 + 0.001
        scales[1::2].neg_()
        qweight = _pack_gptq_uint8(qvalue.to(torch.uint8), pack_dim=0)
        qzeros = torch.full((groups, output_channels // 4), 127, dtype=torch.int32)

        weight, weight_scale = _gptq_w8_to_channelwise_w8a8(
            qweight, qzeros, scales, is_symmetric=True
        )

        self.assertEqual(tuple(weight.shape), (output_channels, input_channels))
        self.assertEqual(weight.dtype, torch.int8)
        self.assertEqual(tuple(weight_scale.shape), (output_channels, 1))
        self.assertEqual(weight_scale.dtype, torch.float32)

        dequantized = (
            (
                (qvalue.reshape(groups, group_size, output_channels) - 128).float()
                * scales.float()[..., None, :]
            )
            .permute(2, 0, 1)
            .reshape(output_channels, input_channels)
        )
        requantized = weight.float() * weight_scale
        relative_error = (requantized - dequantized).norm() / dequantized.norm()
        self.assertLess(relative_error.item(), 0.02)

        expected_scale = dequantized.abs().amax(dim=1, keepdim=True) / 127.0
        torch.testing.assert_close(weight_scale, expected_scale, rtol=1e-5, atol=1e-8)

    def test_asymmetric_conversion_applies_stored_zero_plus_one(self):
        torch.manual_seed(1)
        output_channels, input_channels, group_size = 32, 128, 128
        groups = input_channels // group_size
        actual_zero = torch.randint(
            64, 192, (groups, output_channels), dtype=torch.int32
        )
        qvalue = torch.randint(
            0, 256, (input_channels, output_channels), dtype=torch.int32
        )
        scales = torch.rand(groups, output_channels, dtype=torch.float16) * 0.01
        qweight = _pack_gptq_uint8(qvalue.to(torch.uint8), pack_dim=0)
        qzeros = _pack_gptq_uint8((actual_zero - 1).to(torch.uint8), pack_dim=1)

        weight, weight_scale = _gptq_w8_to_channelwise_w8a8(
            qweight, qzeros, scales, is_symmetric=False
        )

        dequantized = (
            (
                (
                    qvalue.reshape(groups, group_size, output_channels)
                    - actual_zero[:, None, :]
                ).float()
                * scales.float()[..., None, :]
            )
            .permute(2, 0, 1)
            .reshape(output_channels, input_channels)
        )
        requantized = weight.float() * weight_scale
        relative_error = (requantized - dequantized).norm() / dequantized.norm()
        self.assertLess(relative_error.item(), 0.02)

    def test_invalid_shapes_are_rejected(self):
        qweight = torch.zeros((8, 32), dtype=torch.int32)
        scales = torch.ones((2, 32), dtype=torch.float16)
        qzeros = torch.full((2, 8), 127, dtype=torch.int32)

        with self.assertRaisesRegex(ValueError, "scale shape"):
            _gptq_w8_to_channelwise_w8a8(
                qweight,
                qzeros,
                torch.ones((2, 31), dtype=torch.float16),
                is_symmetric=True,
            )
        with self.assertRaisesRegex(ValueError, "qzeros shape"):
            _gptq_w8_to_channelwise_w8a8(
                qweight,
                torch.full((2, 7), 127, dtype=torch.int32),
                scales,
                is_symmetric=True,
            )
        with self.assertRaisesRegex(ValueError, "output channels"):
            _gptq_w8_to_channelwise_w8a8(
                torch.zeros((8, 30), dtype=torch.int32),
                torch.full((2, 7), 127, dtype=torch.int32),
                torch.ones((2, 30), dtype=torch.float16),
                is_symmetric=True,
            )
        with self.assertRaisesRegex(ValueError, "multiple of 4"):
            _gptq_w8_to_channelwise_w8a8(
                torch.zeros((1, 32), dtype=torch.int32),
                torch.full((2, 8), 127, dtype=torch.int32),
                torch.ones((2, 32), dtype=torch.float16),
                is_symmetric=True,
            )


if __name__ == "__main__":
    unittest.main()
